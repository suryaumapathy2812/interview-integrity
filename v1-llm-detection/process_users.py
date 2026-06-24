"""
Interview session analysis pipeline.

Single entry point for the full workflow:
  1. Transcribe — download audio + Sarvam transcription → report.json
  2. Fix speakers — LLM identifies interviewer/candidate → canonical IDs
  3. NLP analysis — statistical feature extraction → nlp_report.json
  4. LLM analysis — semantic analysis on high-scoring sessions (optional)

Usage:
    # Full pipeline on all sessions
    uv run process_users.py

    # Skip transcription (already done), run remaining stages
    uv run process_users.py --skip-transcription

    # Only run NLP + LLM on existing data
    uv run process_users.py --skip-transcription --skip-speaker-fix --run-llm

    # Resume mode — skip stages that already have output files
    uv run process_users.py --resume

    # Process specific users
    uv run process_users.py --users "Deepanshu Gunwant" "Priyanka A"

    # Control concurrency
    uv run process_users.py --concurrency 8
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

from tqdm import tqdm

from shared import (
    build_session_list,
    download_audio,
    ensure_nlp_data,
    find_session_dir,
    get_openrouter_client,
    get_sarvam_client,
    load_users,
)


# ──────────────────────────────────────────────
# Pipeline stage runners
# ──────────────────────────────────────────────

def run_transcription(session: dict, base_dir: Path, sarvam_client) -> Path:
    """Download audio + transcribe + build report. Returns path to report.json."""
    from pipeline import analyze, transcribe

    user_dir = base_dir / session["user_dir_name"]
    session_dir = user_dir / session["session_id"][:len(session["session_id"])]
    session_dir.mkdir(parents=True, exist_ok=True)

    report_path = session_dir / "report.json"
    if report_path.exists():
        return report_path

    # Download
    download_audio(session["audio_url"], session_dir)

    # Transcribe
    audio_files = list(session_dir.glob("*.mp3")) + list(session_dir.glob("*.wav"))
    if not audio_files:
        raise FileNotFoundError("No audio file found after download")
    sarvam_data = transcribe(audio_files[0], client=sarvam_client, lang=None, speakers=None)

    # Save transcript
    transcript_path = session_dir / "transcript.json"
    with open(transcript_path, "w", encoding="utf-8") as f:
        json.dump(sarvam_data, f, indent=2, ensure_ascii=False)

    # Build report
    report = analyze(sarvam_data)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    return report_path


def run_speaker_fix(session: dict, base_dir: Path, openrouter_client, model: str) -> dict:
    """Identify and remap speakers for one session."""
    from fix_speakers import fix_session_speakers

    session_dir = find_session_dir(base_dir, session["user_name"], session["session_id"])
    if not session_dir:
        raise FileNotFoundError("Session directory not found")

    return fix_session_speakers(session_dir, client=openrouter_client, model=model)


def run_nlp(session: dict, base_dir: Path, use_perplexity: bool) -> dict:
    """Run NLP analysis on one session."""
    from analyze_nlp import analyze as nlp_analyze

    session_dir = find_session_dir(base_dir, session["user_name"], session["session_id"])
    if not session_dir:
        raise FileNotFoundError("Session directory not found")

    report_path = session_dir / "report.json"
    if not report_path.exists():
        raise FileNotFoundError("report.json not found")

    with open(report_path) as f:
        report_data = json.load(f)

    result = nlp_analyze(report_data, use_perplexity=use_perplexity)

    # Save per-session
    nlp_path = session_dir / "nlp_report.json"
    with open(nlp_path, "w") as f:
        json.dump(result, f, indent=2)

    return result


def run_llm(session: dict, base_dir: Path, openrouter_client, model: str) -> dict:
    """Run LLM semantic analysis on one session."""
    from llm_detection import analyze_session

    session_dir = find_session_dir(base_dir, session["user_name"], session["session_id"])
    if not session_dir:
        raise FileNotFoundError("Session directory not found")

    report_path = session_dir / "report.json"
    if not report_path.exists():
        raise FileNotFoundError("report.json not found")

    result = analyze_session(report_path, model=model, client=openrouter_client)

    # Save per-session LLM output for downstream assessment
    llm_path = session_dir / "llm.json"
    with open(llm_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    return result


# ──────────────────────────────────────────────
# Concurrent phase runner
# ──────────────────────────────────────────────

def run_phase(
    phase_name: str,
    sessions: list[dict],
    worker_fn,
    concurrency: int,
    desc: str,
) -> tuple[list[dict], list[dict]]:
    """Run a pipeline phase concurrently with tqdm progress."""
    results = []
    errors = []
    lock = Lock()

    if not sessions:
        return results, errors

    workers = min(concurrency, len(sessions))
    print(f"\n{phase_name} ({len(sessions)} sessions, {workers} workers)")
    print(f"{'─'*60}")

    with ThreadPoolExecutor(max_workers=workers) as pool:
        future_map = {pool.submit(worker_fn, s): s for s in sessions}
        for future in tqdm(as_completed(future_map), total=len(future_map), desc=desc, unit="session"):
            session = future_map[future]
            try:
                result = future.result()
                entry = {
                    "user_name": session["user_name"],
                    "session_id": session["session_id"],
                    "result": result,
                }
                with lock:
                    results.append(entry)
            except Exception as e:
                with lock:
                    errors.append({
                        "user_name": session["user_name"],
                        "session_id": session["session_id"],
                        "error": str(e),
                        "traceback": traceback.format_exc(),
                    })
                tqdm.write(f"  ❌ {session['user_name']}/{session['session_id'][:12]}: {e}")

    print(f"\n  Completed: {len(results)}/{len(sessions)}")
    if errors:
        print(f"  Errors: {len(errors)}")

    return results, errors


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Interview session analysis pipeline: transcribe → fix speakers → NLP → LLM"
    )
    parser.add_argument("--users-json", default="users.json", help="Path to users.json")
    parser.add_argument("--output", default="output/", help="Base output directory")
    parser.add_argument("--users", nargs="*", default=None, help="Only process specific users")
    parser.add_argument("--concurrency", type=int, default=5, help="Concurrent workers (default: 5)")

    # Stage flags
    parser.add_argument("--skip-transcription", action="store_true", help="Skip download + transcription")
    parser.add_argument("--skip-speaker-fix", action="store_true", help="Skip speaker identification")
    parser.add_argument("--skip-nlp", action="store_true", help="Skip NLP analysis")
    parser.add_argument("--run-llm", action="store_true", help="Run LLM analysis on high-scoring sessions")
    parser.add_argument("--resume", action="store_true", help="Skip stages that already have output files")

    # Model/quality flags
    parser.add_argument("--no-perplexity", action="store_true", help="Skip GPT-2 perplexity in NLP")
    parser.add_argument("--speaker-model", default="openai/gpt-4o-mini", help="Model for speaker identification")
    parser.add_argument("--llm-model", default="openai/gpt-4o", help="Model for LLM analysis")
    parser.add_argument("--llm-threshold", type=float, default=0.3, help="NLP score threshold for LLM analysis")

    args = parser.parse_args()
    base_dir = Path(args.output)
    base_dir.mkdir(parents=True, exist_ok=True)

    # Load sessions
    users = load_users(Path(args.users_json))
    sessions = build_session_list(users, base_dir, only_users=args.users)

    if not sessions:
        print("No sessions found.")
        sys.exit(1)

    # Categorize by what needs doing
    need_transcription = [s for s in sessions if not s["has_report"] and not args.skip_transcription]
    need_speaker_fix = [s for s in sessions if s["has_report"] and not args.skip_speaker_fix]
    need_nlp = [s for s in sessions if s["has_report"] and not args.skip_nlp]

    if args.resume:
        need_speaker_fix = [s for s in need_speaker_fix if not s["has_report"]]  # always re-check
        need_nlp = [s for s in need_nlp if not s["has_nlp"]]

    # Print overview
    print(f"\n{'='*60}")
    print(f"Interview Analysis Pipeline")
    print(f"{'='*60}")
    print(f"  Sessions: {len(sessions)}")
    print(f"  Stages:")
    print(f"    1. Transcription: {'skip' if args.skip_transcription else f'{len(need_transcription)} pending'}")
    print(f"    2. Speaker fix:   {'skip' if args.skip_speaker_fix else f'{len(need_speaker_fix)} pending'}")
    print(f"    3. NLP analysis:  {'skip' if args.skip_nlp else f'{len(need_nlp)} pending'}")
    print(f"    4. LLM analysis:  {'run (threshold=' + str(args.llm_threshold) + ')' if args.run_llm else 'skip'}")
    print(f"  Concurrency: {args.concurrency}")
    if not args.no_perplexity and not args.skip_nlp:
        print(f"  Perplexity: enabled (GPT-2)")
    print()

    # ── Stage 1: Transcription ──
    transcription_results = []
    transcription_errors = []
    if not args.skip_transcription and need_transcription:
        sarvam_client = get_sarvam_client()
        worker = lambda s: run_transcription(s, base_dir, sarvam_client)
        transcription_results, transcription_errors = run_phase(
            "Stage 1: Download + Transcribe", need_transcription, worker,
            args.concurrency, "Transcribing",
        )
        # Update session status
        for s in need_transcription:
            s["has_report"] = find_session_dir(base_dir, s["user_name"], s["session_id"]) is not None

    # ── Stage 2: Speaker Fix ──
    speaker_results = []
    speaker_errors = []
    if not args.skip_speaker_fix and need_speaker_fix:
        openrouter_client = get_openrouter_client()
        worker = lambda s: run_speaker_fix(s, base_dir, openrouter_client, args.speaker_model)
        speaker_results, speaker_errors = run_phase(
            "Stage 2: Speaker Identification", need_speaker_fix, worker,
            args.concurrency, "Fixing speakers",
        )

    # ── Stage 3: NLP Analysis ──
    nlp_results = []
    nlp_errors = []
    if not args.skip_nlp:
        # Re-check which need NLP after speaker fix
        if args.resume:
            def _has_nlp(sess):
                sd = find_session_dir(base_dir, sess["user_name"], sess["session_id"])
                return sd and ((sd / "nlp.json").exists() or (sd / "nlp_report.json").exists())
            need_nlp = [s for s in sessions if s["has_report"] and not _has_nlp(s)]

        # Pre-load GPT-2 if perplexity enabled
        if not args.no_perplexity:
            print("\n  Pre-loading GPT-2 for perplexity scoring...")
            from analyze_nlp import _get_perplexity_model
            _get_perplexity_model()

        worker = lambda s: run_nlp(s, base_dir, use_perplexity=not args.no_perplexity)
        nlp_raw, nlp_errors = run_phase(
            "Stage 3: NLP Analysis", need_nlp, worker,
            args.concurrency, "NLP analysis",
        )

        # Extract summary scores
        for entry in nlp_raw:
            result = entry["result"]
            student_sp = result.get("student_speaker", "SPEAKER_01")
            summary = result.get("session_analysis", {}).get(student_sp, {})
            nlp_results.append({
                **entry,
                "avg_composite_score": summary.get("avg_composite_score", 0),
                "max_composite_score": summary.get("max_composite_score", 0),
                "outlier_turn_count": summary.get("outlier_turn_count", 0),
                "substantive_turns": summary.get("substantive_turns", 0),
                "register_gap": summary.get("register_gap", {}),
            })
    # Load existing NLP results if stage was skipped but LLM needs them
    if not nlp_results and args.run_llm:
        for session in sessions:
            session_dir = find_session_dir(base_dir, session["user_name"], session["session_id"])
            if session_dir:
                nlp_path = session_dir / "nlp.json"
                if not nlp_path.exists():
                    nlp_path = session_dir / "nlp_report.json"
                if nlp_path.exists():
                    with open(nlp_path) as f:
                        result = json.load(f)
                    student_sp = result.get("student_speaker", "SPEAKER_01")
                    summary = result.get("session_analysis", {}).get(student_sp, {})
                    nlp_results.append({
                        "user_name": session["user_name"],
                        "session_id": session["session_id"],
                        "avg_composite_score": summary.get("avg_composite_score", 0),
                        "max_composite_score": summary.get("max_composite_score", 0),
                        "outlier_turn_count": summary.get("outlier_turn_count", 0),
                        "substantive_turns": summary.get("substantive_turns", 0),
                        "register_gap": summary.get("register_gap", {}),
                    })
        print(f"\n  Loaded {len(nlp_results)} existing NLP results for LLM filtering")

    # ── Stage 4: LLM Analysis (optional) ──
    llm_results = []
    llm_errors = []
    if args.run_llm and nlp_results:
        candidates = [r for r in nlp_results if r["avg_composite_score"] >= args.llm_threshold]
        if candidates:
            openrouter_client = get_openrouter_client()
            # Build session lookup
            session_map = {s["session_id"]: s for s in sessions}
            llm_sessions = [session_map.get(r["session_id"], r) for r in candidates]
            worker = lambda s: run_llm(s, base_dir, openrouter_client, args.llm_model)
            llm_raw, llm_errors = run_phase(
                "Stage 4: LLM Analysis", llm_sessions, worker,
                min(args.concurrency, 4), "LLM analysis",  # cap concurrency for API
            )
            for entry in llm_raw:
                llm_results.append({
                    "user_name": entry["user_name"],
                    "session_id": entry["session_id"],
                    "llm_verdict": entry["result"].get("verdict", {}),
                })
        else:
            print(f"\n  No sessions above LLM threshold ({args.llm_threshold}). Skipping.")

    # ── Print results ──
    if nlp_results:
        nlp_results.sort(key=lambda r: r["avg_composite_score"], reverse=True)
        print(f"\n{'='*60}")
        print(f"NLP Results (top 20 by avg composite score)")
        print(f"{'='*60}")
        print(f"  {'User':<30} {'Round':>5} {'Band':<6} {'Avg':>6} {'Max':>6} {'Out':>4}")
        print(f"  {'─'*30} {'─'*5} {'─'*6} {'─'*6} {'─'*6} {'─'*4}")
        for r in nlp_results[:20]:
            sm = '🔴' if r['avg_composite_score'] >= 0.3 else '🟡' if r['avg_composite_score'] >= 0.2 else '🟢'
            print(
                f"  {sm} {r['user_name']:<28} {'':>5} {'':>6} "
                f"{r['avg_composite_score']:>5.2f} {r['max_composite_score']:>5.2f} "
                f"{r['outlier_turn_count']:>4}"
            )

    if llm_results:
        print(f"\n  LLM Verdicts:")
        for r in llm_results:
            v = r.get("llm_verdict", {})
            print(f"    {r['user_name']}/{r['session_id'][:8]}: {v.get('assessment', '?')} (score={v.get('confidence_score', 0):.2f})")

    # ── Save consolidated results ──
    output_path = base_dir / "pipeline_results.json"
    consolidated = {
        "metadata": {
            "total_sessions": len(sessions),
            "transcribed": len(transcription_results),
            "transcription_errors": len(transcription_errors),
            "speakers_fixed": len(speaker_results),
            "speaker_errors": len(speaker_errors),
            "nlp_analyzed": len(nlp_results),
            "nlp_errors": len(nlp_errors),
            "llm_analyzed": len(llm_results),
            "llm_errors": len(llm_errors),
        },
        "nlp_results": [{k: v for k, v in r.items() if k != "result"} for r in nlp_results],
        "llm_results": llm_results,
        "errors": {
            "transcription": transcription_errors,
            "speakers": speaker_errors,
            "nlp": nlp_errors,
            "llm": llm_errors,
        },
    }
    with open(output_path, "w") as f:
        json.dump(consolidated, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Results saved to: {output_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
