"""Evaluate assessment results against LLM verdicts.

Computes metrics to measure how well deterministic verdicts match LLM labels.
"""

from __future__ import annotations

import json
from pathlib import Path
from collections import Counter


def load_llm_labels(output_dir: Path) -> dict[str, dict]:
    """Load LLM verdicts from llm.json files in output directory."""
    labels = {}
    for user_dir in output_dir.iterdir():
        if not user_dir.is_dir():
            continue
        for session_dir in user_dir.iterdir():
            if not session_dir.is_dir():
                continue
            llm_path = session_dir / "llm.json"
            if not llm_path.exists():
                continue
            llm = json.loads(llm_path.read_text(encoding="utf-8"))
            verdict = llm.get("verdict", {})
            labels[session_dir.name] = {
                "llm_verdict": verdict.get("assessment", "unknown"),
                "llm_score": verdict.get("confidence_score", 0.0),
                "suspicious_turns": verdict.get("suspicious_turn_ids", []),
                "genuine_turns": verdict.get("genuine_turn_ids", []),
            }
    return labels


def load_deterministic_verdicts(results_dir: Path) -> dict[str, dict]:
    """Load deterministic verdicts from results directory."""
    verdicts = {}
    for user_dir in results_dir.iterdir():
        if not user_dir.is_dir() or user_dir.name == "config_used.json":
            continue
        for session_file in user_dir.glob("*.json"):
            if session_file.name == "config_used.json":
                continue
            session = json.loads(session_file.read_text(encoding="utf-8"))
            verdicts[session["session_id"]] = {
                "deterministic_verdict": session["verdict"]["label"],
                "session_score": session["scores"]["session_assistance_score"],
                "max_answer_score": session["scores"]["max_answer_score"],
                "watch_count": session["scores"]["watch_answer_count"],
                "flagged_count": session["scores"]["flagged_answer_count"],
            }
    return verdicts


def normalize_llm_verdict(verdict: str) -> str:
    """Normalize LLM verdict to match our categories."""
    if verdict in ("genuine",):
        return "genuine"
    elif verdict in ("mixed_genuine_and_llm", "pre_prepared_with_llm"):
        return "mixed"
    elif verdict in ("llm_primary",):
        return "ai_assisted"
    else:
        return "unknown"


def compute_metrics(llm_labels: dict, deterministic: dict) -> dict:
    """Compute evaluation metrics."""
    # Find sessions with both LLM and deterministic verdicts
    common_sessions = set(llm_labels.keys()) & set(deterministic.keys())

    if not common_sessions:
        return {"error": "No common sessions found"}

    # Compute agreement
    agreements = 0
    total = len(common_sessions)
    llm_mixed_but_deterministic_genuine = 0
    llm_mixed_but_deterministic_anything = Counter()

    for sid in common_sessions:
        llm_norm = normalize_llm_verdict(llm_labels[sid]["llm_verdict"])
        det_verdict = deterministic[sid]["deterministic_verdict"]

        if llm_norm == det_verdict:
            agreements += 1

        if llm_norm == "mixed":
            llm_mixed_but_deterministic_anything[det_verdict] += 1
            if det_verdict == "genuine":
                llm_mixed_but_deterministic_genuine += 1

    agreement_rate = agreements / total if total else 0.0

    # Compute score separation
    genuine_scores = []
    mixed_scores = []
    ai_scores = []

    for sid in common_sessions:
        llm_norm = normalize_llm_verdict(llm_labels[sid]["llm_verdict"])
        score = deterministic[sid]["session_score"]

        if llm_norm == "genuine":
            genuine_scores.append(score)
        elif llm_norm == "mixed":
            mixed_scores.append(score)
        elif llm_norm == "ai_assisted":
            ai_scores.append(score)

    from statistics import mean, stdev

    genuine_mean = mean(genuine_scores) if genuine_scores else 0.0
    mixed_mean = mean(mixed_scores) if mixed_scores else 0.0
    separation = mixed_mean - genuine_mean

    return {
        "total_sessions": total,
        "agreement_count": agreements,
        "agreement_rate": round(agreement_rate, 4),
        "llm_mixed_but_deterministic_genuine": llm_mixed_but_deterministic_genuine,
        "llm_mixed_verdict_distribution": dict(llm_mixed_but_deterministic_anything),
        "score_statistics": {
            "genuine_mean_score": round(genuine_mean, 4),
            "mixed_mean_score": round(mixed_mean, 4),
            "separation": round(separation, 4),
        },
        "verdict_counts": {
            "genuine": len(genuine_scores),
            "mixed": len(mixed_scores),
            "ai_assisted": len(ai_scores),
        },
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate assessment results")
    parser.add_argument("--results", default="results", help="Results directory")
    parser.add_argument("--output", default="../v1-llm-detection/output", help="V1 output directory")
    args = parser.parse_args()

    results_dir = Path(args.results)
    output_dir = Path(args.output)

    llm_labels = load_llm_labels(output_dir)
    deterministic = load_deterministic_verdicts(results_dir)

    metrics = compute_metrics(llm_labels, deterministic)
    print(json.dumps(metrics, indent=2, sort_keys=True))

    # Save metrics
    metrics_path = results_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    print(f"\nMetrics saved to: {metrics_path}")


if __name__ == "__main__":
    main()
