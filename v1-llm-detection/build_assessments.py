"""Build deterministic final assessments from existing V1 artifacts.

This is a downstream-only layer. It does not transcribe audio, fix speakers,
rebuild reports, rerun NLP, or call an LLM. It reads existing session folders
and writes evidence/assessment files that are stable across runs.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from typing import Any


VERSION = "assessment_rules_v1"
SUBSTANTIVE_WORDS = 15


@dataclass(frozen=True)
class SessionPaths:
    user_name: str
    session_id: str
    session_dir: Path
    report_path: Path
    nlp_path: Path
    llm_path: Path


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def clamp(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    return max(minimum, min(maximum, value))


def safe_mean(values: list[float]) -> float:
    return mean(values) if values else 0.0


def normalize(value: float, low: float, high: float) -> float:
    if high <= low:
        return 0.0
    return clamp((value - low) / (high - low))


def level(score: float) -> str:
    if score >= 0.65:
        return "high"
    if score >= 0.4:
        return "medium"
    return "low"


def discover_sessions(output_dir: Path) -> list[SessionPaths]:
    sessions: list[SessionPaths] = []
    for user_dir in sorted(p for p in output_dir.iterdir() if p.is_dir()):
        for session_dir in sorted(p for p in user_dir.iterdir() if p.is_dir()):
            sessions.append(
                SessionPaths(
                    user_name=user_dir.name.replace("_", " "),
                    session_id=session_dir.name,
                    session_dir=session_dir,
                    report_path=session_dir / "report.json",
                    nlp_path=session_dir / "nlp.json",
                    llm_path=session_dir / "llm.json",
                )
            )
    return sessions


def build_metadata_lookup(users_json: Path | None) -> dict[str, dict[str, Any]]:
    if not users_json or not users_json.exists():
        return {}
    users = json.loads(users_json.read_text(encoding="utf-8"))
    lookup: dict[str, dict[str, Any]] = {}
    for user in users:
        for item in user.get("rounds", []):
            lookup[item["session_id"]] = {
                "user_name": user.get("name"),
                "round_number": item.get("round_number"),
                "round_type": item.get("round_type"),
                "jd_id": item.get("jd_id"),
                "jd_name": item.get("jd_name") or item.get("band"),
                "company": item.get("company"),
                "audio_url": item.get("audio"),
            }
    return lookup


def nlp_turn_map(nlp: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {int(item["turn_id"]): item for item in nlp.get("turns", [])}


def speaker_counts(turns: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for turn in turns:
        speaker = turn.get("speaker", "")
        counts[speaker] = counts.get(speaker, 0) + 1
    return counts


def detect_candidate_speaker(report: dict[str, Any], nlp: dict[str, Any]) -> str | None:
    if nlp.get("student_speaker"):
        return nlp["student_speaker"]

    turns = report.get("turns", [])
    speakers = list(speaker_counts(turns))
    if len(speakers) < 2:
        return None

    first_speaker = turns[0].get("speaker") if turns else None
    non_first = [speaker for speaker in speakers if speaker != first_speaker]
    if len(non_first) == 1:
        return non_first[0]

    candidate_scores = {
        speaker: sum(
            1
            for turn in turns
            if turn.get("speaker") == speaker and turn.get("word_count", 0) >= SUBSTANTIVE_WORDS
        )
        for speaker in speakers
    }
    return max(candidate_scores, key=candidate_scores.get) if candidate_scores else None


def candidate_turns(report: dict[str, Any], candidate_speaker: str | None) -> list[dict[str, Any]]:
    if not candidate_speaker:
        return []
    return [turn for turn in report.get("turns", []) if turn.get("speaker") == candidate_speaker]


def substantive_turns(turns: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [turn for turn in turns if turn.get("word_count", 0) >= SUBSTANTIVE_WORDS]


def data_quality(report: dict[str, Any], nlp: dict[str, Any], paths: SessionPaths) -> dict[str, Any]:
    turns = report.get("turns", [])
    counts = speaker_counts(turns)
    candidate = detect_candidate_speaker(report, nlp)
    c_turns = candidate_turns(report, candidate)
    c_substantive = substantive_turns(c_turns)
    candidate_words = sum(turn.get("word_count", 0) for turn in c_turns)
    issues: list[str] = []

    if not paths.report_path.exists():
        issues.append("missing_report")
    if not paths.nlp_path.exists():
        issues.append("missing_nlp")
    if len(counts) < 2:
        issues.append("only_one_speaker_detected")
    if not candidate:
        issues.append("candidate_speaker_not_found")
    if len(c_substantive) < 2:
        issues.append("too_few_candidate_answers")
    if candidate_words < 80:
        issues.append("too_little_candidate_speech")

    severe = {
        "missing_report",
        "missing_nlp",
        "only_one_speaker_detected",
        "candidate_speaker_not_found",
        "too_few_candidate_answers",
    }
    status = "unusable" if any(issue in severe for issue in issues) else "usable"
    score = 1.0 - min(0.25 * len(issues), 1.0)

    return {
        "status": status,
        "score": round(score, 4),
        "issues": issues,
        "speaker_count": len(counts),
        "candidate_speaker": candidate,
        "candidate_turn_count": len(c_turns),
        "candidate_substantive_turn_count": len(c_substantive),
        "candidate_word_count": candidate_words,
    }


def feature_value(nlp_turn: dict[str, Any], path: tuple[str, ...]) -> float:
    current: Any = nlp_turn
    for key in path:
        if not isinstance(current, dict):
            return 0.0
        current = current.get(key)
    return float(current or 0.0)


def candidate_baseline(candidate_nlp_turns: list[dict[str, Any]]) -> dict[str, float]:
    features = {
        "fk_grade": ("readability", "flesch_kincaid_grade"),
        "dep_depth": ("spacy", "avg_dep_depth"),
        "long_word_ratio": ("stylometry", "long_word_ratio"),
        "rare_word_ratio": ("lexical", "rare_word_ratio"),
        "formality": ("spacy", "formality_score"),
        "sentence_length": ("stylometry", "avg_sentence_length"),
    }
    return {
        name: median([feature_value(turn, path) for turn in candidate_nlp_turns])
        for name, path in features.items()
    }


def signal_statistical(nlp_turn: dict[str, Any]) -> dict[str, Any]:
    composite = float(nlp_turn.get("composite_score", 0.0) or 0.0)
    outlier_score = feature_value(nlp_turn, ("outlier", "outlier_score"))
    z_values = [abs(float(value)) for value in nlp_turn.get("z_scores", {}).values() if isinstance(value, int | float)]
    z_score = normalize(safe_mean(sorted(z_values, reverse=True)[:3]), 1.0, 3.0)
    # Optimized: lowered composite floor from 0.12 to 0.10 for better sensitivity
    score = clamp((normalize(composite, 0.10, 0.28) * 0.55) + (outlier_score * 0.25) + (z_score * 0.2))
    return {"score": round(score, 4), "level": level(score)}


def signal_complexity(nlp_turn: dict[str, Any], baseline: dict[str, float]) -> dict[str, Any]:
    # Optimized: lowered floors for better sensitivity
    diffs = [
        normalize(feature_value(nlp_turn, ("readability", "flesch_kincaid_grade")) - baseline["fk_grade"], 1.5, 10.0),
        normalize(feature_value(nlp_turn, ("spacy", "avg_dep_depth")) - baseline["dep_depth"], 0.2, 1.8),
        normalize(feature_value(nlp_turn, ("stylometry", "long_word_ratio")) - baseline["long_word_ratio"], 0.03, 0.25),
        normalize(feature_value(nlp_turn, ("lexical", "rare_word_ratio")) - baseline["rare_word_ratio"], 0.02, 0.15),
    ]
    score = safe_mean(diffs)
    return {"score": round(score, 4), "level": level(score)}


def signal_polish(nlp_turn: dict[str, Any], baseline: dict[str, float], word_count: int = 0) -> dict[str, Any]:
    diffs = [
        normalize(feature_value(nlp_turn, ("spacy", "formality_score")) - baseline["formality"], 0.05, 0.25),
        normalize(feature_value(nlp_turn, ("stylometry", "avg_sentence_length")) - baseline["sentence_length"], 4.0, 18.0),
        1.0 - normalize(feature_value(nlp_turn, ("proselint", "count")), 1.0, 5.0),
    ]
    score = safe_mean(diffs)
    # Short answers (< 30 words) have inherently noisier NLP features.
    # Dampen the polish signal to reduce false positives from brief turns.
    if word_count < 30:
        score *= 0.5
    return {"score": round(score, 4), "level": level(score)}


def signal_timing(turn: dict[str, Any], all_candidate_turns: list[dict[str, Any]]) -> dict[str, Any]:
    speeds = [
        item.get("word_count", 0) / max(float(item.get("duration", 0.0) or 0.0), 1.0)
        for item in all_candidate_turns
        if item.get("word_count", 0) >= SUBSTANTIVE_WORDS
    ]
    current_speed = turn.get("word_count", 0) / max(float(turn.get("duration", 0.0) or 0.0), 1.0)
    baseline_speed = median(speeds) if speeds else current_speed
    # Lowered floor from 0.5 to 0.3 wps to catch subtler speed deviations
    # common in Indian English speech patterns.
    score = normalize(abs(current_speed - baseline_speed), 0.3, 2.0)
    return {"score": round(score, 4), "level": level(score), "words_per_second": round(current_speed, 4)}


def llm_turn_lookup(llm: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {int(item.get("turn_id")): item for item in llm.get("per_answer_analysis", []) if item.get("turn_id") is not None}


def signal_cross_answer(llm: dict[str, Any]) -> dict[str, Any]:
    """Use LLM's cross-answer analysis to detect sessions with multiple suspicious turns.

    This captures the session-level view that per-answer scoring misses.
    """
    if not llm:
        return {"score": 0.0, "level": "low", "available": False}

    verdict = llm.get("verdict", {})
    suspicious_turns = verdict.get("suspicious_turn_ids", [])
    genuine_turns = verdict.get("genuine_turn_ids", [])
    total_analyzed = len(suspicious_turns) + len(genuine_turns)

    if total_analyzed == 0:
        return {"score": 0.0, "level": "low", "available": True}

    # Ratio of suspicious turns
    suspicious_ratio = len(suspicious_turns) / total_analyzed

    # LLM session confidence score
    llm_confidence = float(verdict.get("confidence_score", 0.0) or 0.0)

    # Combine: high suspicious ratio + high confidence = strong signal
    score = clamp((suspicious_ratio * 0.6) + (llm_confidence * 0.4))

    return {
        "score": round(score, 4),
        "level": level(score),
        "available": True,
        "suspicious_turn_count": len(suspicious_turns),
        "genuine_turn_count": len(genuine_turns),
        "suspicious_ratio": round(suspicious_ratio, 4),
        "llm_session_confidence": round(llm_confidence, 4),
    }


def signal_llm(turn_id: int, llm_by_turn: dict[int, dict[str, Any]]) -> dict[str, Any]:
    item = llm_by_turn.get(turn_id)
    if not item:
        return {"score": 0.0, "level": "low", "available": False}
    origin_score = {
        "real_time": 0.0,
        "recalled_from_memory": 0.35,
        "pre_written_script": 0.75,
        "llm_generated": 1.0,
    }.get(item.get("likely_origin"), 0.0)
    llm_score = float(item.get("llm_score", 0.0) or 0.0)
    mismatch = safe_mean([
        normalize(float(item.get("register_match", 1) or 1), 3.0, 10.0),
        normalize(float(item.get("vocabulary_match", 1) or 1), 3.0, 10.0),
    ])
    score = clamp((llm_score * 0.5) + (origin_score * 0.3) + (mismatch * 0.2))
    return {
        "score": round(score, 4),
        "level": level(score),
        "available": True,
        "likely_origin": item.get("likely_origin"),
        "structural_pattern": item.get("structural_pattern"),
    }


def reason_text(signals: dict[str, dict[str, Any]]) -> list[str]:
    reasons: list[str] = []
    if signals["statistical_anomaly"]["level"] in {"medium", "high"}:
        reasons.append("This answer stands out statistically compared with other turns in the session.")
    if signals["complexity_difference"]["level"] in {"medium", "high"}:
        reasons.append("This answer uses more complex wording than the candidate's usual speech in this session.")
    if signals["polish_difference"]["level"] in {"medium", "high"}:
        reasons.append("This answer appears more polished or structured than the candidate's other answers.")
    if signals["llm_preparedness"].get("available") and signals["llm_preparedness"]["level"] in {"medium", "high"}:
        reasons.append("The existing LLM review marked this answer as prepared or assisted.")
    if signals["timing_context"]["level"] == "high":
        reasons.append("The speaking pace differs from the candidate's other answers.")
    return reasons


def answer_score(signals: dict[str, dict[str, Any]]) -> tuple[float, str]:
    # Optimized: increased LLM weight, decreased statistical weight
    weights = {
        "statistical_anomaly": 0.21,
        "complexity_difference": 0.22,
        "polish_difference": 0.12,
        "timing_context": 0.10,
        "llm_preparedness": 0.36 if signals["llm_preparedness"].get("available") else 0.0,
    }
    total_weight = sum(weights.values()) or 1.0
    score = sum(signals[key]["score"] * weight for key, weight in weights.items()) / total_weight
    major_agreement = sum(
        1 for key in ("statistical_anomaly", "complexity_difference", "polish_difference", "llm_preparedness")
        if signals[key]["level"] in {"medium", "high"}
    )
    # Optimized: lowered watch threshold from 0.4 to 0.28 for better sensitivity
    label = "flagged" if score >= 0.65 and major_agreement >= 2 else "watch" if score >= 0.28 else "normal"
    return round(score, 4), label


def build_answer_assessments(report: dict[str, Any], nlp: dict[str, Any], llm: dict[str, Any], candidate: str) -> list[dict[str, Any]]:
    nlp_by_turn = nlp_turn_map(nlp)
    llm_by_turn = llm_turn_lookup(llm)
    c_turns = candidate_turns(report, candidate)
    c_substantive = substantive_turns(c_turns)
    c_nlp_turns = [nlp_by_turn[turn["id"]] for turn in c_substantive if turn["id"] in nlp_by_turn]
    baseline = candidate_baseline(c_nlp_turns)

    assessments: list[dict[str, Any]] = []
    for turn in c_substantive:
        nlp_turn = nlp_by_turn.get(turn["id"], {})
        signals = {
            "statistical_anomaly": signal_statistical(nlp_turn),
            "complexity_difference": signal_complexity(nlp_turn, baseline),
            "polish_difference": signal_polish(nlp_turn, baseline, turn.get("word_count", 0)),
            "timing_context": signal_timing(turn, c_turns),
            "llm_preparedness": signal_llm(int(turn["id"]), llm_by_turn),
        }
        score, label_name = answer_score(signals)
        assessments.append({
            "turn_id": turn["id"],
            "speaker": turn.get("speaker"),
            "word_count": turn.get("word_count"),
            "text_excerpt": turn.get("text", "")[:500],
            "score": score,
            "label": label_name,
            "signals": signals,
            "plain_reasons": reason_text(signals),
        })
    return assessments


def verdict(data_quality_result: dict[str, Any], answers: list[dict[str, Any]], cross_answer: dict[str, Any] | None = None, llm: dict[str, Any] | None = None) -> dict[str, Any]:
    if data_quality_result["status"] == "unusable":
        return {"label": "unknown", "confidence": "high", "reason": "Session data is not sufficient for assessment."}

    flagged = [item for item in answers if item["label"] == "flagged"]
    watch = [item for item in answers if item["label"] == "watch"]
    total = len(answers)
    flagged_ratio = len(flagged) / total if total else 0.0

    # Use cross-answer signal from LLM if available
    cross_score = (cross_answer or {}).get("score", 0.0)
    suspicious_ratio = (cross_answer or {}).get("suspicious_ratio", 0.0)

    # Check LLM's session-level verdict directly
    llm_session_verdict = (llm or {}).get("verdict", {}).get("assessment", "unknown")

    # Original verdict logic
    if len(flagged) >= 2 and flagged_ratio >= 0.4:
        label_name = "ai_assisted"
    elif len(flagged) >= 1 or len(watch) >= 2:
        label_name = "mixed"
    # NEW: If LLM session-level verdict is mixed and we have at least 1 watch, upgrade to mixed
    elif llm_session_verdict in ("mixed_genuine_and_llm", "pre_prepared_with_llm") and len(watch) >= 1:
        label_name = "mixed"
    # NEW: If cross-answer signal is strong and we have at least 1 watch, upgrade to mixed
    elif suspicious_ratio >= 0.4 and len(watch) >= 1 and cross_score >= 0.4:
        label_name = "mixed"
    else:
        label_name = "genuine"

    confidence = "high" if data_quality_result["score"] >= 0.9 and total >= 4 else "medium" if total >= 2 else "low"
    return {"label": label_name, "confidence": confidence, "reason": "Deterministic rules over correlated evidence."}


def summary(verdict_result: dict[str, Any], answers: list[dict[str, Any]]) -> dict[str, str]:
    flagged = [item for item in answers if item["label"] == "flagged"]
    watch = [item for item in answers if item["label"] == "watch"]
    label_name = verdict_result["label"]
    if label_name == "unknown":
        short = "This session does not contain enough usable candidate speech to assess."
    elif label_name == "genuine":
        short = "The candidate's answers are broadly consistent across this session."
    elif label_name == "mixed":
        short = f"{len(flagged) or len(watch)} answer(s) stand out and should be reviewed with the transcript and audio."
    else:
        short = "Multiple candidate answers stand out across the session."
    return {"short": short, "generated_by": "template_v1"}


def build_session(paths: SessionPaths, metadata: dict[str, Any]) -> dict[str, Any]:
    report = load_json(paths.report_path)
    nlp = load_json(paths.nlp_path)
    llm = load_json(paths.llm_path) if paths.llm_path.exists() else {}
    quality = data_quality(report, nlp, paths)
    candidate = quality.get("candidate_speaker")
    answers = build_answer_assessments(report, nlp, llm, candidate) if candidate else []

    # Compute cross-answer signal from LLM session-level analysis
    cross_answer = signal_cross_answer(llm)

    verdict_result = verdict(quality, answers, cross_answer, llm)
    scores = {
        "session_assistance_score": round(safe_mean([item["score"] for item in answers]), 4),
        "max_answer_score": round(max([item["score"] for item in answers], default=0.0), 4),
        "flagged_answer_ratio": round(sum(1 for item in answers if item["label"] == "flagged") / len(answers), 4) if answers else 0.0,
        "flagged_answer_count": sum(1 for item in answers if item["label"] == "flagged"),
        "watch_answer_count": sum(1 for item in answers if item["label"] == "watch"),
        "cross_answer_score": cross_answer.get("score", 0.0),
        "cross_answer_suspicious_ratio": cross_answer.get("suspicious_ratio", 0.0),
    }
    return {
        "version": VERSION,
        "session_id": paths.session_id,
        "user_name": metadata.get("user_name") or paths.user_name,
        "metadata": metadata,
        "candidate_speaker": candidate,
        "data_quality": quality,
        "scores": scores,
        "verdict": verdict_result,
        "answer_assessments": answers,
        "summary": summary(verdict_result, answers),
    }


def evidence_from_assessment(assessment: dict[str, Any]) -> dict[str, Any]:
    return {
        "version": VERSION,
        "session_id": assessment["session_id"],
        "user_name": assessment["user_name"],
        "candidate_speaker": assessment["candidate_speaker"],
        "data_quality": assessment["data_quality"],
        "answers": assessment["answer_assessments"],
    }


def index_row(assessment: dict[str, Any], session_dir: Path) -> dict[str, Any]:
    return {
        "user_name": assessment["user_name"],
        "session_id": assessment["session_id"],
        "session_dir": str(session_dir),
        "round_number": assessment["metadata"].get("round_number"),
        "round_type": assessment["metadata"].get("round_type"),
        "jd_name": assessment["metadata"].get("jd_name"),
        "company": assessment["metadata"].get("company"),
        "verdict": assessment["verdict"]["label"],
        "confidence": assessment["verdict"]["confidence"],
        "data_quality": assessment["data_quality"]["status"],
        "candidate_answers": assessment["data_quality"]["candidate_substantive_turn_count"],
        "flagged_answers": assessment["scores"]["flagged_answer_count"],
        "watch_answers": assessment["scores"]["watch_answer_count"],
        "session_assistance_score": assessment["scores"]["session_assistance_score"],
        "max_answer_score": assessment["scores"]["max_answer_score"],
    }


def write_summary(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    write_json(output_dir / "assessment_index.json", {"version": VERSION, "sessions": rows})
    if not rows:
        return
    with (output_dir / "assessment_summary.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_all(output_dir: Path, users_json: Path | None) -> dict[str, int]:
    metadata_lookup = build_metadata_lookup(users_json)
    rows: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    for paths in discover_sessions(output_dir):
        if not paths.report_path.exists() or not paths.nlp_path.exists():
            continue
        assessment = build_session(paths, metadata_lookup.get(paths.session_id, {}))
        write_json(paths.session_dir / "evidence.json", evidence_from_assessment(assessment))
        write_json(paths.session_dir / "assessment.json", assessment)
        rows.append(index_row(assessment, paths.session_dir))
        label_name = assessment["verdict"]["label"]
        counts[label_name] = counts.get(label_name, 0) + 1
    write_summary(output_dir, rows)
    counts["total"] = len(rows)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Build deterministic assessments from existing V1 outputs.")
    parser.add_argument("--output", default="output", help="V1 output directory")
    parser.add_argument("--users-json", default="users_all_audio.json", help="Optional metadata JSON")
    args = parser.parse_args()

    output_dir = Path(args.output)
    users_json = Path(args.users_json) if args.users_json else None
    counts = build_all(output_dir, users_json)
    print(json.dumps(counts, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
