"""Run assessment with configurable parameters.

Reads assessment_config.json and applies its parameters to the assessment pipeline.
Outputs verdicts to results/ for evaluation.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Add v1-llm-detection to path for imports
V1_DIR = Path(__file__).parent.parent / "v1-llm-detection"
sys.path.insert(0, str(V1_DIR))

from build_assessments import (
    VERSION,
    SessionPaths,
    build_answer_assessments,
    build_metadata_lookup,
    build_session,
    clamp,
    data_quality,
    detect_candidate_speaker,
    discover_sessions,
    evidence_from_assessment,
    index_row,
    load_json,
    safe_mean,
    signal_complexity,
    signal_llm,
    signal_polish,
    signal_statistical,
    signal_timing,
    summary,
    verdict,
    write_json,
    write_summary,
)


def load_config(config_path: Path) -> dict:
    """Load tunable parameters from config file."""
    return json.loads(config_path.read_text(encoding="utf-8"))


def apply_config_to_signal_statistical(nlp_turn: dict, config: dict) -> dict:
    """Apply config parameters to statistical anomaly signal."""
    params = config["signal_parameters"]
    composite = float(nlp_turn.get("composite_score", 0.0) or 0.0)
    outlier_score = nlp_turn.get("outlier", {}).get("outlier_score", 0.0)

    # Normalize composite score
    if params["statistical_composite_high"] <= params["statistical_composite_low"]:
        norm_composite = 0.0
    else:
        norm_composite = clamp(
            (composite - params["statistical_composite_low"])
            / (params["statistical_composite_high"] - params["statistical_composite_low"])
        )

    # Normalize z-scores
    z_values = [
        abs(float(value))
        for value in nlp_turn.get("z_scores", {}).values()
        if isinstance(value, (int, float))
    ]
    sorted_z = sorted(z_values, reverse=True)[:3]
    mean_top_z = safe_mean(sorted_z)
    if params["statistical_z_ceiling"] <= params["statistical_z_floor"]:
        norm_z = 0.0
    else:
        norm_z = clamp(
            (mean_top_z - params["statistical_z_floor"])
            / (params["statistical_z_ceiling"] - params["statistical_z_floor"])
        )

    score = clamp((norm_composite * 0.55) + (outlier_score * 0.25) + (norm_z * 0.2))
    level = "high" if score >= 0.65 else "medium" if score >= 0.4 else "low"
    return {"score": round(score, 4), "level": level}


def apply_config_to_signal_complexity(nlp_turn: dict, baseline: dict, config: dict) -> dict:
    """Apply config parameters to complexity difference signal."""
    params = config["signal_parameters"]

    def _normalize(value, low, high):
        if high <= low:
            return 0.0
        return clamp((value - low) / (high - low))

    def _feature_value(path):
        current = nlp_turn
        for key in path:
            if not isinstance(current, dict):
                return 0.0
            current = current.get(key)
        return float(current or 0.0)

    diffs = [
        _normalize(
            _feature_value(("readability", "flesch_kincaid_grade")) - baseline["fk_grade"],
            params["complexity_fk_floor"],
            params["complexity_fk_ceiling"],
        ),
        _normalize(
            _feature_value(("spacy", "avg_dep_depth")) - baseline["dep_depth"],
            params["complexity_dep_floor"],
            params["complexity_dep_ceiling"],
        ),
        _normalize(
            _feature_value(("stylometry", "long_word_ratio")) - baseline["long_word_ratio"],
            params["complexity_long_word_floor"],
            params["complexity_long_word_ceiling"],
        ),
        _normalize(
            _feature_value(("lexical", "rare_word_ratio")) - baseline["rare_word_ratio"],
            params["complexity_rare_word_floor"],
            params["complexity_rare_word_ceiling"],
        ),
    ]
    score = safe_mean(diffs)
    level = "high" if score >= 0.65 else "medium" if score >= 0.4 else "low"
    return {"score": round(score, 4), "level": level}


def apply_config_to_signal_polish(nlp_turn: dict, baseline: dict, word_count: int, config: dict) -> dict:
    """Apply config parameters to polish difference signal."""
    params = config["signal_parameters"]

    def _normalize(value, low, high):
        if high <= low:
            return 0.0
        return clamp((value - low) / (high - low))

    def _feature_value(path):
        current = nlp_turn
        for key in path:
            if not isinstance(current, dict):
                return 0.0
            current = current.get(key)
        return float(current or 0.0)

    diffs = [
        _normalize(
            _feature_value(("spacy", "formality_score")) - baseline["formality"],
            params["polish_formality_floor"],
            params["polish_formality_ceiling"],
        ),
        _normalize(
            _feature_value(("stylometry", "avg_sentence_length")) - baseline["sentence_length"],
            params["polish_sentence_length_floor"],
            params["polish_sentence_length_ceiling"],
        ),
        1.0 - _normalize(
            _feature_value(("proselint", "count")),
            params["polish_proselint_floor"],
            params["polish_proselint_ceiling"],
        ),
    ]
    score = safe_mean(diffs)

    # Word count guard
    if word_count < params["polish_word_count_guard"]:
        score *= params["polish_dampening_factor"]

    level = "high" if score >= 0.65 else "medium" if score >= 0.4 else "low"
    return {"score": round(score, 4), "level": level}


def apply_config_to_signal_timing(turn: dict, all_candidate_turns: list, config: dict) -> dict:
    """Apply config parameters to timing context signal."""
    params = config["signal_parameters"]
    min_words = config["data_quality"]["min_substantive_words"]

    speeds = [
        item.get("word_count", 0) / max(float(item.get("duration", 0.0) or 0.0), 1.0)
        for item in all_candidate_turns
        if item.get("word_count", 0) >= min_words
    ]
    current_speed = turn.get("word_count", 0) / max(float(turn.get("duration", 0.0) or 0.0), 1.0)
    baseline_speed = safe_mean(speeds) if speeds else current_speed

    floor = params["timing_normalize_floor"]
    ceiling = params["timing_normalize_ceiling"]
    if ceiling <= floor:
        score = 0.0
    else:
        score = clamp((abs(current_speed - baseline_speed) - floor) / (ceiling - floor))

    level = "high" if score >= 0.65 else "medium" if score >= 0.4 else "low"
    return {"score": round(score, 4), "level": level, "words_per_second": round(current_speed, 4)}


def apply_config_answer_score(signals: dict, config: dict) -> tuple[float, str]:
    """Apply config parameters to answer scoring."""
    weights = config["signal_weights"]
    thresholds = config["answer_thresholds"]

    llm_available = signals["llm_preparedness"].get("available", False)
    effective_weights = {}
    for key, weight in weights.items():
        if key == "llm_preparedness" and not llm_available:
            effective_weights[key] = 0.0
        else:
            effective_weights[key] = weight

    total_weight = sum(effective_weights.values()) or 1.0
    score = sum(signals[key]["score"] * weight for key, weight in effective_weights.items()) / total_weight

    major_agreement = sum(
        1 for key in ("statistical_anomaly", "complexity_difference", "polish_difference", "llm_preparedness")
        if signals[key]["level"] in {"medium", "high"}
    )

    if score >= thresholds["flagged_score"] and major_agreement >= thresholds["flagged_min_major_signals"]:
        label = "flagged"
    elif score >= thresholds["watch_score"]:
        label = "watch"
    else:
        label = "normal"

    return round(score, 4), label


def apply_config_verdict(quality: dict, answers: list, config: dict) -> dict:
    """Apply config parameters to verdict logic."""
    rules = config["verdict_rules"]

    if quality["status"] == "unusable":
        return {"label": "unknown", "confidence": "high", "reason": "Session data is not sufficient for assessment."}

    flagged = [a for a in answers if a["label"] == "flagged"]
    watch = [a for a in answers if a["label"] == "watch"]
    total = len(answers)
    flagged_ratio = len(flagged) / total if total else 0.0

    if len(flagged) >= rules["ai_assisted_min_flagged"] and flagged_ratio >= rules["ai_assisted_min_flagged_ratio"]:
        label = "ai_assisted"
    elif len(flagged) >= rules["mixed_min_flagged"] or len(watch) >= rules["mixed_min_watch"]:
        label = "mixed"
    else:
        label = "genuine"

    confidence = "high" if quality["score"] >= 0.9 and total >= 4 else "medium" if total >= 2 else "low"
    return {"label": label, "confidence": confidence, "reason": "Deterministic rules over correlated evidence."}


def build_session_with_config(paths: SessionPaths, metadata: dict, config: dict) -> dict:
    """Build assessment for a session using config parameters."""
    report = load_json(paths.report_path)
    nlp = load_json(paths.nlp_path)
    llm = load_json(paths.llm_path) if paths.llm_path.exists() else {}
    quality = data_quality(report, nlp, paths)
    candidate = quality.get("candidate_speaker")

    if not candidate:
        answers = []
    else:
        # Build answer assessments with config parameters
        nlp_by_turn = {int(item["turn_id"]): item for item in nlp.get("turns", [])}
        llm_by_turn = {
            int(item.get("turn_id")): item
            for item in llm.get("per_answer_analysis", [])
            if item.get("turn_id") is not None
        }

        # Get candidate turns
        turns = report.get("turns", [])
        c_turns = [t for t in turns if t.get("speaker") == candidate]
        min_words = config["data_quality"]["min_substantive_words"]
        c_substantive = [t for t in c_turns if t.get("word_count", 0) >= min_words]

        # Build baseline from NLP data
        from statistics import median

        c_nlp_turns = [nlp_by_turn[t["id"]] for t in c_substantive if t["id"] in nlp_by_turn]

        def _feature_value(nlp_turn, path):
            current = nlp_turn
            for key in path:
                if not isinstance(current, dict):
                    return 0.0
                current = current.get(key)
            return float(current or 0.0)

        baseline = {
            "fk_grade": median([_feature_value(t, ("readability", "flesch_kincaid_grade")) for t in c_nlp_turns]) if c_nlp_turns else 0.0,
            "dep_depth": median([_feature_value(t, ("spacy", "avg_dep_depth")) for t in c_nlp_turns]) if c_nlp_turns else 0.0,
            "long_word_ratio": median([_feature_value(t, ("stylometry", "long_word_ratio")) for t in c_nlp_turns]) if c_nlp_turns else 0.0,
            "rare_word_ratio": median([_feature_value(t, ("lexical", "rare_word_ratio")) for t in c_nlp_turns]) if c_nlp_turns else 0.0,
            "formality": median([_feature_value(t, ("spacy", "formality_score")) for t in c_nlp_turns]) if c_nlp_turns else 0.0,
            "sentence_length": median([_feature_value(t, ("stylometry", "avg_sentence_length")) for t in c_nlp_turns]) if c_nlp_turns else 0.0,
        }

        # Build assessments
        answers = []
        for turn in c_substantive:
            nlp_turn = nlp_by_turn.get(turn["id"], {})
            signals = {
                "statistical_anomaly": apply_config_to_signal_statistical(nlp_turn, config),
                "complexity_difference": apply_config_to_signal_complexity(nlp_turn, baseline, config),
                "polish_difference": apply_config_to_signal_polish(nlp_turn, baseline, turn.get("word_count", 0), config),
                "timing_context": apply_config_to_signal_timing(turn, c_turns, config),
                "llm_preparedness": signal_llm(int(turn["id"]), llm_by_turn),
            }
            score, label_name = apply_config_answer_score(signals, config)
            answers.append({
                "turn_id": turn["id"],
                "speaker": turn.get("speaker"),
                "word_count": turn.get("word_count"),
                "text_excerpt": turn.get("text", "")[:500],
                "score": score,
                "label": label_name,
                "signals": signals,
                "plain_reasons": [],  # Simplified for config runs
            })

    verdict_result = apply_config_verdict(quality, answers, config)
    scores = {
        "session_assistance_score": round(safe_mean([a["score"] for a in answers]), 4),
        "max_answer_score": round(max([a["score"] for a in answers], default=0.0), 4),
        "flagged_answer_ratio": round(sum(1 for a in answers if a["label"] == "flagged") / len(answers), 4) if answers else 0.0,
        "flagged_answer_count": sum(1 for a in answers if a["label"] == "flagged"),
        "watch_answer_count": sum(1 for a in answers if a["label"] == "watch"),
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


def run_with_config(output_dir: Path, config_path: Path, results_dir: Path | None = None) -> dict:
    """Run assessment with config parameters and return results."""
    config = load_config(config_path)

    if results_dir is None:
        results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Save config used
    write_json(results_dir / "config_used.json", config)

    # Discover sessions
    sessions = discover_sessions(output_dir)
    metadata_lookup = build_metadata_lookup(Path(__file__).parent.parent / "v1-llm-detection" / "users_all_audio.json")

    # Build assessments
    rows = []
    verdicts = {}
    for paths in sessions:
        if not paths.report_path.exists() or not paths.nlp_path.exists():
            continue
        assessment = build_session_with_config(paths, metadata_lookup.get(paths.session_id, {}), config)
        row = index_row(assessment, paths.session_dir)
        rows.append(row)
        label = assessment["verdict"]["label"]
        verdicts[label] = verdicts.get(label, 0) + 1

        # Save per-session result
        session_results = results_dir / paths.user_name.replace(" ", "_")
        session_results.mkdir(parents=True, exist_ok=True)
        write_json(session_results / f"{paths.session_id}.json", assessment)

    # Write summary
    write_summary(results_dir, rows)

    verdicts["total"] = len(rows)
    return verdicts


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run assessment with config parameters")
    parser.add_argument("--config", default="assessment_config.json", help="Config file path")
    parser.add_argument("--output", default="../v1-llm-detection/output", help="V1 output directory")
    parser.add_argument("--results", default=None, help="Results output directory")
    args = parser.parse_args()

    config_path = Path(args.config)
    output_dir = Path(args.output)
    results_dir = Path(args.results) if args.results else None

    verdicts = run_with_config(output_dir, config_path, results_dir)
    print(json.dumps(verdicts, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
