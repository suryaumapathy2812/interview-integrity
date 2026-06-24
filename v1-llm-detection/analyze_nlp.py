"""
NLP analysis on conversation transcripts to detect style anomalies.

Computes per-turn features (stylometry, readability, perplexity, formality,
lexical sophistication, disfluency), detects statistical outliers via
Isolation Forest and z-scores, and produces per-turn + session-level scores.

No hardcoded word lists or phrase lists — every signal is computed
statistically from the speaker's own data using established NLP libraries.

Usage:
    python analyze_nlp.py --input ./output/session/report.json
    python analyze_nlp.py --input ./output/session/report.json --no-perplexity
    python analyze_nlp.py --input ./output/session/report.json --output out.json

Requirements:
    pip install nltk textstat proselint spacy scikit-learn wordfreq transformers torch
    python -m spacy download en_core_web_sm
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import warnings
from pathlib import Path

import numpy as np
import nltk
import spacy
import textstat
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from proselint.checks import __register__
from proselint.registry import CheckRegistry
from proselint.tools import LintFile
from sklearn.ensemble import IsolationForest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from wordfreq import zipf_frequency

warnings.filterwarnings("ignore", category=FutureWarning)

# ──────────────────────────────────────────────
# Setup
# ──────────────────────────────────────────────


def _ensure_nltk_data():
    for resource in ["punkt_tab", "averaged_perceptron_tagger_eng", "stopwords"]:
        try:
            nltk.data.find(
                f"tokenizers/{resource}" if "punkt" in resource else f"corpora/{resource}"
            )
        except LookupError:
            nltk.download(resource, quiet=True)


_ensure_nltk_data()

_check_registry = CheckRegistry()
_check_registry.register_many(__register__)

STOP_WORDS = set(stopwords.words("english"))

_nlp = None


def _get_nlp():
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("en_core_web_sm")
    return _nlp


_perplexity_model = None
_perplexity_tokenizer = None


def _get_perplexity_model():
    global _perplexity_model, _perplexity_tokenizer
    if _perplexity_model is None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print("  Loading GPT-2 for perplexity scoring (one-time)...")
        _perplexity_tokenizer = AutoTokenizer.from_pretrained("gpt2")
        _perplexity_model = AutoModelForCausalLM.from_pretrained("gpt2")
        _perplexity_model.eval()
    return _perplexity_model, _perplexity_tokenizer


# ──────────────────────────────────────────────
# Feature extraction — stylometry (existing, improved)
# ──────────────────────────────────────────────


def extract_stylometry(text: str) -> dict:
    tokens = word_tokenize(text.lower())
    tokens = [t for t in tokens if t.isalpha()]
    if not tokens:
        return {
            "type_token_ratio": 0,
            "hapax_ratio": 0,
            "avg_sentence_length": 0,
            "sentence_length_variance": 0,
            "function_word_ratio": 0,
            "long_word_ratio": 0,
        }

    unique_tokens = set(tokens)
    type_token_ratio = len(unique_tokens) / len(tokens)

    freq_dist = nltk.FreqDist(tokens)
    hapax_count = sum(1 for _, c in freq_dist.items() if c == 1)
    hapax_ratio = hapax_count / len(unique_tokens) if unique_tokens else 0

    sentences = sent_tokenize(text)
    sent_lengths = [len(word_tokenize(s)) for s in sentences if word_tokenize(s)]
    avg_sentence_length = (
        sum(sent_lengths) / len(sent_lengths) if sent_lengths else 0
    )
    if len(sent_lengths) > 1:
        mean = avg_sentence_length
        sentence_length_variance = sum((x - mean) ** 2 for x in sent_lengths) / (
            len(sent_lengths) - 1
        )
    else:
        sentence_length_variance = 0

    function_count = sum(1 for t in tokens if t in STOP_WORDS)
    function_word_ratio = function_count / len(tokens)

    long_word_count = sum(1 for t in tokens if len(t) > 6)
    long_word_ratio = long_word_count / len(tokens)

    return {
        "type_token_ratio": round(type_token_ratio, 4),
        "hapax_ratio": round(hapax_ratio, 4),
        "avg_sentence_length": round(avg_sentence_length, 2),
        "sentence_length_variance": round(sentence_length_variance, 2),
        "function_word_ratio": round(function_word_ratio, 4),
        "long_word_ratio": round(long_word_ratio, 4),
    }


# ──────────────────────────────────────────────
# Feature extraction — readability (existing)
# ──────────────────────────────────────────────


def extract_readability(text: str) -> dict:
    if not text.strip():
        return {
            "flesch_reading_ease": 0,
            "flesch_kincaid_grade": 0,
            "avg_syllables_per_word": 0,
        }
    return {
        "flesch_reading_ease": round(textstat.flesch_reading_ease(text), 2),
        "flesch_kincaid_grade": round(textstat.flesch_kincaid_grade(text), 2),
        "avg_syllables_per_word": round(textstat.avg_syllables_per_word(text), 2),
    }


# ──────────────────────────────────────────────
# Feature extraction — proselint (existing)
# ──────────────────────────────────────────────


def extract_proselint(text: str) -> dict:
    try:
        results = LintFile("input", text).lint()
        issues = []
        for r in results:
            issues.append(
                {
                    "check": r.check_result.check_path,
                    "message": r.check_result.message,
                    "span": list(r.check_result.span),
                }
            )
        return {"count": len(issues), "issues": issues}
    except Exception:
        return {"count": 0, "issues": []}


# ──────────────────────────────────────────────
# Feature extraction — spacy-based (NEW)
# ──────────────────────────────────────────────


def extract_spacy_features(text: str) -> dict:
    """Extract POS-based formality, disfluency ratio, and syntax depth via spacy."""
    nlp = _get_nlp()
    doc = nlp(text)

    pos_counts: dict[str, int] = {}
    for token in doc:
        if not token.is_space:
            pos_counts[token.pos_] = pos_counts.get(token.pos_, 0) + 1

    total = sum(pos_counts.values())
    if total == 0:
        return {
            "formality_score": 0.5,
            "disfluency_ratio": 0,
            "avg_dep_depth": 0,
            "noun_ratio": 0,
            "verb_ratio": 0,
        }

    # POS-based formality (Heylighen & Dewaele, 2002)
    formal = sum(pos_counts.get(p, 0) for p in ("NOUN", "ADJ", "ADP", "DET"))
    informal = sum(pos_counts.get(p, 0) for p in ("PRON", "VERB", "ADV", "INTJ"))
    formality_score = (formal - informal + total) / (2 * total)

    # Disfluency: ratio of interjections (fillers, hesitations)
    intj_count = pos_counts.get("INTJ", 0)
    disfluency_ratio = intj_count / total

    # Syntactic complexity: average dependency tree depth
    dep_depths = []
    for sent in doc.sents:
        for token in sent:
            depth = 0
            current = token
            while current.head != current:
                depth += 1
                current = current.head
            dep_depths.append(depth)
    avg_dep_depth = sum(dep_depths) / len(dep_depths) if dep_depths else 0

    noun_ratio = pos_counts.get("NOUN", 0) / total
    verb_ratio = pos_counts.get("VERB", 0) / total

    return {
        "formality_score": round(formality_score, 4),
        "disfluency_ratio": round(disfluency_ratio, 4),
        "avg_dep_depth": round(avg_dep_depth, 2),
        "noun_ratio": round(noun_ratio, 4),
        "verb_ratio": round(verb_ratio, 4),
    }


# ──────────────────────────────────────────────
# Feature extraction — perplexity (NEW, lazy-loaded)
# ──────────────────────────────────────────────


def extract_perplexity(text: str) -> dict:
    """Compute GPT-2 perplexity. Lower = more predictable = more likely LLM."""
    import torch

    model, tokenizer = _get_perplexity_model()
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
    ppl = torch.exp(outputs.loss).item()
    return {"perplexity": round(ppl, 2)}


# ──────────────────────────────────────────────
# Feature extraction — lexical sophistication (NEW)
# ──────────────────────────────────────────────


def extract_lexical_sophistication(text: str) -> dict:
    """Zipf frequency-based lexical sophistication.
    Lower avg Zipf = rarer words = more formal/sophisticated.
    Uses corpus frequency data — no word lists.
    """
    tokens = [t.lower() for t in text.split() if t.isalpha()]
    if not tokens:
        return {"avg_zipf": 0, "rare_word_ratio": 0}

    zipf_scores = [zipf_frequency(w, "en") for w in tokens]
    avg_zipf = sum(zipf_scores) / len(zipf_scores)

    # Words with Zipf < 4 are relatively rare in everyday English
    rare_count = sum(1 for z in zipf_scores if z < 4.0)
    rare_word_ratio = rare_count / len(tokens)

    return {
        "avg_zipf": round(avg_zipf, 4),
        "rare_word_ratio": round(rare_word_ratio, 4),
    }


# ──────────────────────────────────────────────
# Structural uniformity — TF-IDF cosine similarity (NEW)
# ──────────────────────────────────────────────


def compute_structure_uniformity(answers: list[str]) -> dict:
    """Measure how similar answers are to each other in word patterns.
    High similarity = template-like (suspicious for LLM).
    """
    if len(answers) < 3:
        return {"avg_pairwise_similarity": 0, "template_score": 0}

    tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=500, stop_words="english")
    try:
        matrix = tfidf.fit_transform(answers)
    except ValueError:
        return {"avg_pairwise_similarity": 0, "template_score": 0}

    sim = cosine_similarity(matrix)
    n = len(answers)
    avg_sim = (sim.sum() - n) / (n * (n - 1))

    return {
        "avg_pairwise_similarity": round(float(avg_sim), 4),
        "template_score": round(float(avg_sim), 4),
    }


# ──────────────────────────────────────────────
# Register gap — POS formality comparison (NEW)
# ──────────────────────────────────────────────


def compute_register_gap(turns_data: list[dict], speaker: str) -> dict:
    """Compare register of conversational turns vs. substantive answers.
    Large gap = answers came from a different source than speaker's natural voice.
    """
    conv_turns = [
        t for t in turns_data if t["speaker"] == speaker and t["word_count"] < 15
    ]
    sub_turns = [
        t for t in turns_data if t["speaker"] == speaker and t["word_count"] >= 60
    ]

    if len(conv_turns) < 2 or len(sub_turns) < 2:
        return {"formality_gap": 0, "zipf_gap": 0, "fk_grade_gap": 0, "register_shift_score": 0}

    def median_of(key_chain: list[str], turns: list[dict]) -> float:
        vals = []
        for t in turns:
            obj = t
            for k in key_chain:
                obj = obj[k]
            vals.append(obj)
        vals.sort()
        n = len(vals)
        if n % 2 == 1:
            return vals[n // 2]
        return (vals[n // 2 - 1] + vals[n // 2]) / 2

    formality_gap = median_of(["spacy", "formality_score"], sub_turns) - median_of(
        ["spacy", "formality_score"], conv_turns
    )
    zipf_gap = median_of(["lexical", "avg_zipf"], conv_turns) - median_of(
        ["lexical", "avg_zipf"], sub_turns
    )  # positive = substantive uses rarer words
    fk_gap = median_of(["readability", "flesch_kincaid_grade"], sub_turns) - median_of(
        ["readability", "flesch_kincaid_grade"], conv_turns
    )

    # Normalize each gap to 0-1 range (rough: divide by expected max)
    register_shift_score = (
        min(1.0, max(0, formality_gap * 3))
        + min(1.0, max(0, zipf_gap / 1.5))
        + min(1.0, max(0, fk_gap / 8))
    ) / 3

    return {
        "formality_gap": round(formality_gap, 4),
        "zipf_gap": round(zipf_gap, 4),
        "fk_grade_gap": round(fk_gap, 4),
        "register_shift_score": round(register_shift_score, 4),
    }


# ──────────────────────────────────────────────
# Z-scores and outlier detection (NEW)
# ──────────────────────────────────────────────

# Features to include in outlier detection, mapped to their extraction path
_OUTLIER_FEATURES = [
    ("stylometry", "type_token_ratio"),
    ("stylometry", "long_word_ratio"),
    ("stylometry", "function_word_ratio"),
    ("stylometry", "avg_sentence_length"),
    ("stylometry", "sentence_length_variance"),
    ("readability", "flesch_kincaid_grade"),
    ("readability", "avg_syllables_per_word"),
    ("spacy", "formality_score"),
    ("spacy", "disfluency_ratio"),
    ("lexical", "avg_zipf"),
    ("lexical", "rare_word_ratio"),
]

# Perplexity is stored at td["perplexity"]["perplexity"], not nested like others
_PERPLEXITY_KEY = ("perplexity", "perplexity")


def _get_feature_val(turn: dict, group: str, key: str) -> float:
    return turn.get(group, {}).get(key, 0)


def compute_z_scores(turns_data: list[dict], speaker: str) -> list[dict]:
    """Compute per-turn z-scores for each feature relative to speaker's own distribution."""
    speaker_turns = [t for t in turns_data if t["speaker"] == speaker]
    if len(speaker_turns) < 3:
        return []

    # Build arrays per feature (including perplexity if available)
    all_features = list(_OUTLIER_FEATURES)
    has_perplexity = any("perplexity" in t and isinstance(t["perplexity"], dict) for t in speaker_turns)
    if has_perplexity:
        all_features.append(_PERPLEXITY_KEY)

    feature_vals: dict[str, list[float]] = {f"{g}.{k}": [] for g, k in all_features}
    for t in speaker_turns:
        for g, k in all_features:
            feature_vals[f"{g}.{k}"].append(_get_feature_val(t, g, k))

    # Compute mean and std per feature
    stats = {}
    for key, vals in feature_vals.items():
        arr = np.array(vals)
        m = float(np.mean(arr))
        s = float(np.std(arr))
        if s == 0:
            s = 1.0
        stats[key] = {"mean": m, "std": s}

    # Compute z-scores for each turn
    z_scores_per_turn = []
    for t in speaker_turns:
        z = {}
        for g, k in all_features:
            key = f"{g}.{k}"
            val = _get_feature_val(t, g, k)
            z[f"{g}_{k}"] = round((val - stats[key]["mean"]) / stats[key]["std"], 4)
        z_scores_per_turn.append({"turn_id": t["turn_id"], "z_scores": z})

    return z_scores_per_turn


def detect_outliers_isolation_forest(
    turns_data: list[dict], speaker: str
) -> dict[int, dict]:
    """Use Isolation Forest to find multivariate outlier turns.
    No hardcoded thresholds — the model learns what's normal from the speaker.
    """
    speaker_turns = [t for t in turns_data if t["speaker"] == speaker]
    if len(speaker_turns) < 5:
        return {}

    # Build feature matrix (including perplexity if available)
    all_features = list(_OUTLIER_FEATURES)
    has_perplexity = any("perplexity" in t and isinstance(t["perplexity"], dict) for t in speaker_turns)
    if has_perplexity:
        all_features.append(_PERPLEXITY_KEY)

    feature_matrix = []
    for t in speaker_turns:
        row = [_get_feature_val(t, g, k) for g, k in all_features]
        feature_matrix.append(row)

    X = np.array(feature_matrix)

    # contamination = expected fraction of outliers
    clf = IsolationForest(contamination=0.2, random_state=42, n_estimators=100)
    preds = clf.fit_predict(X)
    decisions = clf.decision_function(X)

    results = {}
    for i, t in enumerate(speaker_turns):
        results[t["turn_id"]] = {
            "is_outlier": bool(preds[i] == -1),
            "outlier_score": round(float(decisions[i]), 4),
        }

    return results

_FEATURE_WEIGHTS = {
    "perplexity_z": 0.20,
    "formality_z": 0.18,
    "register_gap": 0.15,
    "template_score": 0.12,
    "zipf_z": 0.12,
    "disfluency_z": 0.10,
    "sentence_variance_z": 0.08,
    "isolation_outlier": 0.05,
}


def _z_contribution(z_scores: dict, key: str, direction: str, scale: float = 2.0) -> float:
    """Extract a 0-1 contribution from a z-score.
    direction='low' means lower-than-baseline is suspicious (e.g. perplexity, disfluency).
    direction='high' means higher-than-baseline is suspicious (e.g. formality).
    scale controls sensitivity: z=scale maps to contribution=1.0.
    """
    z = z_scores.get(key, 0)
    if direction == "low":
        return min(1.0, max(0, -z / scale))
    else:
        return min(1.0, max(0, z / scale))


def compute_composite_score(
    z_scores: dict[str, float],
    is_outlier: bool,
    register_shift_score: float,
    template_score: float,
) -> float:
    """Combine all signals into a single 0-1 suspicion score."""
    score = 0.0

    # Perplexity: lower than baseline → suspicious (LLM text is more predictable)
    score += _z_contribution(z_scores, "perplexity_perplexity", "low") * _FEATURE_WEIGHTS["perplexity_z"]

    # Formality: higher than baseline → suspicious (LLM text is more formal)
    score += _z_contribution(z_scores, "spacy_formality_score", "high") * _FEATURE_WEIGHTS["formality_z"]

    # Zipf: lower than baseline (rarer words) → suspicious
    score += _z_contribution(z_scores, "lexical_avg_zipf", "low") * _FEATURE_WEIGHTS["zipf_z"]

    # Disfluency: lower than baseline → suspicious (reading, not thinking)
    score += _z_contribution(z_scores, "spacy_disfluency_ratio", "low") * _FEATURE_WEIGHTS["disfluency_z"]

    # Sentence variance: lower than baseline → suspicious (too uniform)
    score += _z_contribution(z_scores, "stylometry_sentence_length_variance", "low") * _FEATURE_WEIGHTS["sentence_variance_z"]

    # Register gap and template score are already 0-1
    score += min(1.0, register_shift_score) * _FEATURE_WEIGHTS["register_gap"]
    score += min(1.0, template_score) * _FEATURE_WEIGHTS["template_score"]

    # Isolation forest outlier
    score += (1.0 if is_outlier else 0.0) * _FEATURE_WEIGHTS["isolation_outlier"]

    return round(min(1.0, score), 4)



# ──────────────────────────────────────────────
# Baseline computation (improved from existing)
# ──────────────────────────────────────────────


def _percentile(sorted_vals: list[float], p: int) -> float:
    if not sorted_vals:
        return 0
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    k = (len(sorted_vals) - 1) * (p / 100)
    f = int(k)
    c = min(f + 1, len(sorted_vals) - 1)
    return sorted_vals[f] + (k - f) * (sorted_vals[c] - sorted_vals[f])


def compute_speaker_baselines(turns_data: list[dict]) -> dict:
    """Compute median/p25/p75 for all features per speaker."""
    speaker_features: dict[str, dict[str, list[float]]] = {}

    for td in turns_data:
        sp = td["speaker"]
        if sp not in speaker_features:
            speaker_features[sp] = {}

        # Stylometry
        for k, v in td.get("stylometry", {}).items():
            speaker_features[sp].setdefault(f"stylometry_{k}", []).append(v)

        # Readability
        for k, v in td.get("readability", {}).items():
            speaker_features[sp].setdefault(f"readability_{k}", []).append(v)

        # Proselint
        speaker_features[sp].setdefault("proselint_count", []).append(
            td.get("proselint", {}).get("count", 0)
        )

        # Spacy
        for k, v in td.get("spacy", {}).items():
            speaker_features[sp].setdefault(f"spacy_{k}", []).append(v)

        # Lexical
        for k, v in td.get("lexical", {}).items():
            speaker_features[sp].setdefault(f"lexical_{k}", []).append(v)

    baselines = {}
    for sp, features in speaker_features.items():
        baselines[sp] = {}
        for key, values in features.items():
            s = sorted(values)
            baselines[sp][key] = {
                "median": round(_percentile(s, 50), 4),
                "p25": round(_percentile(s, 25), 4),
                "p75": round(_percentile(s, 75), 4),
                "mean": round(sum(s) / len(s), 4),
                "std": round(
                    (sum((x - sum(s) / len(s)) ** 2 for x in s) / len(s)) ** 0.5, 4
                ),
            }

    return baselines


# ──────────────────────────────────────────────
# Flagging (improved: both high and low deviations)
# ──────────────────────────────────────────────


def flag_turns(turns_data: list[dict], baselines: dict) -> list[dict]:
    """Flag turns that deviate from speaker's own baseline (both directions)."""
    flags_per_turn = []

    for td in turns_data:
        sp = td["speaker"]
        baseline = baselines.get(sp, {})
        flags = []

        # --- Stylometry: flag both HIGH and LOW deviations ---
        stylometry_checks = [
            ("type_token_ratio", "low"),   # low TTR = repetitive = suspicious
            ("long_word_ratio", "high"),    # too many long words = overly formal
            ("function_word_ratio", "low"), # low function words = formal/LLM
            ("avg_sentence_length", "high"),
        ]

        for key, direction in stylometry_checks:
            bl = baseline.get(f"stylometry_{key}")
            if not bl:
                continue
            val = td["stylometry"].get(key, 0)
            std = bl["std"]
            if std == 0:
                continue
            z = (val - bl["mean"]) / std

            if direction == "high" and z > 2.0:
                flags.append(f"HIGH_{key.upper()}: {round(val, 4)} (z={round(z, 2)})")
            elif direction == "low" and z < -2.0:
                flags.append(f"LOW_{key.upper()}: {round(val, 4)} (z={round(z, 2)})")

        # --- Sentence length variance: LOW is suspicious ---
        slv = td["stylometry"].get("sentence_length_variance", 0)
        slv_bl = baseline.get("stylometry_sentence_length_variance")
        if slv_bl and slv_bl["median"] > 5:
            if slv < slv_bl["median"] * 0.3:
                flags.append(
                    f"LOW_SENTENCE_VARIANCE: {round(slv, 2)} vs median {round(slv_bl['median'], 2)}"
                )

        # --- Readability: significant deviation ---
        fk = td["readability"].get("flesch_kincaid_grade", 0)
        fk_bl = baseline.get("readability_flesch_kincaid_grade")
        if fk_bl and fk_bl["std"] > 0:
            z = (fk - fk_bl["mean"]) / fk_bl["std"]
            if z > 2.0:
                flags.append(f"HIGH_COMPLEXITY: grade {fk} (z={round(z, 2)})")
            elif z < -2.0:
                flags.append(f"LOW_COMPLEXITY: grade {fk} (z={round(z, 2)})")

        # --- Spacy formality: HIGH is suspicious for answers ---
        fs = td.get("spacy", {}).get("formality_score", 0)
        fs_bl = baseline.get("spacy_formality_score")
        if fs_bl and fs_bl["std"] > 0:
            z = (fs - fs_bl["mean"]) / fs_bl["std"]
            if z > 2.0:
                flags.append(f"HIGH_FORMALITY: {round(fs, 4)} (z={round(z, 2)})")

        # --- Spacy disfluency: LOW is suspicious for long answers ---
        dr = td.get("spacy", {}).get("disfluency_ratio", 0)
        dr_bl = baseline.get("spacy_disfluency_ratio")
        if dr_bl and dr_bl["std"] > 0 and td.get("word_count", 0) > 60:
            z = (dr - dr_bl["mean"]) / dr_bl["std"]
            if z < -1.5:
                flags.append(f"LOW_DISFLUENCY: {round(dr, 4)} (z={round(z, 2)})")

        # --- Lexical sophistication: LOW avg_zipf = rare words = suspicious ---
        az = td.get("lexical", {}).get("avg_zipf", 0)
        az_bl = baseline.get("lexical_avg_zipf")
        if az_bl and az_bl["std"] > 0:
            z = (az - az_bl["mean"]) / az_bl["std"]
            if z < -2.0:
                flags.append(f"SOPHISTICATED_VOCABULARY: zipf {round(az, 2)} (z={round(z, 2)})")

        # --- Proselint: spike in issues ---
        pc = td.get("proselint", {}).get("count", 0)
        pc_bl = baseline.get("proselint_count")
        if pc_bl and pc_bl["std"] > 0:
            z = (pc - pc_bl["mean"]) / pc_bl["std"]
            if z > 2.0 and pc > 0:
                cats = set(
                    i["check"].split(".")[0] for i in td["proselint"].get("issues", [])
                )
                flags.append(
                    f"PROSELINT_SPIKE: {pc} issues (categories: {', '.join(sorted(cats))})"
                )

        if flags:
            flags_per_turn.append(
                {"turn_id": td["turn_id"], "speaker": sp, "flags": flags}
            )

    return flags_per_turn


# ──────────────────────────────────────────────
# Analysis pipeline
# ──────────────────────────────────────────────


def analyze(report: dict, use_perplexity: bool = True) -> dict:
    """Run full NLP analysis on report.json."""
    turns = report["turns"]
    MIN_SUBSTANTIVE_WORDS = 15

    # ── Pass 1: Extract all features per turn ──
    turns_data = []
    for turn in turns:
        text = turn["text"]
        word_count = turn.get("word_count", len(text.split()))

        td = {
            "turn_id": turn["id"],
            "speaker": turn["speaker"],
            "text": text,
            "word_count": word_count,
            "turn_type": (
                "substantive" if word_count >= 60
                else "conversational" if word_count < MIN_SUBSTANTIVE_WORDS
                else "short_answer"
            ),
            "stylometry": extract_stylometry(text),
            "readability": extract_readability(text),
            "proselint": extract_proselint(text),
            "spacy": extract_spacy_features(text),
            "lexical": extract_lexical_sophistication(text),
        }

        if use_perplexity:
            td["perplexity"] = extract_perplexity(text)

        turns_data.append(td)

    # ── Pass 2: Structural uniformity (per speaker, across substantive answers) ──
    speakers = {td["speaker"] for td in turns_data}
    structure_scores = {}
    for sp in speakers:
        answers = [
            td["text"]
            for td in turns_data
            if td["speaker"] == sp and td["turn_type"] == "substantive"
        ]
        structure_scores[sp] = compute_structure_uniformity(answers)

    # ── Pass 3: Register gap (per speaker) ──
    register_gaps = {}
    for sp in speakers:
        register_gaps[sp] = compute_register_gap(turns_data, sp)

    # ── Pass 4: Compute baselines ──
    baselines = compute_speaker_baselines(turns_data)

    # ── Pass 5: Z-scores and outlier detection ──
    z_scores_all = {}
    outliers_all = {}
    for sp in speakers:
        z_scores_all[sp] = compute_z_scores(turns_data, sp)
        outliers_all[sp] = detect_outliers_isolation_forest(turns_data, sp)

    # ── Pass 6: Composite scores per turn ──
    for td in turns_data:
        sp = td["speaker"]

        # Find z-scores for this turn
        turn_z = {}
        for zs in z_scores_all.get(sp, []):
            if zs["turn_id"] == td["turn_id"]:
                turn_z = zs["z_scores"]
                break

        outlier_info = outliers_all.get(sp, {}).get(td["turn_id"], {})
        is_outlier = outlier_info.get("is_outlier", False)

        register_shift = register_gaps.get(sp, {}).get("register_shift_score", 0)
        template = structure_scores.get(sp, {}).get("template_score", 0)

        td["z_scores"] = turn_z
        td["outlier"] = outlier_info
        td["composite_score"] = compute_composite_score(
            turn_z, is_outlier, register_shift, template
        )

    # ── Pass 7: Flag turns ──
    flags = flag_turns(turns_data, baselines)

    # ── Pass 8: Session-level summary ──
    session_analysis = {}
    for sp in speakers:
        sp_turns = [td for td in turns_data if td["speaker"] == sp]
        substantive = [td for td in sp_turns if td["turn_type"] == "substantive"]
        scores = [td["composite_score"] for td in substantive]
        outlier_count = sum(
            1 for td in sp_turns
            if td.get("outlier", {}).get("is_outlier", False)
        )

        session_analysis[sp] = {
            "total_turns": len(sp_turns),
            "substantive_turns": len(substantive),
            "avg_composite_score": round(sum(scores) / len(scores), 4) if scores else 0,
            "max_composite_score": round(max(scores), 4) if scores else 0,
            "outlier_turn_count": outlier_count,
            "outlier_turn_ratio": round(outlier_count / len(sp_turns), 4) if sp_turns else 0,
            "register_gap": register_gaps.get(sp, {}),
            "structure_uniformity": structure_scores.get(sp, {}),
        }

    # ── Identify student speaker ──
    # Prefer SPEAKER_01 (canonical candidate ID from speaker fix).
    # Fall back to most-total-words heuristic if canonical IDs not present.
    speaker_names = {td["speaker"] for td in turns_data}
    if "SPEAKER_01" in speaker_names:
        student_speaker = "SPEAKER_01"
    else:
        speaker_word_totals = {}
        for td in turns_data:
            sp = td["speaker"]
            speaker_word_totals[sp] = speaker_word_totals.get(sp, 0) + td["word_count"]
        student_speaker = max(speaker_word_totals, key=speaker_word_totals.get) if speaker_word_totals else None

    return {
        "turns": turns_data,
        "flags": flags,
        "speaker_baselines": baselines,
        "session_analysis": session_analysis,
        "student_speaker": student_speaker,
    }


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="NLP analysis on conversation transcripts"
    )
    parser.add_argument(
        "--input",
        default="./output/test3/report.json",
        help="Path to report.json",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output path (default: same dir as input, named nlp_report.json)",
    )
    parser.add_argument(
        "--no-perplexity",
        action="store_true",
        help="Skip GPT-2 perplexity scoring (faster, slightly less accurate)",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: {input_path} not found")
        sys.exit(1)

    output_path = Path(args.output) if args.output else input_path.parent / "nlp_report.json"

    with open(input_path, "r", encoding="utf-8") as f:
        report = json.load(f)

    print(f"Analyzing {len(report['turns'])} turns...")
    result = analyze(report, use_perplexity=not args.no_perplexity)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    # Print summary
    total_flags = sum(len(f["flags"]) for f in result["flags"])
    print(f"\n{'='*60}")
    print(f"Turns analyzed: {len(result['turns'])}")
    print(f"Flagged turns: {len(result['flags'])} ({total_flags} flags total)")

    for sp, summary in sorted(result["session_analysis"].items()):
        print(f"\n{sp} session summary:")
        print(f"  Substantive turns: {summary['substantive_turns']}")
        print(f"  Avg composite score: {summary['avg_composite_score']}")
        print(f"  Max composite score: {summary['max_composite_score']}")
        print(f"  Outlier turns: {summary['outlier_turn_count']}")
        rg = summary["register_gap"]
        print(f"  Register gap: formality={rg.get('formality_gap', 0)}, zipf={rg.get('zipf_gap', 0)}, fk={rg.get('fk_grade_gap', 0)}")
        su = summary["structure_uniformity"]
        print(f"  Template score: {su.get('template_score', 0)}")

    print(f"\nPer-turn composite scores (substantive answers):")
    for td in result["turns"]:
        if td["turn_type"] == "substantive":
            marker = "🔴" if td["composite_score"] >= 0.5 else "🟡" if td["composite_score"] >= 0.3 else "🟢"
            print(
                f"  {marker} Turn {td['turn_id']:3d} ({td['speaker']}): "
                f"score={td['composite_score']:.2f}  "
                f"outlier={'yes' if td.get('outlier', {}).get('is_outlier') else 'no'}"
            )

    if result["flags"]:
        print(f"\nFlagged turns:")
        for fl in result["flags"]:
            print(f"  Turn {fl['turn_id']} ({fl['speaker']}):")
            for flag in fl["flags"]:
                print(f"    • {flag}")
    else:
        print("\nNo flagged turns.")

    print(f"{'='*60}")
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
