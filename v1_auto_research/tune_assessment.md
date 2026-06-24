# Assessment Parameter Tuning

You are an AI agent tasked with optimizing the deterministic assessment parameters for an interview LLM-detection pipeline. Your goal is to find the best configuration that maximizes agreement with LLM verdicts while maintaining separation between genuine and mixed sessions.

## Context

The pipeline analyzes interview recordings to detect if candidates used LLM assistance. It has:
- **Upstream stages**: transcription, speaker diarization, NLP feature extraction, LLM detection (fixed, do not modify)
- **Downstream assessment**: deterministic verdict based on signal weights and thresholds (this is what you tune)

The downstream assessment reads existing artifacts (`report.json`, `nlp.json`, `llm.json`) and produces verdicts: `genuine`, `mixed`, `ai_assisted`, or `unknown`.

## Your Files

- `assessment_config.json` — **EDIT THIS FILE**. Contains all tunable parameters.
- `run_assessment.py` — Fixed assessment runner. Reads config, runs assessment, outputs results.
- `evaluate.py` — Fixed evaluation script. Computes metrics comparing your verdicts to LLM labels.
- `results/` — Output directory. Your run will create `results/metrics.json` with the evaluation score.

## The Loop

Each iteration follows this pattern:

1. **Read current config**: `cat assessment_config.json`
2. **Modify config**: Edit parameters you think will improve the metric
3. **Run assessment**: `uv run run_assessment.py`
4. **Evaluate**: `uv run evaluate.py`
5. **Check metric**: Look at `results/metrics.json`
6. **Keep or discard**: If metric improved, keep changes. If not, revert.

## What to Tune

### Signal Weights (`signal_weights`)
These control how much each signal contributes to the answer score:
- `statistical_anomaly` (currently 0.36) — NLP composite outlier score
- `complexity_difference` (currently 0.22) — FK grade, dependency depth, rare words
- `polish_difference` (currently 0.12) — formality, sentence length, disfluency
- `timing_context` (currently 0.10) — speaking speed deviation
- `llm_preparedness` (currently 0.20) — LLM detection score (when available)

Weights should sum to ~1.0. The `llm_preparedness` weight only applies when `llm.json` exists.

### Answer Thresholds (`answer_thresholds`)
- `watch_score` (currently 0.4) — answer score above which we flag for review
- `flagged_score` (currently 0.65) — answer score above which we mark as flagged
- `flagged_min_major_signals` (currently 2) — minimum medium/high signals needed for "flagged"

### Verdict Rules (`verdict_rules`)
- `ai_assisted_min_flagged` (currently 2) — minimum flagged answers for ai_assisted verdict
- `ai_assisted_min_flagged_ratio` (currently 0.4) — minimum ratio of flagged answers
- `mixed_min_flagged` (currently 1) — minimum flagged answers for mixed verdict
- `mixed_min_watch` (currently 2) — minimum watch answers for mixed verdict

### Signal Parameters (`signal_parameters`)
Detailed normalization ranges for each signal. Adjust these to change sensitivity.

## Optimization Metric

The primary metric is **agreement rate** — how often your deterministic verdict matches the normalized LLM verdict:

- LLM `genuine` → our `genuine`
- LLM `mixed_genuine_and_llm` → our `mixed`
- LLM `llm_primary` → our `ai_assisted`

Secondary metric is **score separation** — the difference between mean scores of genuine vs mixed sessions. Higher is better.

## Constraints

1. **Do not modify** `run_assessment.py` or `evaluate.py`
2. **Do not modify** files in `../v1-llm-detection/`
3. All weights must be non-negative
4. Thresholds must be between 0 and 1
5. Signal normalization floors must be less than ceilings

## Example Iteration

```bash
# 1. Check current config
cat assessment_config.json

# 2. Edit config (e.g., increase LLM weight)
# Change "llm_preparedness": 0.20 to "llm_preparedness": 0.35

# 3. Run assessment
uv run run_assessment.py

# 4. Evaluate
uv run evaluate.py

# 5. Check results
cat results/metrics.json

# 6. If agreement_rate improved, keep. If not, revert.
```

## Tips

- Start with small changes to weights (±0.05)
- The LLM signal is the most powerful — increasing its weight from 0.20 to 0.30-0.40 often improves agreement
- Lowering `watch_score` from 0.4 to 0.3 makes the system more sensitive
- The `polish_word_count_guard` dampening helps reduce false positives on short answers
- Watch for overfitting — if you optimize too hard on these 46 sessions, you may lose generalization

## Reporting

After each iteration, report:
- Current config changes
- Agreement rate
- Score separation
- Any observations about what worked or didn't
