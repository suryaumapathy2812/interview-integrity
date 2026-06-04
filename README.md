# Veritas

Detect LLM-assisted interview answers by analyzing linguistic patterns, register shifts, and vocabulary anomalies in transcripts.

## How It Works

Veritas uses a two-layer detection approach:

**NLP Layer** (fast, cheap) — Extracts 22 statistical features per turn: stylometry, readability, formality, perplexity, lexical sophistication, and structural uniformity. Computes composite suspicion scores via Isolation Forest and z-scores against each speaker's own baseline.

**LLM Layer** (slow, expensive) — A 3-step chain via OpenRouter: (1) build speaker profile from conversational turns, (2) analyze each substantive answer against that profile, (3) cross-answer coherence analysis for session-level verdict.

The key insight: Veritas doesn't look for "AI patterns" in absolute terms. It detects when a speaker's answers don't match their demonstrated ability — the register gap between how they naturally speak and how they answer questions.

## Pipeline Stages

| Stage | Script | Description |
|-------|--------|-------------|
| 1 | `pipeline.py` | Transcribe audio via Sarvam Saaras v3 with diarization |
| 2 | `fix_speakers.py` | LLM identifies interviewer vs candidate, remaps to canonical IDs |
| 3 | `analyze_nlp.py` | 22-feature extraction + composite scoring per turn |
| 4 | `llm_detection.py` | 3-step LLM chain for semantic analysis |
| — | `process_users.py` | Orchestrator with CLI, concurrency, and resume |

## Quick Start

```bash
# Install dependencies
uv sync

# Run the full pipeline
uv run process_users.py

# Skip transcription (already done), run NLP + LLM
uv run process_users.py --skip-transcription --run-llm

# Process specific users
uv run process_users.py --users "Priyanka A" "Deepanshu Gunwant"

# Resume mode (skip completed stages)
uv run process_users.py --resume
```

## CLI Flags

```
--output DIR           Base output directory (default: output/)
--users NAME [NAME]   Only process specific users
--concurrency N       Concurrent workers (default: 5)

--skip-transcription   Skip download + transcription
--skip-speaker-fix     Skip speaker identification
--skip-nlp             Skip NLP analysis
--run-llm              Run LLM analysis on flagged sessions
--resume               Skip stages with existing output

--no-perplexity        Skip GPT-2 perplexity scoring (faster)
--speaker-model MODEL  Model for speaker ID (default: openai/gpt-4o-mini)
--llm-model MODEL      Model for LLM analysis (default: openai/gpt-4o)
--llm-threshold FLOAT  NLP score threshold for LLM (default: 0.3)
```

## Environment

Required in `.env` or environment:

```
SARVAM_API_KEY=...       # Sarvam AI for transcription
OPENROUTER_API_KEY=...   # OpenRouter for LLM analysis
```

## Output

Each session produces:

- `report.json` — turns, segments, latency, pauses, overlaps, stats
- `transcript.json` — raw Sarvam diarized transcript
- `nlp_report.json` — per-turn features, z-scores, composite scores, flags

Aggregated results:

- `pipeline_results.json` — full pipeline run summary
- `llm_detection_results.json` — LLM verdicts per session
- `speaker_identification_results.json` — speaker remapping results

## Detection Signals

**NLP features** (22 total): type-token ratio, hapax ratio, sentence length/variance, function word ratio, long word ratio, Flesch readability, FK grade, syllables/word, proselint issues, POS formality, disfluency ratio, dependency depth, noun/verb ratio, Zipf frequency, rare word ratio, GPT-2 perplexity, TF-IDF template score, register gap, Isolation Forest outlier score, z-scores, composite score.

**LLM dimensions**: speaker profile, register match, vocabulary match, likely origin, specificity, LLM markers, structural pattern, confidence contradiction, cross-answer consistency, pattern description, overall assessment.

See [`dimensions.md`](dimensions.md) for the full reference.

## Architecture

```
process_users.py          # CLI orchestrator
├── shared.py             # API clients, session discovery
├── pipeline.py           # Transcription + analysis
├── fix_speakers.py       # Speaker role identification
├── analyze_nlp.py        # NLP feature extraction
└── llm_detection.py      # LLM semantic analysis
```

## License

Private — for internal use.
