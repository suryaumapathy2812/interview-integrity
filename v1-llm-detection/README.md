# V1: LLM-Assisted Detection

The original pipeline for detecting whether interview answers were LLM-assisted.

## What It Does

Uses a two-layer approach to flag suspicious answers:

- **NLP Layer** — 22 statistical features per turn: stylometry, readability, formality, perplexity, lexical sophistication, structural uniformity. Computes composite suspicion scores via Isolation Forest and z-scores against each speaker's own baseline.
- **LLM Layer** — A 3-step chain: (1) speaker profile from conversational turns, (2) per-answer analysis against that profile, (3) cross-answer coherence for session-level verdict.

## Key Insight

Does not look for "AI patterns" in absolute terms. Detects when a speaker's answers don't match their demonstrated ability — the register gap between natural speech and interview answers.

## Dimensions

See [`dimensions.md`](dimensions.md) for the full 33-dimension reference (22 programmatic + 11 LLM).

## Files

| File | Description |
|------|-------------|
| `pipeline.py` | Transcription with Sarvam, segment/turn building, timing metrics |
| `shared.py` | API clients, session discovery, download helpers |
| `analyze_nlp.py` | 22-feature NLP extraction + composite scoring |
| `fix_speakers.py` | LLM-based interviewer/candidate identification |
| `llm_detection.py` | 3-step LLM semantic analysis chain |
| `process_users.py` | CLI orchestrator with concurrency and resume |
| `dimensions.md` | Full dimension reference document |
| `users.json` | User/session data for batch processing |

## Quick Start

```bash
uv run process_users.py
uv run process_users.py --skip-transcription --run-llm
uv run process_users.py --users "Priyanka A" --resume
```

## Output Artifacts

Each session produces:
- `report.json` — turns, segments, latency, pauses, overlaps, stats
- `transcript.json` — raw Sarvam diarized transcript
- `nlp_report.json` — per-turn features, z-scores, composite scores, flags

Aggregated:
- `pipeline_results.json`
- `llm_detection_results.json`
- `speaker_identification_results.json`
