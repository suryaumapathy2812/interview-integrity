"""
End-to-end conversation analysis pipeline.

Transcribes with Sarvam Saaras v3 → Analyzes → Outputs report.

Usage as library:
    from shared import get_sarvam_client, download_audio
    from pipeline import transcribe, analyze

    client = get_sarvam_client()
    audio_path = download_audio(url, output_dir)
    sarvam_data = transcribe(audio_path, client)
    report = analyze(sarvam_data)
"""

import json
import sys
import time
from pathlib import Path

from shared import ensure_nlp_data

ensure_nlp_data()


# ──────────────────────────────────────────────
# Step 2: Transcribe with Sarvam
# ──────────────────────────────────────────────

def transcribe(audio_path: Path, client, lang: str | None = None, speakers: int | None = None) -> dict:
    """Run Sarvam batch transcription with diarization.

    Args:
        audio_path: Path to audio file.
        client: SarvamAI client instance.
        lang: Language code (e.g. en-IN, hi-IN). Auto-detect if None.
        speakers: Number of speakers. Auto-detect if None.
    """
    job_params = {
        "model": "saaras:v3",
        "mode": "transcribe",
        "with_diarization": True,
    }
    if lang:
        job_params["language_code"] = lang
    if speakers:
        job_params["num_speakers"] = speakers

    print(f"Transcribing with Sarvam Saaras v3...")
    print(f"  Language: {lang or 'auto-detect'}")
    print(f"  Speakers: {speakers or 'auto-detect'}")

    job = client.speech_to_text_job.create_job(**job_params)
    job.upload_files(file_paths=[str(audio_path)])

    start = time.perf_counter()
    job.start()
    job.wait_until_complete()
    elapsed = time.perf_counter() - start
    print(f"  Done in {elapsed:.1f}s")

    file_results = job.get_file_results()
    if file_results.get("failed"):
        for f in file_results["failed"]:
            print(f"  ✗ {f.get('file_name')}: {f.get('error_message')}")
        sys.exit(1)

    # Get result from SDK response directly
    successful = file_results.get("successful", [])
    if not successful:
        print("No successful results.")
        sys.exit(1)

    # Download to temp dir and load
    tmp_dir = Path("_sarvam_tmp")
    tmp_dir.mkdir(exist_ok=True)
    job.download_outputs(output_dir=str(tmp_dir))

    result_files = sorted(tmp_dir.glob("*.json"))
    if not result_files:
        print("No output JSON from Sarvam.")
        sys.exit(1)

    with open(result_files[0], "r", encoding="utf-8") as f:
        data = json.load(f)

    # Cleanup temp
    for p in result_files:
        p.unlink()
    tmp_dir.rmdir()

    return data


# ──────────────────────────────────────────────
# Step 3: Analyze
# ──────────────────────────────────────────────

def build_segments(entries: list[dict]) -> list[dict]:
    return [
        {
            "speaker": f"SPEAKER_{e['speaker_id']}",
            "start": round(e["start_time_seconds"], 2),
            "end": round(e["end_time_seconds"], 2),
            "text": e["transcript"],
        }
        for e in entries
    ]


def build_turns(segments: list[dict]) -> list[dict]:
    if not segments:
        return []

    turns = []
    current = {
        "speaker": segments[0]["speaker"],
        "start": segments[0]["start"],
        "end": segments[0]["end"],
        "texts": [segments[0]["text"]],
        "segment_ids": [0],
    }

    for i, seg in enumerate(segments[1:], start=1):
        if seg["speaker"] == current["speaker"]:
            current["end"] = seg["end"]
            current["texts"].append(seg["text"])
            current["segment_ids"].append(i)
        else:
            turns.append(_finalize_turn(current, len(turns) + 1))
            current = {
                "speaker": seg["speaker"],
                "start": seg["start"],
                "end": seg["end"],
                "texts": [seg["text"]],
                "segment_ids": [i],
            }

    turns.append(_finalize_turn(current, len(turns) + 1))
    return turns


def _finalize_turn(raw: dict, turn_id: int) -> dict:
    text = " ".join(raw["texts"])
    return {
        "id": turn_id,
        "speaker": raw["speaker"],
        "start": raw["start"],
        "end": raw["end"],
        "segment_ids": raw["segment_ids"],
        "text": text,
        "word_count": len(text.split()),
        "duration": round(raw["end"] - raw["start"], 2),
    }


def build_latency(turns: list[dict]) -> list[dict]:
    latencies = []
    for prev, curr in zip(turns, turns[1:]):
        if prev["speaker"] != curr["speaker"]:
            latencies.append({
                "from_turn_id": prev["id"],
                "to_turn_id": curr["id"],
                "from_speaker": prev["speaker"],
                "to_speaker": curr["speaker"],
                "duration": round(curr["start"] - prev["end"], 2),
            })
    return latencies

def build_pauses(segments: list[dict], turns: list[dict]) -> list[dict]:
    """Compute within-turn pauses: gaps between consecutive segments of the same speaker within a turn."""
    pauses = []
    for turn in turns:
        seg_ids = turn["segment_ids"]
        for prev_id, curr_id in zip(seg_ids, seg_ids[1:]):
            gap = round(segments[curr_id]["start"] - segments[prev_id]["end"], 2)
            if gap > 0:
                pauses.append({
                    "turn_id": turn["id"],
                    "speaker": turn["speaker"],
                    "start": segments[prev_id]["end"],
                    "end": segments[curr_id]["start"],
                    "duration": gap,
                })
    return pauses


def build_overlaps(segments: list[dict]) -> list[dict]:
    overlaps = []
    for i, a in enumerate(segments):
        for j in range(i + 1, len(segments)):
            b = segments[j]
            if b["start"] >= a["end"]:
                break
            if a["speaker"] == b["speaker"]:
                continue
            overlap_start = max(a["start"], b["start"])
            overlap_end = min(a["end"], b["end"])
            if overlap_end > overlap_start:
                overlaps.append({
                    "segment_a": i,
                    "segment_b": j,
                    "speaker_a": a["speaker"],
                    "speaker_b": b["speaker"],
                    "start": round(overlap_start, 2),
                    "end": round(overlap_end, 2),
                    "duration": round(overlap_end - overlap_start, 2),
                })
    return overlaps


def compute_stats(turns: list[dict], latencies: list[dict], pauses: list[dict]) -> dict:
    speakers = {}

    for turn in turns:
        sp = turn["speaker"]
        if sp not in speakers:
            speakers[sp] = {
                "turn_count": 0, "total_speaking_time": 0.0, "total_words": 0,
                "_latencies": [], "_pauses": [], "_word_counts": [], "_durations": [],
            }
        speakers[sp]["turn_count"] += 1
        speakers[sp]["total_speaking_time"] += turn["duration"]
        speakers[sp]["total_words"] += turn["word_count"]
        speakers[sp]["_word_counts"].append(turn["word_count"])
        speakers[sp]["_durations"].append(turn["duration"])

    for lat in latencies:
        to_sp = lat["to_speaker"]
        if to_sp in speakers:
            speakers[to_sp]["_latencies"].append(lat["duration"])

    for pause in pauses:
        sp = pause["speaker"]
        if sp in speakers:
            speakers[sp]["_pauses"].append(pause["duration"])

    result = {}
    for sp, data in speakers.items():
        result[sp] = {
            "turn_count": data["turn_count"],
            "total_speaking_time": round(data["total_speaking_time"], 2),
            "total_words": data["total_words"],
            "latency": _distribution(data["_latencies"]),
            "pause": _distribution(data["_pauses"]),
            "words": _distribution(data["_word_counts"]),
            "duration": _distribution(data["_durations"]),
        }
    return result


def _distribution(values: list[float]) -> dict:
    if not values:
        return {"count": 0, "mean": 0, "median": 0, "p25": 0, "p75": 0, "min": 0, "max": 0, "values": []}

    n = len(values)
    s = sorted(values)
    return {
        "count": n,
        "mean": round(sum(values) / n, 2),
        "median": round(_percentile(s, 50), 2),
        "p25": round(_percentile(s, 25), 2),
        "p75": round(_percentile(s, 75), 2),
        "min": round(s[0], 2),
        "max": round(s[-1], 2),
        "values": [round(v, 2) for v in values],
    }


def _percentile(sorted_vals: list[float], p: int) -> float:
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    k = (len(sorted_vals) - 1) * (p / 100)
    f = int(k)
    c = min(f + 1, len(sorted_vals) - 1)
    return sorted_vals[f] + (k - f) * (sorted_vals[c] - sorted_vals[f])


def analyze(sarvam_data: dict) -> dict:
    entries = sarvam_data["diarized_transcript"]["entries"]
    segments = build_segments(entries)
    turns = build_turns(segments)
    latency = build_latency(turns)
    pauses = build_pauses(segments, turns)
    overlaps = build_overlaps(segments)
    stats = compute_stats(turns, latency, pauses)

    return {
        "segments": segments,
        "turns": turns,
        "latency": latency,
        "pauses": pauses,
        "overlaps": overlaps,
        "stats": stats,
    }
