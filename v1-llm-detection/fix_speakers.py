"""
Speaker role identification using LLM.

Reads each session's transcript.json, sends a sample to an LLM to identify
which speaker is the interviewer and which is the candidate, then remaps
speaker IDs to a canonical format:
  - SPEAKER_00 = interviewer
  - SPEAKER_01 = candidate

Updates both transcript.json and report.json in place.

Usage (as library):
    from fix_speakers import fix_session_speakers, process_session
    result = fix_session_speakers(session_dir)
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

from openai import OpenAI
from pydantic import BaseModel, Field

from shared import get_openrouter_client

# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────

DEFAULT_MODEL = "openai/gpt-4o-mini"
MAX_RETRIES = 3
RETRY_DELAY = 2


# ──────────────────────────────────────────────
# LLM output schema
# ──────────────────────────────────────────────

class SpeakerIdentification(BaseModel):
    interviewer_speaker_id: str = Field(
        description="The original speaker ID of the interviewer (e.g. 'SPEAKER_0', 'SPEAKER_2')"
    )
    candidate_speaker_id: str = Field(
        description="The original speaker ID of the candidate/student (e.g. 'SPEAKER_1', 'SPEAKER_3')"
    )
    confidence: float = Field(
        description="Confidence in the identification, 0.0 to 1.0"
    )
    reasoning: str = Field(
        description="Brief explanation of how you determined the roles"
    )


# ──────────────────────────────────────────────
# JSON extraction
# ──────────────────────────────────────────────

def _extract_json(text: str) -> dict:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    raise ValueError(f"Could not extract JSON from: {text[:200]}")


# ──────────────────────────────────────────────
# Speaker identification
# ──────────────────────────────────────────────

def identify_speakers(client: OpenAI, model: str, turns: list[dict]) -> dict:
    """Ask the LLM to identify which speaker is the interviewer and which is the candidate."""

    # Build a representative sample: first 3 turns per speaker, plus a few from the middle
    speaker_turns: dict[str, list[dict]] = {}
    for t in turns:
        sp = t["speaker"]
        if sp not in speaker_turns:
            speaker_turns[sp] = []
        if len(speaker_turns[sp]) < 5:
            speaker_turns[sp].append(t)

    # Build the sample text
    sample_lines = []
    for sp in sorted(speaker_turns.keys()):
        sample_lines.append(f"\n--- {sp} ---")
        for t in speaker_turns[sp]:
            text = t["text"][:300]  # Trim long turns
            sample_lines.append(f"  [{t.get('word_count', '?')} words]: {text}")

    sample_text = "\n".join(sample_lines)

    # Also add some middle/end turns for context
    if len(turns) > 10:
        mid = len(turns) // 2
        sample_lines.append(f"\n--- Middle of conversation ---")
        for t in turns[mid-2:mid+2]:
            text = t["text"][:300]
            sample_lines.append(f"  {t['speaker']} [{t.get('word_count', '?')} words]: {text}")

    system_prompt = """You are analyzing an interview transcript to identify the speaker roles.

In a diagnostic interview:
- The INTERVIEWER asks questions, provides instructions, gives prompts, and guides the conversation.
- The CANDIDATE/STUDENT answers questions, describes their background, explains their skills, and responds to prompts.

Look at the content and pattern of each speaker's turns to determine who is who.

You MUST respond with a single valid JSON object matching this schema:
{
  "interviewer_speaker_id": "the original speaker ID string (e.g. SPEAKER_0)",
  "candidate_speaker_id": "the original speaker ID string (e.g. SPEAKER_1)",
  "confidence": 0.0 to 1.0,
  "reasoning": "brief explanation"
}

No markdown, no extra text — just the JSON object."""

    user_prompt = f"""Here are sample turns from each speaker in the transcript:

{sample_text}

Identify which speaker is the interviewer and which is the candidate/student."""

    schema = SpeakerIdentification.model_json_schema()
    schema["additionalProperties"] = False

    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "SpeakerIdentification",
                        "strict": True,
                        "schema": schema,
                    },
                },
                temperature=0.0,
                extra_body={
                    "provider": {"require_parameters": True},
                    "plugins": [{"id": "response-healing"}],
                },
            )
            content = response.choices[0].message.content
            if not content:
                raise ValueError("Empty response")
            return SpeakerIdentification.model_validate_json(content).model_dump()
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY * (attempt + 1))
            else:
                raise


# ──────────────────────────────────────────────
# Remapping
# ──────────────────────────────────────────────

def remap_speaker_id(old_id: str, mapping: dict[str, str]) -> str:
    """Remap a speaker ID using the mapping. If not in mapping, keep original."""
    return mapping.get(old_id, old_id)


def _build_mapping(interviewer_id: str, candidate_id: str) -> dict[str, str]:
    """Build a mapping that handles both 'SPEAKER_X' and bare 'X' ID formats."""
    mapping = {
        interviewer_id: "SPEAKER_00",
        candidate_id: "SPEAKER_01",
    }
    # Also handle bare number format (e.g. "0" in transcript vs "SPEAKER_0" in report)
    for orig, canonical in list(mapping.items()):
        if orig.startswith("SPEAKER_"):
            bare = orig.replace("SPEAKER_", "")
            mapping[bare] = canonical
        else:
            mapping["SPEAKER_" + orig] = canonical
    return mapping


def remap_transcript(transcript_path: Path, interviewer_id: str, candidate_id: str) -> dict:
    """Update transcript.json with canonical speaker IDs."""
    with open(transcript_path) as f:
        data = json.load(f)

    mapping = _build_mapping(interviewer_id, candidate_id)

    entries = data.get("diarized_transcript", {}).get("entries", [])
    for entry in entries:
        entry["speaker_id"] = remap_speaker_id(entry["speaker_id"], mapping)

    return data


def _resolve_extra_speaker(extra_speaker: str, turns: list[dict]) -> str:
    """For an unmapped speaker, find the nearest canonical speaker by timing.
    Returns 'SPEAKER_00' or 'SPEAKER_01'."""
    # Find the midpoint of all turns by this speaker
    extra_turns = [t for t in turns if t["speaker"] == extra_speaker]
    if not extra_turns:
        return "SPEAKER_01"  # default to candidate
    extra_midpoints = [(t["start"] + t["end"]) / 2 for t in extra_turns]
    avg_mid = sum(extra_midpoints) / len(extra_midpoints)

    # Find average midpoint of SPEAKER_00 and SPEAKER_01 turns
    canonical = {}
    for sp in ("SPEAKER_00", "SPEAKER_01"):
        sp_turns = [t for t in turns if t["speaker"] == sp]
        if sp_turns:
            canonical[sp] = sum((t["start"] + t["end"]) / 2 for t in sp_turns) / len(sp_turns)

    if not canonical:
        return "SPEAKER_01"

    # Assign to the nearest canonical speaker
    return min(canonical, key=lambda sp: abs(avg_mid - canonical[sp]))


def remap_report(report_path: Path, interviewer_id: str, candidate_id: str) -> dict:
    """Update report.json with canonical speaker IDs, merging extra speakers."""
    with open(report_path) as f:
        data = json.load(f)

    mapping = _build_mapping(interviewer_id, candidate_id)

    # First pass: map interviewer and candidate
    for turn in data.get("turns", []):
        turn["speaker"] = remap_speaker_id(turn["speaker"], mapping)
    for seg in data.get("segments", []):
        seg["speaker"] = remap_speaker_id(seg["speaker"], mapping)

    # Second pass: resolve any remaining non-canonical speakers
    turns = data.get("turns", [])
    remaining = {t["speaker"] for t in turns} - {"SPEAKER_00", "SPEAKER_01"}
    extra_mapping = {}
    for sp in remaining:
        extra_mapping[sp] = _resolve_extra_speaker(sp, turns)

    for turn in turns:
        if turn["speaker"] in extra_mapping:
            turn["speaker"] = extra_mapping[turn["speaker"]]
    for seg in data.get("segments", []):
        if seg["speaker"] in extra_mapping:
            seg["speaker"] = extra_mapping[seg["speaker"]]

    # Merge consecutive turns from the same speaker after remapping
    merged_turns = []
    for turn in turns:
        if merged_turns and merged_turns[-1]["speaker"] == turn["speaker"]:
            merged_turns[-1]["text"] += " " + turn["text"]
            merged_turns[-1]["end"] = turn["end"]
            merged_turns[-1]["duration"] = round(merged_turns[-1]["end"] - merged_turns[-1]["start"], 2)
            merged_turns[-1]["word_count"] = len(merged_turns[-1]["text"].split())
            if "segment_ids" in merged_turns[-1]:
                merged_turns[-1]["segment_ids"] = merged_turns[-1].get("segment_ids", []) + turn.get("segment_ids", [])
        else:
            merged_turns.append(turn)
    # Re-number turns
    for i, turn in enumerate(merged_turns, 1):
        turn["id"] = i
    data["turns"] = merged_turns
    # Remap remaining structures
    for latency in data.get("latency", []):
        latency["from_speaker"] = extra_mapping.get(latency["from_speaker"], latency["from_speaker"])
        latency["to_speaker"] = extra_mapping.get(latency["to_speaker"], latency["to_speaker"])
    for pause in data.get("pauses", []):
        if pause["speaker"] in extra_mapping:
            pause["speaker"] = extra_mapping[pause["speaker"]]
    for overlap in data.get("overlaps", []):
        if overlap["speaker_a"] in extra_mapping:
            overlap["speaker_a"] = extra_mapping[overlap["speaker_a"]]
        if overlap["speaker_b"] in extra_mapping:
            overlap["speaker_b"] = extra_mapping[overlap["speaker_b"]]
    if "stats" in data:
        new_stats = {}
        for sp, val in data["stats"].items():
            canonical = extra_mapping.get(sp, sp)
            if canonical in new_stats:
                # Merge stats — this is a simplification
                continue
            new_stats[canonical] = val
        data["stats"] = new_stats
    return data


# ──────────────────────────────────────────────
# Processing
# ──────────────────────────────────────────────

def process_session(
    session_dir: Path,
    client: OpenAI,
    model: str,
    dry_run: bool = False,
) -> dict:
    """Identify speakers and optionally remap IDs in one session."""
    report_path = session_dir / "report.json"
    transcript_path = session_dir / "transcript.json"

    # Load report to get turns
    with open(report_path) as f:
        report = json.load(f)

    turns = report.get("turns", [])
    if not turns:
        raise ValueError("No turns in report.json")

    # Check if already remapped (all speakers are SPEAKER_00 or SPEAKER_01)
    unique_speakers = {t["speaker"] for t in turns}
    if unique_speakers <= {"SPEAKER_00", "SPEAKER_01"}:
        return {
            "session_dir": str(session_dir),
            "status": "already_remapped",
            "interviewer": "SPEAKER_00",
            "candidate": "SPEAKER_01",
        }

    # Identify speakers
    result = identify_speakers(client, model, turns)

    interviewer_id = result["interviewer_speaker_id"]
    candidate_id = result["candidate_speaker_id"]

    # Validate
    if interviewer_id == candidate_id:
        raise ValueError(f"Interviewer and candidate are the same: {interviewer_id}")

    if interviewer_id not in unique_speakers:
        raise ValueError(f"Interviewer ID '{interviewer_id}' not in speakers {unique_speakers}")
    if candidate_id not in unique_speakers:
        raise ValueError(f"Candidate ID '{candidate_id}' not in speakers {unique_speakers}")

    if not dry_run:
        # Remap and save report.json
        remapped_report = remap_report(report_path, interviewer_id, candidate_id)
        with open(report_path, "w") as f:
            json.dump(remapped_report, f, indent=2, ensure_ascii=False)

        # Remap and save transcript.json
        remapped_transcript = remap_transcript(transcript_path, interviewer_id, candidate_id)
        with open(transcript_path, "w") as f:
            json.dump(remapped_transcript, f, indent=2, ensure_ascii=False)

        # Remove nlp_report.json so it gets re-generated
        nlp_path = session_dir / "nlp_report.json"
        if nlp_path.exists():
            nlp_path.unlink()

    return {
        "session_dir": str(session_dir),
        "status": "remapped" if not dry_run else "identified",
        "original_interviewer": interviewer_id,
        "original_candidate": candidate_id,
        "confidence": result["confidence"],
        "reasoning": result["reasoning"],
    }


# ──────────────────────────────────────────────
# Public entry point
# ──────────────────────────────────────────────

def fix_session_speakers(
    session_dir: Path | str,
    client: OpenAI | None = None,
    model: str = DEFAULT_MODEL,
    dry_run: bool = False,
) -> dict:
    """Public entry point: fix speaker IDs for a single session.

    Creates an OpenRouter client if none provided.

    Args:
        session_dir: Path to the session directory containing report.json and transcript.json.
        client: Optional OpenAI client instance. If None, creates one via get_openrouter_client().
        model: LLM model identifier (default: openai/gpt-4o-mini).
        dry_run: If True, only identify speakers without modifying files.

    Returns:
        dict with session_dir, status, speaker IDs, confidence, and reasoning.
    """
    if client is None:
        client = get_openrouter_client()
    return process_session(Path(session_dir), client, model, dry_run=dry_run)
