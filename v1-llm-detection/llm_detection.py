"""
LLM-assisted interview response detector.

Analyzes interview transcriptions to detect whether a student used LLM
assistance for their answers. Uses a 3-step LLM chain via OpenRouter:

1. Speaker profiling  — learns the student's natural English from their
   conversational / short turns
2. Per-answer analysis — scores each substantive answer against that profile
3. Cross-answer coherence — finds patterns across all answers

Requires: OPENROUTER_API_KEY in env or .env file (via shared.get_openrouter_client).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from openai import OpenAI
from pydantic import BaseModel, Field

from shared import get_openrouter_client

# ─────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────

DEFAULT_MODEL = "openai/gpt-4o"
MAX_RETRIES = 3
RETRY_DELAY = 2

# Turn classification thresholds
MIN_SUBSTANTIVE_WORDS = 15  # turns shorter than this are "conversational"


# ─────────────────────────────────────────────────────────────────
# Pydantic models for structured LLM output
# ─────────────────────────────────────────────────────────────────


class SpeakerProfile(BaseModel):
    """Profile of the speaker's natural English, built from conversational turns."""

    english_proficiency: str = Field(
        description="beginner | elementary | intermediate | upper_intermediate | advanced | native"
    )
    natural_register: str = Field(
        description="The speaker's natural register: very_casual | casual | neutral | formal | very_formal"
    )
    common_grammatical_errors: list[str] = Field(
        description="List of grammatical patterns/errors observed in their natural speech"
    )
    vocabulary_level: str = Field(
        description="basic | intermediate | advanced | academic"
    )
    typical_discourse_markers: list[str] = Field(
        description="Fillers and discourse markers they naturally use (so, like, yeah, actually, etc.)"
    )
    sentence_structure_habits: str = Field(
        description="How they naturally form sentences: fragments | simple | mixed | complex"
    )
    topics_demonstrated: list[str] = Field(
        description="Topics where they show genuine understanding from their conversational turns"
    )
    uncertainty_expression: str = Field(
        description="How they express uncertainty: asks_for_clarification | hedges | trails_off | avoids"
    )
    likely_native_language: str = Field(
        description="Best guess of their native language based on speech patterns"
    )


class AnswerAnalysis(BaseModel):
    """Analysis of a single answer turn."""

    turn_id: int
    register_match: int = Field(
        description="1-10 how well this answer's register matches the speaker's natural register (1=perfect match, 10=completely different)"
    )
    vocabulary_match: int = Field(
        description="1-10 how well the vocabulary matches the speaker's natural level (1=perfect match, 10=obviously different)"
    )
    likely_origin: str = Field(
        description="real_time | recalled_from_memory | pre_written_script | llm_generated"
    )
    specificity: str = Field(
        description="Does the answer contain concrete personal details? high | medium | low | none"
    )
    llm_markers_found: list[str] = Field(
        description="Specific LLM-typical phrases or patterns found in this answer"
    )
    structural_pattern: str = Field(
        description="How the answer is structured: narrative | list | template | textbook | rambling"
    )
    confidence_contradiction: bool = Field(
        description="Does the speaker's confidence in this answer contradict their demonstrated knowledge elsewhere?"
    )
    contradiction_detail: str = Field(
        description="If contradiction is true, explain what contradicts what"
    )
    reasoning: str = Field(
        description="2-3 sentence explanation of your assessment"
    )
    llm_score: float = Field(
        description="0.0 (definitely genuine) to 1.0 (definitely LLM-assisted). Be precise."
    )


class CrossAnswerAnalysis(BaseModel):
    """Analysis of patterns across all answers in a session."""

    consistency_score: float = Field(
        description="0.0 (very inconsistent) to 1.0 (perfectly consistent) — how consistent are all answers with each other"
    )
    register_consistency: str = Field(
        description="Is the register stable across all answers or does it shift? stable | mostly_stable | inconsistent | two_distinct_registers"
    )
    vocabulary_consistency: str = Field(
        description="Is vocabulary level stable? stable | mostly_stable | inconsistent | clear_gap"
    )
    knowledge_consistency: str = Field(
        description="Does demonstrated knowledge level stay consistent? consistent | mostly_consistent | inconsistent | suspicious_pattern"
    )
    pattern_description: str = Field(
        description="Describe the pattern if one exists, e.g., 'concept answers are textbook-perfect while personal answers are rough'"
    )
    suspicious_turns: list[int] = Field(
        description="Turn IDs that are inconsistent with the speaker's overall profile"
    )
    genuine_turns: list[int] = Field(
        description="Turn IDs that clearly match the speaker's natural voice"
    )
    overall_assessment: str = Field(
        description="genuine | llm_primary | mixed_genuine_and_llm | pre_prepared_with_llm | insufficient_data"
    )
    assessment_reasoning: str = Field(
        description="3-5 sentence explanation of the overall assessment"
    )
    session_llm_score: float = Field(
        description="0.0 (fully genuine) to 1.0 (fully LLM-assisted) for the entire session"
    )


# ─────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────


@dataclass
class Turn:
    id: int
    speaker: str
    text: str
    start: float
    end: float
    duration: float
    word_count: int


@dataclass
class ParsedReport:
    """Extracted from report.json."""

    student_speaker: str
    interviewer_speaker: str
    student_turns: list[Turn]
    interviewer_turns: list[Turn]
    all_turns: list[Turn]
    session_duration: float


# ─────────────────────────────────────────────────────────────────
# Report parsing
# ─────────────────────────────────────────────────────────────────


def parse_report(path: Path) -> ParsedReport:
    """Parse a report.json file into structured turns."""
    with open(path) as f:
        data = json.load(f)

    turns_data = data.get("turns", [])
    if not turns_data:
        raise ValueError(f"No turns found in {path}")

    # Determine speaker roles: first turn is usually the interviewer
    first_speaker = turns_data[0]["speaker"]
    interviewer_speaker = first_speaker

    all_turns = []
    for t in turns_data:
        turn = Turn(
            id=t["id"],
            speaker=t["speaker"],
            text=t["text"],
            start=t["start"],
            end=t["end"],
            duration=t["duration"],
            word_count=t["word_count"],
        )
        all_turns.append(turn)

    # The other speaker is the student — use SPEAKER_01 if available, else find dynamically
    speakers = {t.speaker for t in all_turns}
    if "SPEAKER_01" in speakers:
        student_speaker = "SPEAKER_01"
    else:
        student_speaker = [sp for sp in speakers if sp != interviewer_speaker][0]

    student_turns = [t for t in all_turns if t.speaker == student_speaker]
    interviewer_turns = [t for t in all_turns if t.speaker == interviewer_speaker]

    session_duration = max(t.end for t in all_turns) - min(t.start for t in all_turns)

    return ParsedReport(
        student_speaker=student_speaker,
        interviewer_speaker=interviewer_speaker,
        student_turns=student_turns,
        interviewer_turns=interviewer_turns,
        all_turns=all_turns,
        session_duration=session_duration,
    )


def parse_transcript(path: Path) -> ParsedReport:
    """Parse a transcript.json file into structured turns.
    This is a fallback for when report.json isn't available."""
    with open(path) as f:
        data = json.load(f)

    entries = data.get("diarized_transcript", {}).get("entries", [])
    if not entries:
        raise ValueError(f"No diarized entries found in {path}")

    # Merge consecutive entries from the same speaker into turns
    merged: list[dict] = []
    for entry in entries:
        sid = entry["speaker_id"]
        text = entry["transcript"].strip()
        if not text:
            continue
        if merged and merged[-1]["speaker"] == sid:
            merged[-1]["text"] += " " + text
            merged[-1]["end"] = entry["end_time_seconds"]
        else:
            merged.append({
                "speaker": sid,
                "text": text,
                "start": entry["start_time_seconds"],
                "end": entry["end_time_seconds"],
            })

    # Build turns
    all_turns = []
    for i, m in enumerate(merged):
        all_turns.append(Turn(
            id=i + 1,
            speaker=m["speaker"],
            text=m["text"],
            start=m["start"],
            end=m["end"],
            duration=m["end"] - m["start"],
            word_count=len(m["text"].split()),
        ))

    # First speaker is interviewer
    interviewer_speaker = all_turns[0].speaker
    student_speaker = [s for s in {t.speaker for t in all_turns} if s != interviewer_speaker][0]

    student_turns = [t for t in all_turns if t.speaker == student_speaker]
    interviewer_turns = [t for t in all_turns if t.speaker == interviewer_speaker]
    session_duration = max(t.end for t in all_turns) - min(t.start for t in all_turns)

    return ParsedReport(
        student_speaker=student_speaker,
        interviewer_speaker=interviewer_speaker,
        student_turns=student_turns,
        interviewer_turns=interviewer_turns,
        all_turns=all_turns,
        session_duration=session_duration,
    )



def llm_call(
    client: OpenAI,
    model: str,
    system: str,
    user: str,
    response_model: type[BaseModel],
) -> BaseModel:
    """Make an LLM call with structured output via OpenRouter.

    Uses chat.completions.create with response_format=json_schema
    and require_parameters to ensure the provider supports it.
    """
    schema = response_model.model_json_schema()
    schema["additionalProperties"] = False

    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": response_model.__name__,
                        "strict": True,
                        "schema": schema,
                    },
                },
                temperature=0.1,
                extra_body={
                    "provider": {"require_parameters": True},
                    "plugins": [{"id": "response-healing"}],
                },
            )
            content = response.choices[0].message.content
            if not content:
                raise ValueError("Empty response from LLM")
            return response_model.model_validate_json(content)
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                print(f"  [retry {attempt + 1}/{MAX_RETRIES}] {e}")
                time.sleep(RETRY_DELAY * (attempt + 1))
            else:
                raise


# ─────────────────────────────────────────────────────────────────
# Prompt chain
# ─────────────────────────────────────────────────────────────────


SYSTEM_PROMPT = """You are an expert at analyzing interview transcriptions to determine
whether a speaker used AI/LLM assistance for their answers. You analyze linguistic
patterns, vocabulary consistency, register shifts, and content quality to make
your assessments. You are precise, evidence-based, and avoid speculation."""


def build_profile(
    client: OpenAI,
    model: str,
    report: ParsedReport,
) -> SpeakerProfile:
    """Step 1: Build a speaker profile from conversational turns."""

    # Collect conversational turns (short answers, clarifications, filler)
    conversational = [
        t for t in report.student_turns
        if t.word_count < MIN_SUBSTANTIVE_WORDS
    ]
    # Also collect the interviewer's questions for context
    interviewer_context = [
        f"[Interviewer turn {t.id}]: {t.text}"
        for t in report.interviewer_turns[:5]  # first 5 for context
    ]

    conv_text = "\n".join(
        f"[Turn {t.id} ({t.word_count}w)]: {t.text}"
        for t in conversational
    )

    # Also include a few short-to-medium answers for richer profiling
    short_answers = [
        t for t in report.student_turns
        if MIN_SUBSTANTIVE_WORDS <= t.word_count < 60
    ]
    short_text = "\n".join(
        f"[Turn {t.id} ({t.word_count}w)]: {t.text}"
        for t in short_answers[:5]
    )

    user_msg = f"""Analyze this speaker's natural English from their conversational
and short-response turns in an interview. These are turns where they are likely
speaking naturally — asking for clarification, giving brief answers, or making
small talk.

CONVERSATIONAL TURNS (short, natural responses):
{conv_text if conv_text else "(no very short turns found)"}

SHORT ANSWER TURNS (brief substantive responses):
{short_text if short_text else "(no short answers found)"}

INTERVIEWER CONTEXT (first few questions for reference):
{chr(10).join(interviewer_context)}

Based on ONLY these natural/conversational turns, build a detailed profile of
this speaker's genuine English ability. Do NOT use their longer answers for
this profile — those may be assisted."""

    return llm_call(client, model, SYSTEM_PROMPT, user_msg, SpeakerProfile)


def analyze_answer(
    client: OpenAI,
    model: str,
    profile: SpeakerProfile,
    turn: Turn,
    interviewer_question: str,
    turn_number: int,
    total_turns: int,
) -> AnswerAnalysis:
    """Step 2: Analyze a single substantive answer."""

    user_msg = f"""You are analyzing a single answer from an interview. The speaker's
profile (built from their natural conversational turns) is below.

SPEAKER PROFILE:
- English proficiency: {profile.english_proficiency}
- Natural register: {profile.natural_register}
- Common errors: {', '.join(profile.common_grammatical_errors) if profile.common_grammatical_errors else 'none noted'}
- Vocabulary level: {profile.vocabulary_level}
- Discourse markers they use: {', '.join(profile.typical_discourse_markers) if profile.typical_discourse_markers else 'none noted'}
- Sentence habits: {profile.sentence_structure_habits}
- Likely native language: {profile.likely_native_language}
- How they express uncertainty: {profile.uncertainty_expression}

CONTEXT:
- This is answer {turn_number} of {total_turns} substantive answers
- Interviewer question: "{interviewer_question}"
- Turn duration: {turn.duration:.1f}s | Words: {turn.word_count}

THE ANSWER TO ANALYZE:
"{turn.text}"

Analyze this answer against the speaker's profile. Consider:
1. Does the REGISTER (formality, sentence complexity) match the speaker's natural voice?
2. Does the VOCABULARY match their demonstrated level?
3. Is the answer SPECIFIC (real personal details) or GENERIC (textbook advice)?
4. Are there LLM-typical phrases or structures?
5. Is the speaker's confidence level consistent with what they'd naturally express?
6. Does the STRUCTURE (how the answer is organized) look natural or template-like?

Be strict but fair. A well-prepared student who studied hard can sound polished
without LLM help — look for MISMATCHES between their natural level and this answer,
not just polish. The key question is: could THIS speaker have produced THIS answer
from their own knowledge and natural English ability?"""

    return llm_call(client, model, SYSTEM_PROMPT, user_msg, AnswerAnalysis)


def analyze_coherence(
    client: OpenAI,
    model: str,
    profile: SpeakerProfile,
    answer_analyses: list[AnswerAnalysis],
    student_turns: list[Turn],
) -> CrossAnswerAnalysis:
    """Step 3: Cross-answer coherence analysis."""

    # Build a summary of each answer's analysis
    analyses_summary = []
    for a in answer_analyses:
        analyses_summary.append(
            f"Turn {a.turn_id}: origin={a.likely_origin}, "
            f"register_match={a.register_match}/10, "
            f"vocab_match={a.vocabulary_match}/10, "
            f"specificity={a.specificity}, "
            f"llm_score={a.llm_score:.2f}, "
            f"structural={a.structural_pattern}, "
            f"markers={a.llm_markers_found or 'none'}"
        )

    # Build the full answer texts (trimmed to save tokens)
    answers_text = []
    for t in student_turns:
        if t.word_count >= MIN_SUBSTANTIVE_WORDS:
            # Trim very long answers to first 200 + last 50 words
            words = t.text.split()
            if len(words) > 250:
                trimmed = " ".join(words[:200]) + " [...] " + " ".join(words[-50:])
            else:
                trimmed = t.text
            answers_text.append(f"[Turn {t.id} ({t.word_count}w)]: {trimmed}")

    user_msg = f"""You have already analyzed each answer individually. Now look at
the FULL PATTERN across all answers to determine the session-level verdict.

SPEAKER PROFILE:
- English proficiency: {profile.english_proficiency}
- Natural register: {profile.natural_register}
- Vocabulary level: {profile.vocabulary_level}
- Sentence habits: {profile.sentence_structure_habits}
- Likely native language: {profile.likely_native_language}

PER-ANSWER ANALYSES:
{chr(10).join(analyses_summary)}

FULL ANSWER TEXTS (for reference):
{chr(10).join(answers_text)}

Now determine:

1. CONSISTENCY: Are all answers at the same quality level, or are some dramatically
   different from others? A genuine speaker is consistently imperfect. An LLM user
   has some answers that are suspiciously perfect mixed with natural ones.

2. PATTERN: If there's inconsistency, what's the pattern?
   - All answers uniformly polished → likely full LLM
   - Concept-definition answers polished, personal answers rough → LLM for concepts
   - Long answers polished, short answers rough → LLM for complex questions
   - First few answers polished, later ones rough → ran out of prepared material
   - All answers rough but coherent → likely genuine

3. OVERALL: Based on everything — the profile, the individual analyses, and the
   pattern — what is the most likely verdict for this session?

Be precise. "mixed_genuine_and_llm" means the speaker used LLM for SOME answers
but not others. "pre_prepared_with_llm" means they used LLM before the interview
to prepare, then delivered from memory. These are different patterns."""

    return llm_call(client, model, SYSTEM_PROMPT, user_msg, CrossAnswerAnalysis)


# ─────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────


def analyze_session(
    report_path: Path,
    model: str = DEFAULT_MODEL,
    client: OpenAI | None = None,
) -> dict:
    """Run the full detection pipeline on a single session.

    Args:
        report_path: Path to report.json or transcript.json.
        model: OpenRouter model identifier.
        client: Optional OpenAI client. Created via get_openrouter_client() if not provided.
    """

    print(f"\n{'='*60}")
    print(f"Analyzing: {report_path}")
    print(f"{'='*60}")

    # Parse
    if report_path.name == "report.json":
        report = parse_report(report_path)
    elif report_path.name == "transcript.json":
        report = parse_transcript(report_path)
    else:
        raise ValueError(f"Unsupported file: {report_path.name}")

    print(f"  Student: {report.student_speaker} ({len(report.student_turns)} turns)")
    print(f"  Interviewer: {report.interviewer_speaker} ({len(report.interviewer_turns)} turns)")
    print(f"  Session duration: {report.session_duration:.0f}s")

    if client is None:
        client = get_openrouter_client()

    # Step 1: Build speaker profile
    print("\n  [1/3] Building speaker profile...")
    profile = build_profile(client, model, report)
    print(f"    Proficiency: {profile.english_proficiency}")
    print(f"    Register: {profile.natural_register}")
    print(f"    Vocabulary: {profile.vocabulary_level}")
    print(f"    Native lang: {profile.likely_native_language}")

    # Step 2: Analyze each substantive answer
    substantive_turns = [
        t for t in report.student_turns
        if t.word_count >= MIN_SUBSTANTIVE_WORDS
    ]

    # Build a map of turn_id -> interviewer question
    # (the question immediately before the student's answer)
    question_map: dict[int, str] = {}
    for i, turn in enumerate(report.all_turns):
        if turn.speaker == report.interviewer_speaker:
            # Find the next student turn
            for j in range(i + 1, len(report.all_turns)):
                if report.all_turns[j].speaker == report.student_speaker:
                    question_map[report.all_turns[j].id] = turn.text
                    break

    print(f"\n  [2/3] Analyzing {len(substantive_turns)} substantive answers...")
    answer_analyses: list[AnswerAnalysis] = []
    for i, turn in enumerate(substantive_turns, 1):
        question = question_map.get(turn.id, "(question not found)")
        print(f"    Turn {turn.id} ({turn.word_count}w)...", end=" ", flush=True)
        analysis = analyze_answer(
            client, model, profile, turn, question,
            i, len(substantive_turns),
        )
        answer_analyses.append(analysis)
        print(f"score={analysis.llm_score:.2f} origin={analysis.likely_origin}")

    # Step 3: Cross-answer coherence
    print(f"\n  [3/3] Cross-answer coherence analysis...")
    coherence = analyze_coherence(client, model, profile, answer_analyses, report.student_turns)
    print(f"    Overall: {coherence.overall_assessment}")
    print(f"    Score: {coherence.session_llm_score:.2f}")
    print(f"    Suspicious turns: {coherence.suspicious_turns}")
    print(f"    Genuine turns: {coherence.genuine_turns}")

    # Build output
    result = {
        "file": str(report_path),
        "model_used": model,
        "session_duration_seconds": report.session_duration,
        "total_turns": len(report.all_turns),
        "student_turns": len(report.student_turns),
        "substantive_answers_analyzed": len(substantive_turns),
        "speaker_profile": profile.model_dump(),
        "per_answer_analysis": [a.model_dump() for a in answer_analyses],
        "cross_answer_analysis": coherence.model_dump(),
        "verdict": {
            "assessment": coherence.overall_assessment,
            "confidence_score": coherence.session_llm_score,
            "suspicious_turn_ids": coherence.suspicious_turns,
            "genuine_turn_ids": coherence.genuine_turns,
            "reasoning": coherence.assessment_reasoning,
        },
    }

    return result
