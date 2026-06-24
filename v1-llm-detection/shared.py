"""Shared utilities for the interview diagnostics bench."""

from __future__ import annotations

import json
import os
import urllib.request
from pathlib import Path


# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


# ──────────────────────────────────────────────
# .env loader
# ──────────────────────────────────────────────

def _load_dotenv():
    # Check local dir first, then parent bench dir
    for env_path in [Path(__file__).parent / ".env", Path(__file__).parent.parent / ".env"]:
        if not env_path.exists():
            continue
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, _, value = line.partition("=")
                os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))

_load_dotenv()


# ──────────────────────────────────────────────
# API clients
# ──────────────────────────────────────────────

def get_openrouter_client(api_key: str | None = None):
    """Create OpenAI client pointed at OpenRouter."""
    from openai import OpenAI

    if not api_key:
        api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY not found. Set it in env or pass api_key."
        )
    return OpenAI(base_url=OPENROUTER_BASE_URL, api_key=api_key)


def get_sarvam_client(api_key: str | None = None):
    """Create SarvamAI client (lazy import)."""
    from sarvamai import SarvamAI

    if not api_key:
        api_key = os.environ.get("SARVAM_API_KEY")
    if not api_key:
        raise RuntimeError(
            "SARVAM_API_KEY not found. Set it in env or pass api_key."
        )
    return SarvamAI(api_subscription_key=api_key)


# ──────────────────────────────────────────────
# Session discovery
# ──────────────────────────────────────────────

def load_users(users_json_path: Path) -> list[dict]:
    with open(users_json_path) as f:
        return json.load(f)


def find_session_dir(base_dir: Path, user_name: str, session_id: str) -> Path | None:
    """Find the local directory for a session by matching session_id prefix."""
    user_dir = base_dir / user_name.replace(" ", "_")
    if not user_dir.exists():
        return None
    for d in user_dir.iterdir():
        if d.is_dir() and session_id.startswith(d.name):
            return d
    return None


def build_session_list(
    users: list[dict],
    base_dir: Path,
    only_users: list[str] | None = None,
) -> list[dict]:
    """Build flat list of all sessions with their local status."""
    sessions = []
    for user in users:
        name = user["name"]
        if only_users and name not in only_users:
            continue
        for i, round_data in enumerate(user["rounds"]):
            session_dir = find_session_dir(base_dir, name, round_data["session_id"])
            has_report = session_dir and (session_dir / "report.json").exists()
            has_nlp = session_dir and ((session_dir / "nlp.json").exists() or (session_dir / "nlp_report.json").exists())
            has_transcript = session_dir and (session_dir / "transcript.json").exists()

            sessions.append({
                "user_name": name,
                "user_dir_name": name.replace(" ", "_"),
                "session_id": round_data["session_id"],
                "round_index": i,
                "band": round_data.get("band", ""),
                "date": round_data.get("date", ""),
                "salary": round_data.get("salary", ""),
                "audio_url": round_data.get("audio", ""),
                "session_dir": session_dir,
                "has_report": has_report,
                "has_nlp": has_nlp,
                "has_transcript": has_transcript,
            })

    return sessions


# ──────────────────────────────────────────────
# Download
# ──────────────────────────────────────────────

def download_audio(url: str, output_dir: Path) -> Path:
    filename = url.split("/")[-1].split("?")[0] or "audio.mp3"
    if not any(filename.endswith(ext) for ext in (".mp3", ".wav", ".flac", ".ogg", ".m4a")):
        filename += ".mp3"
    dest = output_dir / filename
    if dest.exists():
        return dest
    urllib.request.urlretrieve(url, dest)
    return dest


# ──────────────────────────────────────────────
# NLTK data
# ──────────────────────────────────────────────

def ensure_nlp_data():
    """Download NLTK data if missing."""
    import nltk

    for resource in ["punkt_tab", "averaged_perceptron_tagger_eng", "stopwords"]:
        try:
            nltk.data.find(
                f"tokenizers/{resource}" if "punkt" in resource else f"corpora/{resource}"
            )
        except LookupError:
            nltk.download(resource, quiet=True)
