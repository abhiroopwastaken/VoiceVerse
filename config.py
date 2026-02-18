"""
VoiceVerse Configuration
========================
Central configuration for all modules.
"""

import os

# ─── LLM Settings ───────────────────────────────────────────────
HF_TOKEN = os.environ.get("HF_TOKEN", "")
LLM_MODEL_ID = "Qwen/Qwen2.5-72B-Instruct"
LLM_MAX_NEW_TOKENS = 2048
LLM_TEMPERATURE = 0.7

# ─── Embedding & RAG Settings ───────────────────────────────────
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 500          # characters per chunk
CHUNK_OVERLAP = 50        # overlap between chunks
TOP_K_RETRIEVAL = 5       # number of chunks to retrieve

# ─── Voice Mappings (edge-tts voice IDs) ────────────────────────
VOICE_MAP = {
    "Host A (Male)":        "en-US-GuyNeural",
    "Host B (Female)":      "en-US-JennyNeural",
    "Narrator (British)":   "en-GB-RyanNeural",
    "Professor (Female)":   "en-US-AriaNeural",
    "Storyteller (Male)":   "en-US-DavisNeural",
    "Debater 1 (Male)":     "en-US-ChristopherNeural",
    "Debater 2 (Female)":   "en-US-MichelleNeural",
    "News Anchor (Female)": "en-US-SaraNeural",
}

DEFAULT_VOICES = {
    "podcast":      ("Host A (Male)", "Host B (Female)"),
    "narration":    ("Narrator (British)",),
    "debate":       ("Debater 1 (Male)", "Debater 2 (Female)"),
    "lecture":      ("Professor (Female)",),
    "storytelling": ("Storyteller (Male)",),
}

# ─── Audio Settings ─────────────────────────────────────────────
AUDIO_FORMAT = "mp3"
PAUSE_BETWEEN_SPEAKERS_MS = 400   # silence between speaker turns
SPEECH_RATE = "+0%"               # default speech rate
SPEECH_PITCH = "+0Hz"             # default pitch

# ─── Content Styles ─────────────────────────────────────────────
CONTENT_STYLES = [
    "Podcast",
    "Narration",
    "Debate",
    "Lecture",
    "Storytelling",
]
