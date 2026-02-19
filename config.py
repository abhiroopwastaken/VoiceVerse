"""
VoiceVerse Configuration
========================
Central configuration for all modules.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ─── LLM Settings (Groq API) ────────────────────────────────────
HF_TOKEN = os.environ.get("HF_TOKEN", "")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

# Groq Model — handles both script generation and summarization
# Using llama-3.1-8b-instant: fast, free, high quality
GROQ_MODEL_ID = "llama-3.1-8b-instant"
SCRIPT_MAX_TOKENS = 3000
SCRIPT_TEMPERATURE = 0.5
SUMMARIZATION_MAX_TOKENS = 512
SUMMARIZATION_TEMPERATURE = 0.3

# ─── Embedding & RAG Settings ───────────────────────────────────
# Use local BGE-Small (CPU optimized)
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5" 

# Reranker Model (via HF API)
RERANKER_MODEL = "BAAI/bge-reranker-base"
RERANKER_TOP_K = 8        # Top chunks to keep after reranking

CHUNK_SIZE = 1000          # Larger chunks = more coherent context
CHUNK_OVERLAP = 0          # Semantic splits handle context
SEMANTIC_CHUNK_THRESHOLD = 0.45  
TOP_K_RETRIEVAL = 15       # Initial retrieval (passed to reranker)

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
