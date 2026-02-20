"""
Voice Generator Module
======================
Multi-voice TTS using edge-tts with speaker mapping and tone control.
Generates one audio file per script segment, then merges them.
"""

import asyncio
import tempfile
import os
import re
from typing import List, Tuple, Optional
from config import VOICE_MAP, DEFAULT_VOICES, SPEECH_RATE, SPEECH_PITCH
from modules.audio_utils import merge_audio_segments

# Speaker label → voice name mapping per style
SPEAKER_VOICE_MAPPING = {
    "HOST_A":      "Host A (Male)",
    "HOST_B":      "Host B (Female)",
    "NARRATOR":    "Narrator (British)",
    "PROFESSOR":   "Professor (Female)",
    "STORYTELLER": "Storyteller (Male)",
    "DEBATER_1":   "Debater 1 (Male)",
    "DEBATER_2":   "Debater 2 (Female)",
    "STORY":       "Storyteller (Male)",
    "AUTHOR":      "Narrator (British)",
}


def _clean_text_for_tts(text: str) -> str:
    """
    Strip markdown and special characters that TTS would read verbatim.
    """
    if not text: return ""
    # Remove markdown bold/italic
    text = re.sub(r'\*{1,3}(.*?)\*{1,3}', r'\1', text)
    text = re.sub(r'_{1,2}(.*?)_{1,2}', r'\1', text)
    # Remove markdown headers
    text = re.sub(r'^#+\s*', '', text, flags=re.MULTILINE)
    # Remove inline code and code blocks
    text = re.sub(r'`{1,3}.*?`{1,3}', '', text, flags=re.DOTALL)
    # Remove markdown links [text](url) → text
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
    # Remove leftover brackets and JSON-like artifacts
    text = re.sub(r'[\[\]{}"\\]', '', text)
    # Remove bullet points and numbered lists
    text = re.sub(r'^\s*[-•*]\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*\d+[\.\)]\s+', '', text, flags=re.MULTILINE)
    # Collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


async def _synthesize_segment(
    text: str,
    voice: str,
    output_path: str,
    rate: str = "+0%",
    pitch: str = "+0Hz",
) -> str:
    """Synthesize a single text segment with edge-tts and fallbacks."""
    import edge_tts

    clean = _clean_text_for_tts(text)
    if not clean or len(clean) < 2:
        return None


    # Fallback to standard edge-tts synthesis...
    voices_to_try = [voice, "en-US-GuyNeural", "en-US-JennyNeural"]
    last_err = None

    for v in voices_to_try:
        try:
            communicate = edge_tts.Communicate(
                text=clean,
                voice=v,
                rate=rate,
                pitch=pitch,
            )
            await communicate.save(output_path)
            
            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                return output_path
        except Exception as e:
            last_err = e
            print(f"[VoiceGen] Synthesis failed for voice {v}: {e}")
            continue
            
    if last_err:
        raise last_err
    return None


async def _generate_all_segments(
    script: List[Tuple[str, str]],
    rate: str = "+0%",
    pitch: str = "+0Hz",
    custom_voices: Optional[dict] = None,
) -> List[str]:
    """
    Generate one audio file per script segment using edge-tts.
    Each segment uses the correct voice for its speaker.
    All segments run in parallel.
    """
    temp_dir = tempfile.mkdtemp(prefix="voiceverse_")
    tasks = []
    output_paths = []

    for i, (speaker, text) in enumerate(script):
        # Normalize speaker for lookup
        lookup_speaker = speaker.strip().upper().replace(" ", "_")
        voice_name = SPEAKER_VOICE_MAPPING.get(lookup_speaker, "Narrator (British)")
        
        if custom_voices and lookup_speaker in custom_voices:
            voice_name = custom_voices[lookup_speaker]
        
        voice_id = VOICE_MAP.get(voice_name, "en-US-GuyNeural")
        
        target_voice = voice_id

        # Split very long texts into smaller chunks
        if len(text) > 3500:
            print(f"[VoiceGen] Segment {i} is too long ({len(text)} chars). Chunking...")
            sentences = re.split(r'(?<=[.!?])\s+', text)
            chunks = []
            current_chunk = ""
            for s in sentences:
                if len(current_chunk) + len(s) < 3500:
                    current_chunk += " " + s
                else:
                    if current_chunk: chunks.append(current_chunk.strip())
                    current_chunk = s
            if current_chunk: chunks.append(current_chunk.strip())
            
            for j, chunk_text in enumerate(chunks):
                out = os.path.join(temp_dir, f"segment_{i:03d}_{j:02d}.mp3")
                output_paths.append(out)
                tasks.append(_synthesize_segment(chunk_text, target_voice, out, rate, pitch))
        else:
            out = os.path.join(temp_dir, f"segment_{i:03d}.mp3")
            output_paths.append(out)
            tasks.append(_synthesize_segment(text, target_voice, out, rate, pitch))

    print(f"[VoiceGen] Generating {len(tasks)} tasks (concurrency=5)...")
    
    # Process tasks in batches of 5 to prevent memory OOM or rate limits
    results = []
    concurrency_limit = 5
    for i in range(0, len(tasks), concurrency_limit):
        batch = tasks[i : i + concurrency_limit]
        batch_results = await asyncio.gather(*batch, return_exceptions=True)
        results.extend(batch_results)

    valid_paths = []
    for path, result in zip(output_paths, results):
        if isinstance(result, Exception):
            print(f"[VoiceGen] Segment failed: {result}")
        elif result and os.path.exists(path) and os.path.getsize(path) > 0:
            valid_paths.append(path)

    print(f"[VoiceGen] {len(valid_paths)}/{len(tasks)} segments OK.")
    return valid_paths


def generate_audio(
    script: List[Tuple[str, str]],
    rate: str = "+0%",
    pitch: str = "+0Hz",
    custom_voices: Optional[dict] = None,
) -> str:
    """
    Generate audio for the full script and return path to merged MP3.
    """
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(
                    asyncio.run,
                    _generate_all_segments(script, rate, pitch, custom_voices)
                )
                segment_paths = future.result()
        else:
            segment_paths = loop.run_until_complete(
                _generate_all_segments(script, rate, pitch, custom_voices)
            )
    except RuntimeError:
        segment_paths = asyncio.run(
            _generate_all_segments(script, rate, pitch, custom_voices)
        )

    if not segment_paths:
        raise RuntimeError("No audio segments were generated successfully.")

    final_path = merge_audio_segments(segment_paths)
    return final_path


def get_available_voices() -> List[str]:
    """Return list of available voice names."""
    return list(VOICE_MAP.keys())


def get_default_voices(style: str) -> Tuple:
    """Get default voice names for a content style."""
    return DEFAULT_VOICES.get(style.lower(), ("Narrator (British)",))
