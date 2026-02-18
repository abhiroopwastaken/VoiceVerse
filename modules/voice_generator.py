"""
Voice Generator Module
======================
Multi-voice TTS using edge-tts with speaker mapping and tone control.
"""

import asyncio
import tempfile
import os
from typing import List, Tuple, Optional
from config import VOICE_MAP, DEFAULT_VOICES, SPEECH_RATE, SPEECH_PITCH


# Speaker label → voice name mapping per style
SPEAKER_VOICE_MAPPING = {
    "HOST_A":      "Host A (Male)",
    "HOST_B":      "Host B (Female)",
    "NARRATOR":    "Narrator (British)",
    "PROFESSOR":   "Professor (Female)",
    "STORYTELLER": "Storyteller (Male)",
    "DEBATER_1":   "Debater 1 (Male)",
    "DEBATER_2":   "Debater 2 (Female)",
}


async def _synthesize_segment(
    text: str,
    voice: str,
    output_path: str,
    rate: str = "+0%",
    pitch: str = "+0Hz",
) -> str:
    """Synthesize a single text segment with edge-tts."""
    import edge_tts
    
    communicate = edge_tts.Communicate(
        text=text,
        voice=voice,
        rate=rate,
        pitch=pitch,
    )
    await communicate.save(output_path)
    return output_path


async def _generate_all_segments(
    script: List[Tuple[str, str]],
    rate: str = "+0%",
    pitch: str = "+0Hz",
    custom_voices: Optional[dict] = None,
) -> List[str]:
    """
    Generate audio for all script segments.
    
    Args:
        script: List of (speaker_label, text) tuples
        rate: Speech rate adjustment (e.g., "+10%", "-5%")
        pitch: Pitch adjustment (e.g., "+5Hz", "-3Hz")
        custom_voices: Optional override for speaker→voice mapping
    
    Returns:
        List of paths to individual audio segment files
    """
    segment_paths = []
    temp_dir = tempfile.mkdtemp(prefix="voiceverse_")
    
    for i, (speaker, text) in enumerate(script):
        # Resolve voice ID
        voice_name = SPEAKER_VOICE_MAPPING.get(speaker, "Narrator (British)")
        
        if custom_voices and speaker in custom_voices:
            voice_name = custom_voices[speaker]
        
        voice_id = VOICE_MAP.get(voice_name, "en-US-GuyNeural")
        
        # Generate segment
        seg_path = os.path.join(temp_dir, f"segment_{i:03d}.mp3")
        
        try:
            await _synthesize_segment(text, voice_id, seg_path, rate, pitch)
            segment_paths.append(seg_path)
        except Exception as e:
            print(f"Warning: Failed to synthesize segment {i}: {e}")
            # Try with fallback voice
            try:
                await _synthesize_segment(text, "en-US-GuyNeural", seg_path, rate, pitch)
                segment_paths.append(seg_path)
            except Exception as e2:
                print(f"Error: Fallback also failed for segment {i}: {e2}")
    
    return segment_paths


def generate_audio(
    script: List[Tuple[str, str]],
    rate: str = "+0%",
    pitch: str = "+0Hz",
    custom_voices: Optional[dict] = None,
) -> List[str]:
    """
    Synchronous wrapper for async audio generation.
    
    Returns:
        List of audio segment file paths.
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
                return future.result()
        else:
            return loop.run_until_complete(
                _generate_all_segments(script, rate, pitch, custom_voices)
            )
    except RuntimeError:
        return asyncio.run(
            _generate_all_segments(script, rate, pitch, custom_voices)
        )


def get_available_voices() -> List[str]:
    """Return list of available voice names."""
    return list(VOICE_MAP.keys())


def get_default_voices(style: str) -> Tuple:
    """Get default voice names for a content style."""
    return DEFAULT_VOICES.get(style.lower(), ("Narrator (British)",))
