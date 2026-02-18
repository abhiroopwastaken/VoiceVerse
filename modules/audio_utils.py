"""
Audio Utilities Module
======================
Merge audio segments and export final audio.
Uses binary MP3 concatenation (no ffmpeg dependency).
"""

import os
import io
import struct
import tempfile
from typing import List
from config import PAUSE_BETWEEN_SPEAKERS_MS, AUDIO_FORMAT


def _generate_silence_mp3(duration_ms: int) -> bytes:
    """
    Generate a short silence as MP3-compatible raw bytes.
    Uses edge-tts to generate a silent pause (single space = very short utterance).
    Falls back to empty bytes if that fails.
    """
    if duration_ms <= 0:
        return b""
    
    try:
        import asyncio
        import edge_tts
        
        silence_path = tempfile.mktemp(suffix=".mp3", prefix="silence_")
        
        async def _gen():
            # Generate a very short audio with a pause marker
            communicate = edge_tts.Communicate(
                text="...",
                voice="en-US-GuyNeural",
                rate="-50%",
            )
            await communicate.save(silence_path)
        
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    pool.submit(asyncio.run, _gen()).result()
            else:
                loop.run_until_complete(_gen())
        except RuntimeError:
            asyncio.run(_gen())
        
        if os.path.exists(silence_path):
            with open(silence_path, "rb") as f:
                data = f.read()
            os.remove(silence_path)
            return data
    except Exception:
        pass
    
    return b""


def merge_audio_segments(
    segment_paths: List[str],
    pause_ms: int = PAUSE_BETWEEN_SPEAKERS_MS,
    output_path: str = None,
) -> str:
    """
    Merge multiple MP3 audio segments into a single file.
    Uses binary concatenation (MP3 format supports this natively).
    
    Args:
        segment_paths: List of paths to MP3 audio files
        pause_ms: Milliseconds of silence between segments (approximate)
        output_path: Optional output path. If None, creates temp file.
    
    Returns:
        Path to the merged audio file
    """
    if not segment_paths:
        raise ValueError("No audio segments to merge.")
    
    if output_path is None:
        output_path = tempfile.mktemp(suffix=f".{AUDIO_FORMAT}", prefix="voiceverse_final_")
    
    # Try pydub first (if ffmpeg is available)
    try:
        return _merge_with_pydub(segment_paths, pause_ms, output_path)
    except Exception as pydub_err:
        print(f"pydub merge failed ({pydub_err}), using binary concatenation...")
    
    # Fallback: binary MP3 concatenation
    return _merge_binary(segment_paths, output_path)


def _merge_with_pydub(segment_paths: List[str], pause_ms: int, output_path: str) -> str:
    """Merge using pydub (requires ffmpeg)."""
    from pydub import AudioSegment
    
    pause = AudioSegment.silent(duration=pause_ms)
    combined = AudioSegment.empty()
    
    for i, path in enumerate(segment_paths):
        if not os.path.exists(path):
            continue
        segment = AudioSegment.from_mp3(path)
        combined += segment
        if i < len(segment_paths) - 1:
            combined += pause
    
    if len(combined) == 0:
        raise ValueError("No valid audio segments.")
    
    combined.export(output_path, format=AUDIO_FORMAT)
    return output_path


def _merge_binary(segment_paths: List[str], output_path: str) -> str:
    """
    Merge MP3 files via binary concatenation.
    MP3 is a frame-based format, so concatenation produces valid output.
    """
    valid_segments = [p for p in segment_paths if os.path.exists(p) and os.path.getsize(p) > 0]
    
    if not valid_segments:
        raise ValueError("No valid audio segments could be loaded.")
    
    with open(output_path, "wb") as out_f:
        for i, path in enumerate(valid_segments):
            with open(path, "rb") as seg_f:
                out_f.write(seg_f.read())
    
    return output_path


def get_audio_duration(file_path: str) -> float:
    """
    Estimate duration of an MP3 file.
    Uses file size and average bitrate for estimation if ffmpeg is unavailable.
    """
    try:
        from pydub import AudioSegment
        audio = AudioSegment.from_file(file_path)
        return len(audio) / 1000.0
    except Exception:
        # Estimate from file size (assuming ~128kbps MP3)
        size_bytes = os.path.getsize(file_path)
        # 128 kbps = 16 KB/s
        estimated_seconds = size_bytes / (16 * 1024)
        return estimated_seconds


def cleanup_temp_files(file_paths: List[str]):
    """Remove temporary audio files."""
    for path in file_paths:
        try:
            if os.path.exists(path):
                os.remove(path)
        except OSError:
            pass
