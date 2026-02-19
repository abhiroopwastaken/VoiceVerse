"""
Script Generator Module
=======================
Uses Groq API to generate structured audio scripts from RAG context.
Fast free-tier inference via llama-3.1-8b-instant.
"""

import re
import os
from typing import List, Tuple, Optional
from prompts.templates import build_prompt, SYSTEM_PROMPT
import json
from config import (
    GROQ_API_KEY,
    GROQ_MODEL_ID,
    SCRIPT_MAX_TOKENS, SCRIPT_TEMPERATURE,
    SUMMARIZATION_MAX_TOKENS, SUMMARIZATION_TEMPERATURE
)


def generate_script(
    context: str,
    style: str = "narration",
    custom_focus: str = "",
) -> List[Tuple[str, str]]:
    """
    Generate a structured script from RAG context.

    Args:
        context: Retrieved text from RAG pipeline
        style: One of: podcast, narration, debate, lecture, storytelling
        custom_focus: Optional user-specified topic focus

    Returns:
        List of (speaker_label, text) tuples
    """
    prompt = build_prompt(context, style, custom_focus)

    try:
        raw_response = _call_script_llm(prompt)
    except Exception as e:
        print(f"Script LLM API error: {repr(e)}")
        return [("System", f"API Error: {str(e)}")]

    print(f"DEBUG: LLM Response for {style}:\n{raw_response[:500]}...")
    parsed = _parse_script(raw_response, style)

    if not parsed:
        print(f"ERROR: Failed to parse script for style: {style}")
        return [("System", "Error: LLM output could not be parsed into valid JSON/Script.")]

    # Normalize speakers to uppercase/no spaces
    normalized = []
    for speaker, text in parsed:
        s = speaker.strip().upper().replace(" ", "_")
        normalized.append((s, text))

    return normalized


# ─── Groq Client ────────────────────────────────────────────────────────────

def _get_groq_client():
    """Get a configured Groq client."""
    from groq import Groq
    if not GROQ_API_KEY:
        raise ValueError(
            "GROQ_API_KEY is not set. Please add your Groq API key to the .env file. "
            "Get a free key at: https://console.groq.com"
        )
    return Groq(api_key=GROQ_API_KEY)


def _call_script_llm(prompt: str) -> str:
    """Call Groq API for Script Generation (llama-3.1-8b-instant, fast & free)."""
    client = _get_groq_client()

    response = client.chat.completions.create(
        model=GROQ_MODEL_ID,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        max_tokens=SCRIPT_MAX_TOKENS,
        temperature=SCRIPT_TEMPERATURE,
    )

    return response.choices[0].message.content


def _call_summary_llm(text: str) -> str:
    """Call Groq API for Summarization."""
    client = _get_groq_client()

    response = client.chat.completions.create(
        model=GROQ_MODEL_ID,
        messages=[
            {
                "role": "user",
                "content": (
                    "Summarize the following text in 3-5 concise bullet points. "
                    "Be brief and factual.\n\n" + text
                ),
            },
        ],
        max_tokens=SUMMARIZATION_MAX_TOKENS,
        temperature=SUMMARIZATION_TEMPERATURE,
    )

    return response.choices[0].message.content


# ─── Parsing ────────────────────────────────────────────────────────────────

def _parse_script(raw_text: str, style: str) -> List[Tuple[str, str]]:
    """
    Parse raw LLM output into (speaker, text) tuples.
    Tries JSON first, then Regex fallback.
    """
    # 1. Try JSON Parsing — find the first '[' and match to its closing ']'
    start = raw_text.find('[')
    if start != -1:
        # Walk to find the matching closing bracket
        depth = 0
        end = -1
        for i, ch in enumerate(raw_text[start:], start):
            if ch == '[':
                depth += 1
            elif ch == ']':
                depth -= 1
                if depth == 0:
                    end = i
                    break
        
        if end != -1:
            try:
                json_str = raw_text[start:end + 1]
                data = json.loads(json_str)
                parsed = []
                for item in data:
                    if isinstance(item, dict) and "speaker" in item and "text" in item:
                        parsed.append((str(item["speaker"]), str(item["text"])))
                if parsed:
                    return parsed
            except json.JSONDecodeError:
                pass


    # 2. Fallback to Regex Parsing
    lines = raw_text.strip().split('\n')
    parsed = []

    # More robust pattern: starts with text, ends with colon, then dialogue
    # e.g., "STORYTELLER: Once upon a time..." or "Host A: Hello"
    pattern = r'^([A-Z_a-z0-9 ]+)\s*[:：]\s*(.+)'

    current_speaker = None
    current_text = ""

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Look for [speaker]: [text]
        match = re.match(pattern, line)
        if match:
            if current_speaker and current_text:
                parsed.append((current_speaker, current_text.strip()))
            current_speaker = match.group(1).strip()
            current_text = match.group(2).strip()
        elif current_speaker:
            # Check if this line is just a new speaker with no colon (unlikely but possible)
            # or if it's a continuation of the previous speaker's text
            current_text += " " + line

    if current_speaker and current_text:
        parsed.append((current_speaker, current_text.strip()))

    return parsed


# ─── Template Fallbacks ─────────────────────────────────────────────────────

def _template_fallback(context: str, style: str) -> str:
    """Generate a script using templates when LLM is unavailable."""
    sentences = re.split(r'(?<=[.!?])\s+', context)
    key_points = [s.strip() for s in sentences if len(s.strip()) > 20][:15]

    if style.lower() == "podcast":
        return _build_podcast_template(key_points)
    elif style.lower() == "debate":
        return _build_debate_template(key_points)
    elif style.lower() == "lecture":
        return _build_lecture_template(key_points)
    elif style.lower() == "storytelling":
        return _build_storytelling_template(key_points)
    else:
        return _build_narration_template(key_points)


def _template_fallback_parsed(context: str, style: str) -> List[Tuple[str, str]]:
    """Direct parsed fallback."""
    raw = _template_fallback(context, style)
    parsed = _parse_script(raw, style)
    if not parsed:
        parsed = [("NARRATOR", context[:2000])]
    return parsed


def _build_podcast_template(points: list) -> str:
    lines = [
        "HOST_A: Welcome back everyone to another episode! Today we're diving into a fascinating topic.",
        "HOST_B: Absolutely! I've been looking forward to this one. Let's get right into it.",
    ]
    for i, point in enumerate(points):
        if i % 2 == 0:
            lines.append(f"HOST_A: Here's something really interesting - {point}")
        else:
            lines.append(f"HOST_B: That's a great point. And building on that - {point}")
    lines.append("HOST_A: Well, that's been an incredible discussion today.")
    lines.append("HOST_B: Agreed! Thanks everyone for tuning in. Until next time, stay curious!")
    return "\n".join(lines)


def _build_narration_template(points: list) -> str:
    lines = ["NARRATOR: In a world of constant discovery, some stories demand to be told."]
    for point in points:
        lines.append(f"NARRATOR: {point}")
    lines.append("NARRATOR: And so, these remarkable insights remind us that knowledge transforms our understanding.")
    return "\n".join(lines)


def _build_debate_template(points: list) -> str:
    lines = [
        "DEBATER_1: Thank you for having us today. I'd like to present my perspective.",
        "DEBATER_2: And I look forward to offering an alternative viewpoint.",
    ]
    for i, point in enumerate(points):
        if i % 2 == 0:
            lines.append(f"DEBATER_1: Consider this point - {point}")
        else:
            lines.append(f"DEBATER_2: While that's interesting, I'd argue that {point}")
    lines.append("DEBATER_1: In conclusion, I believe the evidence strongly supports my position.")
    lines.append("DEBATER_2: And I maintain compelling counterarguments. Thank you all.")
    return "\n".join(lines)


def _build_lecture_template(points: list) -> str:
    lines = ["PROFESSOR: Good morning! Today we explore a truly fascinating subject."]
    for i, point in enumerate(points):
        if i == 0:
            lines.append(f"PROFESSOR: Let's start with the fundamentals. {point}")
        elif i == len(points) - 1:
            lines.append(f"PROFESSOR: And finally, this is the key takeaway - {point}")
        else:
            lines.append(f"PROFESSOR: Here's another important concept. {point}")
    lines.append("PROFESSOR: That covers today's material. See you in the next session!")
    return "\n".join(lines)


def _build_storytelling_template(points: list) -> str:
    lines = ["STORYTELLER: Gather round, and let me tell you a tale of discovery and wonder."]
    for i, point in enumerate(points):
        if i == 0:
            lines.append(f"STORYTELLER: It all began when... {point}")
        elif i == len(points) - 1:
            lines.append(f"STORYTELLER: And that, dear listener, is how we arrived at the truth. {point}")
        else:
            lines.append(f"STORYTELLER: And then, something remarkable happened. {point}")
    lines.append("STORYTELLER: And so our story comes to a close. Every ending is a new beginning.")
    return "\n".join(lines)


# ─── Public API ─────────────────────────────────────────────────────────────

def generate_summary(text: str) -> str:
    """Generate a concise bullet-point summary of the provided text."""
    if not text:
        return ""

    # Truncate to stay within token limits
    truncated_text = text[:4000]

    try:
        return _call_summary_llm(truncated_text)
    except Exception as e:
        print(f"Summary generation failed: {e}")
        return f"Summary generation unavailable. Error: {str(e)}"
