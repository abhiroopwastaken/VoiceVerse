"""
Script Generator Module
=======================
Uses HF Inference API to generate structured audio scripts from RAG context.
"""

import re
import os
from typing import List, Tuple, Optional
from prompts.templates import build_prompt, SYSTEM_PROMPT
from config import HF_TOKEN, LLM_MODEL_ID, LLM_MAX_NEW_TOKENS, LLM_TEMPERATURE


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
    
    # Try LLM generation first, fall back to template if API fails
    try:
        raw_script = _call_llm(prompt)
    except Exception as e:
        print(f"LLM API error: {e}. Using template-based generation.")
        raw_script = _template_fallback(context, style)
    
    # Parse the raw script into (speaker, text) tuples
    parsed = _parse_script(raw_script, style)
    
    if not parsed:
        # Last resort fallback
        parsed = _template_fallback_parsed(context, style)
    
    return parsed


def _call_llm(prompt: str) -> str:
    """Call HF Inference API using chat completion."""
    from huggingface_hub import InferenceClient
    
    token = HF_TOKEN
    if not token:
        raise ValueError("HF_TOKEN not set. Please set the HF_TOKEN environment variable.")
    
    client = InferenceClient(token=token)
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    
    response = client.chat_completion(
        messages=messages,
        model=LLM_MODEL_ID,
        max_tokens=LLM_MAX_NEW_TOKENS,
        temperature=LLM_TEMPERATURE,
    )
    
    return response.choices[0].message.content


def _parse_script(raw_text: str, style: str) -> List[Tuple[str, str]]:
    """
    Parse raw LLM output into (speaker, text) tuples.
    Handles formats like:
        HOST_A: Hello everyone...
        NARRATOR: In a world where...
    """
    lines = raw_text.strip().split('\n')
    parsed = []
    
    # Define valid speaker patterns per style
    speaker_patterns = {
        "podcast":      r'^(HOST_A|HOST_B)\s*:\s*(.+)',
        "narration":    r'^(NARRATOR)\s*:\s*(.+)',
        "debate":       r'^(DEBATER_1|DEBATER_2)\s*:\s*(.+)',
        "lecture":      r'^(PROFESSOR)\s*:\s*(.+)',
        "storytelling": r'^(STORYTELLER)\s*:\s*(.+)',
    }
    
    pattern = speaker_patterns.get(style.lower(), r'^([A-Z_]+\d?)\s*:\s*(.+)')
    
    current_speaker = None
    current_text = ""
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        match = re.match(pattern, line)
        if match:
            # Save previous speaker's text
            if current_speaker and current_text:
                parsed.append((current_speaker, current_text.strip()))
            
            current_speaker = match.group(1)
            current_text = match.group(2)
        elif current_speaker:
            # Continuation of previous speaker's text
            current_text += " " + line
    
    # Don't forget the last entry
    if current_speaker and current_text:
        parsed.append((current_speaker, current_text.strip()))
    
    return parsed


def _template_fallback(context: str, style: str) -> str:
    """Generate a script using templates when LLM is unavailable."""
    # Split context into key points
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
        # Ultimate fallback: just narrate the context
        parsed = [("NARRATOR", context[:2000])]
    return parsed


def _build_podcast_template(points: list) -> str:
    lines = []
    lines.append("HOST_A: Welcome back everyone to another episode! Today we're diving into a fascinating topic that I think you're really going to enjoy.")
    lines.append("HOST_B: Absolutely! I've been looking forward to this one. Let's get right into it.")
    
    for i, point in enumerate(points):
        if i % 2 == 0:
            lines.append(f"HOST_A: Here's something really interesting - {point}")
        else:
            lines.append(f"HOST_B: That's a great point. And building on that - {point}")
    
    lines.append("HOST_A: Well, that's been an incredible discussion. So many fascinating insights today.")
    lines.append("HOST_B: Agreed! Thanks everyone for tuning in. Until next time, keep learning and stay curious!")
    return "\n".join(lines)


def _build_narration_template(points: list) -> str:
    lines = []
    lines.append("NARRATOR: In a world of constant discovery, some stories demand to be told. This is one of them.")
    
    for point in points:
        lines.append(f"NARRATOR: {point}")
    
    lines.append("NARRATOR: And so, as we reflect on these remarkable insights, we're reminded that knowledge has the power to transform our understanding of the world.")
    return "\n".join(lines)


def _build_debate_template(points: list) -> str:
    lines = []
    lines.append("DEBATER_1: Thank you for having us today. I'd like to present my perspective on this important topic.")
    lines.append("DEBATER_2: And I look forward to offering an alternative viewpoint. Let's have a productive discussion.")
    
    for i, point in enumerate(points):
        if i % 2 == 0:
            lines.append(f"DEBATER_1: Consider this point - {point}")
        else:
            lines.append(f"DEBATER_2: While that's interesting, I'd argue that {point}")
    
    lines.append("DEBATER_1: In conclusion, I believe the evidence strongly supports my position.")
    lines.append("DEBATER_2: And I maintain that there are compelling counterarguments worth considering. Thank you all.")
    return "\n".join(lines)


def _build_lecture_template(points: list) -> str:
    lines = []
    lines.append("PROFESSOR: Good morning, class! Today we're going to explore a truly fascinating subject. Please pay close attention.")
    
    for i, point in enumerate(points):
        if i == 0:
            lines.append(f"PROFESSOR: Let's start with the fundamentals. {point}")
        elif i == len(points) - 1:
            lines.append(f"PROFESSOR: And finally, this is the key takeaway - {point}")
        else:
            lines.append(f"PROFESSOR: Now, here's another important concept. {point}")
    
    lines.append("PROFESSOR: That covers today's material. Remember to review these points, and I'll see you in the next session!")
    return "\n".join(lines)


def _build_storytelling_template(points: list) -> str:
    lines = []
    lines.append("STORYTELLER: Gather round, and let me tell you a tale unlike any you've heard before. A tale of discovery, wonder, and revelation.")
    
    for i, point in enumerate(points):
        if i == 0:
            lines.append(f"STORYTELLER: It all began when... {point}")
        elif i == len(points) - 1:
            lines.append(f"STORYTELLER: And that, dear listener, is how we arrived at the truth. {point}")
        else:
            lines.append(f"STORYTELLER: And then, something remarkable happened. {point}")
    
    lines.append("STORYTELLER: And so our story comes to a close. But remember, every ending is just a new beginning.")
    return "\n".join(lines)
