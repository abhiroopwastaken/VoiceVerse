"""
Prompt Templates for Script Generation
=======================================
All templates enforce source-grounded output — the LLM must use ONLY
specific facts, names, quotes, and examples from the provided source material.
"""

SYSTEM_PROMPT = """You are a professional script writer for audio content.
Your scripts must be DEEPLY GROUNDED in the provided source material.

CRITICAL RULES:
1. Use SPECIFIC details from the source: exact names, quotes, experiments, statistics, examples.
2. Do NOT invent facts, add generic knowledge, or pad with vague statements.
3. Every claim must be traceable to the source material provided.
4. Produce engaging content that accurately reflects the source.
5. OUTPUT FORMAT: Respond with ONLY a valid JSON array. No markdown, no explanation.
   Format: [{"speaker": "SPEAKER_NAME", "text": "dialogue text"}, ...]"""


PODCAST_TEMPLATE = """Write a podcast script based EXCLUSIVELY on the source material below.

SPEAKERS: HOST_A and HOST_B
STRUCTURE:
1. INTRO (2 exchanges): Introduce the specific topic from the document
2. BODY (12-16 exchanges): Discuss specific concepts, experiments, names, and examples from the source
3. CONCLUSION (2 exchanges): Key takeaways directly from the material

QUALITY RULES:
- MUST reference specific names (e.g. researchers, authors, concepts) from the source
- MUST discuss specific experiments, case studies or findings mentioned in the source
- MUST use language and terminology from the source material
- Do NOT use generic filler like "That's a great point" without substance
- Each HOST turn must contain a specific insight from the source, not vague commentary
- Total: 15-20 exchanges

SOURCE MATERIAL (use ONLY this):
{context}

{custom_focus}

Respond with ONLY the JSON array:"""


NARRATION_TEMPLATE = """Write a narration script based EXCLUSIVELY on the source material below.

SPEAKER: NARRATOR only
STRUCTURE:
1. OPENING HOOK (1-2 paragraphs): Start with a compelling specific fact or quote from the source
2. BODY (8-12 paragraphs): Present specific concepts, names, experiments and findings from the source
3. CLOSING (1-2 paragraphs): End with a key insight or conclusion from the source

QUALITY RULES:
- Open with a specific compelling detail, quote, or fact from the source
- Reference specific researchers, thinkers, experiments, or studies mentioned in the source
- Do NOT use generic phrases — every sentence must reflect the actual content
- Use vivid, documentary-style language to bring the source material to life
- Total: 10-14 paragraphs

SOURCE MATERIAL (use ONLY this):
{context}

{custom_focus}

Respond with ONLY the JSON array:"""


DEBATE_TEMPLATE = """Write a debate script based EXCLUSIVELY on the source material below.

SPEAKERS: DEBATER_1 and DEBATER_2
STRUCTURE:
1. OPENING STATEMENTS (2-3 each): Each debater presents a position grounded in the source
2. MAIN DEBATE (12-16 exchanges): Arguments, counterarguments using specific source evidence
3. CLOSING STATEMENTS (1-2 each): Final position using source-based conclusions

QUALITY RULES:
- Both positions must be derived from the source material itself
- Each argument must cite specific evidence, examples, or ideas from the source
- Counterarguments must engage with the actual content, not generic debate tactics
- Total: 16-22 exchanges

SOURCE MATERIAL (use ONLY this):
{context}

{custom_focus}

Respond with ONLY the JSON array:"""


LECTURE_TEMPLATE = """Write a lecture script based EXCLUSIVELY on the source material below.

SPEAKER: PROFESSOR only
STRUCTURE:
1. INTRODUCTION (2-3 paragraphs): Introduce the topic using the opening ideas from the source
2. MAIN CONTENT (9-13 paragraphs): Teach each key concept, using specific examples and details from the source
3. SUMMARY (1-2 paragraphs): Summarise the core insights as stated in the source

QUALITY RULES:
- Every concept taught must come directly from the source material
- Include specific names, experiments, studies, and findings from the source
- Use clear educational language but grounded in the actual source content
- Total: 12-18 paragraphs

SOURCE MATERIAL (use ONLY this):
{context}

{custom_focus}

Respond with ONLY the JSON array:"""


STORYTELLING_TEMPLATE = """Write a storytelling script based EXCLUSIVELY on the source material below.

SPEAKER: STORYTELLER only
STRUCTURE:
1. THE HOOK (2 paragraphs): Open with a specific intriguing detail or scenario from the source
2. THE JOURNEY (9-13 paragraphs): Unfold the story using actual events, people, and ideas from the source
3. THE REVELATION (1-2 paragraphs): Deliver the key insight exactly as concluded in the source

QUALITY RULES:
- The story MUST be drawn from the actual content of the source, not invented narrative
- Weave in specific researchers, experiments, discoveries, or examples from the source
- Make it compelling and vivid using the real details present in the material
- Total: 6-9 paragraphs

SOURCE MATERIAL (use ONLY this):
{context}

{custom_focus}

Respond with ONLY the JSON array:"""


def get_template(style: str) -> str:
    """Get the prompt template for a given content style."""
    templates = {
        "podcast": PODCAST_TEMPLATE,
        "narration": NARRATION_TEMPLATE,
        "debate": DEBATE_TEMPLATE,
        "lecture": LECTURE_TEMPLATE,
        "storytelling": STORYTELLING_TEMPLATE,
    }
    return templates.get(style.lower(), NARRATION_TEMPLATE)


def build_prompt(context: str, style: str, custom_focus: str = "") -> str:
    """Build the full prompt for script generation safely."""
    template = get_template(style)
    
    focus_instruction = ""
    if custom_focus:
        focus_instruction = f"SPECIAL FOCUS: Emphasise content related to: {custom_focus}"
    
    # Use .replace instead of .format to avoid KeyError/ValueError if context has braces {}
    prompt = template.replace("{context}", context)
    prompt = prompt.replace("{custom_focus}", focus_instruction)
    
    return prompt
