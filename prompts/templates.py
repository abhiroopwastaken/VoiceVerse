"""
Prompt Templates for Script Generation
=======================================
Structured prompts for each content style.
"""

SYSTEM_PROMPT = """You are a professional script writer for audio content. 
You create engaging, well-structured scripts that sound natural when read aloud.
Your scripts must be based ONLY on the provided source material - do not invent facts."""


PODCAST_TEMPLATE = """Write a podcast script for TWO hosts discussing the following material.
The podcast should feel like a natural, engaging conversation between two knowledgeable hosts.

FORMAT RULES (FOLLOW EXACTLY):
- Use EXACTLY this format for each line: SPEAKER: dialogue text
- Use "HOST_A:" and "HOST_B:" as speaker labels
- Start with HOST_A introducing the topic
- Include natural conversational elements (reactions, questions, agreements)
- End with both hosts wrapping up with key takeaways
- Write 15-25 dialogue exchanges total
- Each line should be 1-3 sentences (suitable for spoken delivery)

STRUCTURE:
1. INTRO (2-3 exchanges): Greet listeners, introduce today's topic
2. BODY (10-18 exchanges): Deep dive into the key points from the material
3. CONCLUSION (2-3 exchanges): Summarize takeaways, thank listeners

SOURCE MATERIAL:
{context}

{custom_focus}

Write the podcast script now:"""


NARRATION_TEMPLATE = """Write a narration script for a single narrator presenting the following material.
The narration should be engaging, dramatic, and educational - like a documentary voiceover.

FORMAT RULES (FOLLOW EXACTLY):
- Use EXACTLY this format for each line: NARRATOR: narration text
- Use "NARRATOR:" as the speaker label
- Each paragraph should be 2-4 sentences
- Write 10-15 paragraphs total
- Use vivid language and rhetorical devices

STRUCTURE:
1. OPENING HOOK (1-2 paragraphs): Start with an attention-grabbing statement
2. BODY (7-11 paragraphs): Present the key information in a compelling narrative
3. CLOSING (1-2 paragraphs): End with a thought-provoking conclusion

SOURCE MATERIAL:
{context}

{custom_focus}

Write the narration script now:"""


DEBATE_TEMPLATE = """Write a debate script between TWO debaters discussing the following material.
They should present different perspectives and arguments, but remain respectful.

FORMAT RULES (FOLLOW EXACTLY):
- Use EXACTLY this format for each line: SPEAKER: dialogue text
- Use "DEBATER_1:" and "DEBATER_2:" as speaker labels
- Start with DEBATER_1 presenting an opening argument
- Include counterarguments, rebuttals, and evidence from the source
- End with both debaters giving closing statements
- Write 15-25 exchanges total
- Each turn should be 2-4 sentences

STRUCTURE:
1. OPENING STATEMENTS (2-3 exchanges each)
2. MAIN DEBATE (10-15 exchanges): Arguments and counterarguments
3. CLOSING STATEMENTS (1-2 exchanges each)

SOURCE MATERIAL:
{context}

{custom_focus}

Write the debate script now:"""


LECTURE_TEMPLATE = """Write a lecture script for a professor explaining the following material to students.
The lecture should be clear, educational, and engaging.

FORMAT RULES (FOLLOW EXACTLY):
- Use EXACTLY this format for each line: PROFESSOR: lecture text
- Use "PROFESSOR:" as the speaker label
- Each section should be 2-4 sentences
- Include examples and analogies to explain concepts
- Write 12-18 paragraphs total
- Use a warm, encouraging teaching tone

STRUCTURE:
1. INTRODUCTION (2-3 paragraphs): Welcome students, introduce the topic, explain why it matters
2. MAIN CONTENT (8-12 paragraphs): Teach the key concepts with examples
3. SUMMARY (2-3 paragraphs): Recap key points, preview what's next

SOURCE MATERIAL:
{context}

{custom_focus}

Write the lecture script now:"""


STORYTELLING_TEMPLATE = """Write a storytelling script that weaves the following material into an engaging narrative.
Think of it as telling a fascinating story around a campfire.

FORMAT RULES (FOLLOW EXACTLY):
- Use EXACTLY this format for each line: STORYTELLER: narrative text
- Use "STORYTELLER:" as the speaker label  
- Each section should be 2-4 sentences
- Use descriptive language, suspense, and emotional cues
- Write 12-18 paragraphs total
- Make it captivating and vivid

STRUCTURE:
1. THE HOOK (2-3 paragraphs): Set the scene, create intrigue
2. THE JOURNEY (8-12 paragraphs): Unfold the story based on the material
3. THE REVELATION (2-3 paragraphs): Deliver the key insight, leave the listener inspired

SOURCE MATERIAL:
{context}

{custom_focus}

Write the storytelling script now:"""


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
    """Build the full prompt for script generation."""
    template = get_template(style)
    
    focus_instruction = ""
    if custom_focus:
        focus_instruction = f"SPECIAL FOCUS: Pay particular attention to: {custom_focus}"
    
    return template.format(context=context, custom_focus=focus_instruction)
