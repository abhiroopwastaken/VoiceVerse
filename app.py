"""
VoiceVerse — AI Document-to-Audio Application
==============================================
Transform documents into engaging synthetic audio content:
podcasts, narrations, debates, lectures, and storytelling.

Built with Gradio, sentence-transformers, FAISS, edge-tts.
"""

import os
import sys
import tempfile
import gradio as gr
from typing import List, Tuple, Optional

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import CONTENT_STYLES, VOICE_MAP
from modules.document_loader import extract_text, extract_from_multiple
from modules.rag_pipeline import RAGPipeline
from modules.script_generator import generate_script, generate_summary
from modules.voice_generator import generate_audio, get_available_voices, SPEAKER_VOICE_MAPPING
from modules.audio_utils import merge_audio_segments, get_audio_duration, cleanup_temp_files


# ─── Global State ───────────────────────────────────────────────
rag_pipeline = None


# ─── Core Pipeline Function ────────────────────────────────────

def process_documents(files, text_input):
    """Step 1: Upload and process documents."""
    global rag_pipeline

    try:
        if not files and not text_input.strip():
            raise gr.Error("Please upload a document or enter text.")

        print(f"DEBUG: Starting document processing. Files: {len(files) if files else 0}, Text Input: {len(text_input)} chars")
        
        file_paths = []
        if files:
            for f in files:
                if hasattr(f, 'name'):
                    file_paths.append(f.name)
                elif isinstance(f, str):
                    file_paths.append(f)
                else:
                    file_paths.append(str(f))

        print("DEBUG: Extracting text...")
        combined_text, metadata_list = extract_from_multiple(file_paths)
        
        # Append direct text input
        if text_input.strip():
            combined_text += "\n\n" + text_input
            metadata_list.append({"filename": "Direct Text Input", "type": "TXT", "word_count": len(text_input.split())})

        if not combined_text.strip():
            raise gr.Error("Could not extract any text. Please check your inputs.")

        print(f"DEBUG: Text extraction complete. Total words: {len(combined_text.split())}")
        
        print("DEBUG: Initializing RAG pipeline and ingesting text...")
        rag_pipeline = RAGPipeline()
        num_chunks = rag_pipeline.ingest(combined_text)
        print(f"DEBUG: Ingestion complete. {num_chunks} chunks created.")

        # Build summary
        summary_lines = ["### 📊 Document Processing Summary\n"]
        total_words = 0
        for meta in metadata_list:
            if meta.get("type") == "ERROR":
                summary_lines.append(f"- ❌ **{meta['filename']}**: {meta.get('error', 'Unknown error')}")
            else:
                wc = meta.get('word_count', 0)
                total_words += wc
                summary_lines.append(
                    f"- ✅ **{meta['filename']}** ({meta['type']}) — {wc:,} words"
                )

        summary_lines.append(f"\n**Total**: {total_words:,} words → **{num_chunks} chunks** indexed")
        summary = "\n".join(summary_lines)

        print("DEBUG: Generating content summary...")
        # Generate Content Summary
        content_summary = generate_summary(combined_text)
        print("DEBUG: Content summary generated.")

        return summary, content_summary

    except Exception as e:
        import traceback
        error_msg = f"Error processing documents: {str(e)}"
        print(f"ERROR: {error_msg}")
        print(traceback.format_exc())
        raise gr.Error(error_msg)


def generate_content(style, custom_focus, rate_value, pitch_value):
    """Step 2: Generate script and synthesize audio."""
    global rag_pipeline

    if rag_pipeline is None:
        raise gr.Error("Please upload and process documents first!")

    style_key = style.lower()

    # ── 1. Retrieve relevant context ──
    context = rag_pipeline.get_relevant_context(style_key, k=8, custom_focus=custom_focus)

    if not context.strip():
        context = rag_pipeline.get_full_context(max_chunks=10)

    # ── 2. Generate script ──
    script = generate_script(context, style_key, custom_focus)

    if not script:
        raise gr.Error("Failed to generate script. Please try again or select a different style.")

    # ── 3. Format script for display ──
    script_display = _format_script_display(script, style)

    # ── 4. Synthesize audio ──
    rate_str = f"+{rate_value}%" if rate_value >= 0 else f"{rate_value}%"
    pitch_str = f"+{pitch_value}Hz" if pitch_value >= 0 else f"{pitch_value}Hz"

    segment_paths = generate_audio(script, rate=rate_str, pitch=pitch_str)

    if not segment_paths:
        raise gr.Error("Failed to generate audio. Please try again.")

    # generate_audio now returns the merged final path directly
    if isinstance(segment_paths, list):
        # Backward compat: if a list is returned, merge it
        output_path = tempfile.mktemp(suffix=".mp3", prefix="voiceverse_")
        final_audio = merge_audio_segments(segment_paths, output_path=output_path)
        cleanup_temp_files(segment_paths)
    else:
        # New: single merged path returned directly
        final_audio = segment_paths

    duration = get_audio_duration(final_audio)

    status = f"### ✅ Generation Complete!\n\n"
    status += f"- **Style**: {style}\n"
    status += f"- **Segments**: {len(script)} speaker turns\n"
    status += f"- **Duration**: {duration:.1f} seconds\n"
    status += f"- **Speech Rate**: {rate_str}\n"

    return final_audio, script_display, status


def _format_script_display(script: List[Tuple[str, str]], style: str) -> str:
    """Format the script for display in the UI."""
    lines = [f"## 📝 Generated {style} Script\n"]

    speaker_emoji = {
        "HOST_A": "🎙️",
        "HOST_B": "🎧",
        "NARRATOR": "📖",
        "PROFESSOR": "👩‍🏫",
        "STORYTELLER": "📚",
        "DEBATER_1": "💬",
        "DEBATER_2": "💭",
    }

    for speaker, text in script:
        emoji = speaker_emoji.get(speaker, "🗣️")
        display_name = speaker.replace("_", " ").title()
        lines.append(f"**{emoji} {display_name}:**\n{text}\n")

    return "\n".join(lines)


# ─── Custom CSS ─────────────────────────────────────────────────

CUSTOM_CSS = """
/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ── Global ── */
body, .gradio-container {
    background: #0f172a !important; /* Dark Slate Background */
    color: #e2e8f0 !important;
    font-family: 'Inter', system-ui, sans-serif !important;
}

/* Remove side borders/margins on large screens */
.gradio-container {
    max-width: 100% !important;
    padding: 0 !important;
    margin: 0 !important;
    min-height: 100vh;
    display: flex;
    flex-direction: column;
}

/* Main Content Wrapper to center content nicely */
.main-content {
    max-width: 1200px;
    margin: 0 auto;
    width: 100%;
    padding: 2rem;
    flex: 1;
}

/* ── Typography ── */
h1, h2, h3, h4, h5, h6 {
    font-family: 'Space Grotesk', sans-serif !important;
    color: #f8fafc !important;
}

/* ── Glassmorphism Cards ── */
.glass-card {
    background: rgba(30, 41, 59, 0.7);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid rgba(255, 255, 255, 0.1);
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    border-radius: 16px;
    padding: 2rem;
    margin-bottom: 1.5rem;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}

.glass-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.2), 0 4px 6px -2px rgba(0, 0, 0, 0.1);
    border-color: rgba(99, 102, 241, 0.4); /* Indigo glow */
}

/* ── Header ── */
.app-header {
    display: flex;
    align-items: center;
    gap: 16px;
    padding-bottom: 2rem;
    border-bottom: 1px solid rgba(255, 255, 255, 0.1);
    margin-bottom: 2rem;
}
.app-header .logo-icon {
    font-size: 2rem;
    background: linear-gradient(135deg, #6366f1 0%, #a855f7 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    filter: drop-shadow(0 2px 4px rgba(99, 102, 241, 0.3));
}
.app-header .brand-name {
    font-size: 1.8rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    background: linear-gradient(to right, #f8fafc, #94a3b8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.app-header .tagline {
    margin-left: auto;
    font-size: 0.9rem;
    color: #94a3b8;
    font-weight: 500;
    padding: 0.5rem 1rem;
    background: rgba(255, 255, 255, 0.05);
    border-radius: 999px;
    border: 1px solid rgba(255, 255, 255, 0.1);
}

/* ── Hero Section ── */
.hero-section {
    text-align: center;
    padding: 3rem 1rem;
    margin-bottom: 3rem;
    position: relative;
    overflow: hidden;
}
.hero-section::before {
    content: '';
    position: absolute;
    top: 50%;
    left: 50%;
    width: 600px;
    height: 600px;
    background: radial-gradient(circle, rgba(99, 102, 241, 0.15) 0%, rgba(0, 0, 0, 0) 70%);
    transform: translate(-50%, -50%);
    z-index: -1;
    pointer-events: none;
}
.hero-title {
    font-size: 3.5rem;
    font-weight: 800;
    line-height: 1.1;
    margin-bottom: 1.5rem;
    background: linear-gradient(to bottom right, #ffffff, #cbd5e1);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-shadow: 0 4px 12px rgba(0,0,0,0.5);
}
.hero-subtitle {
    font-size: 1.25rem;
    color: #94a3b8;
    max-width: 600px;
    margin: 0 auto;
    line-height: 1.6;
}

/* ── Stepper ── */
.stepper-container {
    display: flex;
    justify-content: center;
    align-items: center;
    margin-bottom: 3rem;
    gap: 1rem;
    position: relative;
    z-index: 10;
}
.step {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    background: rgba(30, 41, 59, 0.5);
    padding: 0.75rem 1.5rem;
    border-radius: 999px;
    border: 1px solid rgba(255, 255, 255, 0.05);
    color: #64748b;
    font-weight: 600;
    transition: all 0.3s ease;
}
.step.active {
    background: rgba(99, 102, 241, 0.2);
    border-color: rgba(99, 102, 241, 0.5);
    color: #e2e8f0;
    box-shadow: 0 0 15px rgba(99, 102, 241, 0.2);
}
.step-number {
    width: 28px;
    height: 28px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.85rem;
    background: rgba(255, 255, 255, 0.1);
}
.step.active .step-number {
    background: #6366f1;
    color: white;
}
.step-line {
    height: 2px;
    width: 40px;
    background: rgba(255, 255, 255, 0.1);
}

/* ── Components Overrides ── */
.gr-box, .gr-input, .gr-text-input, textarea, .gr-dropdown {
    background-color: rgba(15, 23, 42, 0.6) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    color: #f1f5f9 !important;
}
.gr-input:focus, textarea:focus {
    border-color: #6366f1 !important;
    box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.2) !important;
}
label {
    color: #94a3b8 !important;
    font-weight: 500 !important;
    margin-bottom: 0.5rem !important;
}
.primary-btn {
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%) !important;
    border: none !important;
    color: white !important;
    font-weight: 600 !important;
    padding: 0.75rem 1.5rem !important;
    border-radius: 12px !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 4px 6px -1px rgba(99, 102, 241, 0.3) !important;
}
.primary-btn:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 10px 15px -3px rgba(99, 102, 241, 0.4) !important;
    filter: brightness(110%) !important;
}

/* ── Audio Player ── */
.audio-player {
    background: rgba(15, 23, 42, 0.6) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 12px !important;
}

/* ── Feature Grid ── */
.feature-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
    gap: 1.5rem;
    margin-top: 4rem;
}
.feature-item {
    background: rgba(30, 41, 59, 0.4);
    border: 1px solid rgba(255, 255, 255, 0.05);
    border-radius: 16px;
    padding: 1.5rem;
    transition: all 0.3s ease;
}
.feature-item:hover {
    background: rgba(30, 41, 59, 0.7);
    border-color: rgba(99, 102, 241, 0.3);
    transform: translateY(-5px);
}
.feature-icon-wrapper {
    width: 48px;
    height: 48px;
    border-radius: 12px;
    background: rgba(99, 102, 241, 0.1);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.5rem;
    color: #818cf8;
    margin-bottom: 1rem;
}
.feature-title {
    font-size: 1.1rem;
    font-weight: 700;
    color: #f1f5f9;
    margin-bottom: 0.5rem;
}
.feature-desc {
    color: #94a3b8;
    font-size: 0.95rem;
    line-height: 1.5;
}

/* ── Footer ── */
.app-footer {
    text-align: center;
    padding: 3rem 0;
    margin-top: auto;
    border-top: 1px solid rgba(255, 255, 255, 0.05);
    color: #64748b;
}

/* Make accordion label readable */
.gr-accordion .label-wrap {
    color: #e2e8f0 !important;
}
"""


# ─── Build Gradio App ──────────────────────────────────────────

def create_app():
    with gr.Blocks(
        css=CUSTOM_CSS,
        title="VoiceVerse — AI Document to Audio",
        theme=gr.themes.Base(
            primary_hue="indigo",
            neutral_hue="slate",
            font=("Inter", "sans-serif")
        )
    ) as app:

        with gr.Column(elem_classes=["main-content"]):

            # ── Header Bar ──
            gr.HTML("""
            <div class="app-header">
                <div class="logo-icon">🎙️</div>
                <div class="brand-name">VoiceVerse</div>
                <div class="tagline">AI-Powered Audio Generation</div>
            </div>
            """)

            # ── Hero Section ──
            gr.HTML("""
            <div class="hero-section">
                <h1 class="hero-title">Turn Documents into<br>Lifelike Audio</h1>
                <p class="hero-subtitle">
                    Upload any document and our RAG pipeline extracts key insignts, generates structured scripts, and synthesizes natural-sounding audio with emotion.
                </p>
            </div>
            """)

            # ── Stepper ──
            gr.HTML("""
            <div class="stepper-container">
                <div class="step active">
                    <div class="step-number">1</div>
                    <span>Upload</span>
                </div>
                <div class="step-line"></div>
                <div class="step">
                    <div class="step-number">2</div>
                    <span>Configure</span>
                </div>
                <div class="step-line"></div>
                <div class="step">
                    <div class="step-number">3</div>
                    <span>Generate</span>
                </div>
            </div>
            """)

            # ═══════════════════════════════════════════
            # STEP 1: Document Upload
            # ═══════════════════════════════════════════
            with gr.Column(elem_classes=["glass-card"]):
                gr.HTML("""
                <div class="section-header">
                    <h3 style="margin-bottom: 0.5rem">📄 Upload Your Documents</h3>
                    <p style="color: #94a3b8; margin-bottom: 1.5rem;">
                        Drag & drop PDF, DOCX, or TXT files. Your content will be analyzed and indexed for audio generation.
                    </p>
                </div>
                """)

                with gr.Row():
                    with gr.Column(scale=1):
                        file_upload = gr.File(
                            label="Upload Documents",
                            file_count="multiple",
                            file_types=[".pdf", ".txt", ".docx"],
                            elem_classes=["file-upload-box"],
                        )
                    with gr.Column(scale=1):
                        text_input = gr.Textbox(
                            label="Or Paste Text Directly",
                            placeholder="Enter text here directly...",
                            lines=8,
                            show_copy_button=True,
                        )
                
                with gr.Row():
                     upload_btn = gr.Button(
                        "📥 Process Content",
                        variant="primary",
                        elem_classes=["primary-btn"],
                    )

                doc_summary = gr.Markdown(value="*Upload documents or enter text to see processing summary*")

                with gr.Accordion("📃 Document Summary", open=True):
                    doc_preview = gr.Markdown(value="*AI-generated summary will appear here.*")

            # ═══════════════════════════════════════════
            # STEP 2: Configure & Generate
            # ═══════════════════════════════════════════
            with gr.Column(elem_classes=["glass-card"]):
                gr.HTML("""
                <div class="section-header">
                    <h3 style="margin-bottom: 0.5rem">🎨 Configure Your Audio</h3>
                    <p style="color: #94a3b8; margin-bottom: 1.5rem;">
                        Choose a content style, optionally set a focus topic, and fine-tune voice parameters.
                    </p>
                </div>
                """)

                with gr.Row():
                    with gr.Column(scale=1):
                        style_select = gr.Radio(
                            choices=CONTENT_STYLES,
                            value="Podcast",
                            label="Content Style",
                            info="Choose how your content should be presented",
                        )

                    with gr.Column(scale=1):
                        custom_focus = gr.Textbox(
                            label="Custom Focus (optional)",
                            placeholder="e.g., 'Focus on the impact of AI on healthcare'",
                            info="Guide the script towards specific topics",
                            lines=2,
                        )

                        with gr.Row():
                            rate_slider = gr.Slider(
                                minimum=-30,
                                maximum=30,
                                value=0,
                                step=5,
                                label="Speech Rate (+/- %)",
                            )
                            pitch_slider = gr.Slider(
                                minimum=-10,
                                maximum=10,
                                value=0,
                                step=1,
                                label="Pitch (+/- Hz)",
                            )

                generate_btn = gr.Button(
                    "🎙️ Generate Audio →",
                    variant="primary",
                    elem_classes=["primary-btn", "generate-btn"],
                )

            # ═══════════════════════════════════════════
            # STEP 3: Output & Playback
            # ═══════════════════════════════════════════
            with gr.Column(elem_classes=["glass-card"]):
                gr.HTML("""
                <div class="section-header">
                    <h3 style="margin-bottom: 0.5rem">🔊 Listen & Download</h3>
                    <p style="color: #94a3b8; margin-bottom: 1.5rem;">
                        Your generated audio will appear below. You can play it directly or download the file.
                    </p>
                </div>
                """)

                gen_status = gr.Markdown(value="*Generate audio to see results here*")

                audio_output = gr.Audio(
                    label="Generated Audio",
                    type="filepath",
                    elem_classes=["audio-player"],
                )

                with gr.Accordion("📝 Generated Script", open=False):
                    script_output = gr.Markdown(value="*Script will appear here after generation*")

            # ═══════════════════════════════════════════
            # HOW IT WORKS
            # ═══════════════════════════════════════════
            gr.HTML("""
            <div class="how-it-works">
                <h3>How It Works</h3>
                <p>From raw text to polished audio in four simple steps</p>
                <div class="feature-grid">
                    <div class="feature-item">
                        <div class="feature-icon-wrapper">📄</div>
                        <div class="feature-title">Upload & Ingest</div>
                        <div class="feature-desc">Drag & drop PDF, TXT files or paste links. System chunks text for processing.</div>
                    </div>
                    <div class="feature-item">
                        <div class="feature-icon-wrapper">⚙️</div>
                        <div class="feature-title">RAG Pipeline</div>
                        <div class="feature-desc">Vector search retrieves the most relevant context for your topic.</div>
                    </div>
                    <div class="feature-item">
                        <div class="feature-icon-wrapper">✨</div>
                        <div class="feature-title">Script Generation</div>
                        <div class="feature-desc">LLMs craft a structured script with intro, body, and conclusion.</div>
                    </div>
                    <div class="feature-item">
                        <div class="feature-icon-wrapper">🎙️</div>
                        <div class="feature-title">Voice Synthesis</div>
                        <div class="feature-desc">Neural TTS models generate human-like speech with emotion.</div>
                    </div>
                </div>
            </div>
            """)

            # ── Footer ──
            gr.HTML("""
            <div class="app-footer">
                <p>© 2026 VoiceVerse. Built for the AI Challenge.</p>
                <p style="font-size: 0.75rem; opacity: 0.7;">
                    Models: all-MiniLM-L6-v2 (embeddings) · Qwen2.5-72B-Instruct (scripts) · Microsoft Edge TTS (voice)
                </p>
            </div>
            """)

            # ── Event Handlers ──
            upload_btn.click(
                fn=process_documents,
                inputs=[file_upload, text_input],
                outputs=[doc_summary, doc_preview],
            )

            generate_btn.click(
                fn=generate_content,
                inputs=[style_select, custom_focus, rate_slider, pitch_slider],
                outputs=[audio_output, script_output, gen_status],
            )

    return app


# ─── Launch ─────────────────────────────────────────────────────

app = create_app()

if __name__ == "__main__":
    app.launch(
        server_name="0.0.0.0", 
        server_port=7860, 
        ssr_mode=False, 
        show_api=False
    )

