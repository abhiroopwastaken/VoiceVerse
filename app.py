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
from modules.script_generator import generate_script
from modules.voice_generator import generate_audio, get_available_voices, SPEAKER_VOICE_MAPPING
from modules.audio_utils import merge_audio_segments, get_audio_duration, cleanup_temp_files


# ─── Global State ───────────────────────────────────────────────
rag_pipeline = None


# ─── Core Pipeline Function ────────────────────────────────────

def process_documents(files):
    """Step 1: Upload and process documents."""
    global rag_pipeline

    if not files:
        raise gr.Error("Please upload at least one document (PDF, TXT, or DOCX).")

    file_paths = []
    for f in files:
        if hasattr(f, 'name'):
            file_paths.append(f.name)
        elif isinstance(f, str):
            file_paths.append(f)
        else:
            file_paths.append(str(f))

    combined_text, metadata_list = extract_from_multiple(file_paths)

    if not combined_text.strip():
        raise gr.Error("Could not extract any text from the uploaded documents. Please check your files.")

    rag_pipeline = RAGPipeline(chunk_size=500, chunk_overlap=50)
    num_chunks = rag_pipeline.ingest(combined_text)

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
                f"- ✅ **{meta['filename']}** ({meta['type']}) — {wc:,} words, {meta.get('pages', '?')} page(s)"
            )

    summary_lines.append(f"\n**Total**: {total_words:,} words → **{num_chunks} chunks** indexed")
    summary = "\n".join(summary_lines)

    # Preview first 500 chars
    preview = combined_text[:500] + ("..." if len(combined_text) > 500 else "")

    return summary, preview


def generate_content(style, custom_focus, rate_value, pitch_value):
    """Step 2: Generate script and synthesize audio."""
    global rag_pipeline

    if rag_pipeline is None:
        raise gr.Error("Please upload and process documents first!")

    style_key = style.lower()

    # ── 1. Retrieve relevant context ──
    context = rag_pipeline.get_relevant_context(style_key, k=8)

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

    # ── 5. Merge audio ──
    output_path = tempfile.mktemp(suffix=".mp3", prefix="voiceverse_")
    final_audio = merge_audio_segments(segment_paths, output_path=output_path)

    duration = get_audio_duration(final_audio)

    # Cleanup individual segments
    cleanup_temp_files(segment_paths)

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
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ── Global ── */
.gradio-container {
    max-width: 1100px !important;
    margin: auto !important;
    font-family: 'Inter', 'Segoe UI', system-ui, sans-serif !important;
    background: linear-gradient(180deg, #EEF2FF 0%, #F9FAFB 40%, #FFFFFF 100%) !important;
}

/* Smooth all transitions */
* { transition: all 0.2s ease; }

/* ── Header Bar ── */
.app-header {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 14px;
    padding: 1.4rem 2rem;
    background: #FFFFFF;
    border-radius: 16px;
    margin-bottom: 1.2rem;
    border: 1px solid #E5E7EB;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}
.app-header .logo-icon {
    width: 48px; height: 48px;
    background: linear-gradient(135deg, #6366F1, #818CF8);
    border-radius: 14px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.5rem; color: white;
    box-shadow: 0 4px 12px rgba(99,102,241,0.25);
    flex-shrink: 0;
}
.app-header .brand-name {
    font-size: 1.6rem;
    font-weight: 700;
    color: #6366F1;
    letter-spacing: -0.5px;
}
.app-header .tagline {
    font-size: 0.95rem;
    color: #6B7280;
    font-weight: 400;
    margin-left: 0.5rem;
    padding-left: 0.75rem;
    border-left: 2px solid #E5E7EB;
}

/* ── Hero / Intro Banner ── */
.hero-banner {
    text-align: center;
    padding: 2.5rem 2rem 2rem;
    margin-bottom: 1.5rem;
}
.hero-banner .badge {
    display: inline-block;
    background: #EEF2FF;
    color: #6366F1;
    font-size: 0.78rem;
    font-weight: 600;
    padding: 6px 16px;
    border-radius: 100px;
    margin-bottom: 1rem;
    letter-spacing: 0.3px;
    border: 1px solid #C7D2FE;
}
.hero-banner h2 {
    font-size: 2.2rem;
    font-weight: 800;
    color: #111827;
    margin: 0 0 0.6rem;
    letter-spacing: -0.8px;
    line-height: 1.2;
}
.hero-banner p {
    color: #6B7280;
    font-size: 1.05rem;
    max-width: 600px;
    margin: 0 auto;
    line-height: 1.6;
}

/* ── Stepper ── */
.step-indicator {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 0;
    margin: 0 auto 1.8rem;
    max-width: 500px;
}
.step-item {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 0.88rem;
    font-weight: 600;
    color: #9CA3AF;
}
.step-item.active {
    color: #6366F1;
}
.step-num {
    width: 30px; height: 30px;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.82rem; font-weight: 700;
    background: #F3F4F6;
    color: #9CA3AF;
    border: 2px solid #E5E7EB;
}
.step-item.active .step-num {
    background: #6366F1;
    color: white;
    border-color: #6366F1;
    box-shadow: 0 2px 8px rgba(99,102,241,0.3);
}
.step-item.completed .step-num {
    background: #10B981;
    color: white;
    border-color: #10B981;
}
.step-connector {
    width: 60px;
    height: 2px;
    background: #E5E7EB;
    margin: 0 8px;
    border-radius: 2px;
}
.step-connector.active {
    background: linear-gradient(90deg, #6366F1, #818CF8);
}

/* ── Section Cards ── */
.section-card {
    background: #FFFFFF;
    border-radius: 16px;
    padding: 2rem;
    margin-bottom: 1.2rem;
    border: 1px solid #E5E7EB;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04), 0 1px 2px rgba(0,0,0,0.02);
}
.section-card:hover {
    box-shadow: 0 4px 12px rgba(0,0,0,0.06);
}
.section-title {
    font-size: 1.15rem;
    font-weight: 700;
    color: #111827;
    margin: 0 0 0.3rem;
    display: flex;
    align-items: center;
    gap: 8px;
}
.section-subtitle {
    font-size: 0.88rem;
    color: #6B7280;
    margin: 0;
    line-height: 1.5;
}

/* ── Primary Button (Indigo) ── */
.primary-btn {
    background: linear-gradient(135deg, #6366F1, #818CF8) !important;
    border: none !important;
    color: white !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    padding: 12px 32px !important;
    border-radius: 12px !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 14px rgba(99,102,241,0.25) !important;
    letter-spacing: 0.2px !important;
}
.primary-btn:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(99,102,241,0.35) !important;
    background: linear-gradient(135deg, #4F46E5, #6366F1) !important;
}

/* ── Generate Button (larger) ── */
.generate-btn {
    background: linear-gradient(135deg, #6366F1, #818CF8) !important;
    border: none !important;
    color: white !important;
    font-weight: 700 !important;
    font-size: 1.1rem !important;
    padding: 14px 48px !important;
    border-radius: 14px !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 18px rgba(99,102,241,0.3) !important;
    letter-spacing: 0.3px !important;
}
.generate-btn:hover {
    transform: translateY(-3px) !important;
    box-shadow: 0 8px 25px rgba(99,102,241,0.4) !important;
    background: linear-gradient(135deg, #4F46E5, #6366F1) !important;
}

/* ── Audio Output Card ── */
.audio-player {
    border: 2px solid #C7D2FE !important;
    border-radius: 14px !important;
    padding: 1.2rem !important;
    background: #FAFAFE !important;
}

/* ── Status Card ── */
.status-card {
    background: linear-gradient(135deg, #ECFDF5, #F0FDF4) !important;
    border: 1px solid #A7F3D0 !important;
    border-radius: 12px !important;
    padding: 1rem 1.5rem !important;
}

/* ── How It Works Section ── */
.how-it-works {
    text-align: center;
    padding: 2rem 0 0.5rem;
    margin-top: 1rem;
}
.how-it-works h3 {
    font-size: 1.6rem;
    font-weight: 800;
    color: #111827;
    margin-bottom: 0.4rem;
}
.how-it-works p {
    color: #6B7280;
    font-size: 0.95rem;
    margin-bottom: 1.5rem;
}
.feature-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 16px;
    margin-bottom: 1.5rem;
}
@media (max-width: 768px) {
    .feature-grid { grid-template-columns: repeat(2, 1fr); }
}
@media (max-width: 480px) {
    .feature-grid { grid-template-columns: 1fr; }
}
.feature-card {
    background: #FFFFFF;
    border: 1px solid #E5E7EB;
    border-radius: 14px;
    padding: 1.5rem 1.2rem;
    text-align: left;
    transition: all 0.3s ease;
}
.feature-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 24px rgba(0,0,0,0.06);
    border-color: #C7D2FE;
}
.feature-icon {
    width: 48px; height: 48px;
    border-radius: 12px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.3rem;
    margin-bottom: 1rem;
}
.feature-icon.upload { background: #DBEAFE; color: #3B82F6; }
.feature-icon.rag { background: #F3E8FF; color: #A855F7; }
.feature-icon.script { background: #FCE7F3; color: #EC4899; }
.feature-icon.voice { background: #EEF2FF; color: #6366F1; }
.feature-card h4 {
    font-size: 1rem;
    font-weight: 700;
    color: #111827;
    margin: 0 0 0.5rem;
}
.feature-card p {
    font-size: 0.84rem;
    color: #6B7280;
    margin: 0;
    line-height: 1.5;
}

/* ── Footer ── */
.app-footer {
    text-align: center;
    padding: 1.8rem 1rem;
    margin-top: 0.5rem;
}
.app-footer .footer-logo {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 8px;
    margin-bottom: 0.5rem;
}
.app-footer .footer-logo .f-icon {
    width: 32px; height: 32px;
    background: #F3F4F6;
    border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.95rem;
}
.app-footer .footer-logo span {
    font-weight: 600;
    color: #9CA3AF;
    font-size: 0.95rem;
}
.app-footer p {
    color: #9CA3AF;
    font-size: 0.82rem;
    margin: 0.2rem 0;
}

/* ── Gradio Component Overrides ── */
.gr-group { border-radius: 14px !important; }
.gr-box { border-radius: 12px !important; }
.gr-padded { padding: 1rem !important; }
.gr-form { border-radius: 12px !important; }
.gr-input, .gr-text-input textarea {
    border-radius: 10px !important;
    border-color: #D1D5DB !important;
}
.gr-input:focus, .gr-text-input textarea:focus {
    border-color: #6366F1 !important;
    box-shadow: 0 0 0 3px rgba(99,102,241,0.1) !important;
}

/* Fix Gradio component backgrounds in light mode */
.block { background: transparent !important; }
.wrap { border-radius: 14px !important; }
"""


# ─── Build Gradio App ──────────────────────────────────────────

def create_app():
    with gr.Blocks(
        css=CUSTOM_CSS,
        theme=gr.themes.Soft(
            primary_hue="indigo",
            secondary_hue="purple",
            neutral_hue="slate",
        ),
        title="VoiceVerse — AI Document to Audio",
    ) as app:

        # ── Header Bar ──
        gr.HTML("""
        <div class="app-header">
            <div class="logo-icon">🎙️</div>
            <span class="brand-name">VoiceVerse</span>
            <span class="tagline">AI-Powered Audio Generation</span>
        </div>
        """)

        # ── Hero Banner ──
        gr.HTML("""
        <div class="hero-banner">
            <span class="badge">✨ AI-Powered Audio Generation</span>
            <h2>Turn Documents into Lifelike<br>Audio Experiences</h2>
            <p>Upload any document and our RAG pipeline extracts key insights,
               generates structured scripts, and synthesizes natural-sounding audio.</p>
        </div>
        """)

        # ── Stepper ──
        gr.HTML("""
        <div class="step-indicator">
            <div class="step-item active">
                <div class="step-num">1</div>
                <span>Upload</span>
            </div>
            <div class="step-connector"></div>
            <div class="step-item">
                <div class="step-num">2</div>
                <span>Configure</span>
            </div>
            <div class="step-connector"></div>
            <div class="step-item">
                <div class="step-num">3</div>
                <span>Generate</span>
            </div>
        </div>
        """)

        # ═══════════════════════════════════════════
        # STEP 1: Document Upload
        # ═══════════════════════════════════════════
        gr.HTML("""
        <div class="section-card">
            <div class="section-title">📄 Upload Your Documents</div>
            <div class="section-subtitle">
                Drag & drop PDF, DOCX, or TXT files. Your content will be analyzed and indexed for audio generation.
            </div>
        </div>
        """)

        with gr.Row():
            with gr.Column(scale=2):
                file_upload = gr.File(
                    label="Drag & drop PDF, DOCX, or TXT files",
                    file_count="multiple",
                    file_types=[".pdf", ".txt", ".docx"],
                )
            with gr.Column(scale=1):
                upload_btn = gr.Button(
                    "📥 Process Documents",
                    variant="primary",
                    elem_classes=["primary-btn"],
                )

        doc_summary = gr.Markdown(value="*Upload documents to see processing summary*")

        with gr.Accordion("📃 Document Preview", open=False):
            doc_preview = gr.Textbox(
                label="Extracted Text (first 500 chars)",
                lines=6,
                interactive=False,
                max_lines=10,
            )

        # ═══════════════════════════════════════════
        # STEP 2: Configure & Generate
        # ═══════════════════════════════════════════
        gr.HTML("""
        <div class="section-card" style="margin-top: 0.5rem;">
            <div class="section-title">🎨 Configure Your Audio</div>
            <div class="section-subtitle">
                Choose a content style, optionally set a focus topic, and fine-tune voice parameters.
            </div>
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
                        label="🏃 Speech Rate",
                        info="Faster (+) or slower (-)",
                    )
                    pitch_slider = gr.Slider(
                        minimum=-10,
                        maximum=10,
                        value=0,
                        step=1,
                        label="🎵 Pitch",
                        info="Higher (+) or lower (-)",
                    )

        generate_btn = gr.Button(
            "🎙️ Generate Audio →",
            variant="primary",
            elem_classes=["generate-btn"],
        )

        # ═══════════════════════════════════════════
        # STEP 3: Output & Playback
        # ═══════════════════════════════════════════
        gr.HTML("""
        <div class="section-card" style="margin-top: 0.5rem;">
            <div class="section-title">🔊 Listen & Download</div>
            <div class="section-subtitle">
                Your generated audio will appear below. You can play it directly or download the file.
            </div>
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
                <div class="feature-card">
                    <div class="feature-icon upload">📄</div>
                    <h4>Upload & Ingest</h4>
                    <p>Drag & drop PDF, TXT files or paste links. System chunks text for processing.</p>
                </div>
                <div class="feature-card">
                    <div class="feature-icon rag">⚙️</div>
                    <h4>RAG Pipeline</h4>
                    <p>Vector search retrieves the most relevant context for your topic.</p>
                </div>
                <div class="feature-card">
                    <div class="feature-icon script">✨</div>
                    <h4>Script Generation</h4>
                    <p>LLMs craft a structured script with intro, body, and conclusion.</p>
                </div>
                <div class="feature-card">
                    <div class="feature-icon voice">🎙️</div>
                    <h4>Voice Synthesis</h4>
                    <p>Neural TTS models generate human-like speech with emotion.</p>
                </div>
            </div>
        </div>
        """)

        # ── Footer ──
        gr.HTML("""
        <div class="app-footer">
            <div class="footer-logo">
                <div class="f-icon">🎙️</div>
                <span>VoiceVerse</span>
            </div>
            <p>© 2026 VoiceVerse Sprint. Built for the AI Challenge.</p>
            <p style="margin-top: 0.3rem; font-size: 0.75rem;">
                Models: all-MiniLM-L6-v2 (embeddings) · Qwen2.5-72B-Instruct (scripts) · Microsoft Edge TTS (voice)
            </p>
        </div>
        """)

        # ── Event Handlers ──
        upload_btn.click(
            fn=process_documents,
            inputs=[file_upload],
            outputs=[doc_summary, doc_preview],
        )

        generate_btn.click(
            fn=generate_content,
            inputs=[style_select, custom_focus, rate_slider, pitch_slider],
            outputs=[audio_output, script_output, gen_status],
        )

    return app


# ─── Launch ─────────────────────────────────────────────────────

if __name__ == "__main__":
    app = create_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
    )
