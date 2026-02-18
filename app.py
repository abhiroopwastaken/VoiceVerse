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
/* ── Global ── */
.gradio-container {
    max-width: 1100px !important;
    margin: auto !important;
    font-family: 'Inter', 'Segoe UI', system-ui, sans-serif !important;
    background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%) !important;
}

/* ── Header ── */
.app-header {
    text-align: center;
    padding: 2rem 1rem 1rem;
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    border-radius: 16px;
    margin-bottom: 1.5rem;
    border: 1px solid rgba(255,255,255,0.1);
}
.app-header h1 {
    font-size: 2.4rem;
    background: linear-gradient(90deg, #e94560, #f5a623, #e94560);
    background-size: 200% auto;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: shine 3s ease-in-out infinite;
    margin-bottom: 0.3rem;
}
@keyframes shine {
    to { background-position: 200% center; }
}
.app-header p {
    color: #a0aec0;
    font-size: 1.05rem;
    margin-top: 0;
}

/* ── Buttons ── */
.primary-btn {
    background: linear-gradient(135deg, #e94560, #c23152) !important;
    border: none !important;
    color: white !important;
    font-weight: 600 !important;
    font-size: 1.1rem !important;
    padding: 12px 32px !important;
    border-radius: 10px !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 15px rgba(233,69,96,0.3) !important;
}
.primary-btn:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(233,69,96,0.4) !important;
}

/* ── Audio Player ── */
.audio-player {
    border: 2px solid rgba(233,69,96,0.3) !important;
    border-radius: 12px !important;
    padding: 1rem !important;
}

/* ── Footer ── */
.app-footer {
    text-align: center;
    padding: 1rem;
    color: #718096;
    font-size: 0.85rem;
    border-top: 1px solid rgba(255,255,255,0.05);
    margin-top: 1rem;
}
"""


# ─── Build Gradio App ──────────────────────────────────────────

def create_app():
    with gr.Blocks(
        css=CUSTOM_CSS,
        theme=gr.themes.Soft(
            primary_hue="red",
            secondary_hue="blue",
            neutral_hue="slate",
        ),
        title="VoiceVerse — AI Document to Audio",
    ) as app:

        # ── Header ──
        gr.HTML("""
        <div class="app-header">
            <h1>🎙️ VoiceVerse</h1>
            <p>Transform your documents into captivating audio experiences</p>
            <p style="font-size:0.85rem; color:#718096; margin-top:0.5rem">
                Podcast • Narration • Debate • Lecture • Storytelling
            </p>
        </div>
        """)

        # ═══════════════════════════════════════════
        # STEP 1: Document Upload
        # ═══════════════════════════════════════════
        gr.Markdown("### 📄 Step 1: Upload Your Documents")
        gr.Markdown("*Upload PDF, TXT, or DOCX files. Your content will be analyzed and indexed for audio generation.*")

        with gr.Row():
            with gr.Column(scale=2):
                file_upload = gr.File(
                    label="Drop files here or click to browse",
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

        gr.Markdown("---")

        # ═══════════════════════════════════════════
        # STEP 2: Configure & Generate
        # ═══════════════════════════════════════════
        gr.Markdown("### 🎨 Step 2: Configure Your Audio")

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
            "🎙️ Generate Audio",
            variant="primary",
            elem_classes=["primary-btn"],
        )

        gr.Markdown("---")

        # ═══════════════════════════════════════════
        # STEP 3: Output & Playback
        # ═══════════════════════════════════════════
        gr.Markdown("### 🔊 Step 3: Listen & Download")

        gen_status = gr.Markdown(value="*Generate audio to see results here*")

        audio_output = gr.Audio(
            label="Generated Audio",
            type="filepath",
            elem_classes=["audio-player"],
        )

        with gr.Accordion("📝 Generated Script", open=False):
            script_output = gr.Markdown(value="*Script will appear here after generation*")

        # ── Footer ──
        gr.HTML("""
        <div class="app-footer">
            <p>🎙️ VoiceVerse — Built with Gradio, sentence-transformers, FAISS, and edge-tts</p>
            <p>⚠️ Audio content is synthetically generated. Models: all-MiniLM-L6-v2 (embeddings),
             Qwen2.5-72B-Instruct (script generation), Microsoft Edge TTS (voice synthesis).</p>
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
