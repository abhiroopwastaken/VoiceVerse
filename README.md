---
title: VoiceVerse
emoji: 🎙️
colorFrom: purple
colorTo: blue
sdk: gradio
sdk_version: "5.16.0"
python_version: "3.11"
app_file: app.py
pinned: false
license: mit
secrets:
  - GROQ_API_KEY
---

# 🎙️ VoiceVerse — AI Document-to-Audio Application

Transform your documents into captivating audio experiences using AI.

## Features

- **📄 Multi-Format Upload**: Support for PDF, TXT, and DOCX files
- **🧠 RAG-Powered**: Content grounded in your uploaded documents via retrieval-augmented generation
- **🎨 5 Content Styles**: Podcast, Narration, Debate, Lecture, Storytelling
- **🗣️ Multi-Voice**: Different AI voices per speaker
- **🎵 Customizable**: Adjust speech rate and pitch
- **📝 Script Preview**: Read the generated script before listening

## Setup (HF Spaces)

Add the following as a **Space Secret**:
- `GROQ_API_KEY` — your [Groq API key](https://console.groq.com) (free tier available)

## Technology Stack

| Component | Technology |
|-----------|-----------|
| UI | Gradio |
| LLM | Groq `llama-3.1-8b-instant` |
| Embeddings | `BAAI/bge-small-en-v1.5` |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` (local) |
| Vector Store | FAISS |
| TTS | edge-tts (Microsoft Neural Voices) |
| Audio Processing | pydub |

## Disclaimer

⚠️ All audio content is **synthetically generated** by AI. Content is based on uploaded documents and may not perfectly represent the source material.

## License

MIT License
