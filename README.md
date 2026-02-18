---
title: VoiceVerse
emoji: 🎙️
colorFrom: purple
colorTo: blue
sdk: gradio
sdk_version: "4.36.0"
app_file: app.py
pinned: false
license: mit
---

# 🎙️ VoiceVerse — AI Document-to-Audio Application

Transform your documents into captivating audio experiences using AI.

## Features

- **📄 Multi-Format Upload**: Support for PDF, TXT, and DOCX files
- **🧠 RAG-Powered**: Content is grounded in your uploaded documents using retrieval-augmented generation
- **🎨 5 Content Styles**: Podcast, Narration, Debate, Lecture, Storytelling
- **🗣️ Multi-Voice**: Different AI voices for different speakers
- **🎵 Customizable**: Adjust speech rate and pitch
- **📝 Script Preview**: Read the generated script before listening

## How It Works

1. **Upload** your documents (PDF, TXT, DOCX)
2. **Choose** a content style (Podcast, Narration, Debate, Lecture, Storytelling)
3. **Generate** and listen to your AI-created audio content

## Technology Stack

| Component | Technology |
|-----------|-----------|
| UI | Gradio |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| Vector Store | FAISS |
| LLM | Mistral-7B-Instruct (via HF Inference API) |
| TTS | edge-tts (Microsoft Neural Voices) |
| Audio Processing | pydub |

## Models & Attribution

- **Embedding Model**: [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
- **LLM**: [mistralai/Mistral-7B-Instruct-v0.3](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3)
- **TTS**: [Microsoft Edge TTS](https://github.com/rany2/edge-tts) (Neural Voices)

## Disclaimer

⚠️ All audio content is **synthetically generated** by AI. Content is based on uploaded documents and may not perfectly represent the source material. AI-generated voices are clearly labeled as synthetic.

## License

MIT License
