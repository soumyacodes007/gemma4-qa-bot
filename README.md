---
title: YouTube Shorts Audio QA + Fact Checker
emoji: 🎙
colorFrom: purple
colorTo: blue
sdk: gradio
sdk_version: "4.44.1"
app_file: app.py
pinned: true
license: apache-2.0
short_description: Fact-check YouTube Shorts with Gemma 4 + Tavily
tags:
  - gemma
  - audio
  - fact-checking
  - agents
  - multimodal
  - tool-calling
  - yt-dlp
  - tavily
---

# 🎙 YouTube Shorts Audio QA + Fact Checker

> **First open-source demo** combining **Gemma 4 E2B native audio understanding** + **agentic Tavily tool calling** in a single, no-ASR pipeline.

---

## What this does

Paste a YouTube Shorts URL (or upload an audio file). The app:

1. 📥 Downloads **audio-only** via `yt-dlp` (no video processing overhead)
2. 🧠 Feeds raw WAV to **Gemma 4 E2B** — which natively understands speech (no Whisper needed)
3. 🔍 Gemma extracts **factual claims** and autonomously calls **Tavily search** per claim
4. ✅❌⚠️ Returns a **per-claim verdict** with real source URLs
5. 💬 Or — ask any **freeform QA question** about what was said

---

## Why Gemma 4 E2B?

**Gemma 4 E2B** (2 billion parameters) is the smaller, more efficient version that:
- ✅ Runs on **CPU basic (free tier)** on Hugging Face Spaces
- ✅ Still supports **native audio understanding**
- ✅ Has **tool calling** capabilities
- ✅ Perfect for demos and prototypes

For production with higher traffic, consider upgrading to **Gemma 4 E4B** (4B) with GPU.

---

## What this does

Paste a YouTube Shorts URL (or upload an audio file). The app:

1. 📥 Downloads **audio-only** via `yt-dlp` (no video processing overhead)
2. 🧠 Feeds raw WAV to **Gemma 4 E4B** — which natively understands speech (no Whisper needed)
3. 🔍 Gemma extracts **factual claims** and autonomously calls **Tavily search** per claim
4. ✅❌⚠️ Returns a **per-claim verdict** with real source URLs
5. 💬 Or — ask any **freeform QA question** about what was said

---

## Architecture

```
YouTube URL ──► yt-dlp (audio WAV) ──► Gemma 4 E4B ──► Tavily Search
                                            │
                              ┌─────────────┴──────────────┐
                         Fact Check Mode              QA Mode
                         (agentic loop,           (single-pass,
                          tool calling)            no tools)
                              │                        │
                              └──────────┬─────────────┘
                                         ▼
                                   Gradio UI (ZeroGPU)
```

---

## Why this is novel

| Old pipeline | This app |
|---|---|
| Whisper STT → Text LLM | Gemma 4 E4B processes raw audio natively |
| Hallucinated fact checks | Tavily grounds every claim with web sources |
| Separate models chained | Single model handles audio + reasoning + tool calling |
| Complex RAG/vector stores | Gemma's 256K context replaces retrieval for audio |

---

## Setup

### Required Secrets (set in Space Settings → Secrets)

| Secret | Where to get it |
|---|---|
| `TAVILY_API_KEY` | [app.tavily.com](https://app.tavily.com) — free tier: 1,000 searches/month |
| `HF_TOKEN` | [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) — needed to access gated Gemma 4 model |

### Accepting the Gemma 4 License

Before deploying, visit [google/gemma-4-E2B-it](https://huggingface.co/google/gemma-4-E2B-it) on HuggingFace and accept the license agreement with your HF account.

---

## Local development

```bash
# 1. Clone
git clone https://huggingface.co/spaces/YOUR_USERNAME/shorts-fact-checker
cd shorts-fact-checker

# 2. Install dependencies
pip install -r requirements.txt

# 3. Install system deps (Linux/Mac)
apt-get install ffmpeg   # or: brew install ffmpeg

# 4. Set environment variables
export TAVILY_API_KEY="tvly-..."
export HF_TOKEN="hf_..."

# 5. Run
python app.py
```

---

## Limits & Notes

- **Max audio duration:** 3 minutes (enforced to stay within ZeroGPU timeout)
- **Agentic loop cap:** 3 tool call rounds (balances thoroughness vs. timeout risk)
- **Tavily free tier:** 1,000 searches/month. Each fact-check uses 1–5 searches.
- **Cold start:** ~60 seconds on ZeroGPU A100. Subsequent requests are fast.
- **Audio format:** Always converted to WAV before model inference for compatibility.

---

## Tech Stack

| Component | Library |
|---|---|
| LLM + Audio Understanding | `transformers` + `google/gemma-4-e4b-it` |
| Audio Download | `yt-dlp` |
| Web Search (Fact Grounding) | `tavily-python` |
| UI & Deployment | `gradio` + `spaces` (ZeroGPU) |
| Audio Processing | `ffmpeg` (system) |
| Inference Backend | `torch` + `accelerate` |

---

## License

Apache 2.0 — see [LICENSE](LICENSE).

Model weights: [Gemma Terms of Use](https://ai.google.dev/gemma/terms).
