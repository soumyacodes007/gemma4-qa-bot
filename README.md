---
title: YouTube Shorts Audio QA + Fact Checker
emoji: "🎙️"
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: "4.44.1"
app_file: app.py
pinned: true
license: apache-2.0
short_description: Fact-check short-form audio with Gemma 4 E2B and Tavily.
tags:
  - gemma
  - audio
  - fact-checking
  - multimodal
  - tavily
  - gradio
---

# YouTube Shorts Audio QA + Fact Checker

This Space is built for **Hugging Face Spaces ZeroGPU** and uses:

- `google/gemma-4-E2B-it` for audio understanding
- `Tavily` for claim verification in fact-check mode
- `yt-dlp` + `ffmpeg` for YouTube audio extraction and audio normalization

## What it does

You can either:

- paste a YouTube Shorts URL
- upload an audio file

Then choose one mode:

- `Fact Check`: extracts factual claims and verifies them with Tavily
- `QA`: answers a question using only the audio content

## Why Gemma 4 E2B

This project is tuned for Spaces deployment. `Gemma 4 E2B` is the practical choice here because it keeps the demo lighter while still supporting audio input and tool use. The app is structured around `spaces.GPU` so it works with **ZeroGPU**.

## Required Space secrets

Set these in your Space settings:

- `HF_TOKEN`
- `TAVILY_API_KEY`

The Hugging Face token must belong to an account that has already accepted the license for:

- https://huggingface.co/google/gemma-4-E2B-it

## Local run

```bash
pip install -r requirements.txt
python app.py
```

You also need:

- `ffmpeg` installed on the system
- `HF_TOKEN` in the environment
- `TAVILY_API_KEY` in the environment for fact-check mode

## Deploy to Spaces

```bash
python deploy_to_hf.py
python add_secrets.py
```

Optional environment variables for deployment:

- `SPACE_NAME`
- `SPACE_HARDWARE`
- `SPACE_PRIVATE`

## Notes

- Max audio duration is `180` seconds.
- Uploaded MP3 and M4A files are normalized to WAV before processing.
- Fact-check mode depends on Tavily. QA mode does not.
