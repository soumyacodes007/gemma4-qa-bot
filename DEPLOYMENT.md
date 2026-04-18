# Deployment Guide

## Recommended target

Deploy this project as a **Gradio Space** on **Hugging Face Spaces ZeroGPU**.

## Before deployment

1. Make sure your Hugging Face account has accepted the license for `google/gemma-4-E2B-it`.
2. Export your Hugging Face token:
   - PowerShell: `$env:HF_TOKEN="hf_..."`
3. Export your Tavily key if you want fact-check mode:
   - PowerShell: `$env:TAVILY_API_KEY="tvly-..."`

## Create and upload the Space

```bash
python deploy_to_hf.py
```

By default the script creates:

- a Space named `youtube-shorts-fact-checker`
- a `gradio` Space
- a hardware request for `zero-gpu`

Optional overrides:

```bash
SPACE_NAME=my-space-name
SPACE_HARDWARE=zero-gpu
SPACE_PRIVATE=true
```

## Add secrets

```bash
python add_secrets.py
```

This sets:

- `HF_TOKEN`
- `TAVILY_API_KEY`

## Manual checks in the Space

After upload, confirm:

- hardware is set to `ZeroGPU`
- the secrets are present
- the build log installs `ffmpeg` from `packages.txt`

## Common failure points

- `HF_TOKEN` exists but the account did not accept the Gemma license
- `TAVILY_API_KEY` is missing, so fact-check mode cannot search
- `ffmpeg` is missing from the Space image
- uploaded non-WAV audio was not convertible by ffmpeg
