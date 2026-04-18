# 🚀 Deployment Guide - Hugging Face Spaces

This guide will help you deploy your YouTube Shorts Fact Checker to Hugging Face Spaces using the automated deployment script.

## Prerequisites

1. **Hugging Face Account**
   - Sign up at [huggingface.co](https://huggingface.co/join)

2. **API Tokens**
   - **HF Token**: Get from [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
     - Create a token with "write" access
   - **Tavily API Key**: Get from [app.tavily.com](https://app.tavily.com)
     - Free tier: 1,000 searches/month

3. **Accept Gemma 4 License**
   - Visit [google/gemma-4-e4b-it](https://huggingface.co/google/gemma-4-e4b-it)
   - Click "Agree and access repository"

## Quick Deploy (Automated)

### Step 1: Install deployment dependencies

```bash
pip install huggingface_hub
```

### Step 2: Set your HF token

**Linux/Mac:**
```bash
export HF_TOKEN="hf_your_token_here"
```

**Windows (CMD):**
```cmd
set HF_TOKEN=hf_your_token_here
```

**Windows (PowerShell):**
```powershell
$env:HF_TOKEN="hf_your_token_here"
```

### Step 3: Run the deployment script

```bash
python deploy_to_hf.py
```

The script will:
- ✅ Create a new Space on your HF account
- ✅ Upload all necessary files
- ✅ Configure ZeroGPU hardware
- ✅ Provide next steps for secrets

### Step 4: Add secrets to your Space

1. Go to your Space settings: `https://huggingface.co/spaces/YOUR_USERNAME/youtube-shorts-fact-checker/settings`
2. Scroll to "Repository secrets"
3. Add these secrets:
   - **Name:** `TAVILY_API_KEY` | **Value:** Your Tavily API key
   - **Name:** `HF_TOKEN` | **Value:** Your Hugging Face token

### Step 5: Wait for build

- First build takes 2-3 minutes
- Check the "Logs" tab to monitor progress
- Once complete, your Space will be live!

## Manual Deploy (Alternative)

If you prefer to deploy manually:

### 1. Create Space on HF

```bash
# Install HF CLI
pip install huggingface_hub[cli]

# Login
huggingface-cli login

# Create space
huggingface-cli repo create youtube-shorts-fact-checker --type space --space_sdk gradio
```

### 2. Clone and push

```bash
# Clone your new space
git clone https://huggingface.co/spaces/YOUR_USERNAME/youtube-shorts-fact-checker
cd youtube-shorts-fact-checker

# Copy your files
cp /path/to/your/project/* .

# Commit and push
git add .
git commit -m "Initial commit"
git push
```

### 3. Configure Space

Add this to your `README.md` frontmatter:

```yaml
---
title: YouTube Shorts Fact Checker
emoji: 🎙
colorFrom: purple
colorTo: cyan
sdk: gradio
sdk_version: "4.40.0"
app_file: app.py
pinned: true
license: apache-2.0
hardware: zero-gpu
---
```

## Customization

### Change Space Name

Edit `deploy_to_hf.py`:
```python
SPACE_NAME = "your-custom-name"
```

### Change Hardware

Options in `deploy_to_hf.py`:
- `"cpu-basic"` - Free (slow)
- `"cpu-upgrade"` - $0.03/hour
- `"t4-small"` - $0.60/hour
- `"zero-gpu"` - Pay per use (recommended)

### Make Space Private

Edit `deploy_to_hf.py`:
```python
private=True  # in create_repo()
```

## Troubleshooting

### "Repository already exists"
- The script handles this automatically with `exist_ok=True`
- Or delete the old space first at: `https://huggingface.co/spaces/YOUR_USERNAME/youtube-shorts-fact-checker/settings`

### "Model not found" error
- Make sure you accepted the Gemma 4 license
- Verify your HF_TOKEN has read access
- Check token is set correctly in Space secrets

### "Tavily API error"
- Verify TAVILY_API_KEY is set in Space secrets
- Check you haven't exceeded free tier limit (1,000/month)
- Get a new key at [app.tavily.com](https://app.tavily.com)

### Build fails
- Check "Logs" tab in your Space
- Verify all dependencies in `requirements.txt`
- Ensure `ffmpeg` is in `packages.txt` (already included)

### Cold start is slow
- This is normal for ZeroGPU (~60 seconds)
- Subsequent requests are much faster
- Consider upgrading to dedicated GPU if needed

## Post-Deployment

### Monitor Usage
- Check Space analytics: `https://huggingface.co/spaces/YOUR_USERNAME/youtube-shorts-fact-checker/analytics`
- Monitor Tavily usage: [app.tavily.com/dashboard](https://app.tavily.com/dashboard)

### Update Your Space
```bash
# Make changes locally
# Then re-run deployment script
python deploy_to_hf.py
```

Or use git:
```bash
cd /path/to/space/clone
git pull
# make changes
git add .
git commit -m "Update"
git push
```

## Cost Estimate

**Free Tier:**
- HF Space: Free with ZeroGPU (pay per use)
- Tavily: 1,000 searches/month free
- Estimated: $0-5/month for light usage

**Heavy Usage:**
- ZeroGPU: ~$0.001 per second of GPU time
- Tavily: $0.01 per search after free tier
- Estimated: $20-50/month for 1,000+ fact-checks

## Support

- HF Spaces Docs: [huggingface.co/docs/hub/spaces](https://huggingface.co/docs/hub/spaces)
- Community Forum: [discuss.huggingface.co](https://discuss.huggingface.co)
- Issues: Create an issue in your Space repo

---

**Ready to deploy?** Run `python deploy_to_hf.py` and you'll be live in minutes! 🚀
