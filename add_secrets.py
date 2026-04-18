#!/usr/bin/env python3
"""Add secrets to Hugging Face Space"""

import os
from huggingface_hub import HfApi

# Configuration
SPACE_NAME = "youtube-shorts-fact-checker"
HF_TOKEN = os.environ.get("HF_TOKEN")
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")

def main():
    if not HF_TOKEN:
        print("❌ HF_TOKEN not set")
        return
    
    api = HfApi(token=HF_TOKEN)
    user_info = api.whoami()
    username = user_info["name"]
    repo_id = f"{username}/{SPACE_NAME}"
    
    print(f"🔐 Adding secrets to {repo_id}...")
    
    try:
        # Add TAVILY_API_KEY
        api.add_space_secret(
            repo_id=repo_id,
            key="TAVILY_API_KEY",
            value=TAVILY_API_KEY,
            token=HF_TOKEN,
        )
        print("✅ Added TAVILY_API_KEY")
        
        # Add HF_TOKEN
        api.add_space_secret(
            repo_id=repo_id,
            key="HF_TOKEN",
            value=HF_TOKEN,
            token=HF_TOKEN,
        )
        print("✅ Added HF_TOKEN")
        
        print(f"\n✨ All secrets added successfully!")
        print(f"🌐 Your Space: https://huggingface.co/spaces/{repo_id}")
        
    except Exception as e:
        print(f"❌ Error adding secrets: {e}")

if __name__ == "__main__":
    main()
