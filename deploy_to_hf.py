#!/usr/bin/env python3
"""Create or update a Hugging Face Space for this project."""

import os
from pathlib import Path

from huggingface_hub import HfApi, create_repo

SPACE_NAME = os.environ.get("SPACE_NAME", "youtube-shorts-fact-checker")
SPACE_HARDWARE = os.environ.get("SPACE_HARDWARE", "zero-gpu")
SPACE_PRIVATE = os.environ.get("SPACE_PRIVATE", "false").lower() == "true"

EXCLUDE_PATTERNS = [
    ".git/*",
    ".env",
    "__pycache__/*",
    "*.pyc",
    ".cache/*",
    ".gitignore",
]


def main() -> None:
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise SystemExit("HF_TOKEN is not set.")

    api = HfApi(token=hf_token)
    username = api.whoami()["name"]
    repo_id = f"{username}/{SPACE_NAME}"

    print(f"Creating or updating Space: {repo_id}")
    create_repo(
        repo_id=repo_id,
        repo_type="space",
        space_sdk="gradio",
        private=SPACE_PRIVATE,
        exist_ok=True,
        token=hf_token,
    )

    print("Uploading project files...")
    api.upload_folder(
        folder_path=str(Path.cwd()),
        repo_id=repo_id,
        repo_type="space",
        ignore_patterns=EXCLUDE_PATTERNS,
        token=hf_token,
    )

    try:
        api.request_space_hardware(
            repo_id=repo_id,
            hardware=SPACE_HARDWARE,
            token=hf_token,
        )
        print(f"Requested hardware: {SPACE_HARDWARE}")
    except Exception as exc:
        print(f"Could not set hardware automatically: {exc}")
        print("Set the hardware manually in the Space settings if needed.")

    print(f"Space URL: https://huggingface.co/spaces/{repo_id}")
    print("Next step: add `HF_TOKEN` and `TAVILY_API_KEY` as Space secrets.")
    print("Make sure the token account has accepted the Gemma 4 E2B license.")


if __name__ == "__main__":
    main()
