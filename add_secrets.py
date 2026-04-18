#!/usr/bin/env python3
"""Add required secrets to the Hugging Face Space."""

import os

from huggingface_hub import HfApi

SPACE_NAME = os.environ.get("SPACE_NAME", "youtube-shorts-fact-checker")


def require_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise SystemExit(f"{name} is not set.")
    return value


def main() -> None:
    hf_token = require_env("HF_TOKEN")
    tavily_api_key = require_env("TAVILY_API_KEY")

    api = HfApi(token=hf_token)
    username = api.whoami()["name"]
    repo_id = f"{username}/{SPACE_NAME}"

    api.add_space_secret(repo_id=repo_id, key="HF_TOKEN", value=hf_token, token=hf_token)
    api.add_space_secret(
        repo_id=repo_id,
        key="TAVILY_API_KEY",
        value=tavily_api_key,
        token=hf_token,
    )

    print(f"Secrets added to https://huggingface.co/spaces/{repo_id}")


if __name__ == "__main__":
    main()
