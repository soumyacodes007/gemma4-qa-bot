"""
YouTube Shorts Audio QA + Fact Checker
Optimized for Hugging Face Spaces ZeroGPU with Gemma 4 E2B and Tavily.
"""

import json
import logging
import os
import shutil
import subprocess
import tempfile
import traceback
import wave

import gradio as gr
import spaces
import torch
import yt_dlp
from dotenv import load_dotenv
from tavily import TavilyClient
from transformers import AutoModelForMultimodalLM, AutoProcessor

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

MODEL_ID = "google/gemma-4-E2B-it"
MAX_AUDIO_SECONDS = 180
MAX_TOOL_ROUNDS = 3
TAVILY_MAX_RESULTS = 3
GENERATE_MAX_TOKENS = 768

logger.info("Loading Gemma 4 E2B for ZeroGPU Spaces.")
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForMultimodalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
model.eval()
logger.info("Model loaded.")

TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY", "").strip()
tavily_client = TavilyClient(api_key=TAVILY_API_KEY) if TAVILY_API_KEY else None

TAVILY_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "tavily_search",
        "description": (
            "Search the web for current evidence that verifies or refutes a claim from audio."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "A focused search query for verifying the claim.",
                },
                "claim": {
                    "type": "string",
                    "description": "The claim from the audio that is being checked.",
                },
            },
            "required": ["query", "claim"],
        },
    },
}

FACT_CHECK_SYSTEM_PROMPT = """You are a precise fact-checker with access to web search.

When given audio:
1. Identify factual claims that can be checked.
2. For each important claim, call `tavily_search`.
3. After using search, produce a final report.

Format each checked claim exactly like this:

---
**Claim:** [claim]
**Verdict:** TRUE / FALSE / UNVERIFIABLE
**Evidence:** [1-2 sentence summary]
**Sources:** [URLs]
---

Do not invent evidence. If there is weak or conflicting evidence, say UNVERIFIABLE.
"""

QA_SYSTEM_PROMPT = """You answer questions using only the provided audio.

Rules:
- Do not use outside knowledge.
- If the audio does not contain the answer, say: "The audio doesn't mention this."
- Keep the answer concise.
"""

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;700;800&family=IBM+Plex+Mono:wght@400;500&display=swap');

:root {
    --bg: #0b1117;
    --panel: #111a24;
    --panel-2: #162231;
    --border: #233246;
    --text: #edf4ff;
    --muted: #9fb2c8;
    --accent: #19a974;
    --accent-2: #48c0f7;
    --warn: #f59e0b;
}

body, .gradio-container {
    background:
        radial-gradient(circle at top, rgba(72, 192, 247, 0.12), transparent 35%),
        linear-gradient(180deg, #091018 0%, var(--bg) 100%) !important;
    color: var(--text) !important;
    font-family: 'Manrope', sans-serif !important;
}

.hero {
    padding: 28px 0 12px;
}

.hero h1 {
    margin: 0 0 8px;
    font-size: 2.4rem;
    font-weight: 800;
    color: var(--text);
}

.hero p {
    margin: 0;
    color: var(--muted);
}

.card {
    background: rgba(17, 26, 36, 0.9);
    border: 1px solid var(--border);
    border-radius: 18px;
    padding: 18px;
}

.chip-row {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
    margin-top: 14px;
}

.chip {
    border: 1px solid var(--border);
    background: rgba(22, 34, 49, 0.9);
    border-radius: 999px;
    padding: 6px 12px;
    font-size: 0.8rem;
    color: var(--muted);
}

#status-bar, #result-box {
    border: 1px solid var(--border) !important;
    border-radius: 18px !important;
    background: rgba(17, 26, 36, 0.9) !important;
}

#status-bar {
    padding: 10px 14px !important;
}

#result-box {
    padding: 18px !important;
    min-height: 260px;
}

.gr-button-primary {
    background: linear-gradient(135deg, var(--accent), #0ea5a5) !important;
    border: none !important;
}
"""


def download_audio(url: str) -> str:
    """Download YouTube audio and convert it to wav."""
    logger.info("Downloading audio from %s", url)
    temp_dir = tempfile.mkdtemp(prefix="yt-audio-")
    temp_base = os.path.join(temp_dir, "audio")
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": temp_base + ".%(ext)s",
        "quiet": True,
        "no_warnings": True,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "192",
            }
        ],
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    wav_path = temp_base + ".wav"
    if not os.path.exists(wav_path):
        raise FileNotFoundError(f"Audio download failed for URL: {url}")
    return wav_path


def convert_audio_to_wav(audio_path: str) -> tuple[str, bool]:
    """
    Normalize uploaded audio to mono 16 kHz wav for consistent processing.
    Returns (wav_path, created_temp_copy).
    """
    if audio_path.lower().endswith(".wav"):
        return audio_path, False

    temp_dir = tempfile.mkdtemp(prefix="upload-audio-")
    wav_path = os.path.join(temp_dir, "normalized.wav")
    command = [
        "ffmpeg",
        "-y",
        "-i",
        audio_path,
        "-ac",
        "1",
        "-ar",
        "16000",
        wav_path,
    ]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise FileNotFoundError("ffmpeg is not available. It is required for uploaded MP3/M4A files.") from exc
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").strip()
        raise ValueError(f"Failed to convert uploaded audio to wav. {stderr}") from exc
    return wav_path, True


def validate_audio_length(wav_path: str) -> float:
    """Return the duration in seconds and enforce the project limit."""
    with wave.open(wav_path, "rb") as wf:
        duration = wf.getnframes() / float(wf.getframerate())
    logger.info("Audio duration: %.1fs", duration)
    if duration > MAX_AUDIO_SECONDS:
        raise ValueError(
            f"Audio is {duration:.0f}s long. Maximum allowed is {MAX_AUDIO_SECONDS}s."
        )
    return duration


def run_tavily_search(query: str) -> str:
    """Run Tavily search and return a compact markdown summary."""
    if not tavily_client:
        return "Tavily API key is missing, so web verification is unavailable."
    response = tavily_client.search(
        query=query,
        max_results=TAVILY_MAX_RESULTS,
        search_depth="basic",
        include_answer=True,
    )
    parts = []
    answer = response.get("answer")
    if answer:
        parts.append(f"Answer: {answer}")
    for result in response.get("results", []):
        title = result.get("title", "Untitled")
        snippet = result.get("content", "").strip()[:240]
        url = result.get("url", "")
        parts.append(f"- {title}: {snippet} ({url})")
    return "\n".join(parts) if parts else "No search results found."


def extract_tool_call(response_text: str) -> dict | None:
    """Parse a <tool_call>...</tool_call> block from model output."""
    start_token = "<tool_call>"
    end_token = "</tool_call>"
    start = response_text.find(start_token)
    end = response_text.find(end_token)
    if start == -1 or end == -1:
        return None
    raw = response_text[start + len(start_token):end].strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("Could not parse tool call JSON: %s", raw[:300])
        return None


def build_model_inputs(messages: list[dict], tools: list[dict] | None = None):
    kwargs = {
        "tokenize": True,
        "return_dict": True,
        "return_tensors": "pt",
        "add_generation_prompt": True,
    }
    if tools:
        kwargs["tools"] = tools
    return processor.apply_chat_template(messages, **kwargs).to(model.device)


def decode_new_tokens(output_ids, input_ids) -> str:
    new_tokens = output_ids[0][input_ids.shape[1]:]
    return processor.decode(new_tokens, skip_special_tokens=False)


@spaces.GPU(duration=120)
def fact_check(audio_path: str, progress=gr.Progress()) -> str:
    """Run the claim extraction and verification loop."""
    progress(0.1, desc="Preparing audio for Gemma")
    messages = [
        {"role": "system", "content": FACT_CHECK_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": audio_path},
                {"type": "text", "text": "Fact-check all meaningful claims from this audio."},
            ],
        },
    ]

    for round_idx in range(MAX_TOOL_ROUNDS):
        progress(
            0.2 + (0.15 * round_idx),
            desc=f"Gemma reasoning pass {round_idx + 1}/{MAX_TOOL_ROUNDS}",
        )
        inputs = build_model_inputs(messages, tools=[TAVILY_TOOL_SCHEMA])
        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=GENERATE_MAX_TOKENS,
                do_sample=False,
            )

        response_text = decode_new_tokens(output_ids, inputs["input_ids"])
        tool_call = extract_tool_call(response_text)
        if not tool_call:
            return processor.decode(
                output_ids[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
            ).strip()

        arguments = tool_call.get("arguments", tool_call)
        query = arguments.get("query", "").strip()
        claim = arguments.get("claim", "").strip()
        if not query:
            break

        progress(
            0.45 + (0.15 * round_idx),
            desc=f"Searching evidence for: {query[:50]}",
        )
        search_results = run_tavily_search(query)
        messages.append({"role": "assistant", "content": response_text})
        messages.append(
            {
                "role": "tool",
                "name": "tavily_search",
                "content": f"Claim: {claim}\nQuery: {query}\nResults:\n{search_results}",
            }
        )

    progress(0.95, desc="Compiling final answer")
    final_inputs = build_model_inputs(messages)
    with torch.inference_mode():
        final_output = model.generate(
            **final_inputs,
            max_new_tokens=GENERATE_MAX_TOKENS,
            do_sample=False,
        )
    return processor.decode(
        final_output[0][final_inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    ).strip()


@spaces.GPU(duration=90)
def audio_qa(audio_path: str, question: str, progress=gr.Progress()) -> str:
    """Answer a question using only the uploaded audio."""
    if not question.strip():
        return "Please enter a question in QA mode."

    progress(0.1, desc="Preparing QA inputs")
    messages = [
        {"role": "system", "content": QA_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": audio_path},
                {"type": "text", "text": question.strip()},
            ],
        },
    ]
    inputs = build_model_inputs(messages)
    progress(0.4, desc="Gemma is answering")
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=GENERATE_MAX_TOKENS,
            do_sample=False,
        )
    return processor.decode(
        output_ids[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    ).strip()


def cleanup_temp_roots(paths: list[str]) -> None:
    for root in paths:
        if not root:
            continue
        try:
            if os.path.isdir(root):
                shutil.rmtree(root, ignore_errors=True)
        except OSError:
            pass


def run(
    youtube_url: str,
    uploaded_audio: str | None,
    mode: str,
    question: str,
    progress=gr.Progress(),
) -> tuple[str, str]:
    """Entry point for the Gradio app."""
    cleanup_roots: list[str] = []
    audio_path = None
    try:
        if youtube_url and youtube_url.strip().startswith("http"):
            progress(0.05, desc="Downloading YouTube audio")
            audio_path = download_audio(youtube_url.strip())
            cleanup_roots.append(os.path.dirname(audio_path))
        elif uploaded_audio:
            audio_path, created_temp = convert_audio_to_wav(uploaded_audio)
            if created_temp:
                cleanup_roots.append(os.path.dirname(audio_path))
        else:
            return "", "Please provide a YouTube URL or upload audio."

        progress(0.1, desc="Validating audio")
        duration = validate_audio_length(audio_path)
        status = f"Audio ready: {duration:.0f}s | Mode: **{mode}** | Model: **Gemma 4 E2B**"

        if mode == "Fact Check":
            result = fact_check(audio_path, progress)
        else:
            result = audio_qa(audio_path, question, progress)
        return result, status

    except (FileNotFoundError, ValueError) as exc:
        logger.error(traceback.format_exc())
        return "", f"Error: {exc}"
    except Exception as exc:
        logger.error(traceback.format_exc())
        return "", f"Unexpected error: {exc}"
    finally:
        cleanup_temp_roots(cleanup_roots)


EXAMPLES = [
    ["https://www.youtube.com/shorts/oERLa0AMIA8", None, "Fact Check", ""],
    ["https://www.youtube.com/shorts/2GjvBWgQkXM", None, "Fact Check", ""],
    ["", None, "QA", "What is the main point of the audio?"],
]

with gr.Blocks(
    css=CSS,
    title="YouTube Shorts Fact Checker",
    theme=gr.themes.Base(),
) as demo:
    gr.HTML(
        """
        <div class="hero">
            <h1>YouTube Shorts Fact Checker</h1>
            <p>Gemma 4 E2B audio understanding, Tavily grounding, and a ZeroGPU-friendly Spaces setup.</p>
            <div class="chip-row">
                <span class="chip">Gemma 4 E2B</span>
                <span class="chip">Hugging Face Spaces</span>
                <span class="chip">ZeroGPU</span>
                <span class="chip">Tavily Search</span>
            </div>
        </div>
        """
    )

    with gr.Row(equal_height=False):
        with gr.Column(scale=4, min_width=320):
            with gr.Group(elem_classes="card"):
                youtube_url = gr.Textbox(
                    label="YouTube Shorts URL",
                    placeholder="https://youtube.com/shorts/...",
                )
                uploaded_audio = gr.Audio(
                    label="Upload audio",
                    type="filepath",
                )
                mode = gr.Radio(
                    ["Fact Check", "QA"],
                    value="Fact Check",
                    label="Mode",
                )
                question_box = gr.Textbox(
                    label="Question for QA mode",
                    placeholder="What is being discussed in the audio?",
                    lines=2,
                    visible=False,
                )
                mode.change(
                    fn=lambda selected: gr.update(visible=(selected == "QA")),
                    inputs=mode,
                    outputs=question_box,
                )
                run_btn = gr.Button("Run Analysis", variant="primary")
                gr.Markdown(
                    "Requirements: `HF_TOKEN` must have access to `google/gemma-4-E2B-it`, "
                    "and `TAVILY_API_KEY` is required for fact-check mode."
                )

        with gr.Column(scale=6, min_width=420):
            status_bar = gr.Markdown(
                value="Waiting for input.",
                elem_id="status-bar",
            )
            result_box = gr.Markdown(
                value="",
                elem_id="result-box",
            )

    gr.Examples(
        examples=EXAMPLES,
        inputs=[youtube_url, uploaded_audio, mode, question_box],
        label="Examples",
    )

    run_btn.click(
        fn=run,
        inputs=[youtube_url, uploaded_audio, mode, question_box],
        outputs=[result_box, status_bar],
        show_progress=True,
    )


if __name__ == "__main__":
    demo.launch(show_error=True)
