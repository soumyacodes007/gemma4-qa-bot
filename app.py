"""
🎙 YouTube Shorts Audio QA + Fact Checker
Powered by Gemma 4 E4B (native audio understanding) + Tavily Search
Deployed on Hugging Face Spaces with ZeroGPU
"""

import os
import json
import wave
import time
import tempfile
import logging
import traceback

import gradio as gr
import torch
import spaces
from transformers import AutoProcessor, AutoModelForImageTextToText
from tavily import TavilyClient
import yt_dlp
from dotenv import load_dotenv

load_dotenv()  # Load variables from .env file

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
MODEL_ID = "google/gemma-4-e4b-it"
MAX_AUDIO_SECONDS = 180          # 3-minute hard limit
MAX_TOOL_ROUNDS = 3              # Agentic loop cap (ZeroGPU timeout safety)
TAVILY_MAX_RESULTS = 3
GENERATE_MAX_TOKENS = 768

# ─────────────────────────────────────────────────────────────────────────────
# Model — loaded once at module level, outside GPU functions
# ─────────────────────────────────────────────────────────────────────────────
logger.info("Loading Gemma 4 E4B processor and model...")
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
model.eval()
logger.info("Model loaded ✓")

# ─────────────────────────────────────────────────────────────────────────────
# Tavily Client
# ─────────────────────────────────────────────────────────────────────────────
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY", "")
tavily_client = TavilyClient(api_key=TAVILY_API_KEY) if TAVILY_API_KEY else None

# ─────────────────────────────────────────────────────────────────────────────
# Tavily Tool Schema (Gemma 4 native function calling format)
# ─────────────────────────────────────────────────────────────────────────────
TAVILY_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "tavily_search",
        "description": (
            "Search the web for real-time information to verify a factual claim. "
            "Use this when you need to check if something stated in the audio is true or false."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query to verify the claim. Be specific and concise.",
                },
                "claim": {
                    "type": "string",
                    "description": "The exact claim from the audio that is being verified.",
                },
            },
            "required": ["query", "claim"],
        },
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# System Prompts
# ─────────────────────────────────────────────────────────────────────────────
FACT_CHECK_SYSTEM_PROMPT = """You are a rigorous, precise fact-checker powered by real-time web search.

When given audio:
1. Listen carefully and identify ALL factual claims (statistics, dates, names, events, scientific statements).
2. For EACH claim, call `tavily_search` with a clear, specific query to verify it.
3. After getting search results for every claim, compile your final report.

Format your FINAL response exactly like this for each claim:

---
**Claim:** [exact quote or close paraphrase from audio]
**Verdict:** ✅ TRUE / ❌ FALSE / ⚠️ UNVERIFIABLE
**Evidence:** [1–2 sentence summary of what the search found]
**Sources:** [list URLs]
---

Be honest. If evidence is weak or contradictory, use ⚠️ UNVERIFIABLE. Do not guess."""

QA_SYSTEM_PROMPT = """You are a helpful assistant that answers questions based strictly on what was said in the audio provided.

Rules:
- Only use information from the audio. Do not invent or hallucinate details.
- If the audio does not contain the answer, say: "The audio doesn't mention this."
- Be concise and direct."""


# ─────────────────────────────────────────────────────────────────────────────
# Helper: Download audio from YouTube via yt-dlp
# ─────────────────────────────────────────────────────────────────────────────
def download_audio(url: str) -> str:
    """Download audio-only from YouTube URL. Returns path to .wav file."""
    logger.info(f"Downloading audio from: {url}")
    tmp_base = tempfile.mktemp()
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": tmp_base + ".%(ext)s",
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

    wav_path = tmp_base + ".wav"
    if not os.path.exists(wav_path):
        raise FileNotFoundError(f"Audio download failed. Check the URL: {url}")

    logger.info(f"Audio downloaded to: {wav_path}")
    return wav_path


# ─────────────────────────────────────────────────────────────────────────────
# Helper: Validate audio length
# ─────────────────────────────────────────────────────────────────────────────
def validate_audio_length(wav_path: str) -> float:
    """Returns duration in seconds. Raises if over the limit."""
    with wave.open(wav_path, "r") as wf:
        frames = wf.getnframes()
        rate = wf.getframerate()
        duration = frames / float(rate)
    logger.info(f"Audio duration: {duration:.1f}s")
    if duration > MAX_AUDIO_SECONDS:
        raise ValueError(
            f"Audio is {duration:.0f}s long. Maximum allowed is {MAX_AUDIO_SECONDS}s (3 minutes). "
            "Please use a shorter clip."
        )
    return duration


# ─────────────────────────────────────────────────────────────────────────────
# Helper: Run Tavily search
# ─────────────────────────────────────────────────────────────────────────────
def run_tavily_search(query: str) -> str:
    """Execute a Tavily web search and return formatted results."""
    if not tavily_client:
        return "⚠️ Tavily API key not configured. Cannot perform web search."
    try:
        logger.info(f"Tavily search: {query!r}")
        response = tavily_client.search(
            query=query,
            max_results=TAVILY_MAX_RESULTS,
            search_depth="basic",
            include_answer=True,
        )
        parts = []
        if response.get("answer"):
            parts.append(f"Quick answer: {response['answer']}")
        for r in response.get("results", []):
            snippet = r.get("content", "")[:250].strip()
            parts.append(f"- **{r['title']}**: {snippet}... ({r['url']})")
        return "\n".join(parts) if parts else "No results found."
    except Exception as e:
        logger.warning(f"Tavily search failed: {e}")
        return f"Search error: {e}"


# ─────────────────────────────────────────────────────────────────────────────
# Helper: Extract tool call from Gemma output
# ─────────────────────────────────────────────────────────────────────────────
def extract_tool_call(response_text: str) -> dict | None:
    """
    Parse Gemma's tool call output from between special tokens.
    Returns a dict or None if parsing fails.
    """
    try:
        start_token = "<tool_call>"
        end_token = "</tool_call>"
        start = response_text.find(start_token)
        end = response_text.find(end_token)
        if start == -1 or end == -1:
            return None
        raw = response_text[start + len(start_token):end].strip()
        return json.loads(raw)
    except (json.JSONDecodeError, ValueError) as e:
        logger.warning(f"Failed to parse tool call JSON: {e}\nRaw text: {response_text[:300]}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Core: Fact Check (agentic loop with tool calling)
# ─────────────────────────────────────────────────────────────────────────────
@spaces.GPU
def fact_check(audio_path: str, progress=gr.Progress()) -> str:
    """
    Feed audio to Gemma 4 E4B, let it extract claims and call Tavily per claim.
    Returns a formatted markdown verdict string.
    """
    progress(0.1, desc="Preparing model inputs…")

    messages = [
        {"role": "system", "content": FACT_CHECK_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": audio_path},
                {"type": "text", "text": "Fact-check all claims made in this audio."},
            ],
        },
    ]

    progress(0.2, desc="Running Gemma 4 — identifying claims…")

    for round_idx in range(MAX_TOOL_ROUNDS):
        logger.info(f"Agentic round {round_idx + 1}/{MAX_TOOL_ROUNDS}")

        # Build prompt with tool schema
        text = processor.apply_chat_template(
            messages,
            tools=[TAVILY_TOOL_SCHEMA],
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = processor(text=text, return_tensors="pt").to(model.device)

        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=GENERATE_MAX_TOKENS,
                do_sample=False,
            )

        # Decode only the newly generated tokens
        new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
        response_text = processor.decode(new_tokens, skip_special_tokens=False)
        logger.info(f"Round {round_idx + 1} response (first 300 chars): {response_text[:300]}")

        # Check if Gemma wants to call a tool
        tool_call = extract_tool_call(response_text)

        if tool_call:
            query = tool_call.get("arguments", tool_call).get("query", "")
            claim = tool_call.get("arguments", tool_call).get("claim", "")

            progress(
                0.2 + (0.6 * (round_idx + 1) / MAX_TOOL_ROUNDS),
                desc=f"Searching web for: {query[:50]}…",
            )

            search_results = run_tavily_search(query)
            logger.info(f"Tavily results: {search_results[:200]}")

            # Append assistant tool_call message and the tool result
            messages.append({"role": "assistant", "content": response_text})
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.get("id", str(round_idx)),
                    "content": search_results,
                }
            )
        else:
            # No tool call — this is the final answer
            progress(0.95, desc="Compiling verdict…")
            # Clean up special tokens for display
            clean_response = processor.decode(new_tokens, skip_special_tokens=True).strip()
            return clean_response

    progress(0.95, desc="Finalizing…")
    # If we exhaust all rounds, return whatever the last response was
    final_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    final_inputs = processor(text=final_text, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        final_output = model.generate(**final_inputs, max_new_tokens=GENERATE_MAX_TOKENS, do_sample=False)
    new_tokens = final_output[0][final_inputs["input_ids"].shape[1]:]
    return processor.decode(new_tokens, skip_special_tokens=True).strip()


# ─────────────────────────────────────────────────────────────────────────────
# Core: QA Mode (no tool calls)
# ─────────────────────────────────────────────────────────────────────────────
@spaces.GPU
def audio_qa(audio_path: str, question: str, progress=gr.Progress()) -> str:
    """
    Answer a user question based on audio content only — no tool calling.
    """
    progress(0.1, desc="Preparing model inputs…")

    if not question.strip():
        return "⚠️ Please enter a question in QA mode."

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

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=text, return_tensors="pt").to(model.device)

    progress(0.4, desc="Gemma 4 is thinking…")

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=GENERATE_MAX_TOKENS,
            do_sample=False,
        )

    progress(0.9, desc="Decoding answer…")
    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    return processor.decode(new_tokens, skip_special_tokens=True).strip()


# ─────────────────────────────────────────────────────────────────────────────
# Orchestrator: called by Gradio
# ─────────────────────────────────────────────────────────────────────────────
def run(
    youtube_url: str,
    uploaded_audio,
    mode: str,
    question: str,
    progress=gr.Progress(),
) -> tuple[str, str]:
    """
    Main entry point for the Gradio interface.
    Returns (result_markdown, status_message).
    """
    audio_path = None
    temp_downloaded = False

    try:
        # ── Step 1: Resolve audio source ──────────────────────────────────
        if youtube_url and youtube_url.strip().startswith("http"):
            progress(0.05, desc="Downloading audio from YouTube…")
            audio_path = download_audio(youtube_url.strip())
            temp_downloaded = True
        elif uploaded_audio:
            audio_path = uploaded_audio
        else:
            return "", "⚠️ Please provide a YouTube URL or upload an audio file."

        # ── Step 2: Validate duration ─────────────────────────────────────
        progress(0.1, desc="Validating audio…")
        duration = validate_audio_length(audio_path)
        status = f"✅ Audio loaded — {duration:.0f}s | Mode: **{mode}**"

        # ── Step 3: Run selected mode ──────────────────────────────────────
        if mode == "Fact Check":
            result = fact_check(audio_path, progress)
        else:
            result = audio_qa(audio_path, question, progress)

        return result, status

    except FileNotFoundError as e:
        logger.error(traceback.format_exc())
        return "", f"❌ Download error: {e}"
    except ValueError as e:
        logger.error(traceback.format_exc())
        return "", f"❌ Validation error: {e}"
    except Exception as e:
        logger.error(traceback.format_exc())
        return "", f"❌ Unexpected error: {e}"

    finally:
        # Always clean up temp downloaded files
        if temp_downloaded and audio_path and os.path.exists(audio_path):
            try:
                os.remove(audio_path)
                logger.info(f"Cleaned up temp file: {audio_path}")
            except OSError:
                pass


# ─────────────────────────────────────────────────────────────────────────────
# Gradio UI
# ─────────────────────────────────────────────────────────────────────────────
CSS = """
/* ── Global ─────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --bg-primary: #0d0f14;
    --bg-card: #13161e;
    --bg-card-2: #1a1d27;
    --accent-purple: #7c3aed;
    --accent-purple-light: #a78bfa;
    --accent-teal: #06b6d4;
    --accent-pink: #ec4899;
    --accent-green: #10b981;
    --text-primary: #f1f5f9;
    --text-secondary: #94a3b8;
    --text-muted: #475569;
    --border: #1e2434;
    --border-glow: rgba(124, 58, 237, 0.4);
    --shadow-glow: 0 0 40px rgba(124, 58, 237, 0.15);
    --radius: 14px;
    --radius-sm: 8px;
}

* { box-sizing: border-box; }

body, .gradio-container {
    background: var(--bg-primary) !important;
    font-family: 'Inter', sans-serif !important;
    color: var(--text-primary) !important;
    min-height: 100vh;
}

/* ── Hero Header ─────────────────────────────────────── */
.hero-header {
    text-align: center;
    padding: 40px 20px 24px;
    background: radial-gradient(ellipse 80% 60% at 50% 0%, rgba(124,58,237,0.18) 0%, transparent 70%);
    border-bottom: 1px solid var(--border);
    margin-bottom: 28px;
}
.hero-header h1 {
    font-size: 2.4rem;
    font-weight: 700;
    background: linear-gradient(135deg, #a78bfa, #06b6d4, #ec4899);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 0 10px;
    letter-spacing: -0.02em;
}
.hero-header p {
    color: var(--text-secondary);
    font-size: 1rem;
    margin: 0;
    font-weight: 400;
}
.badge-row {
    display: flex;
    justify-content: center;
    gap: 10px;
    margin-top: 16px;
    flex-wrap: wrap;
}
.badge {
    background: rgba(124,58,237,0.15);
    border: 1px solid rgba(124,58,237,0.35);
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 0.78rem;
    font-weight: 500;
    color: var(--accent-purple-light);
    letter-spacing: 0.02em;
}
.badge.teal { background: rgba(6,182,212,0.1); border-color: rgba(6,182,212,0.3); color: var(--accent-teal); }
.badge.pink  { background: rgba(236,72,153,0.1); border-color: rgba(236,72,153,0.3); color: var(--accent-pink); }

/* ── Cards ───────────────────────────────────────────── */
.card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 20px;
    margin-bottom: 16px;
    transition: border-color 0.2s;
}
.card:hover { border-color: rgba(124,58,237,0.3); }
.card-title {
    font-size: 0.8rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--accent-purple-light);
    margin-bottom: 14px;
    display: flex;
    align-items: center;
    gap: 7px;
}

/* ── Inputs ──────────────────────────────────────────── */
input[type="text"], textarea, .gr-textbox textarea {
    background: var(--bg-card-2) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-sm) !important;
    color: var(--text-primary) !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.95rem !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
}
input[type="text"]:focus, textarea:focus {
    border-color: var(--accent-purple) !important;
    box-shadow: 0 0 0 3px rgba(124,58,237,0.15) !important;
    outline: none !important;
}
label, .gr-form label {
    color: var(--text-secondary) !important;
    font-size: 0.875rem !important;
    font-weight: 500 !important;
}

/* ── Radio ───────────────────────────────────────────── */
.gr-radio label { color: var(--text-primary) !important; }

/* ── Primary Button ──────────────────────────────────── */
button.primary-btn, .gr-button-primary {
    background: linear-gradient(135deg, var(--accent-purple), #9333ea) !important;
    border: none !important;
    color: #fff !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    border-radius: 10px !important;
    padding: 12px 28px !important;
    cursor: pointer !important;
    transition: transform 0.15s, box-shadow 0.15s !important;
    box-shadow: 0 4px 20px rgba(124,58,237,0.3) !important;
    width: 100% !important;
}
button.primary-btn:hover, .gr-button-primary:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 28px rgba(124,58,237,0.45) !important;
}
button.primary-btn:active { transform: translateY(0) !important; }

/* ── Status Bar ──────────────────────────────────────── */
#status-bar {
    background: linear-gradient(90deg, rgba(6,182,212,0.08), rgba(124,58,237,0.08));
    border: 1px solid rgba(6,182,212,0.2);
    border-radius: var(--radius-sm);
    padding: 10px 16px;
    font-size: 0.88rem;
    color: var(--accent-teal);
    min-height: 40px;
}

/* ── Result area ─────────────────────────────────────── */
#result-box {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    padding: 20px !important;
    min-height: 200px;
    font-family: 'Inter', sans-serif !important;
    color: var(--text-primary) !important;
    line-height: 1.7;
}
#result-box hr { border-color: var(--border); }
#result-box strong { color: var(--accent-purple-light) !important; }

/* ── Audio player ────────────────────────────────────── */
audio { border-radius: var(--radius-sm); }

/* ── How it works strip ──────────────────────────────── */
.how-strip {
    display: flex;
    gap: 6px;
    overflow-x: auto;
    padding-bottom: 4px;
    margin: 0 0 20px;
}
.step-pill {
    flex: 0 0 auto;
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 6px 14px;
    font-size: 0.78rem;
    color: var(--text-secondary);
    white-space: nowrap;
}
.step-pill span { color: var(--accent-purple-light); font-weight: 600; }
.step-arrow { flex: 0 0 auto; color: var(--text-muted); padding-top: 5px; font-size: 0.85rem; }
"""

EXAMPLE_URLS = [
    ["https://www.youtube.com/shorts/oERLa0AMIA8", "", "Fact Check", ""],
    ["https://www.youtube.com/shorts/2GjvBWgQkXM", "", "Fact Check", ""],
    ["", "", "QA", "What is the main product being discussed?"],
]

with gr.Blocks(
    css=CSS,
    title="🎙 YouTube Shorts Fact Checker — Gemma 4 + Tavily",
    theme=gr.themes.Base(),
) as demo:

    # ── Hero ──────────────────────────────────────────────
    gr.HTML("""
    <div class="hero-header">
        <h1>🎙 YouTube Shorts Fact Checker</h1>
        <p>Native audio understanding · Agentic fact grounding · Real-time web search</p>
        <div class="badge-row">
            <span class="badge">Gemma 4 E4B</span>
            <span class="badge teal">ZeroGPU · A100</span>
            <span class="badge pink">Tavily Search</span>
            <span class="badge">Native Audio</span>
        </div>
    </div>
    """)

    # ── Flow strip ────────────────────────────────────────
    gr.HTML("""
    <div class="how-strip">
        <div class="step-pill"><span>1</span> Paste URL or upload</div>
        <div class="step-arrow">→</div>
        <div class="step-pill"><span>2</span> yt-dlp extracts audio (WAV)</div>
        <div class="step-arrow">→</div>
        <div class="step-pill"><span>3</span> Gemma 4 listens natively</div>
        <div class="step-arrow">→</div>
        <div class="step-pill"><span>4</span> Calls Tavily per claim</div>
        <div class="step-arrow">→</div>
        <div class="step-pill"><span>5</span> ✅ / ❌ / ⚠️ Verdict + sources</div>
    </div>
    """)

    with gr.Row(equal_height=False):

        # ── Left column: Inputs ────────────────────────────
        with gr.Column(scale=4, min_width=300):

            with gr.Group(elem_classes="card"):
                gr.HTML('<div class="card-title">🔗 Audio Source</div>')
                youtube_url = gr.Textbox(
                    label="YouTube Shorts URL",
                    placeholder="https://youtube.com/shorts/...",
                    lines=1,
                    elem_id="yt-url-input",
                )
                gr.HTML('<div style="text-align:center;color:#475569;font-size:0.8rem;padding:6px 0">— or —</div>')
                uploaded_audio = gr.Audio(
                    label="Upload audio file (WAV / MP3 / M4A)",
                    type="filepath",
                    elem_id="audio-upload",
                )

            with gr.Group(elem_classes="card"):
                gr.HTML('<div class="card-title">⚙️ Mode</div>')
                mode = gr.Radio(
                    ["Fact Check", "QA"],
                    value="Fact Check",
                    label=None,
                    elem_id="mode-radio",
                )
                question_box = gr.Textbox(
                    label="Your question (QA mode only)",
                    placeholder="e.g. What product is being reviewed?",
                    lines=2,
                    visible=False,
                    elem_id="question-input",
                )
                mode.change(
                    fn=lambda m: gr.update(visible=(m == "QA")),
                    inputs=mode,
                    outputs=question_box,
                )

            run_btn = gr.Button("🚀 Run Analysis", variant="primary", elem_id="run-btn")

            gr.HTML("""
            <div style="margin-top:14px;padding:12px 16px;background:#13161e;border:1px solid #1e2434;
                        border-radius:10px;font-size:0.78rem;color:#475569;line-height:1.6;">
                <strong style="color:#7c3aed">⚡ ZeroGPU Note:</strong>
                Cold starts take ~60s. Warm requests are fast.<br>
                <strong style="color:#7c3aed">📏 Limit:</strong>
                Max audio duration is 3 minutes.<br>
                <strong style="color:#7c3aed">🔑 Requires:</strong>
                TAVILY_API_KEY and HF_TOKEN Space secrets.
            </div>
            """)

        # ── Right column: Results ──────────────────────────
        with gr.Column(scale=6, min_width=400):

            status_bar = gr.Markdown(
                value="*Waiting for input…*",
                elem_id="status-bar",
            )

            result_box = gr.Markdown(
                value="",
                label="Result",
                elem_id="result-box",
            )

    # ── Examples ──────────────────────────────────────────
    gr.Examples(
        examples=EXAMPLE_URLS,
        inputs=[youtube_url, uploaded_audio, mode, question_box],
        label="Example URLs",
    )

    # ── Wire up run button ─────────────────────────────────
    run_btn.click(
        fn=run,
        inputs=[youtube_url, uploaded_audio, mode, question_box],
        outputs=[result_box, status_bar],
        show_progress=True,
    )

    gr.HTML("""
    <div style="text-align:center;padding:24px 0 10px;font-size:0.78rem;color:#334155;">
        Built with <strong style="color:#a78bfa">Gemma 4 E4B</strong> (Google DeepMind) ·
        <strong style="color:#06b6d4">Tavily</strong> Search ·
        Deployed on <strong style="color:#ec4899">Hugging Face ZeroGPU</strong>
    </div>
    """)

# ─────────────────────────────────────────────────────────────────────────────
# Launch
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    demo.launch(show_error=True)
