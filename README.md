# Focus

A desktop screen annotation assistant. Draw on your screen and talk to an AI about what you see.

## How it works

1. Press **Alt+D** to enter drawing mode
2. **Draw** on screen with your mouse (red ink) and **speak** to describe your question
3. **Release Alt** to send — the app captures the final full-screen screenshot along with your voice, and sends it to the AI model
4. The model's response appears in the bottom-right corner of your screen and is spoken aloud through MOSS-TTS-Nano

The spoken response uses sentence-level streaming: as soon as the LLM emits a stable sentence fragment, the app starts a MOSS-TTS streaming request instead of waiting for the full answer to finish.

The app runs as a transparent always-on-top overlay. When not in drawing mode, all mouse events pass through to the underlying windows.

## Setup

```bash
npm install
```

Create a `.env` file:

```
LLM_BASE=http://127.0.0.1:1234/v1
LLM_MODEL=qwen3.5-4b
LLM_API_KEY=lm-studio

TAVILY_API_KEY=your_key_here
MOSS_TTS_REPO_PATH=/Users/huanghan/Desktop/Projects/MOSS-TTS-Nano
MOSS_TTS_PYTHON=/Users/huanghan/Desktop/Projects/MOSS-TTS-Nano/.venv/bin/python
MOSS_TTS_CHECKPOINT_PATH=/Users/huanghan/.cache/huggingface/hub/models--OpenMOSS-Team--MOSS-TTS-Nano/snapshots/c3158a0fb0ff379ef79750129152b7730b5fb0f9
MOSS_TTS_AUDIO_TOKENIZER_PATH=/Users/huanghan/.cache/huggingface/hub/models--OpenMOSS-Team--MOSS-Audio-Tokenizer-Nano/snapshots/8ee35ebff3a211e0aad01c1aa7b2076a4310440f

# Optional screenshot preprocessing with OmniParser
OMNIPARSER_REPO_PATH=/Users/huanghan/Desktop/Projects/OmniParser
OMNIPARSER_PYTHON=python3
OMNIPARSER_PORT=18084
```

`TAVILY_API_KEY` is optional, but required if you want the agent to use live web search.
`MOSS_TTS_REPO_PATH` and `MOSS_TTS_PYTHON` default to the local MOSS-TTS-Nano checkout above.
`MOSS_TTS_AUDIO_TOKENIZER_PATH` is optional; if unset, the app will first try the local Hugging Face cache snapshot and then fall back to the remote repo id.
`MOSS_TTS_CHECKPOINT_PATH` is also optional, but a local checkpoint path only works if that directory already contains the TTS text tokenizer files. Otherwise the app falls back to the upstream repo id so MOSS can resolve the tokenizer itself.
The Electron app will auto-start the FastAPI TTS server and use its streaming endpoint for low-latency playback.
If `OMNIPARSER_REPO_PATH` points to a valid OmniParser checkout with weights, the app will auto-start OmniParser and preprocess the latest full-screen screenshot before sending it to the LLM. The LLM then receives the original screenshot, the OmniParser labeled screenshot, and the full text UI element list.
By default, Focus now starts OmniParser with `easyocr` and `ch_sim,en`, so Chinese and English text can both be recognized if the local EasyOCR model files are present.

## Usage

```bash
npm start
```

## Debug

Each annotation session is saved to `tmp/{task_id}/` containing:
- `frame_000.jpg` — final full-screen screenshot
- `audio.webm` — voice recording

## Tech stack

- Electron (transparent overlay, global shortcuts, screen capture)
- OpenAI-compatible local/model endpoint with streaming
- OmniParser for optional screenshot preprocessing and UI element extraction
- Tavily web search tool for live lookups
- Canvas-based drawing with hand-drawn jitter effect
