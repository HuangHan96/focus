# Focus

A desktop screen annotation assistant powered by AI vision models. Draw on your screen and talk to an AI about what you see.

## Features

- **Screen Annotation**: Draw directly on your screen with mouse/trackpad
- **Voice Input**: Speak your questions while drawing
- **AI Vision**: Powered by OpenAI-compatible vision models (local or cloud)
- **Text-to-Speech**: Natural voice responses via MOSS-TTS-Nano
- **UI Understanding**: Optional OmniParser integration for enhanced screen element detection
- **Web Search**: Live web search capability via Tavily API
- **Computer Control**: AI can interact with your computer (click, type, scroll, etc.)

## How it works

1. Press **Alt+D** to enter drawing mode
2. **Draw** on screen with your mouse (red ink) and **speak** to describe your question
3. **Release Alt** to send — the app captures the screenshot with your voice and sends it to the AI
4. The AI's response appears in the bottom-right corner and is spoken aloud

The app runs as a transparent always-on-top overlay. When not in drawing mode, all mouse events pass through to underlying windows.

## Prerequisites

- **Node.js** (v16 or later)
- **Python 3.10+** (for TTS and OmniParser)
- **uv** (recommended, for faster Python dependency installation)
  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```
- **LLM Server**: A running OpenAI-compatible API endpoint (e.g., LM Studio, Ollama, or OpenAI)

## Quick Start

### 1. Clone the repository with submodules

```bash
git clone --recursive https://github.com/yourusername/focus.git
cd focus
```

If you already cloned without `--recursive`:

```bash
git submodule update --init --recursive
```

### 2. Install Node.js dependencies

```bash
npm install
```

### 3. Configure environment

Create a `.env` file in the project root:

```env
# Required: LLM Configuration
LLM_BASE=http://localhost:1234/v1
LLM_MODEL=qwen2.5-vl-7b-instruct
LLM_API_KEY=lm_studio

# Optional: Web Search (required for web search tool)
TAVILY_API_KEY=your_tavily_api_key_here

# Optional: Vision and OmniParser
VISION_ENABLE=true
LLM_INCLUDE_OMNIPARSER_IMAGE=true
LLM_OMNIPARSER_SUMMARY_LIMIT=36
LLM_OMNIPARSER_CONTENT_TRUNCATE=64
```

**Note**: TTS and OmniParser dependencies will be automatically installed on first run. No manual setup required!

### 4. Start the app

```bash
npm start
```

On first launch, the app will:
- Create Python virtual environments for TTS and OmniParser
- Install all required dependencies (this may take 5-10 minutes)
- Download model weights from HuggingFace
- Start the TTS and OmniParser services

Subsequent launches will be much faster as dependencies are already installed.

## Configuration

### LLM Setup

The app works with any OpenAI-compatible API. Popular options:

**LM Studio** (recommended for local models):
1. Download from [lmstudio.ai](https://lmstudio.ai)
2. Load a vision model (e.g., `qwen2.5-vl-7b-instruct`)
3. Start the local server (default: `http://localhost:1234/v1`)

**Ollama**:
```bash
ollama serve
# Set LLM_BASE=http://localhost:11434/v1
```

**OpenAI**:
```env
LLM_BASE=https://api.openai.com/v1
LLM_MODEL=gpt-4o
LLM_API_KEY=sk-your-api-key
```

### Optional Features

**Web Search** (via Tavily):
- Get API key from [tavily.com](https://tavily.com)
- Add `TAVILY_API_KEY` to `.env`

**Computer Control**:
- Enabled by default
- AI can click, type, scroll, and interact with your screen
- Uses OmniParser for UI element detection

## Keyboard Shortcuts

- **Alt+D**: Enter drawing mode
- **Alt (hold)**: Draw on screen
- **Alt (release)**: Send to AI
- **Cmd+Q** (Mac) / **Alt+F4** (Windows): Quit app

## Troubleshooting

### TTS or OmniParser fails to start

The app automatically installs dependencies on first run. If setup fails:

1. **Check Python version**: `python3 --version` (must be 3.10+)
2. **Install uv**: `curl -LsSf https://astral.sh/uv/install.sh | sh`
3. **Manual cleanup**:
   ```bash
   rm -rf MOSS-TTS-Nano/.venv OmniParser/.venv OmniParser/weights
   npm start  # Retry setup
   ```

### Port conflicts

If ports 18083 (TTS) or 18084 (OmniParser) are in use:

```bash
# Kill existing processes
lsof -ti:18083 | xargs kill
lsof -ti:18084 | xargs kill
```

Or set custom ports in `.env`:
```env
MOSS_TTS_PORT=18085
OMNIPARSER_PORT=18086
```

### LLM connection issues

- Verify your LLM server is running
- Check `LLM_BASE` URL is correct
- Test with: `curl http://localhost:1234/v1/models`

## Project Structure

```
focus/
├── main.js              # Electron main process
├── mask.html            # Overlay UI
├── package.json         # Node.js dependencies
├── .env                 # Configuration (create this)
├── MOSS-TTS-Nano/       # TTS submodule (auto-setup)
├── OmniParser/          # UI parser submodule (auto-setup)
└── tmp/                 # Debug output (screenshots, audio)
```

## Debug

Each interaction is saved to `tmp/{timestamp}/`:
- `frame_000.jpg` — Screenshot with annotations
- `audio.webm` — Voice recording
- `debug.json` — Full request/response data

## Tech Stack

- **Electron**: Transparent overlay, screen capture, global shortcuts
- **MOSS-TTS-Nano**: Fast multilingual text-to-speech
- **OmniParser**: UI element detection and screen understanding
- **Tavily**: Web search API
- **OpenAI API**: Compatible with local and cloud LLMs

## License

MIT

## Credits

- [MOSS-TTS-Nano](https://github.com/OpenMOSS/MOSS-TTS-Nano) by OpenMOSS
- [OmniParser](https://github.com/microsoft/OmniParser) by Microsoft
