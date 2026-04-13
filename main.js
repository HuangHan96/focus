const { app, BrowserWindow, screen, desktopCapturer, ipcMain, globalShortcut } = require('electron');
const os = require('os');
const path = require('path');
const fs = require('fs');
const { spawn } = require('child_process');
require('dotenv').config();

const LLM_BASE = process.env.LLM_BASE || 'http://127.0.0.1:1234/v1';
const LLM_MODEL = process.env.LLM_MODEL || 'qwen/qwen3.5-9b';
const LLM_API_KEY = process.env.LLM_API_KEY || 'lm-studio';
const TAVILY_BASE = process.env.TAVILY_BASE || 'https://api.tavily.com/search';
const TAVILY_API_KEY = process.env.TAVILY_API_KEY || '';
const MAX_TOOL_ROUNDS = 4;
const TTS_REPO_PATH = process.env.MOSS_TTS_REPO_PATH || '/Users/huanghan/Desktop/Projects/MOSS-TTS-Nano';
const TTS_HOST = process.env.MOSS_TTS_HOST || '127.0.0.1';
const TTS_PORT = Number.parseInt(process.env.MOSS_TTS_PORT || '18083', 10);
const TTS_BASE = `http://${TTS_HOST}:${TTS_PORT}`;
const TTS_HEALTH_TIMEOUT_MS = 90_000;
const TTS_REQUEST_TIMEOUT_MS = 30_000;
const TTS_STREAM_MIN_CHARS = 18;
const TTS_STREAM_SOFT_CHARS = 36;
const TTS_STREAM_MAX_CHARS = 120;
const TTS_PYTHON = process.env.MOSS_TTS_PYTHON || path.join(TTS_REPO_PATH, '.venv', 'bin', 'python');
const TTS_DEMO_METADATA_PATH = path.join(TTS_REPO_PATH, 'assets', 'demo.jsonl');
const TTS_HF_HOME = process.env.MOSS_TTS_HF_HOME || path.join(__dirname, '.cache', 'moss-huggingface');
const TTS_HF_MODULES_CACHE = process.env.HF_MODULES_CACHE || path.join(TTS_HF_HOME, 'modules');
const TTS_TRANSFORMERS_CACHE = process.env.TRANSFORMERS_CACHE || path.join(TTS_HF_HOME, 'hub');
const OMNIPARSER_REPO_PATH = process.env.OMNIPARSER_REPO_PATH || '/Users/huanghan/Desktop/Projects/OmniParser';
const OMNIPARSER_SERVER_DIR = path.join(OMNIPARSER_REPO_PATH, 'omnitool', 'omniparserserver');
const OMNIPARSER_HOST = process.env.OMNIPARSER_HOST || '127.0.0.1';
const OMNIPARSER_PORT = Number.parseInt(process.env.OMNIPARSER_PORT || '18084', 10);
const OMNIPARSER_BASE = `http://${OMNIPARSER_HOST}:${OMNIPARSER_PORT}`;
const OMNIPARSER_PYTHON = process.env.OMNIPARSER_PYTHON
  || (fs.existsSync(path.join(OMNIPARSER_REPO_PATH, '.venv', 'bin', 'python')) ? path.join(OMNIPARSER_REPO_PATH, '.venv', 'bin', 'python') : 'python3');
const OMNIPARSER_HEALTH_TIMEOUT_MS = 120_000;
const OMNIPARSER_REQUEST_TIMEOUT_MS = 90_000;
const OMNIPARSER_SOM_MODEL_PATH = process.env.OMNIPARSER_SOM_MODEL_PATH || path.join(OMNIPARSER_REPO_PATH, 'weights', 'icon_detect', 'model.pt');
const OMNIPARSER_CAPTION_MODEL_NAME = process.env.OMNIPARSER_CAPTION_MODEL_NAME || 'florence2';
const OMNIPARSER_CAPTION_MODEL_PATH = process.env.OMNIPARSER_CAPTION_MODEL_PATH || path.join(OMNIPARSER_REPO_PATH, 'weights', 'icon_caption_florence');
const OMNIPARSER_DEVICE = process.env.OMNIPARSER_DEVICE || '';
const OMNIPARSER_BOX_THRESHOLD = process.env.OMNIPARSER_BOX_THRESHOLD || '0.05';
const OMNIPARSER_MAX_ELEMENTS = clampEnvInt(process.env.OMNIPARSER_MAX_ELEMENTS, 120);
const HF_CACHE_DIR = path.join(os.homedir(), '.cache', 'huggingface', 'hub');

// 是否发送绘画过程中的截图序列，关闭后只发送最后一张完整截图
const SEND_ALL_FRAMES = false;

app.commandLine.appendSwitch('autoplay-policy', 'no-user-gesture-required');

const TOOLS = [
  {
    type: 'function',
    function: {
      name: 'show_tooltip',
      description: 'Display the current step instruction on screen. Only call this once per step. The user will press "Next" to proceed. When all steps are complete, do NOT call this tool — just reply with text.',
      parameters: {
        type: 'object',
        properties: {
          text: { type: 'string', description: 'Brief instruction for this step in the user\'s language.' },
          x: { type: 'number', description: 'Required X coordinate in absolute screen pixels for the relevant target area.' },
          y: { type: 'number', description: 'Required Y coordinate in absolute screen pixels for the relevant target area.' },
          element_index: { type: 'integer', description: 'Optional OmniParser element index to highlight with a rectangle. Use the same index shown in the OmniParser summary when it matches the target UI element.' }
        },
        required: ['text', 'x', 'y']
      }
    }
  },
  {
    type: 'function',
    function: {
      name: 'tavily_web_search',
      description: 'Search the live web with Tavily when the answer requires up-to-date or external information not visible in the screenshot.',
      parameters: {
        type: 'object',
        properties: {
          query: { type: 'string', description: 'The web search query.' },
          topic: {
            type: 'string',
            enum: ['general', 'news', 'finance'],
            description: 'Search topic. Use news for recent events and finance for market data.'
          },
          search_depth: {
            type: 'string',
            enum: ['basic', 'advanced', 'fast', 'ultra-fast'],
            description: 'Latency vs. relevance tradeoff.'
          },
          max_results: {
            type: 'integer',
            description: 'Maximum number of search results to return, from 1 to 10.'
          },
          time_range: {
            type: 'string',
            enum: ['day', 'week', 'month', 'year', 'd', 'w', 'm', 'y'],
            description: 'Optional recency filter.'
          },
          days: {
            type: 'integer',
            description: 'Optional number of days back for news searches.'
          }
        },
        required: ['query']
      }
    }
  }
];

let maskWindow;
let screenWidth, screenHeight;
let overlayOffsetX = 0;
let overlayOffsetY = 0;
let screenshots = [];
let taskCounter = 0;
let conversationHistory = [];
const MAX_HISTORY = 10;
let stepMessages = [];
let stepNumber = 0;
let ttsProcess = null;
let ttsReadyPromise = null;
let omniParserProcess = null;
let omniParserReadyPromise = null;
let latestOmniParserElements = [];
let activeTtsSession = null;
let ttsPlaybackRunId = 0;
const resolvedLocalTtsCheckpointPath = resolveCachedSnapshotPath('OpenMOSS-Team/MOSS-TTS-Nano');
const TTS_CHECKPOINT_PATH = process.env.MOSS_TTS_CHECKPOINT_PATH || (hasUsableLocalTtsTokenizer(resolvedLocalTtsCheckpointPath) ? resolvedLocalTtsCheckpointPath : 'OpenMOSS-Team/MOSS-TTS-Nano');
const TTS_AUDIO_TOKENIZER_PATH = process.env.MOSS_TTS_AUDIO_TOKENIZER_PATH || resolveCachedSnapshotPath('OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano') || 'OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano';
const ttsDemoEntries = loadTtsDemoEntries();

function clampEnvInt(value, fallback) {
  const num = Number.parseInt(String(value || ''), 10);
  return Number.isFinite(num) && num > 0 ? num : fallback;
}

function resolveCachedSnapshotPath(modelId) {
  const normalizedId = String(modelId || '').trim();
  if (!normalizedId) return '';

  const cacheRoot = path.join(HF_CACHE_DIR, `models--${normalizedId.replace(/\//g, '--')}`);
  const mainRefPath = path.join(cacheRoot, 'refs', 'main');
  try {
    if (fs.existsSync(mainRefPath)) {
      const snapshotId = fs.readFileSync(mainRefPath, 'utf-8').trim();
      const snapshotPath = path.join(cacheRoot, 'snapshots', snapshotId);
      if (fs.existsSync(snapshotPath)) {
        return snapshotPath;
      }
    }

    const snapshotsDir = path.join(cacheRoot, 'snapshots');
    if (!fs.existsSync(snapshotsDir)) return '';
    const snapshotNames = fs.readdirSync(snapshotsDir).filter(Boolean).sort();
    if (!snapshotNames.length) return '';
    return path.join(snapshotsDir, snapshotNames[snapshotNames.length - 1]);
  } catch (error) {
    console.warn(`解析本地缓存模型失败(${normalizedId}):`, error.message);
    return '';
  }
}

function hasUsableLocalTtsTokenizer(rawPath) {
  if (!rawPath) return false;

  try {
    const candidatePath = path.resolve(rawPath);
    return (
      fs.existsSync(path.join(candidatePath, 'tokenizer.model')) ||
      fs.existsSync(path.join(candidatePath, 'hf_tokenizer', 'tokenizer.json')) ||
      fs.existsSync(path.join(candidatePath, 'sentencepiece', 'mossttsnano_spm_bpe.model'))
    );
  } catch {
    return false;
  }
}

function loadTtsDemoEntries() {
  try {
    const raw = fs.readFileSync(TTS_DEMO_METADATA_PATH, 'utf-8');
    return raw
      .split('\n')
      .map(line => line.trim())
      .filter(Boolean)
      .map((line, index) => {
        const payload = JSON.parse(line);
        return {
          id: `demo-${index + 1}`,
          name: String(payload.name || '').trim(),
          role: String(payload.role || '').trim(),
          text: String(payload.text || '').trim()
        };
      });
  } catch (error) {
    console.warn('读取 TTS demo 配置失败:', error.message);
    return [];
  }
}

function categorizeDemoEntry(entry) {
  const name = `${entry.name} ${entry.role}`.toLowerCase();
  if (name.includes('zh_') || entry.name.includes('🇨🇳')) return 'zh';
  if (name.includes('en_') || entry.name.includes('🇺🇸')) return 'en';
  if (entry.name.includes('🇯🇵') || name.includes('jp_') || name.includes('ja_')) return 'ja';
  if (entry.name.includes('🇰🇷') || name.includes('ko_')) return 'ko';
  if (entry.name.includes('🇷🇺')) return 'ru';
  if (entry.name.includes('🇸🇦')) return 'ar';
  if (entry.name.includes('🇮🇷')) return 'fa';
  return 'fallback';
}

function detectTtsLanguage(text) {
  if (!text) return 'zh';
  if (/[\u0600-\u06FF]/.test(text)) return 'ar';
  if (/[\u0400-\u04FF]/.test(text)) return 'ru';
  if (/[\u3040-\u30FF]/.test(text)) return 'ja';
  if (/[\uAC00-\uD7AF]/.test(text)) return 'ko';
  if (/[\u4E00-\u9FFF]/.test(text)) return 'zh';
  if (/[A-Za-z]/.test(text)) return 'en';
  return 'zh';
}

function resolveTtsDemoId(text) {
  if (!ttsDemoEntries.length) return '';

  const preferredLanguage = detectTtsLanguage(text);
  const match = ttsDemoEntries.find(entry => categorizeDemoEntry(entry) === preferredLanguage);
  if (match) return match.id;

  const englishFallback = ttsDemoEntries.find(entry => categorizeDemoEntry(entry) === 'en');
  if (englishFallback) return englishFallback.id;

  return ttsDemoEntries[0].id;
}

function sanitizeTextForTts(text) {
  if (!text || typeof text !== 'string') return '';
  return text
    .replace(/```[\s\S]*?```/g, ' ')
    .replace(/`([^`]+)`/g, ' $1 ')
    .replace(/!\[([^\]]*)\]\([^)]+\)/g, ' $1 ')
    .replace(/\[([^\]]+)\]\([^)]+\)/g, ' $1 ')
    .replace(/<[^>]+>/g, ' ')
    .replace(/^\s{0,3}#{1,6}\s+/gm, '')
    .replace(/^\s{0,3}>\s?/gm, '')
    .replace(/^\s*[-+*]\s+\[[ xX]\]\s+/gm, '')
    .replace(/^\s*[-+*]\s+/gm, '')
    .replace(/^\s*\d+[.)]\s+/gm, '')
    .replace(/^\s*[-*_]{3,}\s*$/gm, ' ')
    .replace(/^\s*\|/gm, '')
    .replace(/\|\s*$/gm, '')
    .replace(/\|/g, '，')
    .replace(/(^|\s)\*{1,3}([^*\n]+)\*{1,3}(?=\s|$)/g, '$1$2')
    .replace(/(^|\s)_{1,3}([^_\n]+)_{1,3}(?=\s|$)/g, '$1$2')
    .replace(/~~([^~\n]+)~~/g, '$1')
    .replace(/\\([\\`*_{}\[\]()#+.!|>\-])/g, '$1')
    .replace(/https?:\/\/\S+/g, ' ')
    .replace(/[*#>_~`]+/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();
}

async function fetchWithTimeout(url, options = {}, timeoutMs = TTS_REQUEST_TIMEOUT_MS) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);

  let cleanupAbortListener = null;
  if (options.signal) {
    const onAbort = () => controller.abort();
    if (options.signal.aborted) {
      controller.abort();
    } else {
      options.signal.addEventListener('abort', onAbort, { once: true });
      cleanupAbortListener = () => options.signal.removeEventListener('abort', onAbort);
    }
  }

  try {
    return await fetch(url, { ...options, signal: controller.signal });
  } finally {
    clearTimeout(timer);
    if (cleanupAbortListener) cleanupAbortListener();
  }
}

function isOmniParserConfigured() {
  return fs.existsSync(OMNIPARSER_SERVER_DIR) && fs.existsSync(OMNIPARSER_SOM_MODEL_PATH) && fs.existsSync(OMNIPARSER_CAPTION_MODEL_PATH);
}

async function isOmniParserHealthy() {
  if (!isOmniParserConfigured()) return false;
  try {
    const response = await fetchWithTimeout(`${OMNIPARSER_BASE}/probe/`, {}, 1500);
    return response.ok;
  } catch {
    return false;
  }
}

function attachOmniParserProcessLogs(child) {
  child.stdout.on('data', chunk => {
    process.stdout.write(`[omniparser] ${chunk.toString()}`);
  });
  child.stderr.on('data', chunk => {
    process.stderr.write(`[omniparser] ${chunk.toString()}`);
  });
}

function spawnOmniParserService() {
  if (!isOmniParserConfigured()) return;
  if (omniParserProcess && !omniParserProcess.killed) return;

  const spawnArgs = [
    'omniparserserver.py',
    '--host', OMNIPARSER_HOST,
    '--port', String(OMNIPARSER_PORT),
    '--som_model_path', OMNIPARSER_SOM_MODEL_PATH,
    '--caption_model_name', OMNIPARSER_CAPTION_MODEL_NAME,
    '--caption_model_path', OMNIPARSER_CAPTION_MODEL_PATH,
    '--BOX_TRESHOLD', String(OMNIPARSER_BOX_THRESHOLD)
  ];

  if (OMNIPARSER_DEVICE) {
    spawnArgs.push('--device', OMNIPARSER_DEVICE);
  }

  console.log(`启动 OmniParser 服务: ${OMNIPARSER_PYTHON} ${spawnArgs.join(' ')}`);
  omniParserProcess = spawn(OMNIPARSER_PYTHON, spawnArgs, {
    cwd: OMNIPARSER_SERVER_DIR,
    env: process.env,
    stdio: ['ignore', 'pipe', 'pipe']
  });

  attachOmniParserProcessLogs(omniParserProcess);

  omniParserProcess.on('exit', (code, signal) => {
    console.log(`OmniParser 服务已退出: code=${code} signal=${signal}`);
    omniParserProcess = null;
    omniParserReadyPromise = null;
  });

  omniParserProcess.on('error', error => {
    console.error('OmniParser 服务启动失败:', error.message);
  });
}

async function waitForOmniParserHealthy(timeoutMs = OMNIPARSER_HEALTH_TIMEOUT_MS) {
  const startedAt = Date.now();
  while (Date.now() - startedAt < timeoutMs) {
    if (await isOmniParserHealthy()) {
      return true;
    }
    await new Promise(resolve => setTimeout(resolve, 1000));
  }
  return false;
}

async function ensureOmniParserReady() {
  if (!isOmniParserConfigured()) {
    throw new Error('OmniParser 未配置完整，缺少服务目录或模型权重');
  }

  if (await isOmniParserHealthy()) return true;

  if (omniParserReadyPromise) {
    try {
      await omniParserReadyPromise;
    } catch {}

    if (await isOmniParserHealthy()) {
      return true;
    }

    omniParserReadyPromise = null;
  }

  if (!omniParserReadyPromise) {
    omniParserReadyPromise = (async () => {
      spawnOmniParserService();
      const ready = await waitForOmniParserHealthy();
      if (!ready) {
        throw new Error(`OmniParser 服务未在 ${OMNIPARSER_HEALTH_TIMEOUT_MS}ms 内就绪`);
      }
      return true;
    })().catch(error => {
      omniParserReadyPromise = null;
      throw error;
    });
  }

  return omniParserReadyPromise;
}

function normalizeImageDataUrl(raw) {
  const url = String(raw || '');
  return url.startsWith('data:') ? url : `data:image/jpeg;base64,${url}`;
}

function imageDataUrlToBase64(dataUrl) {
  const match = String(dataUrl || '').match(/^data:image\/[a-zA-Z0-9.+-]+;base64,(.+)$/);
  return match ? match[1] : String(dataUrl || '');
}

function summarizeOmniParserElements(elements, imageWidth, imageHeight, limit = OMNIPARSER_MAX_ELEMENTS) {
  const list = Array.isArray(elements) ? elements.slice(0, limit) : [];
  const lines = list.map((element, index) => {
    const bbox = Array.isArray(element?.bbox) ? element.bbox : [];
    const x1 = Math.round(Number(bbox[0] || 0) * imageWidth);
    const y1 = Math.round(Number(bbox[1] || 0) * imageHeight);
    const x2 = Math.round(Number(bbox[2] || 0) * imageWidth);
    const y2 = Math.round(Number(bbox[3] || 0) * imageHeight);
    const type = element?.type || 'unknown';
    const interactive = element?.interactivity ? 'interactive' : 'static';
    const content = sanitizeTextForTts(String(element?.content || '').trim()).slice(0, 180);
    return `${index}: type=${type}, ${interactive}, bbox=[${x1},${y1},${x2},${y2}], content="${content}"`;
  }).filter(Boolean);

  if (Array.isArray(elements) && elements.length > list.length) {
    lines.push(`... ${elements.length - list.length} more elements omitted`);
  }

  return lines.join('\n');
}

function normalizeOmniParserElementIndex(value) {
  const index = Number.parseInt(String(value), 10);
  return Number.isFinite(index) && index >= 0 ? index : null;
}

function normalizeMatchText(value) {
  return String(value || '')
    .toLowerCase()
    .replace(/[“”"'\u2018\u2019]/g, '')
    .replace(/[^a-z0-9\u4e00-\u9fff]+/gi, ' ')
    .replace(/\s+/g, ' ')
    .trim();
}

function extractQuotedPhrases(text) {
  const input = String(text || '');
  const patterns = [
    /“([^”]{2,80})”/g,
    /"([^"]{2,80})"/g,
    /'([^']{2,80})'/g,
    /「([^」]{2,80})」/g,
    /『([^』]{2,80})』/g
  ];
  const phrases = [];
  for (const pattern of patterns) {
    for (const match of input.matchAll(pattern)) {
      const phrase = String(match[1] || '').trim();
      if (phrase) phrases.push(phrase);
    }
  }
  return [...new Set(phrases)];
}

function findBestOmniParserElementByText(text) {
  const target = normalizeMatchText(text);
  if (!target) return null;

  let best = null;
  let bestScore = -1;
  const items = Array.isArray(latestOmniParserElements) ? latestOmniParserElements : [];
  for (let index = 0; index < items.length; index += 1) {
    const element = items[index];
    const content = normalizeMatchText(element?.content || '');
    if (!content) continue;

    let score = -1;
    if (content === target) {
      score = 1000;
    } else if (content.startsWith(target)) {
      score = 800 - Math.max(0, content.length - target.length);
    } else if (content.includes(target)) {
      score = 600 - Math.max(0, content.length - target.length);
    } else if (target.includes(content) && content.length >= 4) {
      score = 400 - Math.max(0, target.length - content.length);
    }

    if (score > bestScore) {
      bestScore = score;
      best = { index, element };
    }
  }

  return bestScore >= 0 ? best : null;
}

function resolveTooltipTargetIndex(text, requestedElementIndex) {
  const requestedIndex = normalizeOmniParserElementIndex(requestedElementIndex);
  const quotedPhrases = extractQuotedPhrases(text);
  for (const phrase of quotedPhrases) {
    const bestMatch = findBestOmniParserElementByText(phrase);
    if (!bestMatch) continue;
    if (requestedIndex === null || bestMatch.index !== requestedIndex) {
      return bestMatch.index;
    }
  }
  return requestedIndex;
}

function resolveTooltipHighlight(elementIndex) {
  const normalizedIndex = normalizeOmniParserElementIndex(elementIndex);
  if (normalizedIndex === null) return null;
  const element = Array.isArray(latestOmniParserElements) ? latestOmniParserElements[normalizedIndex] : null;
  const bbox = Array.isArray(element?.bbox) ? element.bbox : null;
  if (!bbox || bbox.length < 4) return null;

  const x1 = Math.round(clampNumber(Number(bbox[0]) * screenWidth, 0, 0, Math.max(screenWidth - 1, 0)));
  const y1 = Math.round(clampNumber(Number(bbox[1]) * screenHeight, 0, 0, Math.max(screenHeight - 1, 0)));
  const x2 = Math.round(clampNumber(Number(bbox[2]) * screenWidth, 0, 0, Math.max(screenWidth - 1, 0)));
  const y2 = Math.round(clampNumber(Number(bbox[3]) * screenHeight, 0, 0, Math.max(screenHeight - 1, 0)));
  const width = Math.max(1, x2 - x1);
  const height = Math.max(1, y2 - y1);
  const renderX = Math.max(0, x1 - overlayOffsetX);
  const renderY = Math.max(0, y1 - overlayOffsetY);

  return {
    index: normalizedIndex,
    x: renderX,
    y: renderY,
    width,
    height,
    centerX: Math.max(0, x1 + Math.round(width / 2) - overlayOffsetX),
    centerY: Math.max(0, y1 + Math.round(height / 2) - overlayOffsetY),
    rawX: x1,
    rawY: y1,
    type: element?.type || 'unknown',
    content: sanitizeTextForTts(String(element?.content || '').trim()).slice(0, 120)
  };
}

async function preprocessScreenshotWithOmniParser(frameDataUrl) {
  await ensureOmniParserReady();

  const normalizedDataUrl = normalizeImageDataUrl(frameDataUrl);
  const response = await fetchWithTimeout(`${OMNIPARSER_BASE}/parse/`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({
      base64_image: imageDataUrlToBase64(normalizedDataUrl)
    })
  }, OMNIPARSER_REQUEST_TIMEOUT_MS);

  if (!response.ok) {
    const errText = await response.text();
    throw new Error(`OmniParser HTTP ${response.status}: ${errText.slice(0, 500)}`);
  }

  const payload = await response.json();
  const parsedImageBase64 = String(payload?.som_image_base64 || '').trim();
  const parsedElements = Array.isArray(payload?.parsed_content_list) ? payload.parsed_content_list : [];

  return {
    parsedImageDataUrl: parsedImageBase64 ? `data:image/png;base64,${parsedImageBase64}` : '',
    parsedElements,
    latency: payload?.latency || null
  };
}

async function isTtsServiceHealthy() {
  try {
    const response = await fetchWithTimeout(`${TTS_BASE}/health`, {}, 1500);
    return response.ok;
  } catch {
    return false;
  }
}

function attachTtsProcessLogs(child) {
  child.stdout.on('data', chunk => {
    process.stdout.write(`[tts] ${chunk.toString()}`);
  });
  child.stderr.on('data', chunk => {
    process.stderr.write(`[tts] ${chunk.toString()}`);
  });
}

function spawnTtsService() {
  if (ttsProcess && !ttsProcess.killed) return;

  const spawnArgs = [
    'app.py',
    '--host', TTS_HOST,
    '--port', String(TTS_PORT),
    '--checkpoint-path', TTS_CHECKPOINT_PATH,
    '--audio-tokenizer-path', TTS_AUDIO_TOKENIZER_PATH
  ];

  console.log(`启动 TTS 服务: ${TTS_PYTHON} ${spawnArgs.join(' ')}`);
  ttsProcess = spawn(TTS_PYTHON, spawnArgs, {
    cwd: TTS_REPO_PATH,
    env: {
      ...process.env,
      HF_HOME: process.env.HF_HOME || TTS_HF_HOME,
      HF_MODULES_CACHE: TTS_HF_MODULES_CACHE,
      TRANSFORMERS_CACHE: TTS_TRANSFORMERS_CACHE
    },
    stdio: ['ignore', 'pipe', 'pipe']
  });

  attachTtsProcessLogs(ttsProcess);

  ttsProcess.on('exit', (code, signal) => {
    console.log(`TTS 服务已退出: code=${code} signal=${signal}`);
    ttsProcess = null;
    ttsReadyPromise = null;
  });

  ttsProcess.on('error', error => {
    console.error('TTS 服务启动失败:', error.message);
  });
}

async function waitForTtsHealthy(timeoutMs = TTS_HEALTH_TIMEOUT_MS) {
  const startedAt = Date.now();
  while (Date.now() - startedAt < timeoutMs) {
    if (await isTtsServiceHealthy()) {
      return true;
    }
    await new Promise(resolve => setTimeout(resolve, 1000));
  }
  return false;
}

async function ensureTtsServiceReady() {
  if (await isTtsServiceHealthy()) return true;

  if (ttsReadyPromise) {
    try {
      await ttsReadyPromise;
    } catch {}

    if (await isTtsServiceHealthy()) {
      return true;
    }

    ttsReadyPromise = null;
  }

  if (!ttsReadyPromise) {
    ttsReadyPromise = (async () => {
      spawnTtsService();
      const ready = await waitForTtsHealthy();
      if (!ready) {
        throw new Error(`TTS 服务未在 ${TTS_HEALTH_TIMEOUT_MS}ms 内就绪`);
      }
      return true;
    })().catch(error => {
      ttsReadyPromise = null;
      throw error;
    });
  }

  return ttsReadyPromise;
}

function resolveTtsUrl(maybeRelativeUrl) {
  if (!maybeRelativeUrl) return '';
  if (/^https?:\/\//i.test(maybeRelativeUrl)) return maybeRelativeUrl;
  return `${TTS_BASE}${maybeRelativeUrl.startsWith('/') ? '' : '/'}${maybeRelativeUrl}`;
}

async function closeTtsStream(streamId) {
  if (!streamId) return;

  try {
    await fetchWithTimeout(`${TTS_BASE}/api/generate-stream/${encodeURIComponent(streamId)}/close`, {
      method: 'POST'
    }, 3000);
  } catch (error) {
    console.warn(`关闭 TTS 流失败(${streamId}):`, error.message);
  }
}

function stopTtsPlayback() {
  const session = activeTtsSession;
  activeTtsSession = null;
  ttsPlaybackRunId += 1;

  if (session) {
    session.cancelled = true;
    session.pendingBuffer = '';
    session.queue.length = 0;
    if (session.currentAbortController) {
      session.currentAbortController.abort();
      session.currentAbortController = null;
    }
    session.currentStreamId = null;
  }

  if (maskWindow && !maskWindow.isDestroyed()) {
    maskWindow.webContents.send('tts-stop');
  }
}

function createTtsSession(source = 'response') {
  stopTtsPlayback();

  const session = {
    id: ttsPlaybackRunId,
    source,
    queue: [],
    pendingBuffer: '',
    processingPromise: null,
    currentAbortController: null,
    currentStreamId: null,
    rendererStarted: false,
    cancelled: false,
    finishRequested: false
  };

  activeTtsSession = session;
  return session;
}

function finalizeTtsSession(session) {
  if (!session || session.cancelled) return;

  session.finishRequested = true;
  if (!session.processingPromise && !session.queue.length) {
    if (session.rendererStarted && maskWindow && !maskWindow.isDestroyed()) {
      maskWindow.webContents.send('tts-end', { id: session.id });
    }
    if (activeTtsSession === session) {
      activeTtsSession = null;
    }
  }
}

function sanitizeTtsQueueText(text) {
  const normalized = sanitizeTextForTts(text);
  return normalized.replace(/\s+/g, ' ').trim();
}

function isStrongTtsBoundary(text, index) {
  const char = text[index];
  if (char === '\n' || char === '\r') return true;
  if ('。！？!?；;'.includes(char)) return true;
  if (char !== '.') return false;

  const nextChar = text[index + 1] || '';
  return !nextChar || /\s|["')\]]/.test(nextChar);
}

function isWeakTtsBoundary(char) {
  return '，,、：:'.includes(char);
}

function extractQueuedTtsSegments(text, force = false) {
  const rawText = String(text || '');
  const segments = [];
  let start = 0;

  while (start < rawText.length) {
    let strongBoundary = -1;
    let weakBoundary = -1;
    let splitIndex = -1;

    for (let index = start; index < rawText.length; index++) {
      if (isStrongTtsBoundary(rawText, index)) {
        strongBoundary = index + 1;
        const candidate = sanitizeTtsQueueText(rawText.slice(start, strongBoundary));
        if (candidate.length >= TTS_STREAM_MIN_CHARS) {
          splitIndex = strongBoundary;
          break;
        }
      } else if (isWeakTtsBoundary(rawText[index])) {
        weakBoundary = index + 1;
      }

      const candidateLength = sanitizeTtsQueueText(rawText.slice(start, index + 1)).length;
      if (candidateLength >= TTS_STREAM_MAX_CHARS) {
        splitIndex = strongBoundary > start
          ? strongBoundary
          : weakBoundary > start
            ? weakBoundary
            : index + 1;
        break;
      }

      if (candidateLength >= TTS_STREAM_SOFT_CHARS && strongBoundary > start) {
        splitIndex = strongBoundary;
        break;
      }
    }

    if (splitIndex === -1) {
      break;
    }

    const segment = sanitizeTtsQueueText(rawText.slice(start, splitIndex));
    if (segment) {
      segments.push(segment);
    }
    start = splitIndex;
  }

  let remainder = rawText.slice(start);
  if (force) {
    const finalSegment = sanitizeTtsQueueText(remainder);
    if (finalSegment) {
      segments.push(finalSegment);
    }
    remainder = '';
  }

  return { segments, remainder };
}

async function streamTtsSegment(session, text) {
  if (!session || session.cancelled || !text) return;

  const abortController = new AbortController();
  let streamId = null;
  session.currentAbortController = abortController;

  try {
    await ensureTtsServiceReady();
    if (session.cancelled || activeTtsSession !== session) return;

    const formData = new URLSearchParams();
    formData.set('text', text);
    const demoId = resolveTtsDemoId(text);
    if (demoId) formData.set('demo_id', demoId);

    const startResponse = await fetchWithTimeout(`${TTS_BASE}/api/generate-stream/start`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/x-www-form-urlencoded'
      },
      body: formData.toString(),
      signal: abortController.signal
    });

    if (!startResponse.ok) {
      const errText = await startResponse.text();
      throw new Error(`TTS start HTTP ${startResponse.status}: ${errText.slice(0, 300)}`);
    }

    const startData = await startResponse.json();
    if (!startData.audio_url || !startData.stream_id) {
      throw new Error('TTS start response missing audio_url/stream_id');
    }

    streamId = String(startData.stream_id);
    session.currentStreamId = streamId;

    if (!session.rendererStarted && maskWindow && !maskWindow.isDestroyed()) {
      maskWindow.webContents.send('tts-start', {
        id: session.id,
        sampleRate: Number.parseInt(startData.sample_rate, 10) || 48000,
        channels: Number.parseInt(startData.channels, 10) || 2,
        source: session.source
      });
      session.rendererStarted = true;
    }

    const audioResponse = await fetch(resolveTtsUrl(startData.audio_url), {
      signal: abortController.signal
    });

    if (!audioResponse.ok) {
      const errText = await audioResponse.text();
      throw new Error(`TTS audio HTTP ${audioResponse.status}: ${errText.slice(0, 300)}`);
    }

    if (!audioResponse.body) {
      throw new Error('TTS audio stream is empty');
    }

    const reader = audioResponse.body.getReader();
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      if (!value?.length) continue;
      if (abortController.signal.aborted || session.cancelled || activeTtsSession !== session) break;

      if (maskWindow && !maskWindow.isDestroyed()) {
        maskWindow.webContents.send('tts-chunk', {
          id: session.id,
          chunkBase64: Buffer.from(value).toString('base64')
        });
      }
    }
  } catch (error) {
    if (!abortController.signal.aborted && !session.cancelled) {
      console.error('TTS 播放失败:', error.message);
      if (maskWindow && !maskWindow.isDestroyed()) {
        maskWindow.webContents.send('tts-error', { message: error.message, source: session.source });
      }
    }
  } finally {
    if (session.currentAbortController === abortController) {
      session.currentAbortController = null;
    }
    if (session.currentStreamId === streamId) {
      session.currentStreamId = null;
    }
    await closeTtsStream(streamId);
  }
}

async function processTtsSessionQueue(session) {
  if (!session || session.cancelled) return;
  if (session.processingPromise) return session.processingPromise;

  session.processingPromise = (async () => {
    while (!session.cancelled && activeTtsSession === session) {
      const nextSegment = session.queue.shift();
      if (!nextSegment) break;
      await streamTtsSegment(session, nextSegment);
    }
  })();

  try {
    await session.processingPromise;
  } finally {
    session.processingPromise = null;
    if (session.finishRequested && !session.cancelled && !session.queue.length) {
      if (session.rendererStarted && maskWindow && !maskWindow.isDestroyed()) {
        maskWindow.webContents.send('tts-end', { id: session.id });
      }
      if (activeTtsSession === session) {
        activeTtsSession = null;
      }
    }
  }
}

function enqueueTtsSegments(session, segments) {
  if (!session || session.cancelled || !Array.isArray(segments) || !segments.length) return;

  session.queue.push(...segments);
  void processTtsSessionQueue(session);
}

function appendStreamingTtsText(session, text) {
  if (!session || session.cancelled || !text) return;

  session.pendingBuffer += text;
  const { segments, remainder } = extractQueuedTtsSegments(session.pendingBuffer, false);
  session.pendingBuffer = remainder;
  enqueueTtsSegments(session, segments);
}

function flushStreamingTtsText(session) {
  if (!session || session.cancelled) return;

  const { segments, remainder } = extractQueuedTtsSegments(session.pendingBuffer, true);
  session.pendingBuffer = remainder;
  enqueueTtsSegments(session, segments);
  finalizeTtsSession(session);
}

function cancelTtsSession(session) {
  if (!session || session.cancelled) return;
  if (activeTtsSession === session) {
    stopTtsPlayback();
    return;
  }

  session.cancelled = true;
  session.pendingBuffer = '';
  session.queue.length = 0;
}

function createStreamingResponseTtsState() {
  let session = null;
  let cancelledForToolCall = false;

  function ensureSession() {
    if (!session || session.cancelled) {
      session = createTtsSession('response');
    }
    return session;
  }

  return {
    pushText(chunk) {
      if (!chunk || cancelledForToolCall) return;
      appendStreamingTtsText(ensureSession(), chunk);
    },
    onToolCall() {
      cancelledForToolCall = true;
      if (session) {
        cancelTtsSession(session);
      }
    },
    finish() {
      if (cancelledForToolCall || !session) return;
      flushStreamingTtsText(session);
    }
  };
}

function speakTtsText(rawText, source = 'response') {
  const text = sanitizeTtsQueueText(rawText);
  if (!text) return;

  const session = createTtsSession(source);
  enqueueTtsSegments(session, [text]);
  finalizeTtsSession(session);
}


async function createMaskWindow() {

  const display = screen.getPrimaryDisplay();
  const { width, height } = display.size; // full screen size, not workArea
  overlayOffsetX = Math.max(0, (display.workArea?.x || 0) - (display.bounds?.x || 0));
  overlayOffsetY = Math.max(0, (display.workArea?.y || 0) - (display.bounds?.y || 0));
  screenWidth = width;
  screenHeight = height;

  maskWindow = new BrowserWindow({
    width: Math.max(1, width - overlayOffsetX),
    height: Math.max(1, height - overlayOffsetY),
    x: (display.bounds?.x || 0) + overlayOffsetX,
    y: (display.bounds?.y || 0) + overlayOffsetY,
    transparent: true, frame: false, alwaysOnTop: true,
    skipTaskbar: true, hasShadow: false,
    webPreferences: { nodeIntegration: true, contextIsolation: false }
  });

  console.log(`遮罩窗口偏移: overlayOffsetX=${overlayOffsetX}, overlayOffsetY=${overlayOffsetY}, screen=${width}x${height}`);

  maskWindow.setAlwaysOnTop(true, 'screen-saver');
  maskWindow.setVisibleOnAllWorkspaces(true);
  maskWindow.setIgnoreMouseEvents(true, { forward: true });
  maskWindow.loadFile('mask.html');

  globalShortcut.register('Alt+D', () => {
    screenshots = [];
    maskWindow.setIgnoreMouseEvents(false);
    maskWindow.focus();
    maskWindow.webContents.send('mode', 'drawing');
  });

  globalShortcut.register('Alt+C', () => {
    conversationHistory = [];
    console.log('上下文记忆已清除');
  });

  ipcMain.on('drawing-done', () => {
    maskWindow.setIgnoreMouseEvents(true, { forward: true });
    maskWindow.webContents.send('mode', 'idle');
  });

  ipcMain.on('set-interactive', () => maskWindow.setIgnoreMouseEvents(false));
  ipcMain.on('set-passthrough', () => maskWindow.setIgnoreMouseEvents(true, { forward: true }));

  ipcMain.on('capture-screenshot', async () => {
    try {
      const dataUrl = await captureScreenJpeg();
      screenshots.push(dataUrl);
      console.log(`截图 #${screenshots.length}`);
    } catch (e) { console.error('截图失败:', e.message); }
  });

  ipcMain.on('capture-final-screenshot', async () => {
    try {
      const dataUrl = await captureFullScreenJpeg();
      screenshots.push(dataUrl);
      console.log(`全屏截图 #${screenshots.length} (final)`);
    } catch (e) { console.error('全屏截图失败:', e.message); }
  });

  ipcMain.on('request-preview', () => maskWindow.webContents.send('show-preview', screenshots));

  ipcMain.on('confirm-send', async (event, audioBase64, userText) => {
    try {
      const frames = [...screenshots];
      screenshots = [];
      console.log(`调用模型... (${frames.length} 帧)`);
      stopTtsPlayback();
      setModelLoading(true);
      saveDebugData(frames, audioBase64);
      await sendToModel(frames, audioBase64, userText);
    } catch (error) {
      console.error('错误:', error.message);
      sendModelError(`请求失败，请重试。\n\n\`${error.message}\``);
      setModelLoading(false);
    }
  });

  ipcMain.on('cancel-send', () => { screenshots = []; console.log('用户取消发送'); });
  ipcMain.on('log', (event, msg) => console.log(msg));

  ipcMain.on('next-step', async () => {
    try {
      console.log('用户点击下一步，截图并继续...');
      stopTtsPlayback();
      setModelLoading(true);
      const dataUrl = await captureFullScreenJpeg();
      await continueStep(dataUrl);
    } catch (error) {
      console.error('下一步错误:', error.message);
      sendModelError(`请求失败，请重试。\n\n\`${error.message}\``);
      setModelLoading(false);
    }
  });

  ipcMain.on('stop-steps', () => {
    stepMessages = [];
    stepNumber = 0;
    stopTtsPlayback();
    console.log('步骤引导已结束');
  });

  // Option+→ global shortcut for Next step
  globalShortcut.register('Alt+Right', () => {
    if (stepNumber > 0) {
      maskWindow.webContents.send('trigger-next');
    }
  });
}

const CROP_W = 800;
const CROP_H = 600;

async function captureScreenJpeg() {
  const cursor = screen.getCursorScreenPoint();
  const display = screen.getPrimaryDisplay();
  const sf = display.scaleFactor;
  const sources = await desktopCapturer.getSources({
    types: ['screen'],
    thumbnailSize: { width: screenWidth * sf, height: screenHeight * sf }
  });
  const fullImage = sources[0].thumbnail;
  const fullW = fullImage.getSize().width;
  const fullH = fullImage.getSize().height;
  const cx = Math.round(cursor.x * sf), cy = Math.round(cursor.y * sf);
  const cw = CROP_W * sf, ch = CROP_H * sf;
  let x = Math.max(0, Math.min(cx - Math.round(cw / 2), fullW - cw));
  let y = Math.max(0, Math.min(cy - Math.round(ch / 2), fullH - ch));
  const cropped = fullImage.crop({ x, y, width: cw, height: ch });
  const pngDataUrl = cropped.toDataURL();
  const jsCode = `
    new Promise(resolve => {
      const img = new Image();
      img.onload = () => {
        const c = document.createElement('canvas');
        c.width = ${CROP_W}; c.height = ${CROP_H};
        c.getContext('2d').drawImage(img, 0, 0, ${CROP_W}, ${CROP_H});
        resolve(c.toDataURL('image/jpeg', 0.8));
      };
      img.src = ${JSON.stringify(pngDataUrl)};
    })
  `;
  return maskWindow.webContents.executeJavaScript(jsCode);
}


async function captureFullScreenJpeg() {
  const sources = await desktopCapturer.getSources({
    types: ['screen'],
    thumbnailSize: { width: screenWidth, height: screenHeight }
  });
  const pngDataUrl = sources[0].thumbnail.toDataURL();
  const jsCode = `
    new Promise(resolve => {
      const img = new Image();
      img.onload = () => {
        const c = document.createElement('canvas');
        c.width = img.width; c.height = img.height;
        c.getContext('2d').drawImage(img, 0, 0);
        resolve(c.toDataURL('image/jpeg', 0.8));
      };
      img.src = ${JSON.stringify(pngDataUrl)};
    })
  `;
  return maskWindow.webContents.executeJavaScript(jsCode);
}

let debugDir = null;

function saveDebugData(frames, audioBase64) {
  const taskId = `${Date.now()}_${++taskCounter}`;
  debugDir = path.join(__dirname, 'tmp', taskId);
  fs.mkdirSync(debugDir, { recursive: true });
  for (let i = 0; i < frames.length; i++) {
    const base64 = frames[i].split(',')[1];
    fs.writeFileSync(path.join(debugDir, `frame_${String(i).padStart(3, '0')}.jpg`), Buffer.from(base64, 'base64'));
  }
  if (audioBase64) {
    fs.writeFileSync(path.join(debugDir, 'audio.webm'), Buffer.from(audioBase64, 'base64'));
  }
  console.log(`调试数据已保存: tmp/${taskId}/ (${frames.length} 帧)`);
}

function writeImageDataUrlToFile(dataUrl, filePath) {
  const normalized = normalizeImageDataUrl(dataUrl);
  const match = normalized.match(/^data:image\/[a-zA-Z0-9.+-]+;base64,(.+)$/);
  if (!match) {
    throw new Error(`Invalid image data URL for ${path.basename(filePath)}`);
  }
  fs.writeFileSync(filePath, Buffer.from(match[1], 'base64'));
}

function saveOmniParserDebugOutputs(tag, omniResult, summaryText = '') {
  if (!debugDir || !omniResult) return;

  try {
    fs.mkdirSync(debugDir, { recursive: true });

    const safeTag = String(tag || 'screen').replace(/[^a-zA-Z0-9_-]+/g, '_');
    const imagePath = path.join(debugDir, `${safeTag}_omniparser.png`);
    const jsonPath = path.join(debugDir, `${safeTag}_omniparser.json`);
    const textPath = path.join(debugDir, `${safeTag}_omniparser.txt`);

    if (omniResult.parsedImageDataUrl) {
      writeImageDataUrlToFile(omniResult.parsedImageDataUrl, imagePath);
    }

    const jsonPayload = {
      latency: omniResult.latency ?? null,
      parsed_content_list: Array.isArray(omniResult.parsedElements) ? omniResult.parsedElements : []
    };
    fs.writeFileSync(jsonPath, JSON.stringify(jsonPayload, null, 2), 'utf-8');

    const textPayload = [
      `latency=${omniResult.latency ?? ''}`,
      '',
      summaryText || summarizeOmniParserElements(omniResult.parsedElements, screenWidth, screenHeight)
    ].join('\n');
    fs.writeFileSync(textPath, textPayload, 'utf-8');
  } catch (error) {
    console.warn('保存 OmniParser 调试输出失败:', error.message);
  }
}

function saveToHistory(frames, cleanResponse) {
  const summary = `[User annotated screen with ${frames.length} screenshots]`;
  conversationHistory.push({ role: 'user', content: summary });
  conversationHistory.push({ role: 'assistant', content: cleanResponse });
  if (conversationHistory.length > MAX_HISTORY * 2) {
    conversationHistory = conversationHistory.slice(-MAX_HISTORY * 2);
  }
}

function setModelLoading(isLoading) {
  if (maskWindow && !maskWindow.isDestroyed()) {
    maskWindow.webContents.send('model-loading', isLoading);
  }
}

function clampNumber(value, fallback, min, max) {
  const num = Number(value);
  if (!Number.isFinite(num)) return fallback;
  return Math.min(Math.max(num, min), max);
}

function normalizeTooltipPayload(rawArgs) {
  const text = typeof rawArgs?.text === 'string' ? rawArgs.text.trim() : '';
  const resolvedElementIndex = resolveTooltipTargetIndex(text, rawArgs?.element_index);
  const highlight = resolveTooltipHighlight(resolvedElementIndex);
  const hasX = Number.isFinite(Number(rawArgs?.x));
  const hasY = Number.isFinite(Number(rawArgs?.y));
  const renderWidth = Math.max(1, screenWidth - overlayOffsetX);
  const renderHeight = Math.max(1, screenHeight - overlayOffsetY);
  const fallbackX = highlight ? highlight.centerX : Math.round(renderWidth / 2);
  const fallbackY = highlight ? highlight.centerY : Math.round(renderHeight / 2);
  const x = hasX
    ? Math.round(clampNumber(Number(rawArgs.x) - overlayOffsetX, fallbackX, 0, Math.max(renderWidth - 1, 0)))
    : fallbackX;
  const y = hasY
    ? Math.round(clampNumber(Number(rawArgs.y) - overlayOffsetY, fallbackY, 0, Math.max(renderHeight - 1, 0)))
    : fallbackY;
  return {
    text,
    x,
    y,
    highlight,
    elementIndex: highlight?.index ?? resolvedElementIndex ?? null,
    hasValidCoordinates: (hasX && hasY) || Boolean(highlight)
  };
}

function sendModelError(message) {
  if (maskWindow && !maskWindow.isDestroyed()) {
    maskWindow.webContents.send('model-error', message);
  }
}

function clampInt(value, fallback, min, max) {
  const num = Number.parseInt(value, 10);
  if (Number.isNaN(num)) return fallback;
  return Math.min(Math.max(num, min), max);
}

function cleanSearchSnippet(text, limit = 600) {
  if (!text || typeof text !== 'string') return '';
  const normalized = text.replace(/\s+/g, ' ').trim();
  return normalized.length > limit ? `${normalized.slice(0, limit)}...` : normalized;
}

async function tavilyWebSearch(rawArgs) {
  if (!TAVILY_API_KEY) {
    throw new Error('TAVILY_API_KEY is not set');
  }

  const query = typeof rawArgs?.query === 'string' ? rawArgs.query.trim() : '';
  if (!query) {
    throw new Error('Missing Tavily search query');
  }

  const allowedTopics = new Set(['general', 'news', 'finance']);
  const allowedDepths = new Set(['basic', 'advanced', 'fast', 'ultra-fast']);
  const allowedRanges = new Set(['day', 'week', 'month', 'year', 'd', 'w', 'm', 'y']);

  const payload = {
    query,
    topic: allowedTopics.has(rawArgs?.topic) ? rawArgs.topic : 'general',
    search_depth: allowedDepths.has(rawArgs?.search_depth) ? rawArgs.search_depth : 'advanced',
    max_results: clampInt(rawArgs?.max_results, 5, 1, 10),
    include_answer: true,
    include_raw_content: false
  };

  if (rawArgs?.time_range && allowedRanges.has(rawArgs.time_range)) {
    payload.time_range = rawArgs.time_range;
  }

  if (rawArgs?.days !== undefined) {
    payload.days = clampInt(rawArgs.days, 7, 1, 365);
  }

  const response = await fetch(TAVILY_BASE, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${TAVILY_API_KEY}`
    },
    body: JSON.stringify(payload)
  });

  if (!response.ok) {
    const errText = await response.text();
    throw new Error(`Tavily HTTP ${response.status}: ${errText.slice(0, 500)}`);
  }

  const data = await response.json();
  const results = Array.isArray(data.results) ? data.results : [];

  return {
    query: data.query || query,
    answer: cleanSearchSnippet(data.answer || '', 1200),
    results: results.slice(0, payload.max_results).map(result => ({
      title: result?.title || '',
      url: result?.url || '',
      content: cleanSearchSnippet(result?.content || ''),
      score: typeof result?.score === 'number' ? result.score : null
    })),
    response_time: data.response_time || null
  };
}

async function callLLM(messages, hooks = {}) {
  const { onContentChunk, onToolCall } = hooks;
  const response = await fetch(`${LLM_BASE}/chat/completions`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${LLM_API_KEY}`
    },
    body: JSON.stringify({
      model: LLM_MODEL,
      messages,
      tools: TOOLS,
      stream: true
    })
  });

  if (!response.ok) {
    const errText = await response.text();
    throw new Error(`HTTP ${response.status}: ${errText.slice(0, 500)}`);
  }

  let textContent = '';
  let inThink = false;
  let toolCalls = [];
  let currentToolCall = null;
  const reader = response.body;
  let buffer = '';
  const decoder = new TextDecoder();

  for await (const chunk of reader) {
    buffer += decoder.decode(chunk, { stream: true });
    const lines = buffer.split('\n');
    buffer = lines.pop();

    for (const line of lines) {
      const trimmed = line.trim();
      if (!trimmed || !trimmed.startsWith('data: ')) continue;
      const data = trimmed.slice(6);
      if (data === '[DONE]') continue;

      let parsed;
      try { parsed = JSON.parse(data); } catch { continue; }

      const delta = parsed.choices?.[0]?.delta;
      if (!delta) continue;

      if (delta.content) {
        let chunk = delta.content;
        // Filter out <think>...</think> blocks from thinking models
        if (chunk.includes('<think>')) inThink = true;
        if (inThink) {
          if (chunk.includes('</think>')) {
            chunk = chunk.split('</think>').pop();
            inThink = false;
          } else {
            chunk = '';
          }
        }
        textContent += chunk;
        if (chunk) {
          maskWindow.webContents.send('model-stream', chunk);
          if (typeof onContentChunk === 'function') {
            try { onContentChunk(chunk); } catch (error) { console.warn('流式 TTS content hook 失败:', error.message); }
          }
        }
      }

      if (delta.tool_calls) {
        if (typeof onToolCall === 'function') {
          try { onToolCall(); } catch (error) { console.warn('流式 TTS tool hook 失败:', error.message); }
        }
        for (const tc of delta.tool_calls) {
          if (tc.index !== undefined) {
            while (toolCalls.length <= tc.index) {
              toolCalls.push({ id: '', type: 'function', function: { name: '', arguments: '' } });
            }
            currentToolCall = toolCalls[tc.index];
          }
          if (tc.id) currentToolCall.id = tc.id;
          if (tc.function?.name) currentToolCall.function.name += tc.function.name;
          if (tc.function?.arguments) currentToolCall.function.arguments += tc.function.arguments;
        }
      }
    }
  }

  return { textContent, toolCalls };
}

function parseToolArgs(raw) {
  // 1. Try direct parse
  try { return JSON.parse(raw); } catch {}

  // 2. Strip <think>...</think> tags (thinking models)
  let cleaned = raw.replace(/<think>[\s\S]*?<\/think>/g, '').trim();
  try { return JSON.parse(cleaned); } catch {}

  // 3. Extract the first JSON object
  const match = cleaned.match(/\{[\s\S]*\}/);
  if (match) {
    try { return JSON.parse(match[0]); } catch {}
  }

  // 4. Regex fallback: extract text
  const textMatch = raw.match(/"text"\s*:\s*"((?:[^"\\]|\\.)*)"/);
  if (textMatch) {
    return { text: textMatch[1] };
  }

  return null;
}

async function executeToolCalls(textContent, toolCalls) {
  console.log(`模型回复: text=${textContent.length}chars, toolCalls=${toolCalls.length}`);

  if (!toolCalls.length) {
    return { hasTooltip: false, needsAnotherRound: false };
  }

  stepMessages.push({
    role: 'assistant',
    content: textContent || null,
    tool_calls: toolCalls
  });

  let hasTooltip = false;
  let needsAnotherRound = false;

  for (const tc of toolCalls) {
    const toolName = tc.function?.name || '';
    const rawArgs = tc.function?.arguments || '{}';
    const args = parseToolArgs(rawArgs);
    let result;

    if (toolName !== 'show_tooltip') {
      needsAnotherRound = true;
    }

    if (!args) {
      console.error('Tool args 解析失败, 原始内容:', rawArgs);
      result = JSON.stringify({ error: 'Invalid tool arguments', raw: rawArgs });
      stepMessages.push({ role: 'tool', tool_call_id: tc.id, content: result });
      continue;
    }

    if (toolName === 'show_tooltip') {
      const tooltip = normalizeTooltipPayload(args);
      if (!tooltip.text) {
        result = JSON.stringify({ error: 'Missing tooltip text' });
        needsAnotherRound = true;
      } else if (!tooltip.hasValidCoordinates) {
        result = JSON.stringify({ error: 'Missing tooltip coordinates x/y. show_tooltip requires absolute screen pixel coordinates, or a valid OmniParser element_index that can be resolved to a target box.' });
        needsAnotherRound = true;
      } else {
        stepNumber++;
        maskWindow.webContents.send('show-tooltip', { ...tooltip, step: stepNumber });
        speakTtsText(tooltip.text, 'tooltip');
        console.log(`显示 tooltip: step ${stepNumber}, "${tooltip.text}"`);
        result = tooltip.highlight
          ? `Tooltip displayed for step ${stepNumber} with highlighted OmniParser element ${tooltip.highlight.index}.`
          : `Tooltip displayed for step ${stepNumber}.`;
        hasTooltip = true;
      }
    } else if (toolName === 'tavily_web_search') {
      try {
        console.log(`Tavily 搜索: "${args.query}"`);
        result = JSON.stringify(await tavilyWebSearch(args));
      } catch (error) {
        console.error('Tavily 搜索失败:', error.message);
        result = JSON.stringify({ error: error.message });
      }
    } else {
      result = JSON.stringify({ error: `Unsupported tool: ${toolName}` });
    }

    stepMessages.push({ role: 'tool', tool_call_id: tc.id, content: result });
  }

  return { hasTooltip, needsAnotherRound };
}

async function runAgentTurn() {
  let sawTooltip = false;

  for (let round = 0; round < MAX_TOOL_ROUNDS; round++) {
    const responseTtsState = createStreamingResponseTtsState();
    const { textContent, toolCalls } = await callLLM(stepMessages, {
      onContentChunk: chunk => responseTtsState.pushText(chunk),
      onToolCall: () => responseTtsState.onToolCall()
    });
    if (toolCalls.length) {
      responseTtsState.onToolCall();
    } else {
      responseTtsState.finish();
    }
    const { hasTooltip, needsAnotherRound } = await executeToolCalls(textContent, toolCalls);

    sawTooltip = sawTooltip || hasTooltip;

    if (!needsAnotherRound) {
      const cleanText = textContent.trim();
      if (cleanText) {
        maskWindow.webContents.send('model-done', cleanText);
      }

      if (!sawTooltip) {
        stepMessages = [];
        stepNumber = 0;
      }

      return { textContent: cleanText, hasTooltip: sawTooltip };
    }
  }

  throw new Error(`Tool loop exceeded ${MAX_TOOL_ROUNDS} rounds`);
}

async function sendToModel(frames, audioBase64, userText) {
  const systemText = `You are a desktop software assistant that helps users navigate and operate software. The user drew RED marks on their screen to highlight areas of interest.

Screen resolution: ${screenWidth}x${screenHeight} pixels.

Your job:
1. **Software guidance**: Identify the software/interface in the screenshot and provide step-by-step instructions for what the user is asking. Use show_tooltip to display the instruction for the FIRST step. Give only ONE step at a time — the user will click "Next" and you will receive an updated screenshot to guide the next step.
2. **Information capture**: If the user highlights text, data, or content, help organize, summarize, or record it. Respond with plain text only (no tools needed).
3. **Live web lookup**: If the user needs current or external information that is not visible on screen, use tavily_web_search before answering.

When giving step-by-step guidance, call show_tooltip for the current step instruction. show_tooltip should include x and y in absolute screen pixels near the center of the relevant target area. If the OmniParser summary already identifies the target element, also pass element_index with the same summary index so the UI can draw a rectangle around it. If you mention a visible label such as "Model Summary", prefer the exact OmniParser item whose content exactly matches that label. Do not choose nearby paragraphs or longer descriptive text when an exact label match exists. Never use null for x or y. Do NOT list all steps upfront — just describe the current step.
Use tavily_web_search sparingly for up-to-date facts, web pages, or recent information. After getting tool results, cite the relevant URLs in your answer when helpful.
For normal text replies, write like a spoken assistant talking to the user in real time. Prefer short, natural conversational sentences and 1 short paragraph by default.
Do not use markdown headings, bullet lists, numbered lists, tables, blockquotes, or code fences unless the user explicitly asks for them.
Avoid document-style wording. Do not structure the reply like an article, report, or notes page.

When all steps are complete, respond with a summary in plain text (no tool calls) to signal you are done.

Always respond in the same language as the user's note. If no note is provided, use Chinese.${userText ? '\n\nUser note: ' + userText : ''}`;

  stepMessages = [];
  stepNumber = 0;

  try {
    const imagesToSend = SEND_ALL_FRAMES ? frames : frames.slice(-1);
    const content = [{ type: 'text', text: systemText }];
    const primaryFrame = imagesToSend[imagesToSend.length - 1] ? normalizeImageDataUrl(imagesToSend[imagesToSend.length - 1]) : '';
    let omniParserSummary = '';
    let omniParserImageDataUrl = '';

    if (primaryFrame) {
      try {
        const omniResult = await preprocessScreenshotWithOmniParser(primaryFrame);
        latestOmniParserElements = Array.isArray(omniResult.parsedElements) ? omniResult.parsedElements : [];
        if (omniResult.parsedImageDataUrl) {
          omniParserImageDataUrl = omniResult.parsedImageDataUrl;
        }
        const parsedSummary = summarizeOmniParserElements(omniResult.parsedElements, screenWidth, screenHeight);
        if (parsedSummary) {
          omniParserSummary = `OmniParser detected UI elements on the current screen. Use this as extra grounding context, but verify against the screenshot.\n${parsedSummary}`;
        }
        saveOmniParserDebugOutputs('initial', omniResult, omniParserSummary);
      } catch (error) {
        latestOmniParserElements = [];
        console.warn('OmniParser 预处理失败，回退到原始截图:', error.message);
      }
    }

    if (omniParserSummary) {
      content.push({ type: 'text', text: omniParserSummary });
    }

    for (const frame of imagesToSend) {
      const url = normalizeImageDataUrl(frame);
      content.push({ type: 'image_url', image_url: { url } });
    }
    if (omniParserImageDataUrl) {
      content.push({
        type: 'text',
        text: 'The next image is the same screen preprocessed by OmniParser, with numbered boxes drawn around detected UI elements.'
      });
      content.push({ type: 'image_url', image_url: { url: omniParserImageDataUrl } });
    }

    stepMessages = [
      ...conversationHistory,
      { role: 'user', content }
    ];

    const { textContent, hasTooltip } = await runAgentTurn();

    if (!hasTooltip) {
      saveToHistory(frames, textContent);
    }
  } catch (error) {
    console.error('API错误:', error.message);
    if (error.cause) console.error('原因:', error.cause);
    console.error('完整错误:', error);
    sendModelError(`请求失败，请重试。\n\n\`${error.message}\``);
  } finally {
    setModelLoading(false);
  }
}

async function continueStep(screenshotDataUrl) {
  try {
    const url = normalizeImageDataUrl(screenshotDataUrl);
    let omniParserSummary = '';
    let omniParserImageDataUrl = '';

    try {
      const omniResult = await preprocessScreenshotWithOmniParser(url);
      latestOmniParserElements = Array.isArray(omniResult.parsedElements) ? omniResult.parsedElements : [];
      if (omniResult.parsedImageDataUrl) {
        omniParserImageDataUrl = omniResult.parsedImageDataUrl;
      }
      const parsedSummary = summarizeOmniParserElements(omniResult.parsedElements, screenWidth, screenHeight);
      if (parsedSummary) {
        omniParserSummary = `OmniParser detected UI elements on the current screen. Use this as extra grounding context, but verify against the screenshot.\n${parsedSummary}`;
      }
      const continueTag = `continue_step_${String(stepNumber + 1).padStart(2, '0')}`;
      saveOmniParserDebugOutputs(continueTag, omniResult, omniParserSummary);
    } catch (error) {
      latestOmniParserElements = [];
      console.warn('OmniParser 续步预处理失败，回退到原始截图:', error.message);
    }

    stepMessages.push({
      role: 'user',
      content: [
        { type: 'text', text: `Here is the current screen after the user clicked "Next". Analyze the screenshot to determine what the user has actually done — do NOT assume the previous step was completed. Based on the current screen state, decide what the user should do next and use show_tooltip. show_tooltip should include x and y in absolute screen pixels near the center of the relevant target area, and x/y must never be null. If the target matches an OmniParser summary item, also include that element_index so the UI can highlight the element with a rectangle. If you refer to a visible label like "Model Summary", prefer the exact OmniParser item whose content exactly matches that label, not a nearby paragraph. If all steps are already done, respond with a summary (no tool call).` },
        ...(omniParserSummary ? [{ type: 'text', text: omniParserSummary }] : []),
        { type: 'image_url', image_url: { url } },
        ...(omniParserImageDataUrl ? [
          { type: 'text', text: 'The next image is the same screen preprocessed by OmniParser, with numbered boxes drawn around detected UI elements.' },
          { type: 'image_url', image_url: { url: omniParserImageDataUrl } }
        ] : [])
      ]
    });

    const { textContent, hasTooltip } = await runAgentTurn();

    if (!hasTooltip) {
      saveToHistory([], textContent);
    }
  } catch (error) {
    console.error('下一步API错误:', error.message);
    console.error('完整错误:', error);
    sendModelError(`请求失败，请重试。\n\n\`${error.message}\``);
  } finally {
    setModelLoading(false);
  }
}

app.whenReady().then(async () => {
  await createMaskWindow();
  ensureTtsServiceReady().catch(error => {
    console.warn('TTS 服务预热失败:', error.message);
  });
  if (isOmniParserConfigured()) {
    ensureOmniParserReady().catch(error => {
      console.warn('OmniParser 服务预热失败:', error.message);
    });
  }
});
app.on('window-all-closed', () => { if (process.platform !== 'darwin') app.quit(); });
app.on('will-quit', () => {
  globalShortcut.unregisterAll();
  stopTtsPlayback();
  if (ttsProcess && !ttsProcess.killed) {
    ttsProcess.kill('SIGTERM');
  }
  if (omniParserProcess && !omniParserProcess.killed) {
    omniParserProcess.kill('SIGTERM');
  }
});
