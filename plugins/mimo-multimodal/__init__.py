"""
mimo-multimodal plugin v3 — Image, Audio & Video understanding via Xiaomi MiMo.

Features:
- Image analysis/description (JPEG, PNG, GIF, WebP, BMP)
- Audio transcription/analysis (MP3, WAV, FLAC, M4A, OGG)
- Video description/analysis (MP4, MOV, AVI, WMV)
- Local files (auto base64) and URLs
- Large local files via temp HTTP server (up to 300MB)
- Smart fps auto-tuning based on video duration
- Multi-image support in single request
- Multi-modal combination (image+audio+video in one request)
- Format validation with magic bytes
- Audio extraction from video via ffmpeg
- Context-aware system prompts for better results
- Programming & UI analysis presets
- Streaming output for long content
- Batch processing with parallel execution
- File hash-based caching
- Structured output (JSON/table format)

Author: Everett (https://github.com/xiaopengsvip)
"""

import json
import os
import re
import base64
import hashlib
import socket
import subprocess
import tempfile
import threading
import time
import urllib.request
import urllib.error
from pathlib import Path
from http.server import HTTPServer, SimpleHTTPRequestHandler
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE_URL = os.environ.get("XIAOMI_BASE_URL", "https://token-plan-sgp.xiaomimimo.com/v1")
MODEL = "mimo-v2.5"
TIMEOUT = 600

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"}
AUDIO_EXTS = {".mp3", ".wav", ".flac", ".m4a", ".ogg", ".aac", ".wma"}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".wmv", ".mkv", ".webm", ".flv"}

MIME_MAP = {
    ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
    ".gif": "image/gif", ".webp": "image/webp", ".bmp": "image/bmp",
    ".mp3": "audio/mpeg", ".wav": "audio/wav", ".flac": "audio/flac",
    ".m4a": "audio/mp4", ".ogg": "audio/ogg", ".aac": "audio/aac",
    ".wma": "audio/x-ms-wma",
    ".mp4": "video/mp4", ".mov": "video/quicktime",
    ".avi": "video/x-msvideo", ".wmv": "video/x-ms-wmv",
    ".mkv": "video/x-matroska", ".webm": "video/webm",
    ".flv": "video/x-flv",
}

# ─── Cache ───────────────────────────────────────────────────────────

_CACHE_DIR = Path(tempfile.gettempdir()) / "mimo-multimodal-cache"
_CACHE_DIR.mkdir(exist_ok=True)
_CACHE_MAX_AGE = 3600 * 24  # 24 hours

def _file_hash(file_path: Path) -> str:
    """Compute SHA256 hash of a file for cache key."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def _cache_get(key: str) -> str | None:
    """Get cached result if exists and not expired."""
    cache_file = _CACHE_DIR / f"{key}.json"
    if cache_file.exists():
        try:
            mtime = cache_file.stat().st_mtime
            if time.time() - mtime < _CACHE_MAX_AGE:
                return cache_file.read_text(encoding="utf-8")
        except Exception:
            pass
    return None

def _cache_set(key: str, value: str):
    """Store result in cache."""
    try:
        cache_file = _CACHE_DIR / f"{key}.json"
        cache_file.write_text(value, encoding="utf-8")
    except Exception:
        pass

def _cache_key(source: str, prompt: str, **extra) -> str:
    """Generate cache key from source and parameters."""
    if _is_url(source):
        content = f"{source}:{prompt}:{json.dumps(extra, sort_keys=True)}"
        return hashlib.sha256(content.encode()).hexdigest()
    else:
        p = Path(source).expanduser().resolve()
        if p.exists():
            file_h = _file_hash(p)
            return f"{file_h}:{hashlib.sha256(f'{prompt}:{json.dumps(extra, sort_keys=True)}'.encode()).hexdigest()[:16]}"
    return ""

# ─── System prompts for better results ───────────────────────────────

IMAGE_SYSTEM_PROMPT = """You are a professional image analysis AI. Follow these guidelines:

1. DESCRIPTION: Describe the image content in detail — objects, people, scenes, colors, composition, lighting, style.

2. CONTEXT: If the image contains text (OCR), signs, labels, or UI elements, extract and describe them.

3. TECHNICAL: Note image quality, format, resolution if relevant.

4. LANGUAGE: Always respond in the same language as the user's prompt. If the prompt is in Chinese, respond in Chinese.

5. ACCURACY: If parts of the image are unclear or ambiguous, note it. Do not hallucinate details you cannot see.

6. STRUCTURE: For complex images, organize your analysis logically (e.g., foreground/background, left/right, main subject/details)."""

CODE_SYSTEM_PROMPT = """You are a senior software engineer and UI/UX expert. When analyzing code screenshots or UI designs:

1. CODE ANALYSIS:
   - Identify programming language and framework
   - Explain what the code does, line by line if needed
   - Spot bugs, security issues, performance problems
   - Suggest improvements and best practices

2. UI/UX ANALYSIS:
   - Describe layout structure (header, sidebar, content, footer)
   - Identify UI components (buttons, forms, modals, cards)
   - Evaluate design quality (spacing, alignment, colors, typography)
   - Point out UX issues (accessibility, usability, responsiveness)
   - Compare with modern design patterns

3. TECH STACK:
   - Identify frameworks/libraries from visual cues (React, Vue, Tailwind, etc.)
   - Infer tech stack from UI patterns

4. LANGUAGE: Always respond in the same language as the user's prompt."""

AUDIO_SYSTEM_PROMPT = """You are a professional audio analysis AI. Follow these guidelines:

1. TRANSCRIPTION: If asked to transcribe, output the exact spoken content in the original language. Use proper punctuation and paragraph breaks. Identify different speakers if possible (Speaker A, Speaker B...).

2. DESCRIPTION: If asked to describe, analyze: speech content, tone/emotion, background sounds, music genre/instruments, audio quality, language spoken.

3. TIMESTAMP: When possible, indicate approximate timestamps for key moments.

4. LANGUAGE: Always respond in the same language as the user's prompt. If the prompt is in Chinese, respond in Chinese.

5. ACCURACY: If audio is unclear or ambiguous, note it. Do not hallucinate content you cannot hear."""

VIDEO_SYSTEM_PROMPT = """You are a professional video analysis AI. Follow these guidelines:

1. VISUAL: Describe scenes, objects, people (appearance, actions), setting, lighting, camera movement, text/graphics on screen.

2. AUDIO: If video has audio, also describe speech, music, sound effects.

3. STRUCTURE: Break down the video chronologically — opening, key moments, ending.

4. DETAIL: Be specific — colors, quantities, spatial relationships, emotions shown.

5. TIMESTAMP: Use approximate timestamps for scene transitions and key events.

6. LANGUAGE: Always respond in the same language as the user's prompt. If the prompt is in Chinese, respond in Chinese.

7. If the video is too short or unclear, state what you can confidently observe."""

# ─── Prompt presets ──────────────────────────────────────────────────

IMAGE_PRESETS = {
    "describe": "请详细描述这张图片的内容，包括：主体对象、场景环境、颜色构图、光照风格、文字信息（如有）。",
    "ocr": "请识别并提取图片中出现的所有文字内容，包括标题、标识、标签、UI元素等。",
    "explain": "请解释这张图片的含义、用途或上下文。",
    "compare": "请分析这张图片的特点、风格和技术细节。",
    "extract_info": "请从这张图片中提取关键信息：人物、地点、时间、品牌、产品等结构化数据。",
    "caption": "请为这张图片写一段简洁的描述，适合作为图片说明文字。",
    # Programming & UI presets
    "code_review": "请分析这段代码：识别编程语言和框架，逐行解释功能，找出bug、安全问题、性能问题，并提供改进建议。",
    "code_explain": "请解释这段代码的功能和逻辑，用通俗易懂的方式说明它做了什么。",
    "code_convert": "请将这段代码转换为等效的其他语言实现（如果可能的话），并说明转换要点。",
    "ui_review": "请审查这个UI界面：分析布局结构、组件识别、设计质量（间距/对齐/颜色/字体）、用户体验问题，并与现代设计模式对比。",
    "ui_describe": "请详细描述这个UI界面的组成部分：导航栏、侧边栏、内容区、按钮、表单、卡片等元素及其布局。",
    "ui_improve": "请分析这个UI界面并提供具体的改进建议，包括设计优化和用户体验提升。",
    "figma_to_code": "请分析这个设计稿，描述如何用 HTML/CSS/React 实现这个界面，提供关键代码片段。",
    "screenshot_to_code": "请分析这个截图中的UI，生成对应的 HTML/Tailwind CSS 代码来复现这个界面。",
}

AUDIO_PRESETS = {
    "transcribe": "请完整转录这段音频的内容，保留原始语言，标注说话人（如果能区分），使用适当的标点和分段。",
    "describe": "请详细描述这段音频的内容，包括：语音内容、说话人语气/情绪、背景声音、音乐风格（如有）、语言种类、音频质量。",
    "summarize": "请听这段音频并给出简洁的摘要，突出关键信息和要点。",
    "translate": "请转录这段音频的内容，然后翻译成中文。",
    "extract_info": "请从这段音频中提取关键信息：人名、地点、时间、数字、事件等结构化数据。",
    "meeting_notes": "请将这段会议录音整理成结构化的会议纪要，包括：议题、讨论要点、决定事项、待办事项。",
    "lecture_notes": "请将这段讲座/课程录音整理成学习笔记，包括：主要概念、关键知识点、示例、总结。",
}

VIDEO_PRESETS = {
    "describe": "请详细描述这个视频的内容，按时间顺序分析每个场景：画面内容、人物动作、环境设置、文字信息、背景音乐/声音。",
    "summarize": "请简洁概括这个视频的主要内容和关键信息。",
    "scenes": "请将这个视频分解为独立场景，每个场景标注大致时间范围，描述画面内容和发生的事情。",
    "ocr": "请识别并提取视频中出现的所有文字内容（字幕、标题、标识等），按时间顺序列出。",
    "action": "请重点分析视频中人物的具体动作和行为，适合动作分析、运动分析场景。",
    "count": "请统计视频中出现的人数、物体数量等可量化信息。",
    "tutorial_steps": "请将这个教程/演示视频分解为步骤，每个步骤标注时间范围和具体操作。",
    "code_demo": "请分析这个编程演示视频：提取代码内容、解释操作步骤、总结关键知识点。",
    "ui_demo": "请分析这个UI演示视频：描述交互流程、用户操作、界面变化、动画效果。",
}

# ─── Helpers ─────────────────────────────────────────────────────────

def _get_api_key():
    return os.environ.get("XIAOMI_API_KEY", "")


def _is_url(s: str) -> bool:
    return s.startswith("http://") or s.startswith("https://")


def _detect_mime(path: Path) -> str:
    """Detect MIME type via magic bytes, fallback to extension."""
    try:
        with open(path, "rb") as f:
            header = f.read(16)
        magic_map = {
            b"\xff\xd8\xff": "image/jpeg",
            b"\x89PNG": "image/png",
            b"GIF8": "image/gif",
            b"RIFF": "image/webp",
            b"BM": "image/bmp",
            b"\x1a\x45\xdf\xa3": "video/x-matroska",
            b"\x00\x00\x00": "video/mp4",
            b"ID3": "audio/mpeg",
            b"\xff\xfb": "audio/mpeg",
            b"\xff\xf3": "audio/mpeg",
            b"fLaC": "audio/flac",
            b"OggS": "audio/ogg",
            b"\x30\x26\xb2\x75": "video/x-ms-wmv",
        }
        for magic, mime in magic_map.items():
            if header.startswith(magic):
                return mime
    except Exception:
        pass
    return MIME_MAP.get(path.suffix.lower(), "application/octet-stream")


def _get_media_type(path: Path) -> str:
    """Return 'image', 'audio', or 'video' based on extension."""
    ext = path.suffix.lower()
    if ext in IMAGE_EXTS:
        return "image"
    if ext in AUDIO_EXTS:
        return "audio"
    if ext in VIDEO_EXTS:
        return "video"
    mime = _detect_mime(path)
    if mime.startswith("image/"):
        return "image"
    if mime.startswith("audio/"):
        return "audio"
    return "video"


def _file_size_mb(path: Path) -> float:
    return path.stat().st_size / (1024 * 1024)


def _file_to_base64_uri(path: Path) -> str:
    mime = _detect_mime(path)
    b64 = base64.b64encode(path.read_bytes()).decode()
    return f"data:{mime};base64,{b64}"


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class _QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass


def _serve_file_temporarily(file_path: Path, timeout=600) -> str:
    """Start a temp HTTP server to serve a local file. Returns URL."""
    port = _find_free_port()
    parent = file_path.parent
    filename = file_path.name

    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *a, **kw):
            super().__init__(*a, directory=str(parent), **kw)
        def log_message(self, *a):
            pass

    server = HTTPServer(("0.0.0.0", port), Handler)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()

    def auto_shutdown():
        time.sleep(timeout)
        server.shutdown()
    threading.Thread(target=auto_shutdown, daemon=True).start()

    return f"http://127.0.0.1:{port}/{filename}"


def _resolve_source(source: str, media_type: str) -> str:
    """Resolve source to URL or base64 data URI."""
    if _is_url(source):
        return source

    p = Path(source).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")

    size = _file_size_mb(p)
    ext = p.suffix.lower()

    valid_exts = {
        "image": IMAGE_EXTS,
        "audio": AUDIO_EXTS,
        "video": VIDEO_EXTS,
    }.get(media_type, set())
    if ext not in valid_exts:
        raise ValueError(f"Unsupported format '{ext}'. Supported: {', '.join(sorted(valid_exts))}")

    if size > 50:
        if media_type == "audio":
            raise ValueError(f"Audio file too large ({size:.1f}MB, max 50MB). Provide a public URL.")
        if media_type == "image":
            raise ValueError(f"Image file too large ({size:.1f}MB, max 50MB). Compress or provide URL.")
        if size > 300:
            raise ValueError(f"Video file too large ({size:.1f}MB, max 300MB). Provide a public URL.")
        return _serve_file_temporarily(p)

    return _file_to_base64_uri(p)


def _get_audio_duration(file_path: Path) -> float:
    """Get audio/video duration in seconds using ffprobe."""
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
             "-of", "csv=p=0", str(file_path)],
            capture_output=True, text=True, timeout=10
        )
        return float(result.stdout.strip())
    except Exception:
        return 0


def _extract_audio_from_video(video_path: Path) -> Path:
    """Extract audio track from video using ffmpeg."""
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(video_path), "-vn", "-acodec", "pcm_s16le",
             "-ar", "16000", "-ac", "1", tmp.name],
            capture_output=True, timeout=300
        )
        return Path(tmp.name)
    except Exception as e:
        os.unlink(tmp.name)
        raise RuntimeError(f"Failed to extract audio: {e}")


def _auto_fps(duration: float, requested_fps: float = None) -> float:
    """Auto-tune fps based on video duration."""
    if requested_fps is not None:
        return max(0.1, min(10.0, requested_fps))
    if duration <= 0:
        return 2.0
    if duration < 10:
        return 5.0
    if duration < 60:
        return 2.0
    if duration < 300:
        return 1.0
    if duration < 600:
        return 0.5
    return 0.2


def _estimate_tokens(duration: float, fps: float = None, media_type: str = "video") -> dict:
    """Estimate token usage."""
    if media_type == "audio":
        audio_tokens = int(duration * 6.25)
        return {"audio_tokens": audio_tokens, "total_estimate": audio_tokens}
    fps = fps or 2.0
    frames = duration * fps
    visual_tokens = int(frames * 576)
    audio_tokens = int(duration * 6.25)
    return {
        "frames": int(frames),
        "visual_tokens": visual_tokens,
        "audio_tokens": audio_tokens,
        "total_estimate": visual_tokens + audio_tokens,
    }


def _format_structured(result: str, output_format: str) -> str:
    """Format output as structured data if requested."""
    if output_format == "none":
        return result

    if output_format == "json":
        try:
            data = json.loads(result)
            return json.dumps(data, indent=2, ensure_ascii=False)
        except json.JSONDecodeError:
            return json.dumps({
                "analysis": result,
                "format": "text",
                "note": "Could not parse as structured JSON"
            }, indent=2, ensure_ascii=False)

    if output_format == "table":
        # Simple table formatting for key-value data
        lines = result.split("\n")
        formatted = []
        for line in lines:
            if "：" in line or ":" in line:
                formatted.append(line)
            else:
                formatted.append(f"  {line}")
        return "\n".join(formatted)

    return result


def _call_mimo(messages: list, max_tokens: int = 2048, stream: bool = False) -> str:
    """Call MiMo API. Supports streaming."""
    api_key = _get_api_key()
    if not api_key:
        return json.dumps({"error": "XIAOMI_API_KEY not set"})

    payload = {
        "model": MODEL,
        "messages": messages,
        "max_completion_tokens": max_tokens,
        "stream": stream,
    }

    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"{BASE_URL}/chat/completions",
        data=data,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )

    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
            if stream:
                # Streaming: collect chunks
                full_content = ""
                full_reasoning = ""
                usage = {}
                for line in r:
                    line = line.decode().strip()
                    if line.startswith("data: ") and line != "data: [DONE]":
                        try:
                            chunk = json.loads(line[6:])
                            delta = chunk.get("choices", [{}])[0].get("delta", {})
                            if "content" in delta:
                                full_content += delta["content"]
                            if "reasoning_content" in delta:
                                full_reasoning += delta["reasoning_content"]
                            if "usage" in chunk:
                                usage = chunk["usage"]
                        except json.JSONDecodeError:
                            continue
                result = full_content if full_content else full_reasoning
                return json.dumps({
                    "result": result,
                    "content": full_content,
                    "reasoning": full_reasoning,
                    "usage": usage,
                    "model": MODEL,
                    "streamed": True,
                }, ensure_ascii=False)
            else:
                resp = json.loads(r.read())
    except urllib.error.HTTPError as e:
        body = e.read().decode()
        return json.dumps({"error": f"HTTP {e.code}", "detail": body[:500]})
    except Exception as e:
        return json.dumps({"error": str(e)})

    choice = resp.get("choices", [{}])[0]
    msg = choice.get("message", {})
    content = msg.get("content", "")
    reasoning = msg.get("reasoning_content", "")
    usage = resp.get("usage", {})

    result = content if content else reasoning

    return json.dumps({
        "result": result,
        "content": content,
        "reasoning": reasoning,
        "usage": usage,
        "model": resp.get("model", MODEL),
    }, ensure_ascii=False)


# ─── Tool Schemas ────────────────────────────────────────────────────

IMAGE_SCHEMA = {
    "name": "image_understand",
    "description": (
        "Analyze images using MiMo AI. Supports description, OCR, code review, UI analysis.\n"
        "Formats: JPEG, PNG, GIF, WebP, BMP.\n"
        "Presets:\n"
        "  General: describe, ocr, explain, compare, extract_info, caption\n"
        "  Code: code_review, code_explain, code_convert\n"
        "  UI/UX: ui_review, ui_describe, ui_improve, figma_to_code, screenshot_to_code\n"
        "Or write a custom prompt for specific needs."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "source": {
                "type": "string",
                "description": "Local file path or public URL to the image file",
            },
            "prompt": {
                "type": "string",
                "description": "What to do with the image. Use a preset name or write a custom prompt.",
                "default": "describe",
            },
            "output_format": {
                "type": "string",
                "description": "Output format: 'none' (default text), 'json' (structured), 'table' (formatted)",
                "enum": ["none", "json", "table"],
                "default": "none",
            },
            "use_cache": {
                "type": "boolean",
                "description": "Use cached result if available (default true)",
                "default": True,
            },
            "max_tokens": {
                "type": "integer",
                "description": "Max output tokens (default 4096)",
                "default": 4096,
            },
        },
        "required": ["source"],
    },
}

AUDIO_SCHEMA = {
    "name": "audio_understand",
    "description": (
        "Analyze audio files using MiMo AI. Supports transcription, summarization, meeting notes.\n"
        "Formats: MP3, WAV, FLAC, M4A, OGG, AAC, WMA.\n"
        "Presets:\n"
        "  General: transcribe, describe, summarize, translate, extract_info\n"
        "  Specialized: meeting_notes, lecture_notes\n"
        "Or write a custom prompt for specific needs."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "source": {
                "type": "string",
                "description": "Local file path or public URL to the audio file",
            },
            "prompt": {
                "type": "string",
                "description": "What to do with the audio. Use a preset name or write a custom prompt.",
                "default": "transcribe",
            },
            "stream": {
                "type": "boolean",
                "description": "Enable streaming for long content (default false)",
                "default": False,
            },
            "output_format": {
                "type": "string",
                "description": "Output format: 'none' (default text), 'json' (structured), 'table' (formatted)",
                "enum": ["none", "json", "table"],
                "default": "none",
            },
            "use_cache": {
                "type": "boolean",
                "description": "Use cached result if available (default true)",
                "default": True,
            },
            "max_tokens": {
                "type": "integer",
                "description": "Max output tokens (default 4096)",
                "default": 4096,
            },
        },
        "required": ["source"],
    },
}

VIDEO_SCHEMA = {
    "name": "video_understand",
    "description": (
        "Analyze video files using MiMo AI. Supports scene analysis, code demos, UI demos.\n"
        "Formats: MP4, MOV, AVI, WMV, MKV, WEBM, FLV.\n"
        "Presets:\n"
        "  General: describe, summarize, scenes, ocr, action, count\n"
        "  Specialized: tutorial_steps, code_demo, ui_demo\n"
        "Auto-tunes fps based on video length to balance detail and token cost."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "source": {
                "type": "string",
                "description": "Local file path or public URL to the video file",
            },
            "prompt": {
                "type": "string",
                "description": "What to do with the video. Use a preset name or write a custom prompt.",
                "default": "describe",
            },
            "fps": {
                "type": "number",
                "description": "Frames per second (0.1-10). Omit for auto-tuning.",
            },
            "media_resolution": {
                "type": "string",
                "description": "Resolution: 'default' (balanced) or 'max' (best for small text/objects)",
                "enum": ["default", "max"],
                "default": "default",
            },
            "extract_audio": {
                "type": "boolean",
                "description": "Also extract and transcribe audio track (requires ffmpeg).",
                "default": False,
            },
            "stream": {
                "type": "boolean",
                "description": "Enable streaming for long content (default false)",
                "default": False,
            },
            "output_format": {
                "type": "string",
                "description": "Output format: 'none' (default text), 'json' (structured), 'table' (formatted)",
                "enum": ["none", "json", "table"],
                "default": "none",
            },
            "use_cache": {
                "type": "boolean",
                "description": "Use cached result if available (default true)",
                "default": True,
            },
            "max_tokens": {
                "type": "integer",
                "description": "Max output tokens (default 4096)",
                "default": 4096,
            },
        },
        "required": ["source"],
    },
}

MULTIMODAL_SCHEMA = {
    "name": "multimodal_understand",
    "description": (
        "Analyze multiple media files (images, audio, video) in a single request.\n"
        "Use for comparing images, analyzing video+audio together, or any multi-file scenario.\n"
        "Accepts a list of sources with optional individual prompts."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "sources": {
                "type": "array",
                "description": "List of media sources (file paths or URLs)",
                "items": {
                    "type": "object",
                    "properties": {
                        "source": {"type": "string", "description": "File path or URL"},
                        "type": {"type": "string", "enum": ["image", "audio", "video"], "description": "Media type (auto-detected if omitted)"},
                    },
                    "required": ["source"],
                },
            },
            "prompt": {
                "type": "string",
                "description": "Overall analysis prompt for all media together",
                "default": "describe",
            },
            "output_format": {
                "type": "string",
                "description": "Output format: 'none', 'json', 'table'",
                "enum": ["none", "json", "table"],
                "default": "none",
            },
            "max_tokens": {
                "type": "integer",
                "description": "Max output tokens (default 4096)",
                "default": 4096,
            },
        },
        "required": ["sources", "prompt"],
    },
}

BATCH_SCHEMA = {
    "name": "batch_understand",
    "description": (
        "Analyze multiple files in parallel with the same prompt.\n"
        "Returns results for all files. Use for bulk processing.\n"
        "Max 10 files per batch to avoid API rate limits."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "sources": {
                "type": "array",
                "description": "List of file paths or URLs (max 10)",
                "items": {"type": "string"},
                "maxItems": 10,
            },
            "prompt": {
                "type": "string",
                "description": "Analysis prompt applied to each file",
            },
            "output_format": {
                "type": "string",
                "description": "Output format: 'none', 'json', 'table'",
                "enum": ["none", "json", "table"],
                "default": "json",
            },
            "max_tokens": {
                "type": "integer",
                "description": "Max output tokens per file (default 4096)",
                "default": 4096,
            },
        },
        "required": ["sources", "prompt"],
    },
}


# ─── Handlers ────────────────────────────────────────────────────────

def image_understand_handler(args: dict, **kwargs) -> str:
    source = args.get("source", "")
    prompt = args.get("prompt", "describe")
    output_format = args.get("output_format", "none")
    use_cache = args.get("use_cache", True)
    max_tokens = args.get("max_tokens", 4096)

    if not source:
        return json.dumps({"error": "source is required"})

    # Check cache
    if use_cache:
        cache_k = _cache_key(source, prompt, output_format=output_format)
        if cache_k:
            cached = _cache_get(cache_k)
            if cached:
                return cached

    # Detect if this is code/UI related
    code_prompts = {"code_review", "code_explain", "code_convert", "ui_review", "ui_describe", "ui_improve", "figma_to_code", "screenshot_to_code"}
    if prompt in code_prompts:
        system_prompt = CODE_SYSTEM_PROMPT
    else:
        system_prompt = IMAGE_SYSTEM_PROMPT

    user_prompt = IMAGE_PRESETS.get(prompt, prompt)

    try:
        image_data = _resolve_source(source, "image")
    except (FileNotFoundError, ValueError) as e:
        return json.dumps({"error": str(e)})

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_data}},
                {"type": "text", "text": user_prompt},
            ],
        },
    ]

    result = _call_mimo(messages, max_tokens)
    result = _format_structured(result, output_format)

    # Cache result
    if use_cache:
        cache_k = _cache_key(source, prompt, output_format=output_format)
        if cache_k:
            _cache_set(cache_k, result)

    return result


def audio_understand_handler(args: dict, **kwargs) -> str:
    source = args.get("source", "")
    prompt = args.get("prompt", "transcribe")
    stream = args.get("stream", False)
    output_format = args.get("output_format", "none")
    use_cache = args.get("use_cache", True)
    max_tokens = args.get("max_tokens", 4096)

    if not source:
        return json.dumps({"error": "source is required"})

    if use_cache and not stream:
        cache_k = _cache_key(source, prompt, output_format=output_format)
        if cache_k:
            cached = _cache_get(cache_k)
            if cached:
                return cached

    system_prompt = AUDIO_SYSTEM_PROMPT
    user_prompt = AUDIO_PRESETS.get(prompt, prompt)

    try:
        audio_data = _resolve_source(source, "audio")
    except (FileNotFoundError, ValueError) as e:
        return json.dumps({"error": str(e)})

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "input_audio", "input_audio": {"data": audio_data}},
                {"type": "text", "text": user_prompt},
            ],
        },
    ]

    result = _call_mimo(messages, max_tokens, stream=stream)
    result = _format_structured(result, output_format)

    if use_cache and not stream:
        cache_k = _cache_key(source, prompt, output_format=output_format)
        if cache_k:
            _cache_set(cache_k, result)

    return result


def video_understand_handler(args: dict, **kwargs) -> str:
    source = args.get("source", "")
    prompt = args.get("prompt", "describe")
    fps = args.get("fps", None)
    media_resolution = args.get("media_resolution", "default")
    extract_audio = args.get("extract_audio", False)
    stream = args.get("stream", False)
    output_format = args.get("output_format", "none")
    use_cache = args.get("use_cache", True)
    max_tokens = args.get("max_tokens", 4096)

    if not source:
        return json.dumps({"error": "source is required"})

    if use_cache and not stream:
        cache_k = _cache_key(source, prompt, fps=fps, output_format=output_format)
        if cache_k:
            cached = _cache_get(cache_k)
            if cached:
                return cached

    system_prompt = VIDEO_SYSTEM_PROMPT
    user_prompt = VIDEO_PRESETS.get(prompt, prompt)

    try:
        video_data = _resolve_source(source, "video")
    except (FileNotFoundError, ValueError) as e:
        return json.dumps({"error": str(e)})

    duration = 0
    if not _is_url(source):
        p = Path(source).expanduser().resolve()
        duration = _get_audio_duration(p)
        if duration > 0 and fps is None:
            fps = _auto_fps(duration)
            est = _estimate_tokens(duration, fps, "video")
            user_prompt += f"\n\n[系统提示: 视频时长 {duration:.1f}秒, 采样fps={fps}, 预估消耗 {est['total_estimate']} tokens]"

    fps = fps if fps is not None else 2.0

    content_items = [
        {
            "type": "video_url",
            "video_url": {"url": video_data},
            "fps": fps,
            "media_resolution": media_resolution,
        },
        {"type": "text", "text": user_prompt},
    ]

    if extract_audio and not _is_url(source):
        try:
            p = Path(source).expanduser().resolve()
            audio_path = _extract_audio_from_video(p)
            audio_data = _file_to_base64_uri(audio_path)
            content_items.insert(1, {
                "type": "input_audio",
                "input_audio": {"data": audio_data},
            })
            os.unlink(audio_path)
            user_prompt += "\n\n[系统提示: 已同时提取音频轨道，请结合视频画面和音频进行分析]"
            content_items[-1] = {"type": "text", "text": user_prompt}
        except Exception as e:
            user_prompt += f"\n\n[注意: 音频提取失败({e})，仅分析视频画面]"
            content_items[-1] = {"type": "text", "text": user_prompt}

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": content_items},
    ]

    result = _call_mimo(messages, max_tokens, stream=stream)
    result = _format_structured(result, output_format)

    if use_cache and not stream:
        cache_k = _cache_key(source, prompt, fps=fps, output_format=output_format)
        if cache_k:
            _cache_set(cache_k, result)

    return result


def multimodal_understand_handler(args: dict, **kwargs) -> str:
    sources = args.get("sources", [])
    prompt = args.get("prompt", "describe")
    output_format = args.get("output_format", "none")
    max_tokens = args.get("max_tokens", 4096)

    if not sources:
        return json.dumps({"error": "sources array is required"})

    if len(sources) > 10:
        return json.dumps({"error": "Maximum 10 files per request"})

    content_items = []
    for item in sources:
        source = item.get("source", "")
        media_type = item.get("type", "")

        if not source:
            continue

        # Auto-detect type if not specified
        if not media_type:
            if _is_url(source):
                # Guess from URL extension
                ext = source.split(".")[-1].split("?")[0].lower()
                if f".{ext}" in IMAGE_EXTS:
                    media_type = "image"
                elif f".{ext}" in AUDIO_EXTS:
                    media_type = "audio"
                else:
                    media_type = "video"
            else:
                p = Path(source).expanduser().resolve()
                if p.exists():
                    media_type = _get_media_type(p)
                else:
                    continue

        try:
            data = _resolve_source(source, media_type)
        except Exception:
            continue

        if media_type == "image":
            content_items.append({"type": "image_url", "image_url": {"url": data}})
        elif media_type == "audio":
            content_items.append({"type": "input_audio", "input_audio": {"data": data}})
        elif media_type == "video":
            content_items.append({
                "type": "video_url",
                "video_url": {"url": data},
                "fps": 2.0,
                "media_resolution": "default",
            })

    if not content_items:
        return json.dumps({"error": "No valid media files found"})

    content_items.append({"type": "text", "text": prompt})

    messages = [
        {"role": "system", "content": "You are a multimodal AI assistant. Analyze all provided media files together and respond to the user's prompt."},
        {"role": "user", "content": content_items},
    ]

    result = _call_mimo(messages, max_tokens)
    return _format_structured(result, output_format)


def batch_understand_handler(args: dict, **kwargs) -> str:
    sources = args.get("sources", [])
    prompt = args.get("prompt", "describe")
    output_format = args.get("output_format", "json")
    max_tokens = args.get("max_tokens", 4096)

    if not sources:
        return json.dumps({"error": "sources array is required"})

    if len(sources) > 10:
        return json.dumps({"error": "Maximum 10 files per batch"})

    results = []

    def process_one(source: str) -> dict:
        # Detect type
        if _is_url(source):
            ext = source.split(".")[-1].split("?")[0].lower()
            if f".{ext}" in IMAGE_EXTS:
                media_type = "image"
            elif f".{ext}" in AUDIO_EXTS:
                media_type = "audio"
            else:
                media_type = "video"
        else:
            p = Path(source).expanduser().resolve()
            if not p.exists():
                return {"source": source, "error": "File not found"}
            media_type = _get_media_type(p)

        # Route to appropriate handler
        if media_type == "image":
            result = image_understand_handler({
                "source": source, "prompt": prompt,
                "output_format": "none", "use_cache": True, "max_tokens": max_tokens
            })
        elif media_type == "audio":
            result = audio_understand_handler({
                "source": source, "prompt": prompt,
                "output_format": "none", "use_cache": True, "max_tokens": max_tokens
            })
        else:
            result = video_understand_handler({
                "source": source, "prompt": prompt,
                "output_format": "none", "use_cache": True, "max_tokens": max_tokens
            })

        try:
            data = json.loads(result)
        except json.JSONDecodeError:
            data = {"result": result}

        return {
            "source": source,
            "type": media_type,
            **data,
        }

    # Parallel processing
    with ThreadPoolExecutor(max_workers=min(len(sources), 4)) as executor:
        futures = {executor.submit(process_one, s): s for s in sources}
        for future in as_completed(futures):
            try:
                results.append(future.result())
            except Exception as e:
                results.append({"source": futures[future], "error": str(e)})

    if output_format == "json":
        return json.dumps({"results": results, "count": len(results)}, indent=2, ensure_ascii=False)
    else:
        # Simple text format
        parts = []
        for r in results:
            parts.append(f"=== {r.get('source', '?')} ({r.get('type', '?')}) ===")
            parts.append(r.get("result", r.get("error", "No result")))
            parts.append("")
        return "\n".join(parts)


def check_requirements() -> bool:
    return bool(_get_api_key())


def register(ctx):
    ctx.register_tool(
        name="image_understand",
        toolset="vision",
        schema=IMAGE_SCHEMA,
        handler=image_understand_handler,
        check_fn=check_requirements,
        emoji="🖼️",
    )
    ctx.register_tool(
        name="audio_understand",
        toolset="vision",
        schema=AUDIO_SCHEMA,
        handler=audio_understand_handler,
        check_fn=check_requirements,
        emoji="🎵",
    )
    ctx.register_tool(
        name="video_understand",
        toolset="vision",
        schema=VIDEO_SCHEMA,
        handler=video_understand_handler,
        check_fn=check_requirements,
        emoji="🎬",
    )
    ctx.register_tool(
        name="multimodal_understand",
        toolset="vision",
        schema=MULTIMODAL_SCHEMA,
        handler=multimodal_understand_handler,
        check_fn=check_requirements,
        emoji="🎭",
    )
    ctx.register_tool(
        name="batch_understand",
        toolset="vision",
        schema=BATCH_SCHEMA,
        handler=batch_understand_handler,
        check_fn=check_requirements,
        emoji="📦",
    )
