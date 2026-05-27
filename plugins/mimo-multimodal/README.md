# mimo-multimodal

Hermes Agent plugin for **image, audio, and video understanding** via Xiaomi MiMo multimodal models.

## Features

- 🖼️ **Image Analysis** — Description, OCR, code review, UI analysis
- 🎵 **Audio Analysis** — Transcription, summarization, meeting notes
- 🎬 **Video Analysis** — Scene description, tutorial breakdown, code/UI demos
- 🎭 **Multi-modal** — Combine images, audio, video in single request
- 📦 **Batch Processing** — Parallel analysis of multiple files
- 💾 **Caching** — File hash-based result caching (24h TTL)
- 📊 **Structured Output** — JSON/table format support
- 🌊 **Streaming** — Real-time output for long content

## Requirements

- Xiaomi MiMo API Key ([Get one here](https://platform.xiaomimimo.com/))
- `ffmpeg` (optional, for audio extraction from video)

## Installation

This plugin is bundled with Hermes Agent. Enable it in your `~/.hermes/config.yaml`:

```yaml
plugins:
  enabled:
    - mimo-multimodal
```

## Configuration

Set your Xiaomi MiMo API key in `~/.hermes/.env`:

```bash
XIAOMI_API_KEY=your_api_key_here
```

## Tools

### 1. image_understand 🖼️

Analyze images with AI.

**Presets:**

| Category | Preset | Description |
|----------|--------|-------------|
| General | `describe` | Detailed image description |
| General | `ocr` | Extract all text from image |
| General | `explain` | Explain meaning/context |
| General | `extract_info` | Extract structured data |
| General | `caption` | Generate image caption |
| Code | `code_review` | Analyze code, find bugs, suggest improvements |
| Code | `code_explain` | Explain what code does |
| Code | `code_convert` | Convert code to other languages |
| UI/UX | `ui_review` | Review UI design quality |
| UI/UX | `ui_describe` | Describe UI components |
| UI/UX | `ui_improve` | Suggest UI improvements |
| UI/UX | `figma_to_code` | Generate code from design |
| UI/UX | `screenshot_to_code` | Generate HTML/CSS from screenshot |

**Examples:**

```python
# Code review
image_understand(source="screenshot.png", prompt="code_review")

# UI analysis
image_understand(source="design.png", prompt="ui_review")

# Generate code from design
image_understand(source="figma-export.png", prompt="screenshot_to_code", output_format="json")
```

### 2. audio_understand 🎵

Analyze audio files.

**Presets:**

| Preset | Description |
|--------|-------------|
| `transcribe` | Full transcription |
| `describe` | Detailed audio description |
| `summarize` | Audio summary |
| `translate` | Transcribe + translate to Chinese |
| `extract_info` | Extract key information |
| `meeting_notes` | Structured meeting minutes |
| `lecture_notes` | Study notes from lecture |

**Examples:**

```python
# Meeting transcription
audio_understand(source="meeting.mp3", prompt="meeting_notes")

# Streaming for long audio
audio_understand(source="lecture.wav", prompt="lecture_notes", stream=True)
```

### 3. video_understand 🎬

Analyze video files.

**Presets:**

| Preset | Description |
|--------|-------------|
| `describe` | Detailed scene description |
| `summarize` | Video summary |
| `scenes` | Scene-by-scene breakdown |
| `ocr` | Extract text/subtitles |
| `action` | Action analysis |
| `count` | Count people/objects |
| `tutorial_steps` | Tutorial step breakdown |
| `code_demo` | Code demo analysis |
| `ui_demo` | UI interaction analysis |

**Examples:**

```python
# Tutorial breakdown
video_understand(source="tutorial.mp4", prompt="tutorial_steps")

# Code demo analysis
video_understand(source="coding-stream.mp4", prompt="code_demo", media_resolution="max")

# With audio extraction
video_understand(source="interview.mp4", extract_audio=True)
```

### 4. multimodal_understand 🎭

Analyze multiple media files together in one request.

**Examples:**

```python
# Compare images
multimodal_understand(
    sources=[
        {"source": "before.png", "type": "image"},
        {"source": "after.png", "type": "image"}
    ],
    prompt="Compare these two screenshots and describe the differences"
)

# Analyze video with separate audio
multimodal_understand(
    sources=[
        {"source": "video.mp4", "type": "video"},
        {"source": "commentary.mp3", "type": "audio"}
    ],
    prompt="Analyze the video and commentary together"
)
```

### 5. batch_understand 📦

Process multiple files in parallel with the same prompt.

**Examples:**

```python
# Batch OCR
batch_understand(
    sources=["page1.png", "page2.png", "page3.png"],
    prompt="ocr",
    output_format="json"
)

# Batch code review
batch_understand(
    sources=["file1.py", "file2.py", "file3.py"],
    prompt="code_review"
)
```

## Advanced Features

### Structured Output

```python
# JSON format
image_understand(source="photo.jpg", prompt="extract_info", output_format="json")

# Table format
audio_understand(source="meeting.mp3", prompt="meeting_notes", output_format="table")
```

### Streaming

```python
# Enable for long content
video_understand(source="long-video.mp4", stream=True)
```

### Cache Control

```python
# Disable cache for fresh analysis
image_understand(source="live.png", prompt="describe", use_cache=False)
```

## File Size Limits

| Type  | Limit | Fallback |
|-------|-------|----------|
| Image | 50MB  | Error    |
| Audio | 50MB  | Error    |
| Video | 50MB  | Temp HTTP server (up to 300MB) |

## Token Usage

Video auto-fps tuning:

| Duration | Auto fps | Estimated Tokens |
|----------|----------|------------------|
| < 10s    | 5.0      | ~3,000           |
| < 1min   | 2.0      | ~7,000           |
| < 5min   | 1.0      | ~35,000          |
| < 10min  | 0.5      | ~35,000          |
| 10min+   | 0.2      | ~14,000          |

## Supported Formats

| Type  | Formats |
|-------|---------|
| Image | JPEG, PNG, GIF, WebP, BMP |
| Audio | MP3, WAV, FLAC, M4A, OGG, AAC, WMA |
| Video | MP4, MOV, AVI, WMV, MKV, WEBM, FLV |

## Author

**Everett** — [GitHub](https://github.com/xiaopengsvip)

## License

MIT
