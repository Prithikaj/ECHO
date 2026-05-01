# 🎙 ECHO — Real-Time Audio Streaming & Speaker Diarization

ECHO captures microphone input, processes it in near real-time, and streams
clean, speaker-labeled audio chunks to a Flask API with a live monitoring UI.

---

## Saved Recordings (Handoff for Next Developer)

Every time the pipeline runs, clean speech chunks are automatically saved to disk:

```
ECHO/recordings/
└── session_20260501_143022/
    ├── manifest.json          ← start here
    ├── Speaker_1/
    │   ├── chunk_0001_0.0-1.5s.wav
    │   ├── chunk_0003_3.0-4.5s.wav
    │   └── ...
    └── Speaker_2/
        ├── chunk_0002_1.5-3.0s.wav
        └── ...
```

### manifest.json structure

```json
{
  "session_id": "session_20260501_143022",
  "started_at": "2026-05-01T14:30:22",
  "ended_at":   "2026-05-01T14:35:10",
  "total_chunks": 42,
  "speakers": {
    "Speaker 1": { "chunk_count": 25, "total_duration": 37.5, "files": ["..."] },
    "Speaker 2": { "chunk_count": 17, "total_duration": 25.5, "files": ["..."] }
  },
  "chunks": [
    {
      "chunk_id": 1,
      "file": "Speaker_1/chunk_0001_0.0-1.5s.wav",
      "speaker": "Speaker 1",
      "start_time": 0.0,
      "end_time": 1.5,
      "duration": 1.5,
      "sample_rate": 16000,
      "channels": 1,
      "latency": 0.42,
      "quality": "clean"
    }
  ]
}
```

### What's saved

- Only **clean** quality chunks (noisy/silence chunks are skipped)
- All audio is **16 kHz mono PCM_16 WAV** — ready for ASR, transcription, or further ML processing
- Each file is named with its chunk ID and timestamp range for easy ordering

### API endpoints for the next developer

| Endpoint | Description |
|----------|-------------|
| `GET /recordings` | List all sessions |
| `GET /recordings/<session_id>/manifest` | Full manifest for a session |

---

## Architecture

```
Mic Input
   │
   ▼
AudioCapture (sounddevice)
   │  1–2 s chunks @ 44.1 kHz
   ▼
Preprocessor
   │  → mono conversion
   │  → resample to 16 kHz
   │  → noise reduction (noisereduce)
   │  → peak normalization
   ▼
VAD (webrtcvad)
   │  filter silence / non-speech
   ▼
Diarization (pyannote.audio / energy fallback)
   │  Speaker 1 / Speaker 2 + timestamps
   ▼
Flask API  POST /audio_chunk
   │
   ▼
SSE  →  Frontend UI
```

---

## Quick Start

### 1. Install dependencies

```bash
cd ECHO
pip install -r requirements.txt
```

> **Note:** `webrtcvad` requires a C compiler on Windows.
> Install [Build Tools for Visual Studio](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
> or use the pre-built wheel: `pip install webrtcvad-wheels`

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env and add your HuggingFace token (optional — enables pyannote diarization)
```

### 3. Run (two options)

#### Option A — Embedded (pipeline + server in one process)

```bash
cd backend
python app.py --embed-pipeline
```

#### Option B — Separate processes (recommended for production)

Terminal 1 — Flask server:
```bash
cd backend
python app.py
```

Terminal 2 — Audio pipeline:
```bash
cd backend
python pipeline.py
```

### 4. Open the UI

Navigate to **http://localhost:5000** in your browser.

---

## API Reference

### `POST /audio_chunk`

Receive a processed audio chunk.

```json
{
  "chunk_id":   1,
  "start_time": 0.0,
  "end_time":   1.5,
  "speaker":    "Speaker 1",
  "audio":      "<base64-encoded WAV>",
  "latency":    0.42,
  "quality":    "clean",
  "channels":   1,
  "segments":   [
    { "speaker": "Speaker 1", "start": 0.0, "end": 1.5 }
  ]
}
```

**Response:**
```json
{ "status": "ok", "chunk_id": 1 }
```

### `GET /status`

Returns pipeline stats and last 50 chunks.

### `GET /stream`

Server-Sent Events stream for real-time frontend updates.

### `POST /pipeline/start` / `POST /pipeline/stop`

Control the embedded pipeline.

### `GET /health`

Health check.

---

## Speaker Diarization

| Mode | Requirement | Accuracy |
|------|-------------|----------|
| **pyannote.audio** | `HF_TOKEN` in `.env` | High |
| **Energy fallback** | None | Demo-quality |

To use pyannote:
1. Create a free account at [huggingface.co](https://huggingface.co)
2. Accept the model terms at [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
3. Generate a token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
4. Add `HF_TOKEN=<your_token>` to `.env`

---

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `chunk_duration` | 1.5 s | Audio chunk length |
| `vad_aggressiveness` | 2 | webrtcvad level (0–3) |
| `vad_threshold` | 0.3 | Min speech frame ratio |
| `sample_rate` (capture) | 44100 Hz | Native mic rate |
| `target_sr` (processing) | 16000 Hz | Processing sample rate |

---

## Project Structure

```
ECHO/
├── backend/
│   ├── app.py            Flask API + SSE server
│   ├── pipeline.py       Pipeline orchestrator
│   ├── audio_capture.py  Mic capture (sounddevice)
│   ├── preprocessor.py   Noise reduction + normalization
│   ├── vad.py            Voice Activity Detection
│   ├── diarization.py    Speaker diarization
│   └── utils.py          Shared helpers
├── frontend/
│   ├── index.html        UI
│   ├── style.css         Dark theme styles
│   └── app.js            SSE client + canvas rendering
├── .env.example
├── requirements.txt
└── README.md
```

---

## Troubleshooting

**`webrtcvad` install fails on Windows**
```bash
pip install webrtcvad-wheels
```

**No audio input detected**
```python
import sounddevice as sd
print(sd.query_devices())  # list available devices
```
Then pass `device=<id>` to `AudioCapture`.

**pyannote not loading**
- Ensure `HF_TOKEN` is set in `.env`
- Accept model terms on HuggingFace
- ECHO will fall back to energy-based diarization automatically

**High latency**
- Reduce `chunk_duration` to `1.0`
- Increase `vad_aggressiveness` to `3` to skip more silence
- Run pipeline and server as separate processes
