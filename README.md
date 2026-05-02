# ECHO - AI Emergency Call Intelligence System

ECHO is a real-time AI-powered call center intelligence platform built for emergency helplines.
It captures voice, transcribes speech in Kannada, Hindi, and English, detects emotion and intent,
identifies crises, and guides agents - all in under 2 seconds per chunk.

Powered by **Gemini 2.5 Flash** for all language and intelligence tasks.

---

## What It Does

| Layer | Capability |
|-------|-----------|
| Voice Capture | Real-time mic streaming, noise reduction, multi-speaker separation |
| Speech-to-Text | Multilingual ASR - Kannada, Hindi, English, code-mixed |
| Language Detection | Auto-detects language per chunk with confidence score |
| Dialect Understanding | Normalizes slang, interprets cultural phrases |
| Emotion Analysis | Classifies fear, panic, anger, confusion, distress, calm |
| Urgency Detection | 0-1 urgency score with trajectory tracking (escalating/stable) |
| Intent Classification | Emergency, complaint, harassment, medical, fire, accident, etc. |
| Entity Extraction | Location, people involved, incident type, landmarks |
| Verification Loop | AI restates the problem and asks for confirmation |
| Crisis Mode | Auto-activates on high distress - escalates to police/ambulance/human |
| TTS Response | Generates calm, language-appropriate spoken responses |
| Agent Dashboard | Live transcript, emotion heatmap, risk level, override controls |
| Post-Call Summary | Auto-generated case report with timeline and follow-up notes |

---

## Architecture

```
Mic Input
   |
   v
AudioCapture (sounddevice)  44.1 kHz
   |
   v
Preprocessor
   +-- mono conversion
   +-- resample to 16 kHz
   +-- noise reduction (noisereduce)
   +-- peak normalization
   |
   v
VAD (Silero / webrtcvad fallback)
   |  strips silence, keeps only voiced segments
   v
Diarization (pyannote.audio / energy fallback)
   |  Speaker 1 / Speaker 2 + timestamps
   v
Gemini 2.5 Flash  <--- INTELLIGENCE LAYER
   +-- transcription (Kannada / Hindi / English / code-mixed)
   +-- language detection + dialect normalization
   +-- emotion + urgency analysis
   +-- intent classification + entity extraction
   +-- crisis detection
   +-- verification loop generation
   +-- TTS response text
   |
   v
Flask API  POST /audio_chunk
   |
   v
SSE stream -> Frontend Dashboard
```

---

## Quick Start

### 1. Install dependencies

```bash
cd ECHO
pip install flask flask-cors sounddevice numpy scipy noisereduce soundfile requests torch google-genai pyannote.audio
```

On Windows, if `webrtcvad` fails:
```bash
pip install webrtcvad-wheels
```

### 2. Configure environment

```bash
cp .env.example .env
```

`.env.example` already contains the Gemini API key. Optionally add a HuggingFace token for better speaker diarization:

```env
GEMINI_API_KEY=your_key_here
HF_TOKEN=your_hf_token_here   # optional
```

### 3. Run

**Option A - Single process (simplest)**
```bash
cd ECHO/backend
python app.py --embed-pipeline
```

**Option B - Two processes (recommended)**

Terminal 1:
```bash
cd ECHO/backend
python app.py
```

Terminal 2:
```bash
cd ECHO/backend
python pipeline.py
```

Disable AI (raw audio only, no Gemini calls):
```bash
python app.py --embed-pipeline --no-ai
```

### 4. Open the UI

**http://localhost:5000** then click **Start**

---

## Frontend Dashboard

The UI is a 3-column real-time intelligence dashboard:

**Left column**
- Live waveform (urgency-colored)
- Live transcription feed with language tag, confidence, normalized text
- Verification loop - AI restatement and clarification questions

**Middle column**
- Emotion panel - emoji, label, urgency bar, sentiment, trajectory chart
- Intent panel - intent badge, risk level, extracted entities
- Speaker activity bars

**Right column**
- Agent dashboard - risk level, crisis status, escalation path, turn count
- AI voice response (TTS text)
- Event log

**Bottom**
- Chunk timeline with language and emotion tags
- Latency chart (pipeline + AI latency)
- Saved recordings info

**Crisis banner** - flashes red across the top when distress/emergency is detected, shows crisis type and escalation path.

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/audio_chunk` | Receive AI-enriched chunk from pipeline |
| GET  | `/stream` | SSE stream for real-time frontend updates |
| GET  | `/status` | Pipeline stats, AI context, recent chunks |
| GET  | `/context` | Current conversation context and AI state |
| POST | `/pipeline/start` | Start the embedded pipeline |
| POST | `/pipeline/stop` | Stop pipeline + return post-call summary |
| POST | `/analyze_text` | On-demand text analysis via Gemini |
| GET  | `/recordings` | List all saved sessions |
| GET  | `/recordings/<id>/manifest` | Full manifest for a session |
| GET  | `/health` | Health check |

### `/analyze_text` example

```bash
curl -X POST http://localhost:5000/analyze_text \
  -H "Content-Type: application/json" \
  -d '{"text": "Someone is following me near Brigade Road", "language": "english"}'
```

---

## Gemini AI Module

All Gemini calls are in `backend/gemini_ai.py`:

| Function | What it does |
|----------|-------------|
| `full_analysis(audio_b64, speaker, history)` | Single call: transcription + emotion + intent + crisis |
| `analyze_emotion(transcript)` | Emotion, urgency, sentiment, implicit meaning |
| `classify_intent_and_extract(transcript, lang)` | Intent + entity extraction |
| `generate_verification(...)` | Verification loop response |
| `detect_crisis(transcript, emotion, intent)` | Crisis mode decision |
| `generate_post_call_summary(history, session_id)` | Post-call case report |
| `generate_tts_response(message, language, ...)` | TTS text with tone guidance |
| `normalize_dialect(text, language)` | Dialect normalization + cultural phrase interpretation |

Quick test:
```bash
cd ECHO/backend
python -c "
import sys; sys.path.insert(0, '.')
from gemini_ai import analyze_emotion, classify_intent_and_extract
print(analyze_emotion('Help me there is fire near MG Road'))
print(classify_intent_and_extract('Someone is following me', 'english'))
"
```

---

## Project Structure

```
ECHO/
├── backend/
|   ├── app.py              Flask API + SSE server
|   ├── pipeline.py         Pipeline orchestrator (mic -> AI -> API)
|   ├── gemini_ai.py        Gemini 2.5 Flash - all AI functions
|   ├── intelligence.py     Conversation context + AI orchestration
|   ├── audio_capture.py    Mic capture (sounddevice)
|   ├── preprocessor.py     Noise reduction + normalization
|   ├── vad.py              Voice Activity Detection (Silero + webrtcvad)
|   ├── diarization.py      Speaker diarization (pyannote + energy fallback)
|   ├── storage.py          Session recording manager
|   └── utils.py            Shared helpers
├── frontend/
|   ├── index.html          3-column AI dashboard
|   ├── style.css           Dark theme
|   └── app.js              SSE client + all AI panel renderers
├── recordings/             Auto-saved session WAV files (gitignored)
├── .env.example            Environment variable template
└── README.md
```

---

## Saved Recordings

Every session saves clean speech chunks to disk:

```
recordings/
└── session_20260501_144420/
    ├── manifest.json
    ├── Speaker_1/
    |   ├── chunk_0001_0.0-10.0s.wav
    |   └── ...
    └── Speaker_2/
        └── ...
```

`manifest.json` includes full metadata per chunk: speaker, timestamps, language, emotion, intent, latency, quality.

---

## Speaker Diarization

| Mode | Requirement | Accuracy |
|------|-------------|----------|
| pyannote.audio 3.1 | `HF_TOKEN` in `.env` | High |
| Energy-based fallback | None | Demo-quality |

To enable pyannote:
1. Accept model terms at https://huggingface.co/pyannote/speaker-diarization-3.1
2. Get a token at https://huggingface.co/settings/tokens
3. Add `HF_TOKEN=<token>` to `.env`

---

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `chunk_duration` | 10.0 s | Audio chunk length |
| `vad_threshold` | 0.5 | Silero VAD confidence threshold |
| `enable_ai` | True | Enable Gemini AI analysis |
| `save_audio` | True | Save clean chunks to disk |
| `sample_rate` (capture) | 44100 Hz | Native mic rate |
| `target_sr` (processing) | 16000 Hz | Processing sample rate |

---

## Troubleshooting

**`webrtcvad` install fails on Windows**
```bash
pip install webrtcvad-wheels
```

**No audio input detected**
```python
import sounddevice as sd
print(sd.query_devices())
```
Pass `device=<id>` to `AudioCapture` in `pipeline.py`.

**Gemini API errors**
- Check `GEMINI_API_KEY` is set in `.env`
- Pipeline continues without AI if Gemini fails - chunks still process, just without transcription/emotion data

**pyannote not loading**
- Set `HF_TOKEN` in `.env` and accept model terms on HuggingFace
- Falls back to energy-based diarization automatically

**High latency**
- Reduce `chunk_duration` to `5.0`
- Run pipeline and server as separate processes
- Gemini AI adds ~1-2 s per chunk - use `--no-ai` to disable if latency is critical