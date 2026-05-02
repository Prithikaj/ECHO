"""
ECHO - Main Pipeline Orchestrator
Mic → Chunk → Preprocess → VAD → Diarize → AI Analysis → POST to Flask API
Runs in a background thread so the Flask server stays non-blocking.

AI Layer (Gemini 2.5 Flash):
  - Transcription (Kannada, Hindi, English, code-mixed)
  - Language detection
  - Emotion & sentiment analysis
  - Intent classification + entity extraction
  - Verification loop generation
  - Crisis detection
  - Post-call summary
"""

import threading
import time
import queue
import requests

from audio_capture import AudioCapture
from preprocessor import preprocess
from vad import extract_speech
from diarization import diarize, dominant_speaker
from storage import AudioStorage
from intelligence import get_engine
from utils import get_logger, audio_to_base64, assess_quality, Timer

logger = get_logger("pipeline")

API_URL = "http://127.0.0.1:5000/audio_chunk"


class EchoPipeline:
    """
    Orchestrates the full ECHO audio processing pipeline.

    Call start() to begin streaming; stop() to shut down cleanly.
    Processed clean chunks are saved to disk via AudioStorage.
    Each chunk is enriched with Gemini AI analysis (transcription,
    language detection, emotion, intent, crisis detection).
    """

    def __init__(
        self,
        chunk_duration: float = 10.0,
        api_url: str = API_URL,
        vad_threshold: float = 0.5,
        save_audio: bool = True,
        enable_ai: bool = True,
    ):
        self.chunk_duration = chunk_duration
        self.api_url = api_url
        self.vad_threshold = vad_threshold
        self.save_audio = save_audio
        self.enable_ai = enable_ai

        self._capture = AudioCapture(chunk_duration=chunk_duration)
        self._storage = AudioStorage() if save_audio else None
        self._intelligence = get_engine() if enable_ai else None
        self._worker_thread: threading.Thread | None = None
        self._running = False
        self._chunk_id = 0
        self._wall_clock_start: float = 0.0
        self._session_id: str = ""

        # Metrics
        self.stats = {
            "chunks_captured": 0,
            "chunks_speech": 0,
            "chunks_silence": 0,
            "chunks_sent": 0,
            "chunks_failed": 0,
            "avg_latency": 0.0,
            "avg_ai_latency": 0.0,
            "transcriptions": 0,
            "languages_detected": {},
            "emotions_detected": {},
            "crisis_events": 0,
        }
        self._latencies: list[float] = []
        self._ai_latencies: list[float] = []

    # ── Lifecycle ──────────────────────────────────────────────────────────────
    def start(self):
        if self._running:
            logger.warning("Pipeline already running.")
            return
        self._running = True
        self._wall_clock_start = time.time()
        self._capture.start()

        # Start storage session
        if self._storage:
            session_dir = self._storage.start_session()
            self._session_id = self._storage._manifest.get("session_id", "")
            logger.info(f"Audio will be saved to: {session_dir}")

        # Start intelligence session
        if self._intelligence and self._session_id:
            self._intelligence.start_session(self._session_id)

        self._worker_thread = threading.Thread(
            target=self._process_loop, daemon=True, name="echo-pipeline"
        )
        self._worker_thread.start()
        logger.info("ECHO pipeline started (AI enabled: %s).", self.enable_ai)

    def stop(self):
        self._running = False
        self._capture.stop()
        if self._worker_thread:
            self._worker_thread.join(timeout=5)

        # Generate post-call summary
        post_call = {}
        if self._intelligence:
            try:
                post_call = self._intelligence.end_session()
                logger.info("Post-call summary generated.")
            except Exception as e:
                logger.warning(f"Post-call summary failed: {e}")

        # Finalise storage session
        if self._storage and self._storage.is_active:
            manifest = self._storage.end_session()
            logger.info(f"Recordings saved → {self._storage.session_dir}")

        logger.info("ECHO pipeline stopped.")
        self._log_stats()
        return post_call

    # ── Processing loop ────────────────────────────────────────────────────────
    def _process_loop(self):
        while self._running:
            try:
                raw_audio, orig_sr, n_channels = self._capture.get_chunk(timeout=2.0)
            except queue.Empty:
                continue

            timer = Timer()
            self.stats["chunks_captured"] += 1
            chunk_start = time.time() - self._wall_clock_start - self.chunk_duration

            # ── 1. Preprocess ──────────────────────────────────────────────────
            try:
                audio, sr, detected_channels = preprocess(raw_audio, orig_sr)
            except Exception as exc:
                logger.error(f"Preprocessing failed: {exc}")
                continue

            # ── 2. VAD ────────────────────────────────────────────────────────
            audio, speech_intervals, has_speech = extract_speech(audio, sr, threshold=self.vad_threshold)
            if not has_speech:
                self.stats["chunks_silence"] += 1
                logger.debug("VAD: no speech — skipping chunk")
                continue

            self.stats["chunks_speech"] += 1

            # ── 3. Diarization ────────────────────────────────────────────────
            try:
                segments = diarize(audio, sr, chunk_start=chunk_start)
                speaker = dominant_speaker(segments)
            except Exception as exc:
                logger.warning(f"Diarization failed: {exc}")
                speaker = "Speaker 1"
                segments = []

            # ── 4. Encode & assess quality ────────────────────────────────────
            try:
                b64_audio = audio_to_base64(audio, sr)
                quality = assess_quality(audio)
            except Exception as exc:
                logger.error(f"Encoding failed: {exc}")
                continue

            latency = timer.elapsed()
            self._latencies.append(latency)
            self.stats["avg_latency"] = round(
                sum(self._latencies[-50:]) / len(self._latencies[-50:]), 4
            )

            # ── 5. AI Analysis (Gemini 2.5 Flash) ────────────────────────────
            ai_analysis = {}
            if self.enable_ai and self._intelligence:
                try:
                    self._chunk_id += 1
                    ai_analysis = self._intelligence.process_chunk(
                        audio_b64=b64_audio,
                        speaker=speaker,
                        chunk_id=self._chunk_id,
                    )
                    # Track AI latency
                    ai_lat = ai_analysis.get("ai_latency", 0)
                    self._ai_latencies.append(ai_lat)
                    self.stats["avg_ai_latency"] = round(
                        sum(self._ai_latencies[-50:]) / len(self._ai_latencies[-50:]), 4
                    )

                    # Track language stats
                    lang = ai_analysis.get("language", "unknown")
                    self.stats["languages_detected"][lang] = (
                        self.stats["languages_detected"].get(lang, 0) + 1
                    )

                    # Track emotion stats
                    emo = ai_analysis.get("emotion", "neutral")
                    self.stats["emotions_detected"][emo] = (
                        self.stats["emotions_detected"].get(emo, 0) + 1
                    )

                    # Track transcriptions
                    if ai_analysis.get("transcript"):
                        self.stats["transcriptions"] += 1

                    # Track crisis events
                    if ai_analysis.get("crisis_activated"):
                        self.stats["crisis_events"] += 1
                        logger.warning(
                            f"🚨 CRISIS DETECTED: {ai_analysis.get('crisis_type')} | "
                            f"severity={ai_analysis.get('crisis_severity')} | "
                            f"bypass_ai={ai_analysis.get('bypass_ai')}"
                        )

                except Exception as exc:
                    logger.error(f"AI analysis failed for chunk {self._chunk_id}: {exc}")
            else:
                self._chunk_id += 1

            # ── 6. Save clean audio to disk ───────────────────────────────────
            if self._storage:
                self._storage.save_chunk(
                    audio=audio,
                    sample_rate=sr,
                    chunk_id=self._chunk_id,
                    start_time=chunk_start,
                    end_time=chunk_start + self.chunk_duration,
                    speaker=speaker,
                    latency=latency,
                    quality=quality,
                    channels=detected_channels,
                )

            # ── 7. Build payload ──────────────────────────────────────────────
            payload = {
                "chunk_id": self._chunk_id,
                "start_time": round(chunk_start, 3),
                "end_time": round(chunk_start + self.chunk_duration, 3),
                "speaker": speaker,
                "audio": b64_audio,
                "latency": latency,
                "quality": quality,
                "channels": detected_channels,
                "segments": segments,
                "speech_intervals": speech_intervals,
                "voiced_duration": round(len(audio) / sr, 3),
                # AI enrichment
                **ai_analysis,
            }

            # ── 8. Send to API ────────────────────────────────────────────────
            self._send_chunk(payload)

    def _send_chunk(self, payload: dict):
        try:
            resp = requests.post(self.api_url, json=payload, timeout=5)
            if resp.status_code == 200:
                self.stats["chunks_sent"] += 1
                logger.info(
                    f"[Chunk {payload['chunk_id']}] {payload['speaker']} | "
                    f"lang={payload.get('language', '?')} | "
                    f"emotion={payload.get('emotion', '?')} | "
                    f"intent={payload.get('intent', '?')} | "
                    f"latency={payload['latency']}s"
                )
            else:
                self.stats["chunks_failed"] += 1
                logger.warning(f"API returned {resp.status_code}: {resp.text}")
        except requests.RequestException as exc:
            self.stats["chunks_failed"] += 1
            logger.error(f"Failed to send chunk: {exc}")

    # ── Stats ──────────────────────────────────────────────────────────────────
    def _log_stats(self):
        s = self.stats
        logger.info(
            f"Pipeline stats — captured={s['chunks_captured']} "
            f"speech={s['chunks_speech']} silence={s['chunks_silence']} "
            f"sent={s['chunks_sent']} failed={s['chunks_failed']} "
            f"avg_latency={s['avg_latency']}s avg_ai_latency={s['avg_ai_latency']}s "
            f"transcriptions={s['transcriptions']} crisis_events={s['crisis_events']}"
        )

    def get_stats(self) -> dict:
        return dict(self.stats)


# ── Standalone entry point ─────────────────────────────────────────────────────
if __name__ == "__main__":
    import signal

    pipeline = EchoPipeline(chunk_duration=10.0, enable_ai=True)
    pipeline.start()

    def _shutdown(sig, frame):
        print("\nShutting down…")
        pipeline.stop()
        raise SystemExit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    logger.info("Pipeline running. Press Ctrl+C to stop.")
    while True:
        time.sleep(1)
