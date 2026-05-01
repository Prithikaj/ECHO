"""
ECHO - Main Pipeline Orchestrator
Mic → Chunk → Preprocess → VAD → Diarize → POST to Flask API
Runs in a background thread so the Flask server stays non-blocking.
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
from utils import get_logger, audio_to_base64, assess_quality, Timer

logger = get_logger("pipeline")

API_URL = "http://127.0.0.1:5000/audio_chunk"


class EchoPipeline:
    """
    Orchestrates the full ECHO audio processing pipeline.

    Call start() to begin streaming; stop() to shut down cleanly.
    Processed clean chunks are saved to disk via AudioStorage so
    downstream processes can consume them without re-running the pipeline.
    """

    def __init__(
        self,
        chunk_duration: float = 10.0,
        api_url: str = API_URL,
        vad_threshold: float = 0.5,       # Silero confidence threshold (0–1)
        save_audio: bool = True,
    ):
        self.chunk_duration = chunk_duration
        self.api_url = api_url
        self.vad_threshold = vad_threshold
        self.save_audio = save_audio

        self._capture = AudioCapture(chunk_duration=chunk_duration)
        self._storage = AudioStorage() if save_audio else None
        self._worker_thread: threading.Thread | None = None
        self._running = False
        self._chunk_id = 0
        self._wall_clock_start: float = 0.0

        # Metrics (thread-safe via GIL for simple reads)
        self.stats = {
            "chunks_captured": 0,
            "chunks_speech": 0,
            "chunks_silence": 0,
            "chunks_sent": 0,
            "chunks_failed": 0,
            "avg_latency": 0.0,
        }
        self._latencies: list[float] = []

    # ── Lifecycle ──────────────────────────────────────────────────────────────
    def start(self):
        if self._running:
            logger.warning("Pipeline already running.")
            return
        self._running = True
        self._wall_clock_start = time.time()
        self._capture.start()

        # Start a new storage session
        if self._storage:
            session_dir = self._storage.start_session()
            logger.info(f"Audio will be saved to: {session_dir}")

        self._worker_thread = threading.Thread(
            target=self._process_loop, daemon=True, name="echo-pipeline"
        )
        self._worker_thread.start()
        logger.info("ECHO pipeline started.")

    def stop(self):
        self._running = False
        self._capture.stop()
        if self._worker_thread:
            self._worker_thread.join(timeout=5)

        # Finalise storage session
        if self._storage and self._storage.is_active:
            manifest = self._storage.end_session()
            logger.info(f"Recordings saved → {self._storage.session_dir}")
            logger.info(f"Manifest: {self._storage.session_dir}/manifest.json")

        logger.info("ECHO pipeline stopped.")
        self._log_stats()

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

            # ── 2. VAD — extract only voiced segments ─────────────────────────
            audio, speech_intervals, has_speech = extract_speech(audio, sr, threshold=self.vad_threshold)
            if not has_speech:
                self.stats["chunks_silence"] += 1
                logger.debug("VAD: no speech — skipping chunk")
                continue

            self.stats["chunks_speech"] += 1
            logger.debug(
                f"VAD: {len(speech_intervals)} speech segment(s), "
                f"{len(audio)/sr:.2f}s voiced audio retained"
            )

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

            # ── 5. Save clean audio to disk ───────────────────────────────────
            if self._storage:
                self._storage.save_chunk(
                    audio=audio,
                    sample_rate=sr,
                    chunk_id=self._chunk_id + 1,   # preview id before increment
                    start_time=chunk_start,
                    end_time=chunk_start + self.chunk_duration,
                    speaker=speaker,
                    latency=latency,
                    quality=quality,
                    channels=detected_channels,
                )

            # ── 7. Build payload ──────────────────────────────────────────────
            self._chunk_id += 1
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
            }

            # ── 8. Send to API ────────────────────────────────────────────────
            self._send_chunk(payload)

    def _send_chunk(self, payload: dict):
        try:
            resp = requests.post(self.api_url, json=payload, timeout=3)
            if resp.status_code == 200:
                self.stats["chunks_sent"] += 1
                logger.info(
                    f"[Chunk {payload['chunk_id']}] {payload['speaker']} | "
                    f"{payload['start_time']:.1f}–{payload['end_time']:.1f}s | "
                    f"latency={payload['latency']}s | quality={payload['quality']}"
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
            f"avg_latency={s['avg_latency']}s"
        )

    def get_stats(self) -> dict:
        return dict(self.stats)


# ── Standalone entry point ─────────────────────────────────────────────────────
if __name__ == "__main__":
    import signal

    pipeline = EchoPipeline(chunk_duration=10.0)
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
