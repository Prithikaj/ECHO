"""
ECHO - Utility helpers
Handles base64 encoding/decoding, logging, and audio format conversions.
"""

import base64
import logging
import time
import numpy as np
import io
import soundfile as sf

# ── Logging setup ──────────────────────────────────────────────────────────────
def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "[%(asctime)s] %(levelname)s [%(name)s] %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
    return logger


logger = get_logger("utils")


# ── Audio encoding / decoding ──────────────────────────────────────────────────
def audio_to_base64(audio: np.ndarray, sample_rate: int = 16000) -> str:
    """Encode a numpy float32 audio array to a base64 WAV string."""
    buf = io.BytesIO()
    sf.write(buf, audio.astype(np.float32), sample_rate, format="WAV", subtype="PCM_16")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def base64_to_audio(b64_string: str):
    """Decode a base64 WAV string back to (numpy array, sample_rate)."""
    raw = base64.b64decode(b64_string)
    buf = io.BytesIO(raw)
    audio, sr = sf.read(buf, dtype="float32")
    return audio, sr


# ── Timing helpers ─────────────────────────────────────────────────────────────
class Timer:
    """Simple wall-clock timer for latency measurement."""

    def __init__(self):
        self._start = time.perf_counter()

    def elapsed(self) -> float:
        return round(time.perf_counter() - self._start, 4)

    def reset(self):
        self._start = time.perf_counter()


# ── Audio quality assessment ───────────────────────────────────────────────────
def assess_quality(audio: np.ndarray, threshold: float = 0.005) -> str:
    """
    Rough quality label based on RMS energy variance.
    Returns 'clean' or 'noisy'.
    Threshold lowered to 0.005 — post-noise-reduction audio has lower RMS.
    """
    rms = float(np.sqrt(np.mean(audio ** 2)))
    return "clean" if rms >= threshold else "noisy"


# ── Channel detection ──────────────────────────────────────────────────────────
def detect_channels(audio: np.ndarray) -> int:
    """Return number of channels (1 = mono, 2 = stereo)."""
    if audio.ndim == 1:
        return 1
    return audio.shape[1] if audio.ndim == 2 else 1


def to_mono(audio: np.ndarray) -> np.ndarray:
    """Convert stereo (or multi-channel) audio to mono by averaging channels."""
    if audio.ndim == 2:
        return audio.mean(axis=1)
    return audio
