"""
ECHO - Voice Activity Detection (VAD)
Primary:  Silero VAD (torch-based, accurate speech timestamps)
Fallback: webrtcvad (fast, frame-level, used if Silero fails to load)

Key function: extract_speech(audio, sr)
  → returns a trimmed numpy array containing ONLY the voiced segments,
    with silence removed. Also returns the speech intervals for logging.
"""

import numpy as np
import torch
import threading
from utils import get_logger

logger = get_logger("vad")

TARGET_SR = 16000  # Silero VAD requires 16 kHz

# ── Silero model (lazy-loaded once) ───────────────────────────────────────────
_silero_model = None
_silero_utils  = None
_silero_lock   = threading.Lock()
_silero_failed = False


def _load_silero():
    """Load Silero VAD from torch.hub (cached after first load)."""
    global _silero_model, _silero_utils, _silero_failed

    with _silero_lock:
        if _silero_model is not None or _silero_failed:
            return _silero_model, _silero_utils

        try:
            logger.info("Loading Silero VAD model…")
            model, utils = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                force_reload=False,
                onnx=False,
                verbose=False,
            )
            _silero_model = model
            _silero_utils  = utils
            logger.info("Silero VAD loaded successfully.")
        except Exception as exc:
            logger.warning(f"Silero VAD failed to load: {exc}. Falling back to webrtcvad.")
            _silero_failed = True

    return _silero_model, _silero_utils


# ── Silero VAD ─────────────────────────────────────────────────────────────────

def _get_speech_intervals_silero(
    audio: np.ndarray,
    sr: int = TARGET_SR,
    threshold: float = 0.5,
    min_speech_ms: int = 250,
    min_silence_ms: int = 300,
    speech_pad_ms: int = 100,
) -> list[dict]:
    """
    Run Silero VAD and return a list of speech intervals.

    Returns:
        [{"start": float, "end": float}, ...]   (seconds)
    """
    model, utils = _load_silero()
    if model is None:
        return []

    try:
        get_speech_timestamps, _, _, _, _ = utils

        # Silero expects a float32 tensor at 16 kHz
        wav = torch.from_numpy(audio.astype(np.float32))

        timestamps = get_speech_timestamps(
            wav,
            model,
            sampling_rate=sr,
            threshold=threshold,
            min_speech_duration_ms=min_speech_ms,
            min_silence_duration_ms=min_silence_ms,
            speech_pad_ms=speech_pad_ms,
            return_seconds=True,
        )

        # timestamps is a list of {"start": float, "end": float}
        return [{"start": t["start"], "end": t["end"]} for t in timestamps]

    except Exception as exc:
        logger.warning(f"Silero inference error: {exc}")
        return []


# ── webrtcvad fallback ─────────────────────────────────────────────────────────

def _get_speech_intervals_webrtcvad(
    audio: np.ndarray,
    sr: int = TARGET_SR,
    aggressiveness: int = 2,
    frame_ms: int = 30,
) -> list[dict]:
    """Frame-level webrtcvad fallback — returns coarse speech intervals."""
    try:
        try:
            import webrtcvad
        except ImportError:
            import webrtcvad_wheels as webrtcvad

        if sr not in {8000, 16000, 32000, 48000}:
            return [{"start": 0.0, "end": len(audio) / sr}]

        vad = webrtcvad.Vad(aggressiveness)
        frame_size = int(sr * frame_ms / 1000)
        intervals = []
        in_speech = False
        seg_start = 0.0

        for i in range(0, len(audio) - frame_size + 1, frame_size):
            frame = audio[i : i + frame_size]
            pcm = (frame * 32767).astype(np.int16).tobytes()
            t = i / sr
            try:
                is_speech = vad.is_speech(pcm, sr)
            except Exception:
                is_speech = False

            if is_speech and not in_speech:
                in_speech = True
                seg_start = t
            elif not is_speech and in_speech:
                in_speech = False
                intervals.append({"start": seg_start, "end": round(t, 3)})

        if in_speech:
            intervals.append({"start": seg_start, "end": round(len(audio) / sr, 3)})

        return intervals

    except Exception as exc:
        logger.warning(f"webrtcvad fallback error: {exc}")
        return [{"start": 0.0, "end": len(audio) / sr}]


# ── Public API ─────────────────────────────────────────────────────────────────

def get_speech_intervals(
    audio: np.ndarray,
    sr: int = TARGET_SR,
    threshold: float = 0.5,
) -> list[dict]:
    """
    Return speech intervals using Silero VAD (falls back to webrtcvad).

    Returns:
        [{"start": float, "end": float}, ...]   (seconds, relative to chunk start)
    """
    intervals = _get_speech_intervals_silero(audio, sr, threshold=threshold)

    if not intervals:
        logger.debug("Silero returned no intervals — trying webrtcvad fallback")
        intervals = _get_speech_intervals_webrtcvad(audio, sr)

    logger.debug(f"VAD intervals: {intervals}")
    return intervals


def extract_speech(
    audio: np.ndarray,
    sr: int = TARGET_SR,
    threshold: float = 0.5,
    min_voiced_ratio: float = 0.05,
) -> tuple[np.ndarray, list[dict], bool]:
    """
    Cut out ONLY the voiced segments from audio.

    Args:
        audio:             mono float32 array at `sr` Hz
        sr:                sample rate (should be 16000)
        threshold:         Silero confidence threshold (0–1)
        min_voiced_ratio:  minimum fraction of audio that must be speech
                           to consider the chunk worth keeping

    Returns:
        trimmed_audio  (np.ndarray): concatenated voiced segments only
        intervals      (list[dict]): speech intervals used for trimming
        has_speech     (bool):       False if chunk should be discarded entirely
    """
    intervals = get_speech_intervals(audio, sr, threshold)

    if not intervals:
        logger.debug("VAD: no speech detected — discarding chunk")
        return np.array([], dtype=np.float32), [], False

    # Concatenate only the voiced segments
    voiced_parts = []
    for seg in intervals:
        start_sample = int(seg["start"] * sr)
        end_sample   = int(seg["end"]   * sr)
        start_sample = max(0, min(start_sample, len(audio)))
        end_sample   = max(0, min(end_sample,   len(audio)))
        if end_sample > start_sample:
            voiced_parts.append(audio[start_sample:end_sample])

    if not voiced_parts:
        return np.array([], dtype=np.float32), intervals, False

    trimmed = np.concatenate(voiced_parts, axis=0)

    # Check voiced ratio
    voiced_ratio = len(trimmed) / max(len(audio), 1)
    if voiced_ratio < min_voiced_ratio:
        logger.debug(f"VAD: voiced ratio {voiced_ratio:.1%} below threshold — discarding")
        return np.array([], dtype=np.float32), intervals, False

    total_voiced_s = len(trimmed) / sr
    total_chunk_s  = len(audio)   / sr
    logger.debug(
        f"VAD: kept {total_voiced_s:.2f}s / {total_chunk_s:.2f}s "
        f"({voiced_ratio:.0%}) across {len(intervals)} segment(s)"
    )
    return trimmed.astype(np.float32), intervals, True


def contains_speech(
    audio: np.ndarray,
    sr: int = TARGET_SR,
    threshold: float = 0.5,
    min_voiced_ratio: float = 0.05,
) -> bool:
    """Convenience wrapper — returns True if the chunk contains enough speech."""
    _, _, has_speech = extract_speech(audio, sr, threshold, min_voiced_ratio)
    return has_speech
