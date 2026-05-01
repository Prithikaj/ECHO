"""
ECHO - Speaker Diarization
Uses pyannote.audio for speaker detection and labeling.
Falls back to a lightweight energy-based heuristic when pyannote is unavailable
(e.g., no HuggingFace token or GPU).
"""

import os
import numpy as np
from utils import get_logger
import io
import soundfile as sf

logger = get_logger("diarization")

# ── pyannote pipeline (lazy-loaded) ───────────────────────────────────────────
_pipeline = None
_pipeline_attempted = False


def _load_pipeline():
    """Attempt to load the pyannote diarization pipeline once."""
    global _pipeline, _pipeline_attempted
    if _pipeline_attempted:
        return _pipeline
    _pipeline_attempted = True

    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        logger.warning(
            "HF_TOKEN not set — pyannote diarization unavailable. "
            "Using energy-based fallback speaker detection."
        )
        return None

    try:
        from pyannote.audio import Pipeline
        logger.info("Loading pyannote speaker-diarization pipeline…")
        _pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=hf_token,
        )
        logger.info("pyannote pipeline loaded successfully.")
    except Exception as exc:
        logger.warning(f"Failed to load pyannote pipeline: {exc}. Using fallback.")
        _pipeline = None

    return _pipeline


# ── pyannote diarization ───────────────────────────────────────────────────────
def _diarize_pyannote(audio: np.ndarray, sr: int, chunk_start: float) -> list[dict]:
    """Run pyannote diarization and return speaker segments."""
    pipeline = _load_pipeline()
    if pipeline is None:
        return []

    try:
        # Write audio to an in-memory WAV buffer
        buf = io.BytesIO()
        sf.write(buf, audio.astype(np.float32), sr, format="WAV", subtype="PCM_16")
        buf.seek(0)

        import torch
        waveform = torch.tensor(audio).unsqueeze(0).float()
        diarization = pipeline({"waveform": waveform, "sample_rate": sr})

        segments = []
        speaker_map = {}
        speaker_counter = 1

        for turn, _, speaker in diarization.itertracks(yield_label=True):
            if speaker not in speaker_map:
                speaker_map[speaker] = f"Speaker {speaker_counter}"
                speaker_counter += 1
            segments.append(
                {
                    "speaker": speaker_map[speaker],
                    "start": round(chunk_start + turn.start, 3),
                    "end": round(chunk_start + turn.end, 3),
                }
            )

        return segments

    except Exception as exc:
        logger.warning(f"pyannote diarization error: {exc}")
        return []


# ── Energy-based fallback ──────────────────────────────────────────────────────
def _diarize_energy_fallback(
    audio: np.ndarray,
    sr: int,
    chunk_start: float,
    window_ms: int = 200,
) -> list[dict]:
    """
    Lightweight heuristic: split audio into windows, cluster by RMS energy
    into two groups (high-energy = Speaker 1, low-energy = Speaker 2).
    Not accurate for real diarization — purely a demo fallback.
    """
    window_size = int(sr * window_ms / 1000)
    windows = [
        audio[i : i + window_size]
        for i in range(0, len(audio) - window_size + 1, window_size)
    ]

    if not windows:
        return [{"speaker": "Speaker 1", "start": chunk_start, "end": chunk_start + len(audio) / sr}]

    rms_values = [float(np.sqrt(np.mean(w ** 2))) for w in windows]
    median_rms = float(np.median(rms_values))

    segments = []
    current_speaker = None
    seg_start = chunk_start

    for idx, rms in enumerate(rms_values):
        speaker = "Speaker 1" if rms >= median_rms else "Speaker 2"
        t = chunk_start + idx * window_ms / 1000

        if speaker != current_speaker:
            if current_speaker is not None:
                segments.append({"speaker": current_speaker, "start": seg_start, "end": round(t, 3)})
            current_speaker = speaker
            seg_start = round(t, 3)

    # Close last segment
    if current_speaker:
        end_t = round(chunk_start + len(audio) / sr, 3)
        segments.append({"speaker": current_speaker, "start": seg_start, "end": end_t})

    return segments


# ── Public API ─────────────────────────────────────────────────────────────────
def diarize(audio: np.ndarray, sr: int, chunk_start: float = 0.0) -> list[dict]:
    """
    Identify speakers in an audio chunk.

    Returns a list of dicts:
        [{"speaker": "Speaker 1", "start": 0.0, "end": 1.2}, ...]

    Tries pyannote first; falls back to energy heuristic.
    """
    segments = _diarize_pyannote(audio, sr, chunk_start)
    if not segments:
        segments = _diarize_energy_fallback(audio, sr, chunk_start)

    logger.debug(f"Diarization → {len(segments)} segment(s): {segments}")
    return segments


def dominant_speaker(segments: list[dict]) -> str:
    """Return the speaker with the most total speaking time in the segment list."""
    if not segments:
        return "Speaker 1"

    durations: dict[str, float] = {}
    for seg in segments:
        spk = seg["speaker"]
        durations[spk] = durations.get(spk, 0.0) + (seg["end"] - seg["start"])

    return max(durations, key=durations.get)
