"""
ECHO - Audio Preprocessor
Handles: mono conversion, resampling to 16 kHz, noise reduction, normalization.
"""

import numpy as np
import noisereduce as nr
import scipy.signal as signal
from utils import get_logger, to_mono, detect_channels

logger = get_logger("preprocessor")

TARGET_SR = 16000  # Hz


def resample(audio: np.ndarray, orig_sr: int, target_sr: int = TARGET_SR) -> np.ndarray:
    """Resample audio from orig_sr to target_sr using scipy."""
    if orig_sr == target_sr:
        return audio
    num_samples = int(len(audio) * target_sr / orig_sr)
    resampled = signal.resample(audio, num_samples)
    logger.debug(f"Resampled {orig_sr}Hz → {target_sr}Hz ({len(audio)} → {len(resampled)} samples)")
    return resampled.astype(np.float32)


def reduce_noise(audio: np.ndarray, sr: int = TARGET_SR) -> np.ndarray:
    """
    Apply spectral noise reduction via noisereduce.
    Uses the first 0.5 s as a noise profile when the clip is long enough.
    """
    try:
        noise_clip_len = int(sr * 0.5)
        if len(audio) > noise_clip_len * 2:
            noise_clip = audio[:noise_clip_len]
            reduced = nr.reduce_noise(y=audio, sr=sr, y_noise=noise_clip, prop_decrease=0.75)
        else:
            reduced = nr.reduce_noise(y=audio, sr=sr, prop_decrease=0.75)
        return reduced.astype(np.float32)
    except Exception as exc:
        logger.warning(f"Noise reduction failed ({exc}), returning original audio")
        return audio


def normalize(audio: np.ndarray, target_peak: float = 0.9) -> np.ndarray:
    """Peak-normalize audio so the loudest sample reaches target_peak."""
    peak = np.max(np.abs(audio))
    if peak < 1e-6:
        return audio  # silence — nothing to normalize
    return (audio / peak * target_peak).astype(np.float32)


def preprocess(audio: np.ndarray, orig_sr: int) -> tuple[np.ndarray, int, int]:
    """
    Full preprocessing pipeline.

    Returns:
        processed_audio (np.ndarray): clean, mono, 16 kHz float32 array
        sample_rate     (int):        always TARGET_SR
        n_channels      (int):        original channel count (for reporting)
    """
    n_channels = detect_channels(audio)
    logger.debug(f"Input: {n_channels}ch, {orig_sr}Hz, {len(audio) if audio.ndim==1 else audio.shape} samples")

    # 1. Convert to mono
    audio = to_mono(audio)

    # 2. Resample to 16 kHz
    audio = resample(audio, orig_sr, TARGET_SR)

    # 3. Noise reduction
    audio = reduce_noise(audio, TARGET_SR)

    # 4. Normalize
    audio = normalize(audio)

    logger.debug(f"Preprocessed → mono, {TARGET_SR}Hz, {len(audio)} samples")
    return audio, TARGET_SR, n_channels
