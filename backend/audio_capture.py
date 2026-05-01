"""
ECHO - Audio Capture
Captures microphone input using sounddevice and yields fixed-duration chunks.
"""

import threading
import queue
import numpy as np
import sounddevice as sd
from utils import get_logger

logger = get_logger("audio_capture")

DEFAULT_CHUNK_DURATION = 10.0   # seconds per chunk
DEFAULT_SAMPLE_RATE    = 44100  # capture at native rate; preprocessor resamples to 16 kHz
DEFAULT_BLOCKSIZE      = 1024   # frames per sounddevice callback


class AudioCapture:
    """
    Continuously captures microphone audio and exposes chunks via a queue.

    Usage:
        capture = AudioCapture(chunk_duration=1.5)
        capture.start()
        while True:
            chunk, sr, channels = capture.get_chunk()   # blocks until available
            ...
        capture.stop()
    """

    def __init__(
        self,
        chunk_duration: float = DEFAULT_CHUNK_DURATION,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        blocksize: int = DEFAULT_BLOCKSIZE,
        device=None,
    ):
        self.chunk_duration = chunk_duration
        self.sample_rate = sample_rate
        self.blocksize = blocksize
        self.device = device

        self._chunk_samples = int(sample_rate * chunk_duration)
        self._buffer: list[np.ndarray] = []
        self._buffer_len = 0
        self._queue: queue.Queue = queue.Queue(maxsize=20)
        self._stream: sd.InputStream | None = None
        self._running = False
        self._lock = threading.Lock()

        # Detect input device channels
        self._channels = self._detect_channels()

    # ── Channel detection ──────────────────────────────────────────────────────
    def _detect_channels(self) -> int:
        try:
            info = sd.query_devices(self.device, "input")
            ch = int(info["max_input_channels"])
            ch = min(ch, 2)  # cap at stereo
            logger.info(f"Detected {ch} input channel(s) on device: {info['name']}")
            return ch
        except Exception as exc:
            logger.warning(f"Could not detect channels ({exc}), defaulting to 1")
            return 1

    # ── sounddevice callback ───────────────────────────────────────────────────
    def _callback(self, indata: np.ndarray, frames: int, time_info, status):
        if status:
            logger.warning(f"sounddevice status: {status}")

        with self._lock:
            self._buffer.append(indata.copy())
            self._buffer_len += frames

            if self._buffer_len >= self._chunk_samples:
                chunk = np.concatenate(self._buffer, axis=0)[: self._chunk_samples]
                self._buffer = [np.concatenate(self._buffer, axis=0)[self._chunk_samples :]]
                self._buffer_len = max(0, self._buffer_len - self._chunk_samples)

                try:
                    self._queue.put_nowait((chunk, self.sample_rate, self._channels))
                except queue.Full:
                    logger.warning("Audio queue full — dropping chunk")

    # ── Public interface ───────────────────────────────────────────────────────
    def start(self):
        if self._running:
            return
        self._running = True
        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=self._channels,
            blocksize=self.blocksize,
            device=self.device,
            dtype="float32",
            callback=self._callback,
        )
        self._stream.start()
        logger.info(
            f"Audio capture started — {self._channels}ch @ {self.sample_rate}Hz, "
            f"{self.chunk_duration}s chunks"
        )

    def stop(self):
        self._running = False
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        logger.info("Audio capture stopped.")

    def get_chunk(self, timeout: float = 5.0):
        """
        Block until a chunk is available and return (audio, sample_rate, channels).
        Raises queue.Empty on timeout.
        """
        return self._queue.get(timeout=timeout)

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def queue_size(self) -> int:
        return self._queue.qsize()
