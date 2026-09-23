"""Per-connection asynchronous openWakeWord detector for PCM16/16 kHz/mono."""
from __future__ import annotations

import asyncio
import contextlib
import logging
import os
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Awaitable, Callable

import numpy as np

LOG = logging.getLogger("volc_stt_adapter.kws")
SAMPLE_RATE = 16_000
FRAME_SAMPLES = 1_280  # openWakeWord's fixed 80 ms frame
FRAME_BYTES = FRAME_SAMPLES * 2


@dataclass(frozen=True)
class WakeEvent:
    score: float
    sample_index: int
    timestamp_ms: float
    preroll_pcm: bytes
    detector_generation: int = 0


@dataclass(frozen=True)
class DetectorStats:
    queued_frames: int
    dropped_frames: int


@dataclass(frozen=True)
class _Frame:
    pcm: bytes
    end_sample: int
    generation: int


class WakeWordDetector:
    """One model, frame queue, and inference worker per client connection."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        threshold: float = 0.5,
        preroll_seconds: float = 1.5,
        queue_frames: int = 32,
        on_wake: Callable[[WakeEvent], Awaitable[None]] | None = None,
        model_factory=None,
    ) -> None:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be in [0, 1]")
        if preroll_seconds < 0 or queue_frames <= 0:
            raise ValueError("preroll_seconds must be non-negative and queue_frames positive")
        self.model_path = Path(model_path)
        self.threshold = threshold
        self.preroll_samples = int(preroll_seconds * SAMPLE_RATE)
        self.on_wake = on_wake
        self._model_factory = model_factory
        self._model = None
        self._queue: asyncio.Queue[_Frame] = asyncio.Queue(maxsize=queue_frames)
        self._worker: asyncio.Task[None] | None = None
        self._model_lock = asyncio.Lock()
        self._pending = bytearray()
        self._next_sample = 0
        self._generation = 0
        self._inference_preroll: deque[tuple[int, bytes]] = deque()
        self._inference_preroll_samples = 0
        self._last_inference_end_sample: int | None = None
        self._above_threshold = False
        self._dropped_frames = 0
        # Debug telemetry: track the peak KWS score over a rolling window and
        # emit one INFO line per window when the peak is non-trivial. This lets
        # us diagnose "the mic is on but no wake fired" without dumping every
        # frame's score. Env-configurable via WAKE_DEBUG_WINDOW_S /
        # WAKE_DEBUG_MIN_SCORE.
        try:
            self._debug_window_s = float(os.getenv("WAKE_DEBUG_WINDOW_S", "1.0"))
        except ValueError:
            self._debug_window_s = 1.0
        try:
            self._debug_min_score = float(os.getenv("WAKE_DEBUG_MIN_SCORE", "0.05"))
        except ValueError:
            self._debug_min_score = 0.05
        self._debug_window_frames = max(
            1, int(self._debug_window_s * SAMPLE_RATE / FRAME_SAMPLES)
        )
        self._debug_frames_seen = 0
        self._debug_window_peak = 0.0

    @property
    def stats(self) -> DetectorStats:
        return DetectorStats(self._queue.qsize(), self._dropped_frames)

    @property
    def sample_index(self) -> int:
        return self._next_sample

    @property
    def generation(self) -> int:
        return self._generation

    async def start(self) -> None:
        if self._worker is not None:
            return
        # Publish the model only after construction succeeds: callers can safely
        # downgrade the whole connection to off without a half-initialized worker.
        model = await asyncio.to_thread(self._build_model)
        self._model = model
        self._worker = asyncio.create_task(self._run(), name="openwakeword-inference")

    def _build_model(self):
        if self._model_factory is not None:
            return self._model_factory()

        runtime = os.getenv("KWS_RUNTIME", "openwakeword").strip().lower()
        if runtime == "microwakeword":
            # Lazy import so environments without tflite/pymicro-features can
            # still load this module when using the openwakeword backend.
            from microwakeword_runtime import build_microwakeword_model

            return build_microwakeword_model(self.model_path, cutoff=self.threshold)
        if runtime not in {"openwakeword", ""}:
            raise ValueError(
                f"Unsupported KWS_RUNTIME={runtime!r}; expected 'openwakeword' or 'microwakeword'"
            )

        from openwakeword.model import Model

        feature_dir = Path(__file__).resolve().with_name("models") / "openwakeword"
        return Model(
            wakeword_models=[str(self.model_path)],
            inference_framework="onnx",
            melspec_model_path=str(feature_dir / "melspectrogram.onnx"),
            embedding_model_path=str(feature_dir / "embedding_model.onnx"),
        )

    def append(self, pcm: bytes) -> int:
        """Split even a very large append into fixed frames without blocking.

        Returns the number of frames dropped because the bounded queue was full.
        The global sample index always advances, including dropped frames.
        """
        if len(pcm) % 2:
            raise ValueError("PCM16 byte length must be even")
        self._pending.extend(pcm)
        dropped_now = 0
        while len(self._pending) >= FRAME_BYTES:
            frame = bytes(self._pending[:FRAME_BYTES])
            del self._pending[:FRAME_BYTES]
            self._next_sample += FRAME_SAMPLES
            try:
                self._queue.put_nowait(_Frame(frame, self._next_sample, self._generation))
            except asyncio.QueueFull:
                self._dropped_frames += 1
                dropped_now += 1
        if dropped_now:
            LOG.warning(
                "KWS inference queue full; dropped %d frame(s), total_dropped=%d",
                dropped_now,
                self._dropped_frames,
            )
        return dropped_now

    async def _run(self) -> None:
        assert self._model is not None
        while True:
            frame = await self._queue.get()
            try:
                if frame.generation != self._generation:
                    continue
                gap = (
                    self._last_inference_end_sample is not None
                    and frame.end_sample != self._last_inference_end_sample + FRAME_SAMPLES
                )
                if gap:
                    self._above_threshold = False
                    if hasattr(self._model, "reset"):
                        async with self._model_lock:
                            await asyncio.to_thread(self._model.reset)
                self._remember(frame)
                samples = np.frombuffer(frame.pcm, dtype="<i2")
                async with self._model_lock:
                    scores = await asyncio.to_thread(self._model.predict, samples)
                if frame.generation != self._generation:
                    continue
                score = max((float(value) for value in scores.values()), default=0.0)
                # Sub-threshold telemetry: emit one line per rolling window
                # whenever the peak score in that window is non-trivial. This
                # is critical for tuning KWS_THRESHOLD from real data.
                if score > self._debug_window_peak:
                    self._debug_window_peak = score
                self._debug_frames_seen += 1
                if self._debug_frames_seen >= self._debug_window_frames:
                    if self._debug_window_peak >= self._debug_min_score:
                        LOG.info(
                            "KWS score peak: %.3f (window=%.1fs, threshold=%.3f)",
                            self._debug_window_peak,
                            self._debug_window_s,
                            self.threshold,
                        )
                    self._debug_frames_seen = 0
                    self._debug_window_peak = 0.0
                crossed_threshold = score >= self.threshold and not self._above_threshold
                self._above_threshold = score >= self.threshold
                if crossed_threshold and self.on_wake is not None:
                    event = WakeEvent(
                        score=score,
                        sample_index=frame.end_sample,
                        timestamp_ms=frame.end_sample * 1000.0 / SAMPLE_RATE,
                        preroll_pcm=self.preroll_pcm,
                        detector_generation=frame.generation,
                    )
                    await self.on_wake(event)
            finally:
                self._queue.task_done()

    def _remember(self, frame: _Frame) -> None:
        if (
            self._last_inference_end_sample is not None
            and frame.end_sample != self._last_inference_end_sample + FRAME_SAMPLES
        ):
            # Never fabricate contiguous preroll across frames dropped by the
            # bounded queue.
            self._inference_preroll.clear()
            self._inference_preroll_samples = 0
        self._last_inference_end_sample = frame.end_sample
        self._inference_preroll.append((FRAME_SAMPLES, frame.pcm))
        self._inference_preroll_samples += FRAME_SAMPLES
        while self._inference_preroll and self._inference_preroll_samples > self.preroll_samples:
            count, data = self._inference_preroll[0]
            excess = self._inference_preroll_samples - self.preroll_samples
            if excess >= count:
                self._inference_preroll.popleft()
                self._inference_preroll_samples -= count
            else:
                self._inference_preroll[0] = (count - excess, data[excess * 2 :])
                self._inference_preroll_samples -= excess

    @property
    def preroll_pcm(self) -> bytes:
        return b"".join(data for _, data in self._inference_preroll)

    async def reset(self) -> None:
        self._generation += 1
        self._pending.clear()
        self._next_sample = 0
        self._inference_preroll.clear()
        self._inference_preroll_samples = 0
        self._last_inference_end_sample = None
        self._above_threshold = False
        while True:
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            else:
                self._queue.task_done()
        if self._model is not None and hasattr(self._model, "reset"):
            async with self._model_lock:
                await asyncio.to_thread(self._model.reset)

    async def close(self) -> None:
        if self._worker is not None:
            self._worker.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._worker
            self._worker = None
        self._model = None
        await self.reset()
