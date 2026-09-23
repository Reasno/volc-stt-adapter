"""microWakeWord streaming inference runtime.

Provides a drop-in replacement for the openWakeWord model used by
``WakeWordDetector``. Uses ``pymicro-features`` for spectrogram extraction and
``tflite-runtime`` for inference. The public surface matches openWakeWord's
``Model`` closely enough for the async detector to remain agnostic:

* ``predict(samples: np.ndarray) -> dict[str, float]`` — returns the current
  smoothed wake probability keyed by the wakeword name. ``samples`` is a
  little-endian int16 mono 16 kHz PCM chunk of arbitrary length.
* ``reset() -> None`` — clears the feature buffer, sliding window, and any
  streaming state so the next ``predict`` call starts fresh.

The runtime is intentionally cheap to construct in tests. Both the tflite
interpreter and the microfrontend can be injected through factories so the
tests can validate the streaming/sliding-window bookkeeping without pulling
either native dependency in.
"""

from __future__ import annotations

import logging
import os
from collections import deque
from pathlib import Path
from typing import Any, Callable, Deque, Iterable, Sequence

import numpy as np

LOG = logging.getLogger("volc_stt_adapter.kws.microwakeword")

# pymicro-features consumes samples in fixed 160-sample (10 ms @ 16 kHz)
# chunks; each processed chunk may emit one 40-dim log-mel feature vector
# after a ~30 ms warm-up window is filled.
_FEATURE_HOP_SAMPLES = 160
_FEATURE_HOP_BYTES = _FEATURE_HOP_SAMPLES * 2  # int16
_FEATURE_MS_PER_HOP = 10  # ms

# Match microwakeword's uint16 → float32 rescaling. pymicro-features emits raw
# int16-ish features (see ``audio_utils.generate_features_for_clip``); the
# canonical scaling divides by 256 so features land in ~[0, 256).
_FEATURE_SCALE = 0.0390625  # 1 / 25.6, matches microWakeWord's reference


class MicroWakeWordModel:
    """tflite-runtime backed microWakeWord detector.

    Args:
        tflite_model_path: Path to the microWakeWord ``.tflite`` model file.
        wakeword_name: Name reported in ``predict`` output. Defaults to the
            tflite file stem.
        cutoff: Probability threshold above which a smoothed score is
            considered a wake. This is intentionally identical to the value
            surfaced through ``KWS_THRESHOLD``: ``WakeWordDetector`` compares
            the returned smoothed score against its own ``threshold``, so the
            cutoff is a *documented* expectation rather than an internal gate.
        sliding_window: Number of most-recent per-inference probabilities to
            average before returning a smoothed score. Matches
            microWakeWord/ESPHome's ``sliding_window_average``.
        step_ms: Effective hop between inference invocations (ms). Features
            are always produced every 10 ms by pymicro-features; the model is
            invoked whenever ``step_ms // 10`` new features have accumulated
            and the buffer holds at least ``input_feature_slices`` features.
            ``40`` matches the reference "40 ms step" streaming setup.
        interpreter_factory: Optional callable returning a duck-typed tflite
            interpreter (``set_tensor``/``invoke``/``get_tensor`` /
            ``allocate_tensors`` /``get_input_details``/``get_output_details``).
            Used by tests to avoid the tflite-runtime dependency.
        frontend_factory: Optional callable returning a duck-typed
            microfrontend (``process_samples`` /``reset``). Used by tests.
    """

    def __init__(
        self,
        tflite_model_path: str | Path,
        *,
        wakeword_name: str | None = None,
        cutoff: float = 0.95,
        sliding_window: int = 5,
        step_ms: int = 40,
        interpreter_factory: Callable[[], Any] | None = None,
        frontend_factory: Callable[[], Any] | None = None,
    ) -> None:
        if sliding_window <= 0:
            raise ValueError("sliding_window must be a positive integer")
        if step_ms <= 0 or step_ms % _FEATURE_MS_PER_HOP != 0:
            raise ValueError(
                f"step_ms must be a positive multiple of {_FEATURE_MS_PER_HOP} ms"
            )
        if not 0.0 <= cutoff <= 1.0:
            raise ValueError("cutoff must be in [0, 1]")

        self.model_path = Path(tflite_model_path)
        self.wakeword_name = wakeword_name or self.model_path.stem
        self.cutoff = cutoff
        self.sliding_window_size = sliding_window
        self.step_ms = step_ms
        self.stride_features = step_ms // _FEATURE_MS_PER_HOP

        self._frontend_factory = frontend_factory or self._default_frontend
        self._interpreter_factory = interpreter_factory or (
            lambda: self._default_interpreter(self.model_path)
        )

        self._frontend = self._frontend_factory()
        self._interpreter = self._interpreter_factory()

        # Populate interpreter I/O metadata following microwakeword's
        # reference Model.__init__ (zero-initialize every input tensor, quantize
        # aware).
        self._input_details = list(self._interpreter.get_input_details())
        self._output_details = list(self._interpreter.get_output_details())
        if not self._input_details or not self._output_details:
            raise ValueError("tflite model exposes no input/output tensors")
        primary_shape = tuple(self._input_details[0]["shape"])
        if len(primary_shape) < 2:
            raise ValueError(
                f"unexpected primary input tensor shape {primary_shape}; expected"
                " at least [batch, num_features] rank-2 tensor"
            )
        # microWakeWord input tensors are shaped [1, feature_slices, 40]. We
        # track the slice count (time dim) and permit any feature dim so we
        # can support models trained with a different num_channels if needed.
        self.input_feature_slices = int(primary_shape[1])
        self._is_quantized = self._input_details[0]["dtype"] == np.int8
        self._interpreter.allocate_tensors()
        self._zero_all_inputs()

        # Streaming state.
        self._pcm_pending = bytearray()
        # Feature buffer: only ever holds up to ``input_feature_slices``
        # entries (the current inference window). Older features are evicted
        # to keep memory bounded.
        self._feature_buffer: Deque[np.ndarray] = deque(
            maxlen=self.input_feature_slices
        )
        # Number of features consumed towards the next inference stride.
        self._features_since_last_predict = 0
        # Sliding window of raw per-invocation probabilities.
        self._score_history: Deque[float] = deque(maxlen=self.sliding_window_size)

    # ------------------------------------------------------------------
    # Default factories — lazy imports so tests never need the native deps.
    # ------------------------------------------------------------------
    @staticmethod
    def _default_frontend() -> Any:
        from pymicro_features import MicroFrontend

        return MicroFrontend()

    @staticmethod
    def _default_interpreter(model_path: Path) -> Any:
        try:
            from tflite_runtime.interpreter import Interpreter  # type: ignore
        except ImportError:  # pragma: no cover - covered on target device only
            try:
                from ai_edge_litert.interpreter import Interpreter  # type: ignore
            except ImportError as exc:  # pragma: no cover - dev machines only
                raise RuntimeError(
                    "microwakeword runtime requires tflite-runtime (or "
                    "ai-edge-litert). Install one on the target device."
                ) from exc
        return Interpreter(model_path=str(model_path))

    # ------------------------------------------------------------------
    # Public API — matches openwakeword.Model surface used by WakeWordDetector.
    # ------------------------------------------------------------------
    def predict(self, samples: np.ndarray) -> dict[str, float]:
        """Advance the streaming pipeline by ``samples`` and return the smoothed
        wake probability.

        The returned score is the arithmetic mean of the most recent
        ``sliding_window`` per-invocation probabilities. Before the sliding
        window has been filled, missing slots contribute ``0.0`` so the smoothed
        score is deliberately conservative and cannot fire prematurely from a
        single high inference output.
        """

        if samples.dtype != np.int16:
            samples = samples.astype(np.int16, copy=False)
        pcm = samples.tobytes()
        if pcm:
            self._pcm_pending.extend(pcm)
            self._drain_pcm_to_features()
        return {self.wakeword_name: self._current_score()}

    def reset(self) -> None:
        """Reset streaming state so the next ``predict`` starts fresh.

        Clears the PCM/feature buffers, the sliding window, and re-zeros every
        interpreter input tensor. The frontend is reset when it exposes a
        ``reset`` method (pymicro-features does).
        """

        self._pcm_pending.clear()
        self._feature_buffer.clear()
        self._features_since_last_predict = 0
        self._score_history.clear()
        reset_fn = getattr(self._frontend, "reset", None)
        if callable(reset_fn):
            reset_fn()
        self._zero_all_inputs()

    # ------------------------------------------------------------------
    # Internal helpers.
    # ------------------------------------------------------------------
    def _drain_pcm_to_features(self) -> None:
        """Feed pending PCM through the microfrontend, triggering inferences
        whenever the stride threshold is crossed."""

        # pymicro-features enforces a minimum call size of 160 samples
        # (320 bytes). We deliberately feed exactly one hop per call so
        # ``samples_read`` stays predictable and the loop always makes forward
        # progress.
        while len(self._pcm_pending) >= _FEATURE_HOP_BYTES:
            chunk = bytes(self._pcm_pending[:_FEATURE_HOP_BYTES])
            del self._pcm_pending[:_FEATURE_HOP_BYTES]
            result = self._frontend.process_samples(chunk)
            samples_read = getattr(result, "samples_read", _FEATURE_HOP_SAMPLES)
            if samples_read != _FEATURE_HOP_SAMPLES:
                # The frontend is supposed to consume the whole 10 ms hop when
                # given exactly one. If it consumed less we would drift; log
                # once so the operator can catch a version regression.
                LOG.debug(
                    "microfrontend consumed %d samples, expected %d",
                    samples_read,
                    _FEATURE_HOP_SAMPLES,
                )
            features = getattr(result, "features", None)
            if not features:
                continue
            feature_vector = self._prepare_feature_vector(features)
            self._feature_buffer.append(feature_vector)
            self._features_since_last_predict += 1
            self._maybe_invoke()

    def _prepare_feature_vector(self, features: Sequence[float]) -> np.ndarray:
        """Scale raw microfrontend features to the float32 form the reference
        microWakeWord ``Model`` uses (see ``predict_spectrogram``)."""

        arr = np.asarray(features, dtype=np.float32)
        arr *= _FEATURE_SCALE
        return arr

    def _maybe_invoke(self) -> None:
        # Need at least ``input_feature_slices`` features buffered and enough
        # newly accumulated ones to satisfy the requested stride. Once we hit
        # both, invoke and reset the stride counter (features remain in the
        # sliding buffer so the next inference sees an overlapping window).
        if len(self._feature_buffer) < self.input_feature_slices:
            return
        if self._features_since_last_predict < self.stride_features:
            return
        self._features_since_last_predict = 0

        window = np.stack(tuple(self._feature_buffer), axis=0)
        assert window.shape[0] == self.input_feature_slices, window.shape
        score = self._invoke_interpreter(window)
        # ``sliding_window`` is a rolling arithmetic mean, matching ESPHome's
        # `sliding_window_average` mode used with microwakeword.
        self._score_history.append(score)

    def _invoke_interpreter(self, window: np.ndarray) -> float:
        primary = self._input_details[0]
        tensor_shape = tuple(primary["shape"])
        payload = window.reshape(tensor_shape)
        if self._is_quantized:
            payload = self._quantize(payload, primary)
        self._interpreter.set_tensor(primary["index"], payload)
        self._interpreter.invoke()
        raw = self._interpreter.get_tensor(self._output_details[0]["index"])
        raw_value = float(np.asarray(raw).flatten()[0])
        if self._is_quantized:
            raw_value = self._dequantize(raw_value, self._output_details[0])
        # Some quantized models emit scores in [0, 255]; clamp to [0, 1] so
        # sliding_window smoothing stays interpretable.
        return max(0.0, min(1.0, raw_value))

    def _quantize(self, data: np.ndarray, details: dict) -> np.ndarray:
        params = details.get("quantization_parameters", {})
        scales = params.get("scales", [1.0])
        zero_points = params.get("zero_points", [0])
        scale = float(scales[0]) if len(scales) else 1.0
        zero_point = int(zero_points[0]) if len(zero_points) else 0
        quantized = np.round(data / scale + zero_point)
        info = np.iinfo(details["dtype"])
        quantized = np.clip(quantized, info.min, info.max)
        return quantized.astype(details["dtype"])

    def _dequantize(self, value: float, details: dict) -> float:
        # Matches microWakeWord's reference dequantize_output_data. It assumes
        # a fixed uint8-style scale of 1/255 with the tensor's own zero point.
        params = details.get("quantization_parameters", {})
        zero_points = params.get("zero_points", [0])
        zero_point = int(zero_points[0]) if len(zero_points) else 0
        return (value - zero_point) / 255.0

    def _zero_all_inputs(self) -> None:
        for detail in self._input_details:
            shape = tuple(detail["shape"])
            dtype = detail["dtype"]
            self._interpreter.set_tensor(detail["index"], np.zeros(shape, dtype=dtype))

    def _current_score(self) -> float:
        if not self._score_history:
            return 0.0
        # The mean is taken over the *current* history length rather than a
        # zero-padded sliding window: pre-warmup detection is intentionally
        # impossible because ``_score_history`` cannot exceed
        # ``sliding_window_size`` entries and we would rather stay silent than
        # gamble on a partial window. Once the window is full the mean tracks
        # ESPHome's ``sliding_window_average`` exactly.
        if len(self._score_history) < self.sliding_window_size:
            return 0.0
        return float(sum(self._score_history) / len(self._score_history))


def _bool_env(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


def build_microwakeword_model(
    model_path: str | Path,
    *,
    cutoff: float,
) -> MicroWakeWordModel:
    """Factory used by ``WakeWordDetector`` when ``KWS_RUNTIME=microwakeword``.

    Wire-up centralized here so environment variables don't leak into the
    async detector.
    """

    try:
        sliding_window = int(os.getenv("KWS_MICROWAKEWORD_SLIDING_WINDOW", "5"))
    except ValueError as exc:
        raise RuntimeError("KWS_MICROWAKEWORD_SLIDING_WINDOW must be an integer") from exc
    try:
        step_ms = int(os.getenv("KWS_MICROWAKEWORD_STEP_MS", "40"))
    except ValueError as exc:
        raise RuntimeError("KWS_MICROWAKEWORD_STEP_MS must be an integer") from exc
    wakeword_name = os.getenv("KWS_WAKEWORD_NAME", "").strip() or None
    return MicroWakeWordModel(
        model_path,
        wakeword_name=wakeword_name,
        cutoff=cutoff,
        sliding_window=sliding_window,
        step_ms=step_ms,
    )
