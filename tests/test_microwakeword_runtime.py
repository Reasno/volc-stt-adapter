"""Tests for the microWakeWord streaming runtime.

The tests deliberately avoid importing tflite-runtime or ai-edge-litert. They
supply fake interpreter and frontend factories to ``MicroWakeWordModel`` so the
sliding-window/streaming bookkeeping can be validated on any dev machine.
"""

from __future__ import annotations

import unittest
from pathlib import Path
from typing import Sequence

import numpy as np

from microwakeword_runtime import (
    MicroWakeWordModel,
    _FEATURE_HOP_SAMPLES,
    _FEATURE_SCALE,
)


class FakeFrontend:
    """Deterministic microfrontend: emits a rising feature vector per hop."""

    def __init__(self, warmup_hops: int = 0) -> None:
        # ``warmup_hops`` mirrors pymicro-features' ~30 ms warmup (2 hops of
        # missing output before the first feature).
        self.warmup_hops = warmup_hops
        self.calls_received = 0
        self.feature_index = 0
        self.reset_count = 0
        self.last_chunks: list[bytes] = []

    def process_samples(self, chunk: bytes):
        self.calls_received += 1
        self.last_chunks.append(bytes(chunk))
        result = _FrontendResult(samples_read=_FEATURE_HOP_SAMPLES, features=())
        if self.calls_received <= self.warmup_hops:
            return result
        # 40-dim feature vector; use a monotonically increasing scalar so
        # tests can prove which frames reached the interpreter.
        self.feature_index += 1
        feature = tuple(float(self.feature_index) for _ in range(40))
        return _FrontendResult(samples_read=_FEATURE_HOP_SAMPLES, features=feature)

    def reset(self) -> None:
        self.reset_count += 1
        self.feature_index = 0


class _FrontendResult:
    __slots__ = ("samples_read", "features")

    def __init__(self, samples_read: int, features: Sequence[float]) -> None:
        self.samples_read = samples_read
        self.features = features


class FakeInterpreter:
    """Duck-typed tflite interpreter that records invocations."""

    def __init__(
        self,
        *,
        feature_slices: int = 3,
        num_channels: int = 40,
        scores: Sequence[float] | None = None,
        dtype: np.dtype = np.float32,
    ) -> None:
        self._input_details = [
            {
                "index": 0,
                "shape": np.array([1, feature_slices, num_channels]),
                "dtype": dtype,
                "quantization_parameters": {"scales": [1.0], "zero_points": [0]},
            }
        ]
        self._output_details = [
            {
                "index": 100,
                "shape": np.array([1, 1]),
                "dtype": dtype,
                "quantization_parameters": {"scales": [1.0], "zero_points": [0]},
            }
        ]
        self._tensors: dict[int, np.ndarray] = {}
        self._scores = list(scores) if scores is not None else []
        self._score_cursor = 0
        self.allocate_calls = 0
        self.invocations: list[np.ndarray] = []
        self.set_tensor_history: list[tuple[int, np.ndarray]] = []

    def get_input_details(self):
        return self._input_details

    def get_output_details(self):
        return self._output_details

    def allocate_tensors(self):
        self.allocate_calls += 1

    def set_tensor(self, index: int, value: np.ndarray) -> None:
        self._tensors[index] = np.array(value, copy=True)
        self.set_tensor_history.append((index, np.array(value, copy=True)))

    def invoke(self) -> None:
        input_tensor = self._tensors[self._input_details[0]["index"]]
        self.invocations.append(input_tensor.copy())
        if self._score_cursor < len(self._scores):
            score = self._scores[self._score_cursor]
            self._score_cursor += 1
        else:
            score = 0.0
        self._tensors[self._output_details[0]["index"]] = np.array(
            [[score]], dtype=np.float32
        )

    def get_tensor(self, index: int) -> np.ndarray:
        return self._tensors[index]


def _pcm_from_hops(count: int) -> np.ndarray:
    """Return ``count`` * 10 ms of int16 zero samples."""

    return np.zeros(count * _FEATURE_HOP_SAMPLES, dtype=np.int16)


class MicroWakeWordModelTest(unittest.TestCase):
    def _build_model(
        self,
        *,
        scores: Sequence[float] | None = None,
        feature_slices: int = 3,
        sliding_window: int = 5,
        step_ms: int = 40,
        warmup_hops: int = 0,
        cutoff: float = 0.95,
    ):
        frontend = FakeFrontend(warmup_hops=warmup_hops)
        interpreter = FakeInterpreter(feature_slices=feature_slices, scores=scores)
        model = MicroWakeWordModel(
            Path("fake.tflite"),
            cutoff=cutoff,
            sliding_window=sliding_window,
            step_ms=step_ms,
            interpreter_factory=lambda: interpreter,
            frontend_factory=lambda: frontend,
        )
        return model, interpreter, frontend

    def test_predict_returns_wakeword_name_from_model_path(self):
        model, _, _ = self._build_model()
        result = model.predict(np.zeros(0, dtype=np.int16))
        self.assertEqual(list(result.keys()), ["fake"])
        self.assertEqual(result["fake"], 0.0)

    def test_invalid_step_ms_rejected(self):
        with self.assertRaises(ValueError):
            self._build_model(step_ms=25)  # not a multiple of 10
        with self.assertRaises(ValueError):
            self._build_model(step_ms=0)

    def test_input_gain_scales_pcm_before_feature_extraction(self):
        # Feed one hop of a low-amplitude constant PCM signal; the gain must
        # be applied before the microfrontend sees the bytes, and out-of-range
        # values must be clipped to int16 min/max instead of wrapping.
        frontend = FakeFrontend()
        interpreter = FakeInterpreter(feature_slices=1, scores=[0.0])
        model = MicroWakeWordModel(
            Path("fake.tflite"),
            cutoff=0.5,
            sliding_window=1,
            step_ms=10,
            input_gain=6.0,
            interpreter_factory=lambda: interpreter,
            frontend_factory=lambda: frontend,
        )
        # Values chosen so 100 * 6 fits (=600) but 6000 * 6 = 36000 must clip
        # to int16 max 32767.
        samples = np.array([100, -100, 6000, -6000] * (_FEATURE_HOP_SAMPLES // 4), dtype=np.int16)
        model.predict(samples)
        seen = np.frombuffer(frontend.last_chunks[-1], dtype=np.int16)
        self.assertEqual(int(seen[0]), 600)
        self.assertEqual(int(seen[1]), -600)
        self.assertEqual(int(seen[2]), 32767)
        self.assertEqual(int(seen[3]), -32768)

    def test_input_gain_default_is_identity(self):
        frontend = FakeFrontend()
        interpreter = FakeInterpreter(feature_slices=1, scores=[0.0])
        model = MicroWakeWordModel(
            Path("fake.tflite"),
            cutoff=0.5,
            sliding_window=1,
            step_ms=10,
            interpreter_factory=lambda: interpreter,
            frontend_factory=lambda: frontend,
        )
        samples = np.array([100, -100] * (_FEATURE_HOP_SAMPLES // 2), dtype=np.int16)
        model.predict(samples)
        seen = np.frombuffer(frontend.last_chunks[-1], dtype=np.int16)
        self.assertEqual(int(seen[0]), 100)
        self.assertEqual(int(seen[1]), -100)

    def test_input_gain_must_be_positive(self):
        with self.assertRaises(ValueError):
            MicroWakeWordModel(
                Path("fake.tflite"),
                cutoff=0.5,
                sliding_window=1,
                step_ms=10,
                input_gain=0.0,
                interpreter_factory=lambda: FakeInterpreter(feature_slices=1, scores=[0.0]),
                frontend_factory=lambda: FakeFrontend(),
            )

    def test_stride_matches_step_ms(self):
        # step_ms=40, feature_hop=10 → stride_features=4.
        model, interpreter, _ = self._build_model(
            scores=[0.1, 0.2, 0.3], feature_slices=3, step_ms=40
        )
        # Feed 80 ms of audio (8 hops). After the buffer fills 3 features,
        # inferences fire every 4 additional features:
        #   feature 3 → 1st invoke (buffer=3, since_last=3<4? actually we need
        #   4 accumulated since last predict). Detail: the first inference is
        #   triggered as soon as we have ``input_feature_slices`` features AND
        #   ``stride_features`` accumulated. Both conditions are equivalent for
        #   the first fire.
        # We ingest enough audio to fire 2 inferences and verify.
        model.predict(_pcm_from_hops(8))
        # We should have at most 2 invocations after 8 features:
        # features 4 (first) and 8 (second, +stride=4).
        self.assertEqual(len(interpreter.invocations), 2)

    def test_smoothed_score_hidden_until_sliding_window_full(self):
        # Feed a stream where every inference returns 1.0. Until the window
        # has ``sliding_window`` samples, the returned smoothed score MUST be
        # 0.0 — an anti-false-positive rule that matches the runtime docstring.
        model, interpreter, _ = self._build_model(
            scores=[1.0] * 10, feature_slices=3, sliding_window=5, step_ms=40
        )
        # stride=4 features (40 ms). First inference fires when the buffer
        # first reaches ``input_feature_slices`` AND ``stride_features``
        # accumulated → at feature #4. Subsequent inferences at features
        # 8, 12, 16, 20. Reaching a full sliding_window=5 needs 20 features.
        result = model.predict(_pcm_from_hops(19))
        self.assertEqual(len(interpreter.invocations), 4)
        self.assertEqual(result["fake"], 0.0)  # only 4/5 inferences ran
        # One more inference (add stride=4 hops → hop 20) fills the window.
        result = model.predict(_pcm_from_hops(1))
        self.assertEqual(len(interpreter.invocations), 5)
        self.assertAlmostEqual(result["fake"], 1.0)

    def test_smoothed_score_is_moving_average(self):
        model, _, _ = self._build_model(
            scores=[0.2, 0.4, 0.6, 0.8, 1.0, 0.0],
            feature_slices=3,
            sliding_window=5,
            step_ms=40,
        )
        # 5 inferences: buffer warmup + inferences at features 4,8,12,16,20.
        result = model.predict(_pcm_from_hops(20))
        expected = (0.2 + 0.4 + 0.6 + 0.8 + 1.0) / 5
        self.assertAlmostEqual(result["fake"], expected)
        # Sixth inference (feature 24) drops 0.2 and adds 0.0.
        result = model.predict(_pcm_from_hops(4))
        expected = (0.4 + 0.6 + 0.8 + 1.0 + 0.0) / 5
        self.assertAlmostEqual(result["fake"], expected)

    def test_reset_clears_streaming_state(self):
        model, interpreter, frontend = self._build_model(
            scores=[1.0] * 10, sliding_window=5, step_ms=40
        )
        model.predict(_pcm_from_hops(20))
        invocations_before = len(interpreter.invocations)
        # Reset must clear the sliding window and re-zero every input tensor.
        model.reset()
        self.assertEqual(frontend.reset_count, 1)
        # Feed zero samples: cannot fire since buffer is empty.
        result = model.predict(np.zeros(0, dtype=np.int16))
        self.assertEqual(result["fake"], 0.0)
        # No new invocations from empty predict.
        self.assertEqual(len(interpreter.invocations), invocations_before)

    def test_partial_pcm_is_buffered_between_predict_calls(self):
        # Feeding half a hop (80 samples = 5 ms) should not lose data; the
        # frontend fires on the next call when the running buffer reaches
        # one full hop.
        model, _, frontend = self._build_model()
        first = np.zeros(80, dtype=np.int16)
        # Second call adds another 80 samples so the accumulated buffer
        # reaches exactly one 160-sample hop (2 halves + no more).
        second = np.zeros(80, dtype=np.int16)
        model.predict(first)
        self.assertEqual(frontend.calls_received, 0)
        model.predict(second)
        # Combined 160 samples → exactly one hop consumed.
        self.assertEqual(frontend.calls_received, 1)

    def test_feature_vector_scaled_like_reference(self):
        # microwakeword.audio.audio_utils multiplies raw microfrontend outputs
        # by 1/25.6 when their dtype is uint16. Confirm we do the same.
        scores = [0.5]
        frontend = FakeFrontend()
        interpreter = FakeInterpreter(feature_slices=1, scores=scores)
        model = MicroWakeWordModel(
            Path("fake.tflite"),
            cutoff=0.5,
            sliding_window=1,
            step_ms=10,  # trigger on every single feature
            interpreter_factory=lambda: interpreter,
            frontend_factory=lambda: frontend,
        )
        model.predict(_pcm_from_hops(1))
        (window,) = interpreter.invocations
        # Feature vector was 1.0 → scaled to 0.0390625.
        expected = np.full((1, 1, 40), _FEATURE_SCALE, dtype=np.float32)
        np.testing.assert_allclose(window, expected)

    def test_quantized_model_roundtrips_through_int8(self):
        # Verify the int8 quantize/dequantize path exercises the reference
        # microwakeword formulas without crashing. We use a 1-slice model so
        # the arithmetic is easy to reason about.
        frontend = FakeFrontend()
        interpreter = FakeInterpreter(feature_slices=1, dtype=np.int8, scores=[10.0])
        model = MicroWakeWordModel(
            Path("fake.tflite"),
            cutoff=0.1,
            sliding_window=1,
            step_ms=10,
            interpreter_factory=lambda: interpreter,
            frontend_factory=lambda: frontend,
        )
        result = model.predict(_pcm_from_hops(1))
        self.assertGreaterEqual(result["fake"], 0.0)
        self.assertLessEqual(result["fake"], 1.0)
        # The tensor set on the interpreter must be int8.
        primary_index = interpreter.get_input_details()[0]["index"]
        for index, tensor in interpreter.set_tensor_history:
            if index == primary_index and tensor.shape == (1, 1, 40):
                self.assertEqual(tensor.dtype, np.int8)


class MicroWakeWordDetectorIntegrationTest(unittest.IsolatedAsyncioTestCase):
    """End-to-end: feed the microwakeword model through ``WakeWordDetector``.

    The detector calls ``predict()`` on each 80 ms frame; sustained high
    smoothed scores must produce exactly one rising-edge wake event, matching
    the openwakeword behavior guaranteed elsewhere in the test suite.
    """

    async def test_detector_fires_once_on_sustained_high_score(self):
        from wake_word import FRAME_BYTES, FRAME_SAMPLES, WakeWordDetector

        events = []

        async def on_wake(event):
            events.append(event)

        # Score list long enough to stay saturated for many detector frames.
        interpreter = FakeInterpreter(feature_slices=3, scores=[1.0] * 200)
        frontend = FakeFrontend()

        def factory():
            return MicroWakeWordModel(
                Path("fake.tflite"),
                cutoff=0.95,
                sliding_window=5,
                step_ms=40,
                interpreter_factory=lambda: interpreter,
                frontend_factory=lambda: frontend,
            )

        detector = WakeWordDetector(
            "fake.tflite",
            threshold=0.95,
            preroll_seconds=0.16,
            queue_frames=32,
            model_factory=factory,
            on_wake=on_wake,
        )
        await detector.start()
        # Feed 1 second of 80 ms frames → the sliding window fills well within
        # this budget and the rising edge fires exactly once.
        for _ in range(int(16000 / FRAME_SAMPLES)):
            detector.append(bytes(FRAME_BYTES))
        await detector._queue.join()
        self.assertEqual(len(events), 1)
        await detector.close()


if __name__ == "__main__":
    unittest.main()
