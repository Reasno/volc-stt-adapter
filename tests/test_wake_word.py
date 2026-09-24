from __future__ import annotations

import asyncio
import unittest

import numpy as np

from wake_word import FRAME_BYTES, FRAME_SAMPLES, WakeWordDetector


class FakeModel:
    def __init__(self, score=0.9, scores=None):
        self.score = score
        self.scores = iter(scores) if scores is not None else None
        self.frames = []
        self.reset_count = 0
    def predict(self, samples):
        self.frames.append(samples.copy())
        score = next(self.scores) if self.scores is not None else self.score
        return {"reachy": score}
    def reset(self): self.reset_count += 1


class WakeWordDetectorTest(unittest.IsolatedAsyncioTestCase):
    async def test_large_append_is_split_and_preroll_is_sample_bounded(self):
        model = FakeModel(scores=[0.0, 0.0, 0.0, 0.0, 0.9])
        events = []
        async def on_wake(event):
            events.append(event)
        detector = WakeWordDetector("model.onnx", threshold=0.5, preroll_seconds=0.16,
                                    queue_frames=10, model_factory=lambda: model,
                                    on_wake=on_wake)
        await detector.start()
        pcm = b"".join((np.full(FRAME_SAMPLES, i, dtype="<i2").tobytes() for i in range(5)))
        self.assertEqual(detector.append(pcm), 0)
        await detector._queue.join()
        self.assertEqual(len(model.frames), 5)
        self.assertEqual([e.sample_index for e in events], [6400])
        self.assertEqual(len(events[0].preroll_pcm), 2 * FRAME_BYTES)
        self.assertEqual(np.frombuffer(events[0].preroll_pcm, dtype="<i2")[0], 3)
        await detector.close()

    async def test_queue_is_bounded_and_reports_drops(self):
        detector = WakeWordDetector("model.onnx", queue_frames=2, model_factory=FakeModel)
        # Intentionally append before start so no worker drains the queue.
        dropped = detector.append(bytes(FRAME_BYTES * 5))
        self.assertEqual(dropped, 3)
        self.assertEqual(detector.stats.queued_frames, 2)
        self.assertEqual(detector.stats.dropped_frames, 3)
        await detector.close()

    async def test_sustained_high_score_emits_one_rising_edge(self):
        events = []
        async def on_wake(event):
            events.append(event)
        detector = WakeWordDetector(
            "model.onnx", queue_frames=4, model_factory=FakeModel, on_wake=on_wake
        )
        await detector.start()
        detector.append(bytes(FRAME_BYTES * 3))
        await detector._queue.join()
        self.assertEqual([event.sample_index for event in events], [FRAME_SAMPLES])
        await detector.close()

    async def test_logs_every_inferred_frame_score_at_debug(self):
        detector = WakeWordDetector(
            "model.onnx",
            queue_frames=4,
            model_factory=lambda: FakeModel(scores=[0.0, 0.0123456, 0.9]),
        )
        await detector.start()
        with self.assertLogs("volc_stt_adapter.kws", level="DEBUG") as captured:
            detector.append(bytes(FRAME_BYTES * 3))
            await detector._queue.join()
        score_lines = [line for line in captured.output if "KWS score:" in line]
        self.assertEqual(len(score_lines), 3)
        self.assertIn("0.000000 amp_peak=0 sample_index=1280 threshold=0.500", score_lines[0])
        self.assertIn("0.012346 amp_peak=0 sample_index=2560 threshold=0.500", score_lines[1])
        self.assertIn("0.900000 amp_peak=0 sample_index=3840 threshold=0.500", score_lines[2])
        await detector.close()

    async def test_connections_do_not_share_model_state(self):
        models = []
        def factory():
            model = FakeModel(); models.append(model); return model
        first = WakeWordDetector("model.onnx", model_factory=factory)
        second = WakeWordDetector("model.onnx", model_factory=factory)
        await first.start(); await second.start()
        first.append(bytes(FRAME_BYTES)); second.append(bytes(FRAME_BYTES))
        await first._queue.join(); await second._queue.join()
        self.assertEqual(len(models), 2)
        self.assertIsNot(models[0], models[1])
        self.assertEqual([len(model.frames) for model in models], [1, 1])
        await first.close(); await second.close()


if __name__ == "__main__":
    unittest.main()
