from __future__ import annotations

import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from audio_gate import GateMode
from volc_stt_adapter import RealtimeAdapterConnection, VolcengineStream

from test_connection_gate import Sink, append_message, settings


class BlockingStartStream:
    instances = []

    def __init__(self, *args, timeline_origin_sample, **kwargs):
        self.item_id = args[2]
        self.generation = 1
        self.timeline_origin_sample = timeline_origin_sample
        self.start_entered = asyncio.Event()
        self.release_start = asyncio.Event()
        self.close_called = asyncio.Event()
        self.__class__.instances.append(self)

    async def start(self, initial_pcm):
        self.start_entered.set()
        await self.release_start.wait()

    async def close(self):
        self.close_called.set()
        self.release_start.set()


class RecordingStream:
    def __init__(self):
        self.generation = 1
        self.received = []
        self.closed = False

    async def send_audio(self, pcm):
        self.received.append(pcm)

    async def close(self):
        self.closed = True


class RecordingDetector:
    def __init__(self):
        self.received = []
        self.generation = 0

    def append(self, pcm):
        self.received.append(pcm)

    async def reset(self):
        self.generation += 1


class StreamLifecycleRegressionTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        BlockingStartStream.instances.clear()

    async def test_clear_closes_starting_stream_and_prevents_republish(self):
        connection = RealtimeAdapterConnection(Sink(), settings("off"))
        with patch("volc_stt_adapter.VolcengineStream", BlockingStartStream):
            start_task = asyncio.create_task(
                connection._start_stream(b"\x01\x00", timeline_origin_sample=0)
            )
            while not BlockingStartStream.instances:
                await asyncio.sleep(0)
            candidate = BlockingStartStream.instances[0]
            await candidate.start_entered.wait()
            await connection.clear_audio(emit_confirmation=False)
            await start_task

        self.assertTrue(candidate.close_called.is_set())
        self.assertIsNone(connection.stream)
        self.assertIsNone(connection._starting_stream)

    async def test_cancel_closes_starting_stream_before_propagating(self):
        connection = RealtimeAdapterConnection(Sink(), settings("off"))
        with patch("volc_stt_adapter.VolcengineStream", BlockingStartStream):
            start_task = asyncio.create_task(
                connection._start_stream(b"\x01\x00", timeline_origin_sample=0)
            )
            while not BlockingStartStream.instances:
                await asyncio.sleep(0)
            candidate = BlockingStartStream.instances[0]
            await candidate.start_entered.wait()
            start_task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await start_task

        self.assertTrue(candidate.close_called.is_set())
        self.assertIsNone(connection.stream)
        self.assertIsNone(connection._starting_stream)


class TimelineAccountingRegressionTest(unittest.IsolatedAsyncioTestCase):
    def make_stream(self):
        origins = []
        stream = VolcengineStream(
            SimpleNamespace(sample_rate=16000),
            "zh-CN",
            "test",
            AsyncMock(),
            lambda generation, origin: origins.append((generation, origin)),
            timeline_origin_sample=100,
        )
        return stream, origins

    async def test_mixed_send_results_advance_next_generation_origin(self):
        stream, origins = self.make_stream()
        stream._send = AsyncMock(side_effect=[True, False, True])
        await stream.send_audio(b"\x00\x00" * 3)
        await stream.send_audio(b"\x00\x00" * 5)
        await stream.send_audio(b"\x00\x00" * 2)
        self.assertEqual((stream.sent_samples, stream.unsent_samples), (5, 5))

        stream._begin_generation()
        self.assertEqual(stream.timeline_origin_sample, 110)
        self.assertEqual((stream.sent_samples, stream.unsent_samples), (0, 0))
        self.assertEqual(origins, [(1, 110)])

    async def test_reconnect_advances_origin_without_resending_initial_pcm(self):
        stream, origins = self.make_stream()
        initial_pcm = b"\x01\x00" * 4
        stream._connect_upstream = AsyncMock(side_effect=lambda: stream._begin_generation())
        stream._receive_loop = AsyncMock(return_value=None)
        stream._send = AsyncMock(return_value=True)

        # initial_pcm belongs exclusively to start(): one audio send and one
        # accounting entry after generation 1 is established.
        await stream.start(initial_pcm)
        await asyncio.sleep(0)
        self.assertEqual(stream._send.await_count, 1)
        self.assertEqual(stream.sent_samples, 4)
        self.assertEqual(origins, [(1, 100)])

        # Reconnect invokes only the no-argument handshake/connect path. It
        # advances origin by generation 1 accounting and sends no PCM itself.
        stream.RECONNECT_BACKOFF_S = (0,)
        with patch("volc_stt_adapter.asyncio.sleep", new=AsyncMock()):
            self.assertTrue(await stream._reconnect("test disconnect"))

        self.assertEqual(stream._connect_upstream.await_count, 2)
        self.assertEqual(stream._send.await_count, 1)
        self.assertEqual(stream.timeline_origin_sample, 104)
        self.assertEqual((stream.sent_samples, stream.unsent_samples), (0, 0))
        self.assertEqual(origins, [(1, 100), (2, 104)])

    async def test_failed_initial_pcm_is_unsent_and_advances_origin(self):
        stream, origins = self.make_stream()
        stream._connect_upstream = AsyncMock(side_effect=lambda: stream._begin_generation())
        stream._receive_loop = AsyncMock(return_value=None)
        stream._send = AsyncMock(return_value=False)
        await stream.start(b"\x01\x00" * 4)
        await asyncio.sleep(0)
        self.assertEqual((stream.sent_samples, stream.unsent_samples), (0, 4))

        stream._begin_generation()
        self.assertEqual(stream.timeline_origin_sample, 104)
        self.assertEqual(origins, [(1, 100), (2, 104)])


class PcmAlignmentRegressionTest(unittest.IsolatedAsyncioTestCase):
    async def test_samples_split_across_odd_appends_are_identical(self):
        connection = RealtimeAdapterConnection(Sink(), settings("shadow"))
        stream = RecordingStream()
        detector = RecordingDetector()
        connection.stream = stream
        connection.detector = detector
        original = b"\x11\x22\x33\x44"

        for chunk in (original[:1], original[1:3], b"", original[3:]):
            await connection.handle_audio_append(append_message(chunk))

        self.assertEqual(b"".join(stream.received), original)
        self.assertEqual(b"".join(detector.received), original)
        self.assertEqual(stream.received, detector.received)
        self.assertEqual(connection._pcm_leftover, b"")

    async def test_clear_discards_old_leftover(self):
        connection = RealtimeAdapterConnection(Sink(), settings("shadow"))
        old_stream = RecordingStream()
        detector = RecordingDetector()
        connection.stream = old_stream
        connection.detector = detector
        await connection.handle_audio_append(append_message(b"\xaa"))
        self.assertEqual(connection._pcm_leftover, b"\xaa")

        await connection.clear_audio(emit_confirmation=False)
        new_stream = RecordingStream()
        connection.stream = new_stream
        await connection.handle_audio_append(append_message(b"\xbb\xcc"))

        self.assertTrue(old_stream.closed)
        self.assertEqual(new_stream.received, [b"\xbb\xcc"])
        self.assertEqual(detector.received, [b"\xbb\xcc"])
