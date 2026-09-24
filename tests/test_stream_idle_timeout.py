"""Feature 1: Stream idle timeout (30s post-KWS conversation window).

Verifies that:
* `_arm_stream_idle()` seeds the deadline from `stream_idle_timeout_seconds`
  and starts the background watchdog.
* `_refresh_stream_idle()` bumps the deadline forward on activity.
* An `_on_native_utterance` call refreshes the idle deadline.
* An upstream `response.done` event refreshes the idle deadline.
* `STREAM_IDLE_TIMEOUT_SECONDS=0` disables the mechanism entirely.
"""
from __future__ import annotations

import asyncio
import json
import time
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock

from audio_gate import GateMode, Utterance
from volc_stt_adapter import RealtimeAdapterConnection


def _settings(idle_timeout: float = 30.0) -> SimpleNamespace:
    return SimpleNamespace(
        kws_mode=GateMode("enforce"),
        kws_trigger_timeout_seconds=3,
        kws_speaker_window_seconds=30,
        kws_match_tolerance_ms=100,
        stop_talking_words=(),
        speaker_context_enabled=False,
        speaker_context_window_seconds=30,
        stream_idle_timeout_seconds=idle_timeout,
    )


class _FakeVolcStream:
    """Stand-in for VolcengineStream that exposes the minimum surface needed
    by the stream-idle refresh paths."""

    def __init__(self):
        self.latest_speaker_context = None
        self.armed = False

    def speaker_context_if_fresh(self):
        return None

    def arm_speaker_context_expiration(self):
        self.armed = True


class StreamIdleRefreshTest(unittest.IsolatedAsyncioTestCase):
    async def _make_connection(self, timeout: float = 30.0):
        conn = RealtimeAdapterConnection(SimpleNamespace(), _settings(timeout))
        conn._send_upstream = AsyncMock()  # type: ignore[attr-defined]
        conn.stream = _FakeVolcStream()  # type: ignore[assignment]
        return conn

    async def test_refresh_updates_deadline(self):
        conn = await self._make_connection(30.0)
        # Simulate "stream is running": mark starting flag so refresh is armed
        # even without a real stream.
        conn._stream_starting = True
        conn._refresh_stream_idle()
        deadline_1 = conn._stream_idle_deadline
        self.assertGreater(deadline_1, time.monotonic())
        await asyncio.sleep(0.05)
        conn._refresh_stream_idle()
        deadline_2 = conn._stream_idle_deadline
        self.assertGreater(deadline_2, deadline_1)

    async def test_zero_timeout_disables_refresh(self):
        conn = await self._make_connection(0.0)
        conn._stream_starting = True
        conn._refresh_stream_idle()
        self.assertEqual(conn._stream_idle_deadline, 0.0)

    async def test_no_stream_disables_refresh(self):
        conn = await self._make_connection(30.0)
        conn.stream = None
        conn._stream_starting = False
        conn._refresh_stream_idle()
        self.assertEqual(conn._stream_idle_deadline, 0.0)

    async def test_on_native_utterance_refreshes_idle(self):
        conn = await self._make_connection(30.0)
        conn._stream_starting = True
        before = time.monotonic()
        # Empty transcript short-circuits BEFORE refresh; feed a real one.
        await conn._on_native_utterance(
            Utterance("你好", "speaker-a", 1, 100, 400)
        )
        # The refresh must fire even if the text gate later rejects the line.
        self.assertGreaterEqual(conn._stream_idle_deadline, before + 29.0)

    async def test_upstream_response_done_refreshes_idle(self):
        """Feed a fake `response.done` through `_upstream_loop`'s event handling.

        We drive it by directly invoking the branch: setup upstream, prime a
        message pending, then use the shortest-path — pump one event through
        the loop by patching upstream to a single-event iterator.
        """
        conn = await self._make_connection(30.0)
        conn._stream_starting = True
        # Pre-seed a stale (near-expired) deadline to prove response.done
        # pushes it forward.
        conn._stream_idle_deadline = time.monotonic() + 0.01

        class _OneShotUpstream:
            def __init__(self, frames):
                self._frames = list(frames)

            def __aiter__(self):
                return self

            async def __anext__(self):
                if not self._frames:
                    raise StopAsyncIteration
                return self._frames.pop(0)

        conn.upstream = _OneShotUpstream(
            [json.dumps({"type": "response.done", "response": {}})]
        )
        # Downstream sink swallows the forwarded frame.
        sent = []

        class _Downstream:
            async def send(self, raw):
                sent.append(raw)

        conn.websocket = _Downstream()
        await conn._upstream_loop()
        self.assertGreater(
            conn._stream_idle_deadline, time.monotonic() + 25.0
        )

    async def test_arm_stream_idle_starts_watchdog_task(self):
        conn = await self._make_connection(30.0)
        conn._arm_stream_idle()
        self.assertIsNotNone(conn._stream_watchdog_task)
        self.assertFalse(conn._stream_watchdog_task.done())
        conn._cancel_stream_idle()
        # Cancelling should release the task.
        await asyncio.sleep(0)
        self.assertIsNone(conn._stream_watchdog_task)

    async def test_zero_timeout_does_not_start_watchdog(self):
        conn = await self._make_connection(0.0)
        conn._arm_stream_idle()
        self.assertIsNone(conn._stream_watchdog_task)


if __name__ == "__main__":
    unittest.main()
