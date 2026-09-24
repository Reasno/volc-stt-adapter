"""Tests for `_downstream_loop`'s frame parsing tolerance.

Regression coverage for the 45000081 no-packet Volcengine stream death:
Reachy's OpenAI-realtime SDK can send multiple JSON events back-to-back in a
single WebSocket text frame (e.g. `conversation.item.create` followed by
`response.create`). Naive `json.loads` on such a frame raises `Extra data`
and, if the adapter bounces an "Invalid JSON" error every time, Reachy's SDK
output queue backs up so severely that its microphone uploader stops, killing
the upstream ASR stream.

The fix in `_downstream_loop` parses each frame iteratively with
`JSONDecoder.raw_decode`. These tests lock in that behavior.
"""
from __future__ import annotations

import json
import unittest
from types import SimpleNamespace

from audio_gate import GateMode
from volc_stt_adapter import RealtimeAdapterConnection


class UpstreamSink:
    def __init__(self) -> None:
        self.messages: list[dict] = []

    async def send(self, raw: str) -> None:
        self.messages.append(json.loads(raw))


class FakeDownstreamWebsocket:
    """Serves as both the incoming iterator (for frames) and the outgoing
    sender (for `emit()`) — mirroring the real `websockets.server` object
    the adapter holds in `self.websocket`.
    """

    def __init__(self, frames: list) -> None:
        self._frames = list(frames)
        self.sent_raw: list[str] = []
        self.sent: list[dict] = []

    def __aiter__(self):
        return self._iter()

    async def _iter(self):
        for frame in self._frames:
            yield frame

    async def send(self, raw: str) -> None:
        self.sent_raw.append(raw)
        try:
            self.sent.append(json.loads(raw))
        except json.JSONDecodeError:
            # Shouldn't happen; adapter always sends JSON.
            self.sent.append({"__unparseable__": raw})


def _settings() -> SimpleNamespace:
    return SimpleNamespace(
        kws_mode=GateMode("enforce"),
        kws_trigger_timeout_seconds=3,
        kws_speaker_window_seconds=30,
        kws_match_tolerance_ms=100,
        keyword_gate_enabled=True,
        keyword_gate_prefix_words=(),
        speaker_context_enabled=True,
    )


class DownstreamLoopFrameParsingTest(unittest.IsolatedAsyncioTestCase):
    async def test_single_event_frame_is_handled_and_forwarded(self) -> None:
        upstream = UpstreamSink()
        frame = json.dumps({"type": "session.update", "session": {"language": "zh-CN"}})
        ws = FakeDownstreamWebsocket([frame])
        connection = RealtimeAdapterConnection(ws, _settings())
        connection.upstream = upstream

        await connection._downstream_loop()

        # session.update flows upstream; no error events bounce back.
        self.assertEqual(len(upstream.messages), 1)
        self.assertEqual(upstream.messages[0]["type"], "session.update")
        error_msgs = [m for m in ws.sent if m.get("type") == "error"]
        self.assertEqual(error_msgs, [])

    async def test_concatenated_events_in_one_frame_are_all_handled(self) -> None:
        """The regression: two events in one WS text frame must both be
        dispatched, and no `Invalid JSON` error may be emitted downstream."""
        upstream = UpstreamSink()
        event_a = json.dumps(
            {
                "type": "conversation.item.create",
                "item": {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "hello"}],
                },
            }
        )
        event_b = json.dumps(
            {"type": "response.create", "event_id": "evt_test"}
        )
        # No delimiter, no whitespace — exactly what the Reachy SDK sometimes
        # produces when two send() calls happen back-to-back inside the same
        # frame boundary.
        ws = FakeDownstreamWebsocket([event_a + event_b])
        connection = RealtimeAdapterConnection(ws, _settings())
        connection.upstream = upstream

        await connection._downstream_loop()

        upstream_types = [m.get("type") for m in upstream.messages]
        self.assertEqual(
            upstream_types,
            ["conversation.item.create", "response.create"],
        )
        # Absolutely no `Invalid JSON` error bounced back.
        error_msgs = [m for m in ws.sent if m.get("type") == "error"]
        self.assertEqual(error_msgs, [])

    async def test_concatenated_events_with_whitespace_are_all_handled(self) -> None:
        upstream = UpstreamSink()
        event_a = json.dumps({"type": "session.update", "session": {}})
        event_b = json.dumps({"type": "response.create"})
        ws = FakeDownstreamWebsocket([event_a + "\n" + event_b])
        connection = RealtimeAdapterConnection(ws, _settings())
        connection.upstream = upstream

        await connection._downstream_loop()

        self.assertEqual(
            [m.get("type") for m in upstream.messages],
            ["session.update", "response.create"],
        )
        error_msgs = [m for m in ws.sent if m.get("type") == "error"]
        self.assertEqual(error_msgs, [])

    async def test_truly_malformed_frame_still_errors(self) -> None:
        ws = FakeDownstreamWebsocket(["not json at all"])
        connection = RealtimeAdapterConnection(ws, _settings())

        await connection._downstream_loop()

        error_msgs = [m for m in ws.sent if m.get("type") == "error"]
        self.assertEqual(len(error_msgs), 1)
        self.assertEqual(error_msgs[0]["error"]["message"], "Invalid JSON")

    async def test_partially_malformed_frame_errors_after_valid_prefix(self) -> None:
        """A valid event followed by garbage: the valid one is handled, then
        exactly one Invalid JSON error is emitted for the trailing garbage."""
        upstream = UpstreamSink()
        good = json.dumps({"type": "session.update", "session": {}})
        ws = FakeDownstreamWebsocket([good + "not-json-tail"])
        connection = RealtimeAdapterConnection(ws, _settings())
        connection.upstream = upstream

        await connection._downstream_loop()

        self.assertEqual(
            [m.get("type") for m in upstream.messages], ["session.update"]
        )
        error_msgs = [m for m in ws.sent if m.get("type") == "error"]
        self.assertEqual(len(error_msgs), 1)


if __name__ == "__main__":
    unittest.main()
