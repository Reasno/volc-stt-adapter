"""Tests for ASR speaker-perception context injection into session prompt.

Covers the raw pass-through path: Volcengine ASR sends `additions` with
`speaker_id` / `gender` / `age` / `emotion` on definite utterances; the
adapter records the latest snapshot on `VolcengineStream.latest_speaker_context`
and, on every `session.update`, replaces (never accumulates) a single
`[Speaker context] ...` line in the outgoing `session.instructions`.

No family-member mapping or thresholding — the LLM sees the raw fields.
"""
from __future__ import annotations

import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock

from audio_gate import GateMode
from volc_stt_adapter import RealtimeAdapterConnection, VolcengineStream


def _settings(speaker_context_enabled: bool = True) -> SimpleNamespace:
    return SimpleNamespace(
        kws_mode=GateMode("enforce"),
        kws_trigger_timeout_seconds=3,
        kws_speaker_window_seconds=30,
        kws_match_tolerance_ms=100,
        speaker_context_enabled=speaker_context_enabled,
        speaker_context_window_seconds=30,
        stream_idle_timeout_seconds=30,
    )


class _FakeStream:
    """Stand-in for VolcengineStream exposing only latest_speaker_context."""

    def __init__(self, context):
        self.latest_speaker_context = context

    def speaker_context_if_fresh(self):
        return self.latest_speaker_context


def _make_connection(context=None, enabled=True) -> RealtimeAdapterConnection:
    conn = RealtimeAdapterConnection(SimpleNamespace(), _settings(enabled))
    conn._send_upstream = AsyncMock()  # type: ignore[attr-defined]
    if context is not None:
        conn.stream = _FakeStream(context)  # type: ignore[assignment]
    return conn


class MaybeUpdateSpeakerContextTest(unittest.TestCase):
    """Unit-test the VolcengineStream helper without spinning up a websocket."""

    def _new_stream(self) -> VolcengineStream:
        stream = VolcengineStream.__new__(VolcengineStream)
        stream.latest_speaker_context = None
        stream._speaker_context_expires_at = None
        stream.settings = SimpleNamespace(speaker_context_window_seconds=30)
        return stream

    def test_extracts_all_fields_from_additions(self):
        stream = self._new_stream()
        utterance = {"text": "hi"}
        additions = {
            "speaker_id": "0",
            "gender": "male",
            "age": 34.0,
            "emotion": "neutral",
        }
        stream._maybe_update_speaker_context(utterance, additions, "0")
        self.assertEqual(
            stream.latest_speaker_context,
            {"speaker_id": "0", "gender": "male", "age": 34.0, "emotion": "neutral"},
        )

    def test_missing_fields_are_omitted(self):
        stream = self._new_stream()
        stream._maybe_update_speaker_context(
            {"text": "hi"}, {"gender": "female"}, "1"
        )
        self.assertEqual(
            stream.latest_speaker_context, {"speaker_id": "1", "gender": "female"}
        )

    def test_no_recognized_fields_leaves_previous_context(self):
        stream = self._new_stream()
        stream.latest_speaker_context = {"speaker_id": "0", "gender": "male"}
        stream._maybe_update_speaker_context({"text": "hi"}, None, None)
        self.assertEqual(
            stream.latest_speaker_context, {"speaker_id": "0", "gender": "male"}
        )

    def test_latest_snapshot_overwrites(self):
        stream = self._new_stream()
        stream._maybe_update_speaker_context(
            {}, {"gender": "male", "age": 34.0, "emotion": "neutral"}, "0"
        )
        stream._maybe_update_speaker_context(
            {}, {"gender": "female", "age": 8.0, "emotion": "happy"}, "1"
        )
        self.assertEqual(
            stream.latest_speaker_context,
            {"speaker_id": "1", "gender": "female", "age": 8.0, "emotion": "happy"},
        )

    def test_flat_utterance_fields_are_used_as_fallback(self):
        stream = self._new_stream()
        stream._maybe_update_speaker_context(
            {"gender": "male", "emotion": "angry"}, None, "5"
        )
        self.assertEqual(
            stream.latest_speaker_context,
            {"speaker_id": "5", "gender": "male", "emotion": "angry"},
        )


class InjectSpeakerContextTest(unittest.IsolatedAsyncioTestCase):
    async def test_no_context_and_no_instructions_key_is_noop(self):
        conn = _make_connection(context=None)
        session = {"audio": {}}
        conn._maybe_inject_speaker_context(session)
        self.assertNotIn("instructions", session)

    async def test_no_context_strips_previous_line_from_instructions(self):
        conn = _make_connection(context=None)
        session = {
            "instructions": "You are Reachy.\n[Speaker context] speaker_id: 0"
        }
        conn._maybe_inject_speaker_context(session)
        self.assertEqual(session["instructions"], "You are Reachy.")

    async def test_appends_context_line(self):
        conn = _make_connection(
            context={
                "speaker_id": "0",
                "gender": "male",
                "age": 34.0,
                "emotion": "neutral",
            }
        )
        session = {"instructions": "You are Reachy."}
        conn._maybe_inject_speaker_context(session)
        self.assertEqual(
            session["instructions"],
            "You are Reachy.\n[Speaker context] speaker_id: 0, gender: male, age: 34.0, emotion: neutral",
        )

    async def test_repeated_update_is_idempotent(self):
        """Second call with same context must not stack lines."""
        conn = _make_connection(context={"speaker_id": "0", "gender": "male"})
        session = {"instructions": "You are Reachy."}
        conn._maybe_inject_speaker_context(session)
        conn._maybe_inject_speaker_context(session)
        self.assertEqual(
            session["instructions"],
            "You are Reachy.\n[Speaker context] speaker_id: 0, gender: male",
        )

    async def test_latest_context_replaces_stale_line(self):
        conn = _make_connection(context={"speaker_id": "1", "gender": "female"})
        session = {
            "instructions": "Base prompt.\n[Speaker context] speaker_id: 0, gender: male"
        }
        conn._maybe_inject_speaker_context(session)
        self.assertEqual(
            session["instructions"],
            "Base prompt.\n[Speaker context] speaker_id: 1, gender: female",
        )

    async def test_disabled_via_settings_strips_but_does_not_inject(self):
        conn = _make_connection(
            context={"speaker_id": "0"}, enabled=False
        )
        session = {
            "instructions": "Base.\n[Speaker context] speaker_id: 0"
        }
        conn._maybe_inject_speaker_context(session)
        self.assertEqual(session["instructions"], "Base.")

    async def test_missing_instructions_key_is_untouched_even_with_context(self):
        """Upstream keeps its previously-negotiated instructions when the client
        doesn't send a fresh `instructions` field."""
        conn = _make_connection(context={"speaker_id": "0"})
        session = {"audio": {}}
        conn._maybe_inject_speaker_context(session)
        self.assertNotIn("instructions", session)

    async def test_handle_session_update_forwards_injected_message(self):
        conn = _make_connection(context={"speaker_id": "0", "gender": "male"})
        message = {
            "type": "session.update",
            "session": {"instructions": "You are Reachy."},
        }
        await conn.handle_session_update(message)
        conn._send_upstream.assert_awaited_once()
        forwarded = conn._send_upstream.await_args.args[0]
        self.assertIn(
            "[Speaker context] speaker_id: 0, gender: male",
            forwarded["session"]["instructions"],
        )


class InjectIntoResponseCreateTest(unittest.IsolatedAsyncioTestCase):
    """Feature 2 (方案 A): on every `response.create`, `_send_upstream` must
    push a preceding `session.update` carrying the freshest speaker-context
    line, then forward the `response.create` frame untouched.
    """

    def _fresh_upstream(self):
        import asyncio as _asyncio

        upstream = SimpleNamespace(send=AsyncMock())
        return upstream, _asyncio.Lock()

    def _make_conn(self, *, context=None, enabled=True):
        """Like the module-level `_make_connection` but without the
        `_send_upstream` shadow-mock, since these tests exercise the real
        `_send_upstream` and observe `conn.upstream.send`.
        """
        conn = RealtimeAdapterConnection(SimpleNamespace(), _settings(enabled))
        if context is not None:
            conn.stream = _FakeStream(context)  # type: ignore[assignment]
        return conn

    async def test_response_create_prepended_by_session_update(self):
        conn = self._make_conn(
            context={"speaker_id": "0", "gender": "male"}
        )
        conn.session = {"instructions": "You are Reachy."}
        conn.upstream, conn._upstream_send_lock = self._fresh_upstream()
        msg = {"type": "response.create", "event_id": "x"}
        await conn._send_upstream(msg)
        # Two frames sent, in order: session.update then response.create.
        self.assertEqual(conn.upstream.send.await_count, 2)
        first, second = conn.upstream.send.await_args_list
        first_frame = json.loads(first.args[0])
        second_frame = json.loads(second.args[0])
        self.assertEqual(first_frame["type"], "session.update")
        self.assertEqual(
            first_frame["session"]["instructions"],
            "You are Reachy.\n\n[Speaker context] speaker_id: 0, gender: male",
        )
        # response.create passes through untouched (no per-response override).
        self.assertEqual(second_frame["type"], "response.create")
        self.assertNotIn("response", second_frame)

    async def test_response_create_no_context_passes_through(self):
        conn = self._make_conn(context=None)
        conn.session = {"instructions": "You are Reachy."}
        conn.upstream, conn._upstream_send_lock = self._fresh_upstream()
        msg = {"type": "response.create", "event_id": "x"}
        await conn._send_upstream(msg)
        # No fresh context → NO preceding session.update, just the response.create.
        conn.upstream.send.assert_awaited_once()
        forwarded = json.loads(conn.upstream.send.await_args.args[0])
        self.assertEqual(forwarded["type"], "response.create")
        self.assertNotIn("response", forwarded)
        # msg not mutated.
        self.assertNotIn("response", msg)

    async def test_response_create_disabled_via_settings(self):
        conn = self._make_conn(
            context={"speaker_id": "0"}, enabled=False
        )
        conn.session = {"instructions": "Base."}
        conn.upstream, conn._upstream_send_lock = self._fresh_upstream()
        msg = {"type": "response.create", "event_id": "x"}
        await conn._send_upstream(msg)
        # Feature disabled → no session.update injected, just the response.create.
        conn.upstream.send.assert_awaited_once()
        forwarded = json.loads(conn.upstream.send.await_args.args[0])
        self.assertEqual(forwarded["type"], "response.create")

    async def test_response_create_no_client_instructions_yet(self):
        """Before the client sends its first session.update we still stamp a
        bare `[Speaker context] ...` line (no empty prefix)."""
        conn = self._make_conn(context={"speaker_id": "0", "gender": "male"})
        conn.session = {}  # no `instructions` key
        conn.upstream, conn._upstream_send_lock = self._fresh_upstream()
        msg = {"type": "response.create"}
        await conn._send_upstream(msg)
        self.assertEqual(conn.upstream.send.await_count, 2)
        first = json.loads(conn.upstream.send.await_args_list[0].args[0])
        self.assertEqual(first["type"], "session.update")
        self.assertEqual(
            first["session"]["instructions"],
            "[Speaker context] speaker_id: 0, gender: male",
        )

    async def test_response_create_repeated_is_idempotent(self):
        """Repeated response.create with the same tracked base must not
        accumulate stacked `[Speaker context]` lines in the emitted
        session.update."""
        conn = self._make_conn(context={"speaker_id": "0"})
        conn.session = {
            "instructions": "Base.\n[Speaker context] speaker_id: 9",
        }
        conn.upstream, conn._upstream_send_lock = self._fresh_upstream()
        await conn._send_upstream({"type": "response.create"})
        await conn._send_upstream({"type": "response.create"})
        # 2 response.create × 2 frames each = 4 sends.
        self.assertEqual(conn.upstream.send.await_count, 4)
        for i in (0, 2):
            frame = json.loads(
                conn.upstream.send.await_args_list[i].args[0]
            )
            self.assertEqual(frame["type"], "session.update")
            self.assertEqual(
                frame["session"]["instructions"],
                "Base.\n\n[Speaker context] speaker_id: 0",
            )

    async def test_non_response_create_frames_pass_through_without_extra(self):
        conn = self._make_conn(context={"speaker_id": "0"})
        conn.session = {"instructions": "Base."}
        conn.upstream, conn._upstream_send_lock = self._fresh_upstream()
        msg = {"type": "conversation.item.create", "item": {}}
        await conn._send_upstream(msg)
        conn.upstream.send.assert_awaited_once()
        forwarded = json.loads(conn.upstream.send.await_args.args[0])
        self.assertEqual(forwarded["type"], "conversation.item.create")
        self.assertNotIn("response", msg)


class SpeakerContextExpirationTest(unittest.TestCase):
    """Verify the post-response 30s TTL on `latest_speaker_context`."""

    def _new_stream(self, window: float = 30.0) -> VolcengineStream:
        stream = VolcengineStream.__new__(VolcengineStream)
        stream.latest_speaker_context = None
        stream._speaker_context_expires_at = None
        stream.settings = SimpleNamespace(speaker_context_window_seconds=window)
        return stream

    def test_arm_without_context_is_noop(self):
        stream = self._new_stream()
        stream.arm_speaker_context_expiration()
        self.assertIsNone(stream._speaker_context_expires_at)

    def test_fresh_before_deadline(self):
        stream = self._new_stream(window=60)
        stream._maybe_update_speaker_context({}, {"gender": "male"}, "0")
        stream.arm_speaker_context_expiration()
        self.assertIsNotNone(stream._speaker_context_expires_at)
        self.assertEqual(
            stream.speaker_context_if_fresh(),
            {"speaker_id": "0", "gender": "male"},
        )

    def test_expired_clears_and_returns_none(self):
        import time as _time

        stream = self._new_stream(window=30)
        stream._maybe_update_speaker_context({}, {"gender": "male"}, "0")
        # Simulate deadline already in the past.
        stream._speaker_context_expires_at = _time.monotonic() - 1
        self.assertIsNone(stream.speaker_context_if_fresh())
        # And the stored snapshot is dropped so future injects see nothing.
        self.assertIsNone(stream.latest_speaker_context)
        self.assertIsNone(stream._speaker_context_expires_at)

    def test_new_utterance_clears_pending_expiration(self):
        stream = self._new_stream(window=30)
        stream._maybe_update_speaker_context({}, {"gender": "male"}, "0")
        stream.arm_speaker_context_expiration()
        self.assertIsNotNone(stream._speaker_context_expires_at)
        # A fresh utterance should reset the TTL — the new snapshot is valid
        # for the turn it just produced.
        stream._maybe_update_speaker_context({}, {"gender": "female"}, "1")
        self.assertIsNone(stream._speaker_context_expires_at)

    def test_zero_window_disables_ttl(self):
        stream = self._new_stream(window=0)
        stream._maybe_update_speaker_context({}, {"gender": "male"}, "0")
        stream.arm_speaker_context_expiration()
        self.assertIsNone(stream._speaker_context_expires_at)
        self.assertEqual(
            stream.speaker_context_if_fresh(),
            {"speaker_id": "0", "gender": "male"},
        )


if __name__ == "__main__":
    unittest.main()
