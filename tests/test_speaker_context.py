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
        keyword_gate_enabled=True,
        keyword_gate_prefix_words=(),
        stop_talking_words=(),
        speaker_context_enabled=speaker_context_enabled,
        speaker_context_window_seconds=30,
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
