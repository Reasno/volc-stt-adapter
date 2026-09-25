"""Non-negotiable regression contracts for conversation admission and lifecycle.

These tests intentionally overlap lower-level unit tests.  They name the product
contracts directly so a future refactor cannot accidentally restore deleted
speaker-ID admission or stop-word behavior while keeping implementation tests
green.
"""
from __future__ import annotations

import ast
import asyncio
import inspect
import textwrap
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from audio_gate import GateMode, Utterance
from volc_stt_adapter import RealtimeAdapterConnection


ROOT = Path(__file__).resolve().parents[1]
RUNTIME_FILES = (ROOT / "volc_stt_adapter.py", ROOT / "audio_gate.py")


def settings(timeout: float = 30.0) -> SimpleNamespace:
    return SimpleNamespace(
        kws_mode=GateMode.ENFORCE,
        kws_trigger_timeout_seconds=3,
        kws_speaker_window_seconds=30,
        kws_match_tolerance_ms=100,
        speaker_context_enabled=False,
        speaker_context_window_seconds=30,
        stream_idle_timeout_seconds=timeout,
    )


class Sink:
    def __init__(self):
        self.messages = []

    async def send(self, raw):
        self.messages.append(raw)


class RegressionContractsTest(unittest.IsolatedAsyncioTestCase):
    def make_connection(self, timeout: float = 30.0) -> RealtimeAdapterConnection:
        return RealtimeAdapterConnection(Sink(), settings(timeout))

    def test_conversation_gate_is_only_admission_policy_and_ignores_speaker_id(self):
        connection = self.make_connection()
        utterances = (
            Utterance("甲", "wake-speaker", 1, 0, 100),
            Utterance("乙", "different-speaker", 99, 200, 300),
            Utterance("丙", None, 2, None, None),
        )

        for utterance in utterances:
            self.assertIsNone(connection._decide_text_gate(utterance))

        connection._conversation_gate_open = True
        for utterance in utterances:
            self.assertEqual(
                connection._decide_text_gate(utterance), "conversation_active"
            )

    def test_deleted_per_speaker_rejection_reasons_are_absent(self):
        runtime_source = "\n".join(path.read_text() for path in RUNTIME_FILES)
        self.assertNotIn("gate_sleeping", runtime_source)
        self.assertNotIn("speaker_not_authorized", runtime_source)

    async def test_soft_close_revokes_authorization(self):
        connection = self.make_connection()
        connection._conversation_gate_open = True
        connection.stream = SimpleNamespace(close=AsyncMock(), item_id="stream")

        await connection.clear_audio(
            emit_confirmation=False,
            revoke_authorization=True,
            reason="stream_idle_soft_close",
        )

        self.assertFalse(connection._conversation_gate_open)

    async def test_explicit_clear_revokes_authorization(self):
        connection = self.make_connection()
        connection._conversation_gate_open = True
        await connection.clear_audio(emit_confirmation=False)
        self.assertFalse(connection._conversation_gate_open)

    def test_connection_closed_cleanup_uses_revoking_clear(self):
        tree = ast.parse(textwrap.dedent(inspect.getsource(RealtimeAdapterConnection.run)))
        calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "clear_audio"
        ]
        connection_closed = [
            call
            for call in calls
            if any(
                keyword.arg == "reason"
                and isinstance(keyword.value, ast.Constant)
                and keyword.value.value == "connection_closed"
                for keyword in call.keywords
            )
        ]
        self.assertEqual(len(connection_closed), 1)
        revoke_keywords = {
            keyword.arg: keyword.value for keyword in connection_closed[0].keywords
        }
        self.assertNotIn("revoke_authorization", revoke_keywords)
        self.assertTrue(
            inspect.signature(RealtimeAdapterConnection.clear_audio)
            .parameters["revoke_authorization"]
            .default
        )

    def test_default_idle_timeout_is_exactly_30_seconds(self):
        connection = self.make_connection(30.0)
        connection.stream = SimpleNamespace()
        with patch("volc_stt_adapter.time.monotonic", return_value=100.0):
            connection._refresh_stream_idle()
        self.assertEqual(connection._stream_idle_deadline, 130.0)

    def test_user_speech_activity_restarts_full_idle_window(self):
        connection = self.make_connection(30.0)
        connection.stream = SimpleNamespace()
        with patch("volc_stt_adapter.time.monotonic", side_effect=[100.0, 129.0]):
            connection._refresh_stream_idle()
            first_deadline = connection._stream_idle_deadline
            connection._refresh_stream_idle()
        self.assertEqual(first_deadline, 130.0)
        self.assertEqual(connection._stream_idle_deadline, 159.0)

    async def test_bot_response_blocks_timeout_then_restarts_countdown(self):
        connection = self.make_connection(0.02)
        connection.stream = SimpleNamespace()
        connection.clear_audio = AsyncMock()
        connection.upstream_response_active = True
        connection._arm_stream_idle()
        await asyncio.sleep(0.06)
        connection.clear_audio.assert_not_awaited()

        connection.upstream_response_active = False
        connection._refresh_stream_idle()
        await asyncio.sleep(0.01)
        connection.clear_audio.assert_not_awaited()
        for _ in range(10):
            if connection.clear_audio.await_count:
                break
            await asyncio.sleep(0.01)
        connection.clear_audio.assert_awaited_once_with(
            emit_confirmation=False,
            revoke_authorization=True,
            reason="stream_idle_soft_close",
        )
        connection._cancel_stream_idle()

    def test_stop_talking_interception_is_absent_from_runtime(self):
        runtime_source = "\n".join(path.read_text() for path in RUNTIME_FILES)
        self.assertNotIn("stop_talking", runtime_source)
        self.assertNotIn("停止词", runtime_source)


if __name__ == "__main__":
    unittest.main()
