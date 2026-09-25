from __future__ import annotations

import asyncio
import base64
import json
import unittest
from types import SimpleNamespace

from unittest.mock import AsyncMock, MagicMock

from audio_gate import GateMode, Utterance, WakeMarker
from volc_stt_adapter import LiveConnectionRegistry, RealtimeAdapterConnection
from wake_word import WakeEvent


class Sink:
    def __init__(self): self.messages = []
    async def send(self, raw): self.messages.append(json.loads(raw))


class BlockingAudioStream:
    def __init__(self):
        self.generation = 1
        self.sent = []
        self.closed = False
        self.first_send_started = asyncio.Event()
        self.release_first_send = asyncio.Event()

    async def send_audio(self, pcm):
        self.sent.append(pcm)
        if len(self.sent) == 1:
            self.first_send_started.set()
            await self.release_first_send.wait()

    async def close(self):
        self.closed = True


def append_message(pcm):
    return {
        "type": "input_audio_buffer.append",
        "audio": base64.b64encode(pcm).decode("ascii"),
    }


def settings(mode):
    return SimpleNamespace(
        kws_mode=GateMode(mode),
        kws_trigger_timeout_seconds=3,
        kws_speaker_window_seconds=30,
        kws_match_tolerance_ms=100,
        stop_talking_words=("闭嘴", "住嘴", "停嘴", "安静", "别说了", "停止", "停下", "够了"),
        speaker_context_enabled=True,
    )


class ConnectionGateTest(unittest.IsolatedAsyncioTestCase):
    async def test_allowed_utterance_injects_exact_text_and_events(self):
        downstream = Sink()
        connection = RealtimeAdapterConnection(downstream, settings("enforce"))
        upstream = Sink(); connection.upstream = upstream
        connection._on_stream_generation(7, 0)
        connection.gate.on_wake(WakeMarker(7, 16000, 1000))
        # KWS wake opens the conversation gate in production (_start_after_wake
        # / arm_gate_for_reply); mirror that here since the test builds state
        # manually.
        connection._conversation_gate_open = True
        await connection._on_native_utterance(
            Utterance("瑞奇 请打开客厅灯", "speaker-a", 7, 850, 1400)
        )
        self.assertEqual(
            [message["type"] for message in downstream.messages],
            [
                "input_audio_buffer.speech_started",
                "conversation.item.input_audio_transcription.delta",
                "input_audio_buffer.speech_stopped",
                "conversation.item.input_audio_transcription.completed",
            ],
        )
        self.assertEqual(downstream.messages[1]["delta"], "瑞奇 请打开客厅灯")
        self.assertEqual(downstream.messages[3]["transcript"], "瑞奇 请打开客厅灯")
        self.assertEqual(
            upstream.messages[0]["item"]["content"][0]["text"], "瑞奇 请打开客厅灯"
        )
        self.assertEqual(upstream.messages[1]["type"], "response.create")

    async def test_append_during_blocked_drain_is_sent_in_next_round(self):
        connection = RealtimeAdapterConnection(Sink(), settings("enforce"))
        stream = BlockingAudioStream()
        stream_start_entered = asyncio.Event()
        release_stream_start = asyncio.Event()

        async def fake_start_stream(initial_pcm, *, timeline_origin_sample):
            self.assertEqual(initial_pcm, b"preroll!")
            self.assertEqual(timeline_origin_sample, 15996)
            connection.stream = stream
            stream_start_entered.set()
            await release_stream_start.wait()

        connection._start_stream = fake_start_stream
        wake = WakeEvent(
            score=0.9,
            sample_index=16000,
            timestamp_ms=1000.0,
            preroll_pcm=b"preroll!",
        )
        start_task = asyncio.create_task(connection._start_after_wake(wake))
        await stream_start_entered.wait()

        first_pcm = b"\x01\x00\x02\x00"
        second_pcm = b"\x03\x00\x04\x00"
        await connection.handle_audio_append(append_message(first_pcm))
        release_stream_start.set()
        await stream.first_send_started.wait()

        # This append happens while the first drain batch is blocked in
        # send_audio. It must remain in the swapped-in buffer for round two.
        await connection.handle_audio_append(append_message(second_pcm))
        clear_task = asyncio.create_task(connection.clear_audio(emit_confirmation=False))
        await asyncio.sleep(0)
        self.assertFalse(clear_task.done())

        stream.release_first_send.set()
        await start_task
        await clear_task

        self.assertEqual(stream.sent, [first_pcm, second_pcm])
        self.assertTrue(stream.closed)
        self.assertIsNone(connection.stream)
        self.assertFalse(connection._stream_starting)
        self.assertEqual(bytes(connection._pending_stream_audio), b"")

    async def test_proactive_open_starts_with_empty_pcm_not_detector_preroll(self):
        connection = RealtimeAdapterConnection(Sink(), settings("enforce"))
        connection.detector = SimpleNamespace(sample_index=321, preroll_pcm=b"tts-echo")
        stream = BlockingAudioStream()

        async def start(initial_pcm, *, timeline_origin_sample):
            self.assertEqual(initial_pcm, b"")
            self.assertEqual(timeline_origin_sample, 321)
            connection._on_stream_generation(stream.generation, timeline_origin_sample)
            connection.stream = stream

        connection._start_stream = start
        self.assertTrue(await connection.arm_gate_for_reply())
        self.assertEqual(stream.sent, [])
        self.assertTrue(connection.gate.proactive_reply_armed)

    async def test_proactive_open_reuses_existing_stream(self):
        connection = RealtimeAdapterConnection(Sink(), settings("enforce"))
        stream = BlockingAudioStream()
        connection.stream = stream
        connection._on_stream_generation(1, 0)
        connection._start_stream = AsyncMock()
        self.assertTrue(await connection.arm_gate_for_reply())
        connection._start_stream.assert_not_awaited()

    async def test_proactive_start_clear_race_does_not_arm(self):
        connection = RealtimeAdapterConnection(Sink(), settings("enforce"))
        entered = asyncio.Event()
        release = asyncio.Event()

        async def start(initial_pcm, *, timeline_origin_sample):
            entered.set()
            await release.wait()

        connection._start_stream = start
        task = asyncio.create_task(connection.arm_gate_for_reply())
        await entered.wait()
        await connection.clear_audio(emit_confirmation=False)
        release.set()
        self.assertFalse(await task)
        self.assertFalse(connection.gate.proactive_reply_armed)

    async def test_registry_cardinality_mode_and_unregister(self):
        registry = LiveConnectionRegistry("enforce")
        self.assertEqual(await registry.open_gate(), (False, "no_active_connection"))
        first = RealtimeAdapterConnection(Sink(), settings("enforce"))
        first.arm_gate_for_reply = AsyncMock(return_value=True)
        registry.register(first)
        self.assertEqual(await registry.open_gate(), (True, "proactive_reply_armed"))
        second = RealtimeAdapterConnection(Sink(), settings("enforce"))
        registry.register(second)
        self.assertEqual(await registry.open_gate(), (False, "ambiguous_connections"))
        registry.unregister(second)
        self.assertEqual(len(registry.snapshot()), 1)
        off = LiveConnectionRegistry("off")
        off.register(first)
        self.assertEqual(await off.open_gate(), (False, "gate_not_enforced"))

    async def test_open_conversation_admits_every_speaker(self):
        downstream = Sink()
        connection = RealtimeAdapterConnection(downstream, settings("enforce"))
        upstream = Sink(); connection.upstream = upstream
        connection._conversation_gate_open = True

        utterances = (
            Utterance("第一句", "alice", 2, 900, 1300),
            Utterance("第二句", "bob", 2, 1400, 1800),
            Utterance("第三句", None, 99, 1900, 2200),
        )
        for utterance in utterances:
            await connection._on_native_utterance(utterance)

        injected = [
            message["item"]["content"][0]["text"]
            for message in upstream.messages
            if message["type"] == "conversation.item.create"
        ]
        self.assertEqual(injected, [utterance.text for utterance in utterances])

    async def test_closed_conversation_suppresses_with_only_closed_reason(self):
        downstream = Sink()
        connection = RealtimeAdapterConnection(downstream, settings("enforce"))
        upstream = Sink(); connection.upstream = upstream

        with self.assertLogs("volc_stt_adapter", level="INFO") as captured:
            await connection._on_native_utterance(
                Utterance("关闭时不注入", "speaker-3", 7, 100, 500)
            )

        self.assertEqual(downstream.messages, [])
        self.assertEqual(upstream.messages, [])
        logs = "\n".join(captured.output)
        self.assertIn("reason=conversation_closed", logs)
        self.assertNotIn("speaker_id=", logs)

    async def test_stop_talking_cancels_upstream_and_closes_gate(self):
        """Stop-talking keyword mid-conversation cancels active response,
        closes the conversation gate, and does not inject the utterance."""
        connection = RealtimeAdapterConnection(Sink(), settings("enforce"))
        upstream = Sink(); connection.upstream = upstream
        # Simulate an active conversation + active upstream response.
        connection._conversation_gate_open = True
        connection.upstream_response_active = True

        await connection._on_native_utterance(
            Utterance("闭嘴", "alice", 1, 100, 500)
        )

        # response.cancel should have been sent; no conversation.item.create.
        types = [m["type"] for m in upstream.messages]
        self.assertIn("response.cancel", types)
        self.assertNotIn("conversation.item.create", types)
        self.assertNotIn("response.create", types)
        # Gate must be closed so next turn requires a fresh KWS.
        self.assertFalse(connection._conversation_gate_open)
        self.assertFalse(connection.upstream_response_active)

    async def test_stop_talking_matches_substring_and_ignores_punctuation(self):
        """Stop-talking match works even when the keyword appears embedded
        in a longer utterance with punctuation."""
        connection = RealtimeAdapterConnection(Sink(), settings("enforce"))
        upstream = Sink(); connection.upstream = upstream
        connection._conversation_gate_open = True
        connection.upstream_response_active = True

        await connection._on_native_utterance(
            Utterance("好啦，够了，别念了。", "alice", 1, 100, 500)
        )

        self.assertIn("response.cancel", [m["type"] for m in upstream.messages])
        self.assertFalse(connection._conversation_gate_open)

    async def test_stop_talking_without_active_response_still_closes_gate(self):
        """Even if no upstream response is active, matching a stop keyword
        must still close the gate (and must not send response.cancel)."""
        connection = RealtimeAdapterConnection(Sink(), settings("enforce"))
        upstream = Sink(); connection.upstream = upstream
        connection._conversation_gate_open = True
        connection.upstream_response_active = False

        await connection._on_native_utterance(
            Utterance("安静", "alice", 1, 100, 500)
        )

        self.assertEqual(upstream.messages, [])
        self.assertFalse(connection._conversation_gate_open)

    async def test_soft_stream_close_preserves_boolean_conversation_gate(self):
        connection = RealtimeAdapterConnection(Sink(), settings("enforce"))
        stream = BlockingAudioStream()
        stream.item_id = "old-stream"
        connection.stream = stream
        connection._conversation_gate_open = True

        await connection.clear_audio(
            emit_confirmation=False,
            revoke_authorization=False,
            reason="stream_idle_soft_close",
        )

        self.assertTrue(connection._conversation_gate_open)
        connection._on_stream_generation(2, 16000)
        self.assertEqual(
            connection._decide_text_gate(
                Utterance("继续", "another-speaker", 2, 1200, 1400)
            ),
            "conversation_active",
        )

    async def test_explicit_clear_revokes_authorization(self):
        connection = RealtimeAdapterConnection(Sink(), settings("enforce"))
        connection._on_stream_generation(1, 0)
        connection.gate.on_wake(WakeMarker(1, 16000, 1000))
        connection._conversation_gate_open = True

        await connection.clear_audio(emit_confirmation=False)

        self.assertFalse(connection._conversation_gate_open)
        self.assertIsNone(
            connection._decide_text_gate(Utterance("继续", "alice", 1, 1200, 1400))
        )

    async def test_wake_emotion_skipped_when_stream_already_active(self):
        """A KWS wake fired while a Volcengine stream is already open must
        not replay the wake emotion — that's just a gate re-arm, not a fresh
        conversation."""
        connection = RealtimeAdapterConnection(Sink(), settings("enforce"))
        connection._trigger_wake_emotion = MagicMock()  # type: ignore[assignment]
        connection.stream = SimpleNamespace(generation=3)  # type: ignore[assignment]

        await connection._on_wake(
            WakeEvent(
                detector_generation=0,
                sample_index=16000,
                timestamp_ms=1000,
                score=0.9,
                preroll_pcm=b"",
            )
        )
        connection._trigger_wake_emotion.assert_not_called()

    async def test_wake_emotion_fires_on_first_wake(self):
        """First wake (no active stream) plays the wake emotion."""
        connection = RealtimeAdapterConnection(Sink(), settings("enforce"))
        connection._trigger_wake_emotion = MagicMock()  # type: ignore[assignment]
        connection._start_after_wake = AsyncMock()  # type: ignore[assignment]
        self.assertIsNone(connection.stream)

        await connection._on_wake(
            WakeEvent(
                detector_generation=0,
                sample_index=16000,
                timestamp_ms=1000,
                score=0.9,
                preroll_pcm=b"",
            )
        )
        connection._trigger_wake_emotion.assert_called_once()


if __name__ == "__main__":
    unittest.main()
