from __future__ import annotations

import asyncio
import base64
import json
import unittest
from types import SimpleNamespace

from audio_gate import GateMode, Utterance, WakeMarker
from volc_stt_adapter import RealtimeAdapterConnection
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
    )


class ConnectionGateTest(unittest.IsolatedAsyncioTestCase):
    async def test_allowed_utterance_injects_exact_text_and_events(self):
        downstream = Sink()
        connection = RealtimeAdapterConnection(downstream, settings("enforce"))
        upstream = Sink(); connection.upstream = upstream
        connection._on_stream_generation(7, 0)
        connection.gate.on_wake(WakeMarker(7, 16000, 1000))
        await connection._on_native_utterance(
            Utterance("请打开客厅灯", "speaker-a", 7, 850, 1400)
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
        self.assertEqual(downstream.messages[1]["delta"], "请打开客厅灯")
        self.assertEqual(downstream.messages[3]["transcript"], "请打开客厅灯")
        self.assertEqual(upstream.messages[0]["item"]["content"][0]["text"], "请打开客厅灯")
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

    async def test_other_speaker_is_not_injected(self):
        downstream = Sink()
        connection = RealtimeAdapterConnection(downstream, settings("enforce"))
        upstream = Sink(); connection.upstream = upstream
        connection._on_stream_generation(2, 0)
        connection.gate.on_wake(WakeMarker(2, 16000, 1000))
        await connection._on_native_utterance(Utterance("唤醒首句", "alice", 2, 900, 1300))
        downstream.messages.clear(); upstream.messages.clear()
        await connection._on_native_utterance(Utterance("不应注入的文本", "bob", 2, 1400, 1800))
        self.assertEqual(downstream.messages, [])
        self.assertEqual(upstream.messages, [])


if __name__ == "__main__":
    unittest.main()
