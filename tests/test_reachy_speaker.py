from __future__ import annotations

import asyncio
import json
import struct
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock

import aiohttp
from aiohttp.test_utils import TestClient, TestServer

from reachy_speaker import (
    AUDIO_ONLY_SERVER,
    CONNECTION_FINISHED,
    CONNECTION_STARTED,
    ERROR_RESPONSE,
    FULL_SERVER_RESPONSE,
    SESSION_FINISHED,
    SESSION_STARTED,
    WITH_EVENT,
    ConversationError,
    ConversationSayClient,
    DaemonSoundClient,
    FallbackDisabledError,
    ReachySpeaker,
    SpeakResult,
    SpeakerError,
    TtsProtocolError,
    VolcengineTtsClient,
    create_speak_app,
    decode_tts_frame,
)


class AsyncContext:
    def __init__(self, value):
        self.value = value

    async def __aenter__(self):
        return self.value

    async def __aexit__(self, *_):
        return False


class FakeWs:
    def __init__(self, messages):
        self.messages = list(messages)
        self.sent_json = []
        self.sent_bytes = []

    async def send_json(self, value):
        self.sent_json.append(value)

    async def send_bytes(self, value):
        self.sent_bytes.append(value)

    async def receive(self):
        return self.messages.pop(0)


class FakeSession:
    def __init__(self, ws=None, posts=None, delete=None, ws_error=None):
        self.ws = ws
        self.posts = list(posts or [])
        self.delete_response = delete
        self.ws_error = ws_error
        self.ws_calls = []
        self.post_calls = []
        self.delete_calls = []

    def ws_connect(self, *args, **kwargs):
        self.ws_calls.append((args, kwargs))
        if self.ws_error:
            raise self.ws_error
        return AsyncContext(self.ws)

    def post(self, url, **kwargs):
        self.post_calls.append((url, kwargs))
        return AsyncContext(self.posts.pop(0))

    def delete(self, url, **kwargs):
        self.delete_calls.append((url, kwargs))
        return AsyncContext(self.delete_response)


class FakeResponse:
    def __init__(self, status=200, payload=None):
        self.status = status
        self.payload = payload or {}

    async def json(self):
        return self.payload


def ws_text(value):
    return SimpleNamespace(type=aiohttp.WSMsgType.TEXT, data=json.dumps(value))


def server_event(event, payload=b"", session_id="s"):
    data = bytearray((0x11, (FULL_SERVER_RESPONSE << 4) | WITH_EVENT, 0, 0))
    data.extend(struct.pack(">i", event))
    if event not in {50, 51, 52}:
        sid = session_id.encode()
        data.extend(struct.pack(">I", len(sid)))
        data.extend(sid)
    else:
        data.extend(struct.pack(">I", 0))
    data.extend(struct.pack(">I", len(payload)))
    data.extend(payload)
    return SimpleNamespace(type=aiohttp.WSMsgType.BINARY, data=bytes(data))


def audio_frame(payload):
    data = bytes((0x11, AUDIO_ONLY_SERVER << 4, 0, 0)) + struct.pack(">I", len(payload)) + payload
    return SimpleNamespace(type=aiohttp.WSMsgType.BINARY, data=data)


def error_frame(code, payload=b"bad"):
    data = bytes((0x11, ERROR_RESPONSE << 4, 0, 0)) + struct.pack(">I", code) + struct.pack(">I", len(payload)) + payload
    return SimpleNamespace(type=aiohttp.WSMsgType.BINARY, data=data)


class ConversationTests(unittest.IsolatedAsyncioTestCase):
    async def test_ignores_notifications_and_matches_id(self):
        ws = FakeWs([])
        session = FakeSession(ws=ws)
        client = ConversationSayClient(session, "ws://reachy/rpc")
        original_send = ws.send_json

        async def send(value):
            await original_send(value)
            ws.messages.extend([
                ws_text({"jsonrpc": "2.0", "method": "notice", "params": {}}),
                ws_text({"jsonrpc": "2.0", "id": "other", "result": {}}),
                ws_text({"jsonrpc": "2.0", "id": value["id"], "result": {"accepted": True}}),
            ])

        ws.send_json = send
        await client.say("你好")
        self.assertEqual(ws.sent_json[0]["method"], "conversation.say")

    async def test_rpc_error_is_failure(self):
        ws = FakeWs([])
        client = ConversationSayClient(FakeSession(ws=ws), "ws://reachy/rpc")

        async def send(value):
            ws.messages.append(ws_text({"jsonrpc": "2.0", "id": value["id"], "error": {"message": "offline", "data": {"reason": "not_running"}}}))

        ws.send_json = send
        with self.assertRaisesRegex(ConversationError, "not_running"):
            await client.say("hello")


class AggregationTests(unittest.IsolatedAsyncioTestCase):
    async def test_conversation_success_never_falls_back(self):
        conversation = SimpleNamespace(say=AsyncMock())
        tts = SimpleNamespace(synthesize=AsyncMock())
        daemon = SimpleNamespace(upload_and_play=AsyncMock(), close=AsyncMock())
        route = await ReachySpeaker(conversation, tts, daemon).speak("hi", "r1")
        self.assertEqual(route, "conversation")
        tts.synthesize.assert_not_awaited()

    async def test_connection_failure_falls_back(self):
        conversation = SimpleNamespace(say=AsyncMock(side_effect=ConversationError("connect")))
        tts = SimpleNamespace(synthesize=AsyncMock(return_value=b"mp3"))
        daemon = SimpleNamespace(upload_and_play=AsyncMock(), close=AsyncMock())
        route = await ReachySpeaker(conversation, tts, daemon).speak("hi", "r1")
        self.assertEqual(route, "daemon_tts")
        daemon.upload_and_play.assert_awaited_once_with(b"mp3", "r1")

    async def test_fallback_disabled_never_calls_tts_or_daemon(self):
        conversation = SimpleNamespace(say=AsyncMock(side_effect=ConversationError("not_running")))
        tts = SimpleNamespace(synthesize=AsyncMock())
        daemon = SimpleNamespace(upload_and_play=AsyncMock(), close=AsyncMock())
        with self.assertRaises(FallbackDisabledError):
            await ReachySpeaker(conversation, tts, daemon).speak(
                "hi", "r1", allow_tts_fallback=False
            )
        tts.synthesize.assert_not_awaited()
        daemon.upload_and_play.assert_not_awaited()

    async def test_fallback_disabled_conversation_success(self):
        conversation = SimpleNamespace(say=AsyncMock())
        tts = SimpleNamespace(synthesize=AsyncMock())
        daemon = SimpleNamespace(upload_and_play=AsyncMock(), close=AsyncMock())
        route = await ReachySpeaker(conversation, tts, daemon).speak(
            "hi", "r1", allow_tts_fallback=False
        )
        self.assertEqual(route, "conversation")
        tts.synthesize.assert_not_awaited()

    async def test_rpc_and_fallback_failure(self):
        conversation = SimpleNamespace(say=AsyncMock(side_effect=ConversationError("rpc")))
        tts = SimpleNamespace(synthesize=AsyncMock(side_effect=TtsProtocolError("tts")))
        daemon = SimpleNamespace(upload_and_play=AsyncMock(), close=AsyncMock())
        with self.assertRaisesRegex(SpeakerError, "both speech routes failed"):
            await ReachySpeaker(conversation, tts, daemon).speak("hi", "r1")

    async def test_fallback_cache_hit_synthesizes_once_but_plays_twice(self):
        conversation = SimpleNamespace(say=AsyncMock(side_effect=ConversationError("offline")))
        tts = SimpleNamespace(
            synthesize=AsyncMock(return_value=b"audio"),
            resource_id="resource-a",
            voice="voice-a",
            max_audio_bytes=100,
        )
        daemon = SimpleNamespace(upload_and_play=AsyncMock(), close=AsyncMock())
        speaker = ReachySpeaker(conversation, tts, daemon)

        first = await speaker.speak("same", "r1")
        second = await speaker.speak("same", "r2")

        self.assertFalse(first.tts_cache_hit)
        self.assertTrue(second.tts_cache_hit)
        tts.synthesize.assert_awaited_once_with("same", "r1")
        self.assertEqual(daemon.upload_and_play.await_count, 2)
        self.assertIs(daemon.upload_and_play.await_args_list[0].args[0], b"audio")
        self.assertIs(
            daemon.upload_and_play.await_args_list[0].args[0],
            daemon.upload_and_play.await_args_list[1].args[0],
        )

    async def test_cache_key_separates_text_voice_and_resource(self):
        conversation = SimpleNamespace(say=AsyncMock(side_effect=ConversationError("offline")))
        tts = SimpleNamespace(
            synthesize=AsyncMock(return_value=b"audio"),
            resource_id="resource-a",
            voice="voice-a",
            max_audio_bytes=100,
        )
        speaker = ReachySpeaker(
            conversation,
            tts,
            SimpleNamespace(upload_and_play=AsyncMock(), close=AsyncMock()),
        )
        await speaker.speak("one", "r1")
        await speaker.speak("two", "r2")
        tts.voice = "voice-b"
        await speaker.speak("one", "r3")
        tts.resource_id = "resource-b"
        await speaker.speak("one", "r4")
        self.assertEqual(tts.synthesize.await_count, 4)

    async def test_lru_hit_moves_to_mru_and_evicts_oldest(self):
        conversation = SimpleNamespace(say=AsyncMock(side_effect=ConversationError("offline")))
        tts = SimpleNamespace(
            synthesize=AsyncMock(side_effect=lambda text, _: text.encode()),
            resource_id="resource",
            voice="voice",
            max_audio_bytes=100,
        )
        speaker = ReachySpeaker(
            conversation,
            tts,
            SimpleNamespace(upload_and_play=AsyncMock(), close=AsyncMock()),
            cache_entries=2,
        )
        await speaker.speak("a", "r1")
        await speaker.speak("b", "r2")
        self.assertTrue((await speaker.speak("a", "r3")).tts_cache_hit)
        await speaker.speak("c", "r4")
        self.assertFalse((await speaker.speak("b", "r5")).tts_cache_hit)
        self.assertEqual(speaker.tts_cache.stats["evictions"], 2)

    async def test_default_cache_evicts_on_101st_entry(self):
        conversation = SimpleNamespace(say=AsyncMock(side_effect=ConversationError("offline")))
        tts = SimpleNamespace(
            synthesize=AsyncMock(return_value=b"audio"),
            resource_id="resource",
            voice="voice",
            max_audio_bytes=100,
        )
        speaker = ReachySpeaker(
            conversation,
            tts,
            SimpleNamespace(upload_and_play=AsyncMock(), close=AsyncMock()),
        )
        for index in range(101):
            await speaker.speak(str(index), f"r{index}")
        self.assertEqual(speaker.tts_cache.size, 100)
        self.assertEqual(speaker.tts_cache.stats["evictions"], 1)
        self.assertFalse((await speaker.speak("0", "again")).tts_cache_hit)

    async def test_zero_disables_cache_and_failures_are_not_cached(self):
        conversation = SimpleNamespace(say=AsyncMock(side_effect=ConversationError("offline")))
        tts = SimpleNamespace(
            synthesize=AsyncMock(return_value=b"audio"),
            resource_id="resource",
            voice="voice",
            max_audio_bytes=100,
        )
        daemon = SimpleNamespace(upload_and_play=AsyncMock(), close=AsyncMock())
        disabled = ReachySpeaker(conversation, tts, daemon, cache_entries=0)
        await disabled.speak("same", "r1")
        await disabled.speak("same", "r2")
        self.assertEqual(tts.synthesize.await_count, 2)
        self.assertEqual(disabled.tts_cache.size, 0)

        tts.synthesize.reset_mock()
        tts.synthesize.side_effect = [TtsProtocolError("failed"), b"ok"]
        enabled = ReachySpeaker(conversation, tts, daemon)
        with self.assertRaises(SpeakerError):
            await enabled.speak("retry", "r3")
        result = await enabled.speak("retry", "r4")
        self.assertFalse(result.tts_cache_hit)
        self.assertEqual(tts.synthesize.await_count, 2)

    async def test_close_clears_cache(self):
        conversation = SimpleNamespace(say=AsyncMock(side_effect=ConversationError("offline")))
        tts = SimpleNamespace(
            synthesize=AsyncMock(return_value=b"audio"),
            resource_id="resource",
            voice="voice",
            max_audio_bytes=100,
        )
        daemon = SimpleNamespace(upload_and_play=AsyncMock(), close=AsyncMock())
        speaker = ReachySpeaker(conversation, tts, daemon)
        await speaker.speak("cached", "r1")
        await speaker.close()
        self.assertEqual(speaker.tts_cache.size, 0)
        daemon.close.assert_awaited_once()


class TtsTests(unittest.IsolatedAsyncioTestCase):
    def make_client(self, ws, maximum=20):
        return VolcengineTtsClient(FakeSession(ws=ws), url="wss://tts", app_id="app", access_key="key", resource_id="seed-tts-2.0", voice="voice", timeout_s=1, max_audio_bytes=maximum)

    async def test_accumulates_audio(self):
        ws = FakeWs([
            server_event(CONNECTION_STARTED),
            server_event(SESSION_STARTED),
            audio_frame(b"abc"),
            audio_frame(b"def"),
            server_event(SESSION_FINISHED),
            server_event(CONNECTION_FINISHED),
        ])
        client = self.make_client(ws)
        result = await client.synthesize("hello", "request")
        self.assertEqual(result, b"abcdef")
        self.assertEqual(len(ws.sent_bytes), 5)
        self.assertEqual([decode_tts_frame(frame).event for frame in ws.sent_bytes], [1, 100, 200, 102, 2])
        headers = client.session.ws_calls[0][1]["headers"]
        self.assertEqual(headers["X-Api-App-Key"], "app")
        self.assertNotIn("X-Api-App-Id", headers)
        self.assertNotIn("X-Api-Connect-Id", headers)
        start_payload = json.loads(decode_tts_frame(ws.sent_bytes[1]).payload)
        task_payload = json.loads(decode_tts_frame(ws.sent_bytes[2]).payload)
        self.assertTrue(start_payload["req_params"]["audio_params"]["enable_timestamp"])
        self.assertEqual(
            json.loads(task_payload["req_params"]["additions"]),
            {"disable_markdown_filter": False},
        )
        self.assertEqual(task_payload["req_params"]["text"], "hello")

    async def test_audio_limit(self):
        ws = FakeWs([server_event(CONNECTION_STARTED), server_event(SESSION_STARTED), audio_frame(b"123456")])
        with self.assertRaisesRegex(TtsProtocolError, "exceeds"):
            await self.make_client(ws, maximum=5).synthesize("hello", "request")

    async def test_protocol_error(self):
        ws = FakeWs([server_event(CONNECTION_STARTED), server_event(SESSION_STARTED), error_frame(45000002)])
        with self.assertRaisesRegex(TtsProtocolError, "45000002"):
            await self.make_client(ws).synthesize("hello", "request")

    def test_decode_rejects_truncated_frame(self):
        with self.assertRaises(TtsProtocolError):
            decode_tts_frame(b"\x11")


class DaemonTests(unittest.IsolatedAsyncioTestCase):
    async def test_upload_play_and_delayed_cleanup(self):
        session = FakeSession(posts=[FakeResponse(payload={"status": "ok", "path": "sounds/x.mp3"}), FakeResponse()], delete=FakeResponse())
        client = DaemonSoundClient(session, "http://daemon", cleanup_delay_s=0)
        await client.upload_and_play(b"audio", "safe-id")
        await asyncio.sleep(0.01)
        self.assertTrue(session.post_calls[0][0].endswith("/api/media/sounds/upload"))
        self.assertIn("data", session.post_calls[0][1])
        self.assertIsInstance(session.post_calls[0][1]["timeout"], aiohttp.ClientTimeout)
        self.assertEqual(session.post_calls[0][1]["timeout"].total, 10.0)
        self.assertEqual(session.post_calls[1][1]["json"], {"file": "sounds/x.mp3"})
        self.assertEqual(session.post_calls[1][1]["timeout"].total, 10.0)
        self.assertIn("active-speech-safe-id-", session.delete_calls[0][0])
        self.assertEqual(session.delete_calls[0][1]["timeout"].total, 10.0)
        await client.close()

    async def test_close_cancels_cleanup_tasks(self):
        session = FakeSession(posts=[FakeResponse(payload={"path": "x.mp3"}), FakeResponse()], delete=FakeResponse())
        client = DaemonSoundClient(session, "http://daemon", cleanup_delay_s=3600)
        await client.upload_and_play(b"audio", "r")
        self.assertEqual(len(client._cleanup_tasks), 1)
        await client.close()
        self.assertFalse(client._cleanup_tasks)
        self.assertFalse(session.delete_calls)


class HttpTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.speaker = SimpleNamespace(speak=AsyncMock(return_value="conversation"))
        self.gate_opener = AsyncMock(return_value=(True, "proactive_reply_armed"))
        self.server = TestServer(create_speak_app(
            self.speaker,
            total_timeout_s=1,
            max_text_bytes=10,
            gate_opener=self.gate_opener,
        ))
        self.client = TestClient(self.server)
        await self.client.start_server()

    async def asyncTearDown(self):
        await self.client.close()

    async def test_speak_requires_no_authorization(self):
        response = await self.client.post("/speak", json={"text": " hi "})
        self.assertEqual(response.status, 200)
        payload = await response.json()
        self.assertEqual(payload["route"], "conversation")
        self.assertFalse(payload["tts_cache_hit"])
        self.speaker.speak.assert_awaited_once()

    async def test_success_reports_tts_cache_hit(self):
        self.speaker.speak.return_value = SpeakResult("daemon_tts", tts_cache_hit=True)
        response = await self.client.post(
            "/speak", json={"text": "hi"}
        )
        payload = await response.json()
        self.assertEqual(response.status, 200)
        self.assertEqual(payload["route"], "daemon_tts")
        self.assertTrue(payload["tts_cache_hit"])

    async def test_empty_and_too_long(self):
        self.assertEqual((await self.client.post("/speak", json={"text": "  "})).status, 400)
        self.assertEqual((await self.client.post("/speak", json={"text": "中文中文"})).status, 400)

    async def test_open_gate_default_false_and_true(self):
        response = await self.client.post("/speak", json={"text": "hi"})
        payload = await response.json()
        self.assertFalse(payload["gate_requested"])
        self.assertFalse(payload["gate_opened"])
        self.assertEqual(payload["fallback_allowed"], True)
        self.gate_opener.assert_not_awaited()

        response = await self.client.post(
            "/speak", json={"text": "hi", "open_gate": True}
        )
        payload = await response.json()
        self.assertTrue(payload["gate_opened"])
        self.assertEqual(payload["gate_reason"], "proactive_reply_armed")

    async def test_gate_cardinality_and_non_enforced_reasons_are_success(self):
        for result in (
            (False, "no_active_connection"),
            (False, "ambiguous_connections"),
            (False, "gate_not_enforced"),
        ):
            self.gate_opener.return_value = result
            response = await self.client.post(
                "/speak", json={"text": "hi", "open_gate": True}
            )
            self.assertEqual(response.status, 200)
            self.assertEqual((await response.json())["gate_reason"], result[1])

    async def test_invalid_boolean_types(self):
        for body in (
            {"text": "hi", "open_gate": "true"},
            {"text": "hi", "allow_tts_fallback": "false"},
        ):
            self.assertEqual((await self.client.post("/speak", json=body)).status, 400)
        self.speaker.speak.assert_not_awaited()

    async def test_fallback_disabled_failure_does_not_open_gate(self):
        self.speaker.speak.side_effect = FallbackDisabledError("disabled")
        response = await self.client.post(
            "/speak",
            json={"text": "hi", "open_gate": True, "allow_tts_fallback": False},
        )
        payload = await response.json()
        self.assertEqual(response.status, 502)
        self.assertEqual(payload["reason"], "fallback_disabled")
        self.assertFalse(payload["fallback_allowed"])
        self.gate_opener.assert_not_awaited()

    async def test_gate_opener_exception_still_returns_200_and_speaks_once(self):
        self.gate_opener.side_effect = RuntimeError("boom")
        response = await self.client.post(
            "/speak",
            json={"text": "hi", "open_gate": True, "allow_tts_fallback": False},
        )
        self.assertEqual(response.status, 200)
        payload = await response.json()
        self.assertEqual(payload["gate_reason"], "gate_open_failed")
        self.assertFalse(payload["fallback_allowed"])
        self.speaker.speak.assert_awaited_once()

    async def test_gate_opener_timeout_error_is_not_mislabeled(self):
        self.gate_opener.side_effect = TimeoutError("gate deadline")
        response = await self.client.post(
            "/speak",
            json={"text": "hi", "open_gate": True},
        )
        self.assertEqual(response.status, 200)
        payload = await response.json()
        self.assertFalse(payload["gate_opened"])
        self.assertEqual(payload["gate_reason"], "gate_open_timeout")
        self.speaker.speak.assert_awaited_once()

    async def test_gate_opener_timeout_still_returns_200_and_speaks_once(self):
        async def slow_gate():
            await asyncio.sleep(1)
            return True, "late"

        await self.client.close()
        self.server = TestServer(create_speak_app(
            self.speaker,
            total_timeout_s=0.01,
            gate_opener=slow_gate,
        ))
        self.client = TestClient(self.server)
        await self.client.start_server()
        response = await self.client.post(
            "/speak",
            json={"text": "hi", "open_gate": True},
        )
        self.assertEqual(response.status, 200)
        self.assertEqual((await response.json())["gate_reason"], "gate_open_timeout")
        self.speaker.speak.assert_awaited_once()

    async def test_health_has_no_external_calls(self):
        response = await self.client.get("/health")
        self.assertEqual(response.status, 200)
        self.speaker.speak.assert_not_awaited()

    async def test_both_routes_failed_is_502(self):
        self.speaker.speak.side_effect = SpeakerError("both failed")
        response = await self.client.post("/speak", json={"text": "hi"})
        self.assertEqual(response.status, 502)


if __name__ == "__main__":
    unittest.main()
