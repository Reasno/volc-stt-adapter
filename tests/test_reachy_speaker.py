import asyncio
import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock

from aiohttp import WSMsgType
from aiohttp.test_utils import TestClient, TestServer

from reachy_speaker import (
    ConversationError,
    ConversationSayClient,
    ReachySpeaker,
    SpeakerError,
    create_speak_app,
)


class AsyncContext:
    def __init__(self, value):
        self.value = value

    async def __aenter__(self):
        return self.value

    async def __aexit__(self, *_):
        return None


class FakeWs:
    def __init__(self, messages):
        self.messages = list(messages)
        self.sent_json = []

    async def send_json(self, value):
        self.sent_json.append(value)

    async def receive(self):
        return self.messages.pop(0)


class FakeSession:
    def __init__(self, ws):
        self.ws = ws

    def ws_connect(self, *_args, **_kwargs):
        return AsyncContext(self.ws)


def ws_text(payload):
    return SimpleNamespace(type=WSMsgType.TEXT, data=json.dumps(payload))


class ConversationTests(unittest.IsolatedAsyncioTestCase):
    async def test_say_uses_conversation_rpc(self):
        ws = FakeWs([])
        client = ConversationSayClient(FakeSession(ws), "ws://reachy/rpc")

        async def send(value):
            ws.sent_json.append(value)
            ws.messages.append(ws_text({"jsonrpc": "2.0", "id": value["id"], "result": {}}))

        ws.send_json = send
        await client.say("你好")
        self.assertEqual(ws.sent_json[0]["method"], "conversation.say")

    async def test_rpc_error_is_failure(self):
        ws = FakeWs([])
        client = ConversationSayClient(FakeSession(ws), "ws://reachy/rpc")

        async def send(value):
            ws.messages.append(ws_text({"jsonrpc": "2.0", "id": value["id"], "error": {"message": "offline", "data": {"reason": "not_running"}}}))

        ws.send_json = send
        with self.assertRaisesRegex(ConversationError, "not_running"):
            await client.say("hello")


class SpeakerTests(unittest.IsolatedAsyncioTestCase):
    async def test_success_uses_conversation_only(self):
        conversation = SimpleNamespace(say=AsyncMock())
        route = await ReachySpeaker(conversation).speak("hi", "r1")
        self.assertEqual(route, "conversation")
        conversation.say.assert_awaited_once_with("hi")

    async def test_conversation_failure_is_returned_without_fallback(self):
        conversation = SimpleNamespace(say=AsyncMock(side_effect=ConversationError("not_running")))
        with self.assertRaisesRegex(SpeakerError, "conversation.say failed: not_running"):
            await ReachySpeaker(conversation).speak("hi", "r1")


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

    async def test_success(self):
        response = await self.client.post("/speak", json={"text": " hi "})
        self.assertEqual(response.status, 200)
        payload = await response.json()
        self.assertEqual(payload["route"], "conversation")
        self.assertNotIn("fallback_allowed", payload)
        self.assertNotIn("tts_cache_hit", payload)
        self.speaker.speak.assert_awaited_once()

    async def test_legacy_fallback_option_is_ignored(self):
        response = await self.client.post(
            "/speak", json={"text": "hi", "allow_tts_fallback": True}
        )
        self.assertEqual(response.status, 200)
        self.speaker.speak.assert_awaited_once()

    async def test_empty_and_too_long(self):
        self.assertEqual((await self.client.post("/speak", json={"text": "  "})).status, 400)
        self.assertEqual((await self.client.post("/speak", json={"text": "中文中文"})).status, 400)

    async def test_open_gate(self):
        response = await self.client.post(
            "/speak", json={"text": "hi", "open_gate": True}
        )
        payload = await response.json()
        self.assertTrue(payload["gate_opened"])
        self.assertEqual(payload["gate_reason"], "proactive_reply_armed")

    async def test_invalid_open_gate_type(self):
        response = await self.client.post(
            "/speak", json={"text": "hi", "open_gate": "true"}
        )
        self.assertEqual(response.status, 400)
        self.speaker.speak.assert_not_awaited()

    async def test_conversation_failure_does_not_open_gate(self):
        self.speaker.speak.side_effect = SpeakerError("conversation.say failed: offline")
        response = await self.client.post(
            "/speak", json={"text": "hi", "open_gate": True}
        )
        payload = await response.json()
        self.assertEqual(response.status, 502)
        self.assertIn("conversation.say failed", payload["error"])
        self.assertFalse(payload["gate_opened"])
        self.gate_opener.assert_not_awaited()

    async def test_gate_timeout_after_speech_is_success(self):
        self.gate_opener.side_effect = TimeoutError("gate deadline")
        response = await self.client.post(
            "/speak", json={"text": "hi", "open_gate": True}
        )
        self.assertEqual(response.status, 200)
        self.assertEqual((await response.json())["gate_reason"], "gate_open_timeout")


if __name__ == "__main__":
    unittest.main()
