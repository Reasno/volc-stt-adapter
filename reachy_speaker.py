"""Active speech routes for Reachy Mini.

The Seed-TTS v3 framing below is a deliberately small local implementation
aligned first with the Volcengine HA integration deployed by the user
(``custom_components/volcengine_voice_assistant/sdk/tts.py``), with the
verified ``doubao-speech`` implementation as a secondary cross-check. Neither
package is a runtime dependency.
"""
from __future__ import annotations

import asyncio
import contextlib
import gzip
import hashlib
import json
import logging
import struct
import uuid
from collections import OrderedDict
from dataclasses import dataclass
from threading import Lock
from typing import Any, Awaitable, Callable, Hashable

import aiohttp
from aiohttp import web

LOG = logging.getLogger("volc_stt_adapter.speaker")


class SpeakerError(RuntimeError):
    """An external speech route failed in a diagnosable way."""


class ConversationError(SpeakerError):
    pass


class TtsProtocolError(SpeakerError):
    pass


class DaemonSoundError(SpeakerError):
    pass


class FallbackDisabledError(SpeakerError):
    """conversation.say failed and the caller prohibited TTS fallback."""

    reason = "fallback_disabled"


# Volcengine v3 bidirectional TTS constants.
FULL_CLIENT_REQUEST = 0x1
FULL_SERVER_RESPONSE = 0x9
AUDIO_ONLY_SERVER = 0xB
ERROR_RESPONSE = 0xF
WITH_EVENT = 0x4
SERIALIZATION_JSON = 0x1
START_CONNECTION = 1
FINISH_CONNECTION = 2
CONNECTION_STARTED = 50
CONNECTION_FAILED = 51
CONNECTION_FINISHED = 52
START_SESSION = 100
FINISH_SESSION = 102
SESSION_STARTED = 150
SESSION_FINISHED = 152
SESSION_FAILED = 153
TASK_REQUEST = 200
TTS_SENTENCE_START = 350
TTS_SENTENCE_END = 351
TTS_RESPONSE = 352
TTS_ENDED = 359
CONNECTION_EVENTS = {START_CONNECTION, FINISH_CONNECTION, CONNECTION_STARTED, CONNECTION_FAILED, CONNECTION_FINISHED}


@dataclass(frozen=True)
class TtsFrame:
    message_type: int
    event: int = 0
    payload: bytes = b""
    session_id: str = ""
    error_code: int = 0


def encode_tts_event(event: int, payload: dict[str, Any] | bytes, session_id: str = "") -> bytes:
    raw = payload if isinstance(payload, bytes) else json.dumps(payload, ensure_ascii=False).encode("utf-8")
    output = bytearray((0x11, (FULL_CLIENT_REQUEST << 4) | WITH_EVENT, SERIALIZATION_JSON << 4, 0))
    output.extend(struct.pack(">i", event))
    if event not in CONNECTION_EVENTS:
        sid = session_id.encode("utf-8")
        output.extend(struct.pack(">I", len(sid)))
        output.extend(sid)
    output.extend(struct.pack(">I", len(raw)))
    output.extend(raw)
    return bytes(output)


def decode_tts_frame(data: bytes) -> TtsFrame:
    """Parse the event/audio/error subset emitted by bidirectional TTS."""
    if len(data) < 4 or data[0] >> 4 != 1:
        raise TtsProtocolError("TTS returned a malformed protocol header")
    offset = (data[0] & 0x0F) * 4
    if offset < 4 or offset > len(data):
        raise TtsProtocolError("TTS returned an invalid header size")
    message_type = data[1] >> 4
    flags = data[1] & 0x0F
    compression = data[2] & 0x0F
    event = error_code = 0
    session_id = ""

    def take(size: int) -> bytes:
        nonlocal offset
        if size < 0 or offset + size > len(data):
            raise TtsProtocolError("TTS returned a truncated frame")
        value = data[offset : offset + size]
        offset += size
        return value

    # Match the deployed HA parser: sequence/error fields precede event fields
    # on received frames. Event-only control frames are unaffected.
    if message_type == ERROR_RESPONSE:
        error_code = struct.unpack(">I", take(4))[0]
    elif flags in (1, 3) and message_type in (FULL_SERVER_RESPONSE, AUDIO_ONLY_SERVER):
        take(4)  # sequence
    if flags == WITH_EVENT:
        event = struct.unpack(">i", take(4))[0]
        if event not in CONNECTION_EVENTS:
            sid_len = struct.unpack(">I", take(4))[0]
            session_id = take(sid_len).decode("utf-8", errors="replace")
        elif message_type == FULL_SERVER_RESPONSE:
            # Connection responses carry connect_id, which is not needed here.
            connect_len = struct.unpack(">I", take(4))[0]
            take(connect_len)
    payload_len = struct.unpack(">I", take(4))[0]
    payload = take(payload_len)
    if compression == 1 and payload:
        try:
            payload = gzip.decompress(payload)
        except OSError as exc:
            raise TtsProtocolError("TTS returned invalid gzip data") from exc
    return TtsFrame(message_type, event, payload, session_id, error_code)


class ConversationSayClient:
    def __init__(self, session: aiohttp.ClientSession, url: str, timeout_s: float = 5.0):
        self.session = session
        self.url = url
        self.timeout_s = timeout_s

    async def say(self, text: str) -> None:
        request_id = uuid.uuid4().hex
        request = {"jsonrpc": "2.0", "id": request_id, "method": "conversation.say", "params": {"text": text}}
        try:
            async with asyncio.timeout(self.timeout_s):
                async with self.session.ws_connect(self.url, heartbeat=20) as ws:
                    await ws.send_json(request)
                    while True:
                        message = await ws.receive()
                        if message.type == aiohttp.WSMsgType.TEXT:
                            try:
                                response = json.loads(message.data)
                            except json.JSONDecodeError:
                                continue
                            # Notifications and responses for unrelated calls are ignored.
                            if response.get("id") != request_id:
                                continue
                            if "error" in response:
                                error = response.get("error") or {}
                                reason = error.get("data", {}).get("reason") if isinstance(error.get("data"), dict) else None
                                raise ConversationError(f"conversation.say RPC error: {reason or error.get('message') or 'unknown'}")
                            if "result" not in response:
                                raise ConversationError("conversation.say returned an invalid JSON-RPC response")
                            return
                        if message.type in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                            raise ConversationError("conversation WebSocket closed before a matching response")
        except ConversationError:
            raise
        except TimeoutError as exc:
            raise ConversationError("conversation.say timed out") from exc
        except (aiohttp.ClientError, OSError) as exc:
            raise ConversationError(f"conversation connection failed: {type(exc).__name__}") from exc


class VolcengineTtsClient:
    def __init__(self, session: aiohttp.ClientSession, *, url: str, app_id: str, access_key: str,
                 resource_id: str, voice: str, timeout_s: float, max_audio_bytes: int):
        self.session = session
        self.url = url
        self.app_id = app_id
        self.access_key = access_key
        self.resource_id = resource_id
        self.voice = voice
        self.timeout_s = timeout_s
        self.max_audio_bytes = max_audio_bytes

    def _auth_cache_identity(self) -> tuple[str, str]:
        """Return a non-secret auth mode/fingerprint for cache partitioning."""
        mode = "app_access_key" if self.access_key else "api_key"
        material = f"{mode}\0{self.app_id}\0{self.access_key}".encode("utf-8")
        return mode, hashlib.sha256(material).hexdigest()

    def _build_headers(self, request_id: str) -> dict[str, str]:
        if self.access_key:
            return {
                "X-Api-App-Key": self.app_id,
                "X-Api-Access-Key": self.access_key,
                "X-Api-Resource-Id": self.resource_id,
                "X-Api-Request-Id": request_id,
            }
        return {
            "X-Api-Key": self.app_id,
            "X-Api-Resource-Id": self.resource_id,
            "X-Api-Connect-Id": str(uuid.uuid4()),
        }

    def cache_key(self, text: str) -> tuple[Hashable, ...]:
        """Identify every configured input that can affect synthesized bytes."""
        auth_mode, auth_identity = self._auth_cache_identity()
        return (
            self.url,
            auth_mode,
            auth_identity,
            self.resource_id,
            self.voice,
            "mp3",
            24000,
            True,
            False,  # disable_markdown_filter
            text,
        )

    async def synthesize(self, text: str, request_id: str) -> bytes:
        if not self.voice:
            raise TtsProtocolError("VOLC_TTS_VOICE is required for daemon TTS fallback")
        if not self.app_id:
            raise TtsProtocolError("Volcengine TTS app key is not configured")
        headers = self._build_headers(request_id)
        session_id = str(uuid.uuid4())
        audio_params = {
            "format": "mp3",
            "sample_rate": 24000,
            "enable_timestamp": True,
        }
        req_params = {
            "speaker": self.voice,
            "audio_params": audio_params,
            "additions": json.dumps({"disable_markdown_filter": False}),
        }
        base = {"user": {"uid": request_id}, "namespace": "BidirectionalTTS", "req_params": req_params}
        try:
            async with asyncio.timeout(self.timeout_s):
                async with self.session.ws_connect(self.url, headers=headers, heartbeat=20, max_msg_size=self.max_audio_bytes + 65536) as ws:
                    await ws.send_bytes(encode_tts_event(START_CONNECTION, b"{}"))
                    await self._expect(ws, CONNECTION_STARTED)
                    await ws.send_bytes(encode_tts_event(START_SESSION, {**base, "event": START_SESSION}, session_id))
                    await self._expect(ws, SESSION_STARTED)
                    task = {**base, "event": TASK_REQUEST, "req_params": {**req_params, "text": text}}
                    await ws.send_bytes(encode_tts_event(TASK_REQUEST, task, session_id))
                    await ws.send_bytes(encode_tts_event(FINISH_SESSION, b"{}", session_id))
                    audio = bytearray()
                    while True:
                        frame = await self._receive(ws)
                        if frame.message_type == AUDIO_ONLY_SERVER:
                            if len(audio) + len(frame.payload) > self.max_audio_bytes:
                                raise TtsProtocolError(f"TTS audio exceeds {self.max_audio_bytes} byte limit")
                            audio.extend(frame.payload)
                        elif frame.event in (TTS_ENDED, SESSION_FINISHED):
                            break
                    if not audio:
                        raise TtsProtocolError("TTS returned no audio")
                    # HA waits for ConnectionFinished after SessionFinished;
                    # this catches protocol errors during orderly teardown.
                    await ws.send_bytes(encode_tts_event(FINISH_CONNECTION, b"{}"))
                    await self._expect(ws, CONNECTION_FINISHED)
                    return bytes(audio)
        except TtsProtocolError:
            raise
        except TimeoutError as exc:
            raise TtsProtocolError("TTS request timed out") from exc
        except (aiohttp.ClientError, OSError) as exc:
            raise TtsProtocolError(f"TTS connection failed: {type(exc).__name__}") from exc

    async def _expect(self, ws: aiohttp.ClientWebSocketResponse, event: int) -> None:
        frame = await self._receive(ws)
        if frame.event != event:
            raise TtsProtocolError(f"TTS expected event {event}, got {frame.event}")

    async def _receive(self, ws: aiohttp.ClientWebSocketResponse) -> TtsFrame:
        message = await ws.receive()
        if message.type != aiohttp.WSMsgType.BINARY:
            raise TtsProtocolError(f"TTS returned unexpected WebSocket message type {message.type}")
        frame = decode_tts_frame(message.data)
        if frame.message_type == ERROR_RESPONSE or frame.event in (CONNECTION_FAILED, SESSION_FAILED):
            detail = frame.payload.decode("utf-8", errors="replace")[:500]
            raise TtsProtocolError(f"TTS protocol error code={frame.error_code} event={frame.event}: {detail}")
        return frame


class DaemonSoundClient:
    def __init__(
        self,
        session: aiohttp.ClientSession,
        base_url: str,
        cleanup_delay_s: float = 300.0,
        timeout_s: float = 10.0,
    ):
        self.session = session
        self.base_url = base_url.rstrip("/")
        self.cleanup_delay_s = cleanup_delay_s
        self.request_timeout = aiohttp.ClientTimeout(total=timeout_s)
        self._cleanup_tasks: set[asyncio.Task[None]] = set()

    async def upload_and_play(self, audio: bytes, request_id: str) -> None:
        filename = f"active-speech-{request_id}-{uuid.uuid4().hex}.mp3"
        form = aiohttp.FormData()
        form.add_field("file", audio, filename=filename, content_type="audio/mpeg")
        try:
            async with self.session.post(
                f"{self.base_url}/api/media/sounds/upload",
                data=form,
                timeout=self.request_timeout,
            ) as response:
                if response.status >= 400:
                    raise DaemonSoundError(f"daemon upload failed: HTTP {response.status}")
                payload = await response.json()
            path = payload.get("path") if isinstance(payload, dict) else None
            if not isinstance(path, str) or not path:
                raise DaemonSoundError("daemon upload response has no path")
            async with self.session.post(
                f"{self.base_url}/api/media/play_sound",
                json={"file": path},
                timeout=self.request_timeout,
            ) as response:
                if response.status >= 400:
                    raise DaemonSoundError(f"daemon play_sound failed: HTTP {response.status}")
            task = asyncio.create_task(self._cleanup(filename), name=f"sound-cleanup-{request_id}")
            self._cleanup_tasks.add(task)
            task.add_done_callback(self._cleanup_tasks.discard)
        except DaemonSoundError:
            raise
        except (aiohttp.ClientError, OSError, ValueError) as exc:
            raise DaemonSoundError(f"daemon request failed: {type(exc).__name__}") from exc

    async def _cleanup(self, filename: str) -> None:
        try:
            await asyncio.sleep(self.cleanup_delay_s)
            async with self.session.delete(
                f"{self.base_url}/api/media/sounds/{filename}",
                timeout=self.request_timeout,
            ) as response:
                if response.status >= 400:
                    LOG.warning("Delayed daemon sound cleanup failed: HTTP %d", response.status)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            LOG.warning("Delayed daemon sound cleanup failed: %s", type(exc).__name__)

    async def close(self) -> None:
        tasks = tuple(self._cleanup_tasks)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._cleanup_tasks.clear()


class AudioLruCache:
    """Small thread-safe LRU that retains immutable audio bytes by reference."""

    def __init__(self, max_entries: int):
        if max_entries < 0:
            raise ValueError("max_entries must be non-negative")
        self.max_entries = max_entries
        self._items: OrderedDict[Hashable, bytes] = OrderedDict()
        self._lock = Lock()
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    def get(self, key: Hashable) -> bytes | None:
        with self._lock:
            if self.max_entries == 0 or key not in self._items:
                self.misses += 1
                return None
            audio = self._items.pop(key)
            self._items[key] = audio
            self.hits += 1
            return audio

    def put(self, key: Hashable, audio: bytes) -> None:
        if self.max_entries == 0:
            return
        with self._lock:
            if key in self._items:
                self._items.pop(key)
            self._items[key] = audio
            while len(self._items) > self.max_entries:
                self._items.popitem(last=False)
                self.evictions += 1

    def clear(self) -> None:
        with self._lock:
            self._items.clear()

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._items)

    @property
    def stats(self) -> dict[str, int]:
        with self._lock:
            return {
                "size": len(self._items),
                "hits": self.hits,
                "misses": self.misses,
                "evictions": self.evictions,
            }


class SpeakResult(str):
    """String-compatible route carrying per-request cache observability."""

    tts_cache_hit: bool

    def __new__(cls, route: str, *, tts_cache_hit: bool = False) -> "SpeakResult":
        result = super().__new__(cls, route)
        result.tts_cache_hit = tts_cache_hit
        return result


class ReachySpeaker:
    def __init__(
        self,
        conversation: ConversationSayClient,
        tts: VolcengineTtsClient,
        daemon: DaemonSoundClient,
        *,
        cache_entries: int = 100,
    ):
        self.conversation = conversation
        self.tts = tts
        self.daemon = daemon
        self.tts_cache = AudioLruCache(cache_entries)
        self._lock = asyncio.Lock()

    def _tts_cache_key(self, text: str) -> Hashable:
        key_factory = getattr(self.tts, "cache_key", None)
        if callable(key_factory):
            return key_factory(text)
        return (
            getattr(self.tts, "resource_id", None),
            getattr(self.tts, "voice", None),
            text,
        )

    async def speak(
        self,
        text: str,
        request_id: str,
        *,
        allow_tts_fallback: bool = True,
    ) -> SpeakResult:
        async with self._lock:
            conversation_failure = "unknown"
            try:
                await self.conversation.say(text)
                LOG.info("Speech route ready: request_id=%s route=conversation tts_cache_hit=false", request_id)
                return SpeakResult("conversation")
            except ConversationError as conversation_error:
                conversation_failure = str(conversation_error)
                if not allow_tts_fallback:
                    raise FallbackDisabledError(
                        f"conversation.say failed and TTS fallback is disabled: {conversation_failure}"
                    ) from conversation_error
                LOG.info("conversation.say unavailable; using daemon TTS fallback: %s", conversation_failure)
            try:
                cache_key = self._tts_cache_key(text)
                audio = self.tts_cache.get(cache_key)
                cache_hit = audio is not None
                if audio is None:
                    audio = await self.tts.synthesize(text, request_id)
                    if audio and len(audio) <= getattr(self.tts, "max_audio_bytes", len(audio)):
                        self.tts_cache.put(cache_key, audio)
                LOG.info("TTS fallback ready: request_id=%s tts_cache_hit=%s", request_id, cache_hit)
                await self.daemon.upload_and_play(audio, request_id)
                return SpeakResult("daemon_tts", tts_cache_hit=cache_hit)
            except Exception as fallback_error:
                raise SpeakerError(
                    f"both speech routes failed: conversation={conversation_failure}; fallback={fallback_error}"
                ) from fallback_error

    async def close(self) -> None:
        self.tts_cache.clear()
        await self.daemon.close()


def create_speak_app(
    speaker: ReachySpeaker,
    *,
    total_timeout_s: float,
    max_text_bytes: int = 4000,
    gate_opener: Callable[[], Awaitable[tuple[bool, str]]] | None = None,
) -> web.Application:
    app = web.Application(client_max_size=16 * 1024)

    async def health(_: web.Request) -> web.Response:
        return web.json_response({"ok": True})

    async def speak(request: web.Request) -> web.Response:
        request_id = uuid.uuid4().hex
        try:
            body = await request.json()
        except (json.JSONDecodeError, ValueError):
            return web.json_response({"ok": False, "error": "request body must be JSON", "route": None, "tts_cache_hit": False, "fallback_allowed": True, "request_id": request_id}, status=400)
        text = body.get("text") if isinstance(body, dict) else None
        open_gate = body.get("open_gate", False) if isinstance(body, dict) else False
        allow_tts_fallback = (
            body.get("allow_tts_fallback", True) if isinstance(body, dict) else True
        )
        if not isinstance(open_gate, bool):
            return web.json_response({"ok": False, "error": "open_gate must be a JSON boolean", "route": None, "tts_cache_hit": False, "fallback_allowed": allow_tts_fallback if isinstance(allow_tts_fallback, bool) else True, "request_id": request_id}, status=400)
        if not isinstance(allow_tts_fallback, bool):
            return web.json_response({"ok": False, "error": "allow_tts_fallback must be a JSON boolean", "route": None, "tts_cache_hit": False, "fallback_allowed": True, "request_id": request_id}, status=400)
        if not isinstance(text, str) or not text.strip():
            return web.json_response({"ok": False, "error": "text must be a non-empty string", "route": None, "tts_cache_hit": False, "fallback_allowed": allow_tts_fallback, "request_id": request_id}, status=400)
        text = text.strip()
        if len(text.encode("utf-8")) > max_text_bytes:
            return web.json_response({"ok": False, "error": f"text exceeds {max_text_bytes} UTF-8 bytes", "route": None, "tts_cache_hit": False, "fallback_allowed": allow_tts_fallback, "request_id": request_id}, status=400)
        route: str | None = None
        try:
            async with asyncio.timeout(total_timeout_s):
                route = await speaker.speak(
                    text,
                    request_id,
                    allow_tts_fallback=allow_tts_fallback,
                )
                gate_opened = False
                gate_reason = "not_requested"
                if open_gate:
                    if gate_opener is None:
                        gate_reason = "no_active_connection"
                    else:
                        try:
                            gate_opened, gate_reason = await gate_opener()
                        except (TimeoutError, asyncio.CancelledError):
                            raise
                        except Exception as exc:
                            LOG.warning("Gate opening failed after successful speech: %s", type(exc).__name__)
                            gate_reason = "gate_open_failed"
            return web.json_response({
                "ok": True,
                "route": route,
                "tts_cache_hit": bool(getattr(route, "tts_cache_hit", False)),
                "request_id": request_id,
                "fallback_allowed": allow_tts_fallback,
                "gate_requested": open_gate,
                "gate_opened": gate_opened,
                "gate_reason": gate_reason,
            })
        except TimeoutError:
            if route is not None:
                return web.json_response({
                    "ok": True,
                    "route": route,
                    "tts_cache_hit": bool(getattr(route, "tts_cache_hit", False)),
                    "request_id": request_id,
                    "fallback_allowed": allow_tts_fallback,
                    "gate_requested": open_gate,
                    "gate_opened": False,
                    "gate_reason": "gate_open_timeout",
                })
            return web.json_response({"ok": False, "error": "speech request timed out", "route": None, "tts_cache_hit": False, "fallback_allowed": allow_tts_fallback, "request_id": request_id}, status=502)
        except FallbackDisabledError as exc:
            return web.json_response({"ok": False, "error": str(exc), "reason": exc.reason, "route": None, "tts_cache_hit": False, "fallback_allowed": False, "gate_requested": open_gate, "gate_opened": False, "gate_reason": "speech_failed", "request_id": request_id}, status=502)
        except SpeakerError as exc:
            return web.json_response({"ok": False, "error": str(exc), "route": None, "tts_cache_hit": False, "fallback_allowed": allow_tts_fallback, "gate_requested": open_gate, "gate_opened": False, "gate_reason": "speech_failed", "request_id": request_id}, status=502)

    app.router.add_get("/health", health)
    app.router.add_post("/speak", speak)
    return app
