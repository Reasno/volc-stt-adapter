#!/usr/bin/env python3
"""OpenAI Realtime-compatible STT adapter backed by Volcengine streaming ASR."""

from __future__ import annotations

import asyncio
import base64
import contextlib
import gzip
import json
import logging
import os
import signal
import struct
import time
import uuid
import wave
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

import aiohttp
from websockets.asyncio.client import connect
from websockets.asyncio.server import ServerConnection, serve
from websockets.exceptions import ConnectionClosed

LOG = logging.getLogger("volc_stt_adapter")

# Volcengine binary protocol constants.
CLIENT_FULL_REQUEST = 0x1
CLIENT_AUDIO_ONLY_REQUEST = 0x2
SERVER_FULL_RESPONSE = 0x9
SERVER_ERROR_RESPONSE = 0xF
POS_SEQUENCE = 0x1
NEG_WITH_SEQUENCE = 0x3
SERIALIZATION_JSON = 0x1
COMPRESSION_GZIP = 0x1

KEYWORD_GATE_WORDS = ("reachy", "瑞奇", "瑞琪", "瑞吉", "richie", "ricky", "richey", "riche", "reach")
# Prefix-only wake words: open the gate only when the utterance *starts* with
# one of these (common ASR mishearings of the wake phrase); a mid-sentence
# occurrence must not count. Override with VOLC_GATE_PREFIX_WORDS (comma-separated).
DEFAULT_KEYWORD_GATE_PREFIX_WORDS = ("语音", "微信", "一起", "云溪", "微启", "允许", "机器", "运气")
KEYWORD_GATE_WINDOW_S = 30.0


def load_dotenv(path: str | None) -> None:
    """Load a small KEY=VALUE file without adding python-dotenv as a dependency."""
    if not path:
        return
    env_path = Path(path)
    if not env_path.is_file():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key, value)


def credentials_from_ha(path: str) -> dict[str, str]:
    """Read the first Volcengine STT subentry without logging secret values."""
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    for entry in payload.get("data", {}).get("entries", []):
        if "volcengine" not in str(entry.get("domain", "")).lower():
            continue
        subentries = entry.get("subentries", [])
        if isinstance(subentries, dict):
            subentries = subentries.values()
        for subentry in subentries:
            if subentry.get("subentry_type") != "stt":
                continue
            data = subentry.get("data", {})
            return {
                "app_key": str(data.get("app_key", "")),
                "access_key": str(data.get("access_key", "")),
                "resource_id": str(data.get("resource_id", "")),
                "url": str(data.get("url", "")),
            }
    raise RuntimeError(f"No Volcengine STT subentry found in {path}")


@dataclass(frozen=True)
class Settings:
    host: str
    port: int
    path: str
    volc_url: str
    app_key: str
    access_key: str
    resource_id: str
    sample_rate: int
    volc_timeout_s: float
    boosting_table_id: str
    keyword_gate_enabled: bool
    keyword_gate_prefix_words: tuple[str, ...]
    upstream_mode: str
    upstream_url: str
    upstream_session_url: str
    hf_token: str
    reachy_daemon_url: str
    upstream_open_timeout_s: float

    @classmethod
    def from_environment(cls) -> "Settings":
        load_dotenv(os.getenv("ENV_FILE", str(Path(__file__).with_name(".env"))))
        app_key = os.getenv("VOLC_APP_KEY", "").strip()
        access_key = os.getenv("VOLC_ACCESS_KEY", "").strip()
        resource_id = os.getenv("VOLC_RESOURCE_ID", "").strip()
        volc_url = os.getenv("VOLC_ASR_URL", "").strip()

        ha_path = os.getenv("HA_CONFIG_ENTRIES", "").strip()
        if (not app_key or not resource_id) and ha_path:
            from_ha = credentials_from_ha(ha_path)
            app_key = app_key or from_ha["app_key"]
            access_key = access_key or from_ha["access_key"]
            resource_id = resource_id or from_ha["resource_id"]
            volc_url = volc_url or from_ha["url"]

        missing = [
            name
            for name, value in (("VOLC_APP_KEY", app_key), ("VOLC_RESOURCE_ID", resource_id))
            if not value
        ]
        if missing:
            raise RuntimeError(f"Missing required credentials: {', '.join(missing)}")

        prefix_words_raw = os.getenv("VOLC_GATE_PREFIX_WORDS", "").replace("，", ",")
        keyword_gate_prefix_words = tuple(
            word.strip() for word in prefix_words_raw.split(",") if word.strip()
        ) or DEFAULT_KEYWORD_GATE_PREFIX_WORDS

        return cls(
            host=os.getenv("ADAPTER_HOST", "0.0.0.0"),
            port=int(os.getenv("ADAPTER_PORT", "8765")),
            path=os.getenv("ADAPTER_PATH", "/v1/realtime"),
            volc_url=volc_url or "wss://openspeech.bytedance.com/api/v3/sauc/bigmodel_async",
            app_key=app_key,
            access_key=access_key,
            resource_id=resource_id,
            sample_rate=int(os.getenv("AUDIO_SAMPLE_RATE", "16000")),
            volc_timeout_s=float(os.getenv("VOLC_TIMEOUT_SECONDS", "30")),
            boosting_table_id=os.getenv("VOLC_BOOSTING_TABLE_ID", "").strip(),
            keyword_gate_enabled=os.getenv("KEYWORD_GATE_ENABLED", "true").strip().lower()
            not in {"0", "false", "no", "off"},
            keyword_gate_prefix_words=keyword_gate_prefix_words,
            upstream_mode=os.getenv("UPSTREAM_MODE", "allocator").strip().lower(),
            upstream_url=os.getenv("UPSTREAM_REALTIME_URL", "").strip(),
            upstream_session_url=os.getenv(
                "UPSTREAM_SESSION_URL",
                "https://pollen-robotics-reachy-mini-realtime-url.hf.space/session",
            ).strip(),
            hf_token=os.getenv("HF_TOKEN", "").strip(),
            reachy_daemon_url=os.getenv("REACHY_DAEMON_URL", "http://192.168.31.94:8000").rstrip("/"),
            upstream_open_timeout_s=float(os.getenv("UPSTREAM_OPEN_TIMEOUT_SECONDS", "20")),
        )


def protocol_header(message_type: int, flags: int, serialization: int, compression: int) -> bytes:
    return bytes((0x11, (message_type << 4) | flags, (serialization << 4) | compression, 0x00))


def encode_request(
    message_type: int,
    flags: int,
    sequence: int,
    payload: bytes,
    *,
    serialization: int = 0,
) -> bytes:
    compressed = gzip.compress(payload)
    seq = -sequence if flags == NEG_WITH_SEQUENCE else sequence
    return (
        protocol_header(message_type, flags, serialization, COMPRESSION_GZIP)
        + struct.pack(">i", seq)
        + struct.pack(">I", len(compressed))
        + compressed
    )


def parse_response(message: bytes) -> dict[str, Any]:
    if len(message) < 4:
        raise RuntimeError("Volcengine returned a truncated frame")
    header_words = message[0] & 0x0F
    message_type = message[1] >> 4
    flags = message[1] & 0x0F
    serialization = message[2] >> 4
    compression = message[2] & 0x0F
    payload = message[header_words * 4 :]
    sequence = None
    code = 0

    if flags & 0x01:
        if len(payload) < 4:
            raise RuntimeError("Volcengine response has no sequence")
        sequence = struct.unpack(">i", payload[:4])[0]
        payload = payload[4:]
    if flags & 0x04:
        if len(payload) < 4:
            raise RuntimeError("Volcengine response has no event")
        payload = payload[4:]

    if message_type == SERVER_FULL_RESPONSE:
        if len(payload) < 4:
            raise RuntimeError("Volcengine response has no payload size")
        size = struct.unpack(">I", payload[:4])[0]
        payload = payload[4 : 4 + size]
    elif message_type == SERVER_ERROR_RESPONSE:
        if len(payload) < 8:
            raise RuntimeError("Volcengine error response is truncated")
        code = struct.unpack(">i", payload[:4])[0]
        size = struct.unpack(">I", payload[4:8])[0]
        payload = payload[8 : 8 + size]

    if payload and compression == COMPRESSION_GZIP:
        payload = gzip.decompress(payload)
    decoded: Any = None
    if payload:
        decoded = json.loads(payload.decode("utf-8")) if serialization == SERIALIZATION_JSON else payload
    return {
        "message_type": message_type,
        "is_last": bool(flags & 0x02),
        "sequence": sequence,
        "code": code,
        "payload": decoded,
    }


def wav_chunk(pcm: bytes, sample_rate: int) -> bytes:
    output = BytesIO()
    with wave.open(output, "wb") as wav_file:
        wav_file.setframerate(sample_rate)
        wav_file.setsampwidth(2)
        wav_file.setnchannels(1)
        wav_file.writeframes(pcm)
    return output.getvalue()


class VolcengineStream:
    def __init__(self, settings: Settings, language: str, stream_id: str, on_definite):
        self.settings = settings
        self.language = language
        self.item_id = stream_id
        self.on_definite = on_definite
        self.http: aiohttp.ClientSession | None = None
        self.ws: aiohttp.ClientWebSocketResponse | None = None
        self.sequence = 1
        self.receiver: asyncio.Task[None] | None = None
        self.latest_text = ""
        self.last_emitted_text = ""
        self.error: Exception | None = None
        self._closed = False

    async def start(self, initial_pcm: bytes) -> None:
        headers = {
            "X-Api-Resource-Id": self.settings.resource_id,
            "X-Api-Connect-Id": str(uuid.uuid4()),
        }
        if self.settings.access_key:
            headers["X-Api-App-Key"] = self.settings.app_key
            headers["X-Api-Access-Key"] = self.settings.access_key
        else:
            headers["X-Api-Key"] = self.settings.app_key

        timeout = aiohttp.ClientTimeout(total=None, sock_connect=10, sock_read=self.settings.volc_timeout_s)
        self.http = aiohttp.ClientSession(timeout=timeout)
        self.ws = await self.http.ws_connect(self.settings.volc_url, headers=headers, heartbeat=20)
        request = {
            "user": {"uid": self.item_id},
            "audio": {
                "language": self.language,
                "format": "wav",
                "codec": "raw",
                "rate": self.settings.sample_rate,
                "bits": 16,
                "channel": 1,
            },
            "request": {
                "model_name": "bigmodel",
                "enable_itn": True,
                "enable_punc": True,
                "enable_ddc": True,
                "show_utterances": True,
                "result_type": "single",
                "enable_nonstream": True,
                "enable_speaker_info": True,
                # ASR2.0 (Resource ID volc.seedasr.sauc.duration) + SSD 200
                # 大模型 SSD 短对话聚类模式（<=5 人非会议场景）。ssd_mode=0
                # 适用于 3 分钟以内短交互；ssd_version=200 是官方要求 speaker
                # 分离必须搭配的字段，1.0 上不推荐、2.0 上才是推荐用法。
                "ssd_version": "200",
                "ssd_mode": 0,
                # 缩短 VAD 判停（默认 800ms → 600ms），让分句更早切开，
                # 减少跨说话人合并到同一 utterance 里而被聚成新 speaker_id。
                "end_window_size": 600,
            },
        }
        if self.settings.boosting_table_id:
            request["request"]["corpus"] = {
                "boosting_table_id": self.settings.boosting_table_id,
            }
        await self._send(CLIENT_FULL_REQUEST, POS_SEQUENCE, json.dumps(request).encode(), SERIALIZATION_JSON)
        acknowledgement = await self._receive_one()
        self._raise_for_error(acknowledgement)
        self.receiver = asyncio.create_task(self._receive_loop(), name=f"volc-recv-{self.item_id}")
        if initial_pcm:
            await self.send_audio(wav_chunk(initial_pcm, self.settings.sample_rate))

    async def _send(self, message_type: int, flags: int, payload: bytes, serialization: int = 0) -> None:
        if not self.ws:
            raise RuntimeError("Volcengine stream is not connected")
        frame = encode_request(message_type, flags, self.sequence, payload, serialization=serialization)
        self.sequence += 1
        await self.ws.send_bytes(frame)

    async def send_audio(self, pcm: bytes) -> None:
        if not self._closed:
            await self._send(CLIENT_AUDIO_ONLY_REQUEST, POS_SEQUENCE, pcm)

    async def finish(self) -> str:
        if self._closed:
            return self.latest_text
        self._closed = True
        if self.ws and not self.ws.closed:
            await self._send(CLIENT_AUDIO_ONLY_REQUEST, NEG_WITH_SEQUENCE, b"")
        if self.receiver:
            try:
                await asyncio.wait_for(self.receiver, timeout=self.settings.volc_timeout_s)
            except asyncio.TimeoutError:
                self.receiver.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await self.receiver
                raise RuntimeError("Timed out waiting for Volcengine final result")
        if self.error:
            raise self.error
        return self.latest_text

    async def close(self) -> None:
        if self.receiver and not self.receiver.done():
            self.receiver.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self.receiver
        if self.ws and not self.ws.closed:
            await self.ws.close()
        if self.http and not self.http.closed:
            await self.http.close()

    async def _receive_one(self) -> dict[str, Any]:
        if not self.ws:
            raise RuntimeError("Volcengine stream is not connected")
        message = await self.ws.receive()
        if message.type != aiohttp.WSMsgType.BINARY:
            raise RuntimeError(f"Unexpected Volcengine WebSocket message: {message.type}")
        return parse_response(message.data)

    @staticmethod
    def _raise_for_error(response: dict[str, Any]) -> None:
        if response["message_type"] == SERVER_ERROR_RESPONSE or response["code"]:
            raise RuntimeError(f"Volcengine ASR error: code={response['code']} payload={response['payload']}")

    async def _receive_loop(self) -> None:
        try:
            while True:
                response = await self._receive_one()
                self._raise_for_error(response)
                payload = response.get("payload")
                if isinstance(payload, dict):
                    result = payload.get("result")
                    if isinstance(result, dict):
                        LOG.debug("[%s] Volcengine result: %s", self.item_id, result)
                        text = str(result.get("text") or "").strip()
                        if text:
                            self.latest_text = text
                        utterances = result.get("utterances") or []
                        for utterance in utterances:
                            if not isinstance(utterance, dict) or not utterance.get("definite"):
                                continue
                            utterance_text = str(utterance.get("text") or "").strip()
                            if not utterance_text:
                                continue
                            additions = utterance.get("additions")
                            LOG.debug("[%s] utterance additions raw: %s | utterance keys: %s", self.item_id, additions, list(utterance.keys()))
                            speaker_id = None
                            if isinstance(additions, dict):
                                raw = additions.get("speaker_id") or additions.get("speaker")
                                if raw is not None:
                                    speaker_id = str(raw).strip() or None
                            if speaker_id is None:
                                raw = utterance.get("speaker_id")
                                if raw is not None:
                                    speaker_id = str(raw).strip() or None
                            LOG.info(
                                "[%s] native VAD definite: speaker_id=%s text=%s",
                                self.item_id,
                                speaker_id,
                                utterance_text,
                            )
                            await self.on_definite(utterance_text, speaker_id)
                if response["is_last"]:
                    return
        except asyncio.CancelledError:
            raise
        except ConnectionClosed:
            return
        except Exception as exc:
            self.error = exc
            LOG.exception("[%s] Volcengine receive failed", self.item_id)


class RealtimeAdapterConnection:
    def __init__(self, websocket: ServerConnection, settings: Settings):
        self.websocket = websocket
        self.settings = settings
        self.upstream: Any = None
        self._downstream_send_lock = asyncio.Lock()
        self._upstream_send_lock = asyncio.Lock()
        self.language = "zh-CN"
        self.session: dict[str, Any] = {}
        self.turn_detection: dict[str, Any] = {"type": "server_vad"}
        self.stream: VolcengineStream | None = None
        self.item_id: str | None = None
        self.upstream_response_active = False
        # Multi-speaker gate: any speaker who says a wake word gets their own
        # 30s activation window. Multiple speakers can be active concurrently;
        # each speaker's window is refreshed independently. Key is the
        # speaker_id string, or "__unknown__" when the ASR did not label the
        # utterance with a speaker.
        self.keyword_gate_speakers: dict[str, float] = {}

    async def emit(self, event_type: str, **fields: Any) -> None:
        event = {"event_id": f"event_{uuid.uuid4().hex}", "type": event_type, **fields}
        async with self._downstream_send_lock:
            await self.websocket.send(json.dumps(event, ensure_ascii=False))

    async def _resolve_upstream_url(self) -> str:
        if self.settings.upstream_mode == "direct":
            if not self.settings.upstream_url:
                raise RuntimeError("UPSTREAM_REALTIME_URL is required when UPSTREAM_MODE=direct")
            return self.settings.upstream_url
        if self.settings.upstream_mode != "allocator":
            raise RuntimeError(f"Unsupported UPSTREAM_MODE: {self.settings.upstream_mode}")

        hardware_id = ""
        timeout = aiohttp.ClientTimeout(total=self.settings.upstream_open_timeout_s)
        async with aiohttp.ClientSession(timeout=timeout) as http:
            if self.settings.reachy_daemon_url:
                try:
                    async with http.get(f"{self.settings.reachy_daemon_url}/api/daemon/status") as response:
                        response.raise_for_status()
                        status = await response.json()
                        hardware_id = str(status.get("hardware_id") or "")
                except Exception as exc:
                    LOG.warning("Could not read Reachy hardware id: %s", exc)

            headers = {"User-Agent": "reachy-mini-conversation-app"}
            if self.settings.hf_token:
                headers["X-Reachy-Mini-Authorization"] = f"Bearer {self.settings.hf_token}"
            payload = {"hardware_id": hardware_id} if hardware_id else {}
            async with http.post(self.settings.upstream_session_url, headers=headers, json=payload) as response:
                response.raise_for_status()
                allocation = await response.json()
        connect_url = allocation.get("connect_url")
        if not isinstance(connect_url, str) or not connect_url:
            raise RuntimeError("Upstream allocator did not return connect_url")
        LOG.info("Allocated upstream Realtime session: %s", allocation.get("session_id", "<unknown>"))
        return connect_url

    async def _send_upstream(self, message: dict[str, Any]) -> None:
        if self.upstream is None:
            raise RuntimeError("Upstream Realtime connection is not ready")
        async with self._upstream_send_lock:
            await self.upstream.send(json.dumps(message, ensure_ascii=False))

    async def _downstream_loop(self) -> None:
        async for raw in self.websocket:
            if not isinstance(raw, str):
                await self.error("invalid_request_error", "Only JSON text frames are accepted")
                continue
            try:
                message = json.loads(raw)
            except json.JSONDecodeError:
                await self.error("invalid_request_error", "Invalid JSON")
                continue
            await self.handle(message)

    async def _upstream_loop(self) -> None:
        async for raw in self.upstream:
            if not isinstance(raw, str):
                async with self._downstream_send_lock:
                    await self.websocket.send(raw)
                continue
            try:
                message = json.loads(raw)
            except json.JSONDecodeError:
                LOG.warning("Dropping non-JSON upstream text frame")
                continue
            event_type = str(message.get("type", ""))
            if event_type == "response.created":
                self.upstream_response_active = True
            elif event_type in {"response.done", "response.cancelled"}:
                self.upstream_response_active = False
            if event_type.startswith("conversation.item.input_audio_transcription."):
                LOG.debug("Suppressing upstream transcription event: %s", event_type)
                continue
            async with self._downstream_send_lock:
                await self.websocket.send(raw)

    async def run(self) -> None:
        upstream_url = await self._resolve_upstream_url()
        headers = {
            "Authorization": f"Bearer {self.settings.hf_token or 'DUMMY'}",
            "OpenAI-Beta": "realtime=v1",
        }
        LOG.info("Connecting to upstream Realtime service")
        try:
            async with connect(
                upstream_url,
                additional_headers=headers,
                open_timeout=self.settings.upstream_open_timeout_s,
                max_size=16 * 1024 * 1024,
                ping_interval=20,
                ping_timeout=20,
            ) as upstream:
                self.upstream = upstream
                LOG.info("Upstream Realtime connection established")
                downstream_task = asyncio.create_task(self._downstream_loop(), name="reachy-to-upstream")
                upstream_task = asyncio.create_task(self._upstream_loop(), name="upstream-to-reachy")
                done, pending = await asyncio.wait(
                    {downstream_task, upstream_task}, return_when=asyncio.FIRST_COMPLETED
                )
                for task in pending:
                    task.cancel()
                for task in pending:
                    with contextlib.suppress(asyncio.CancelledError):
                        await task
                for task in done:
                    task.result()
        except ConnectionClosed:
            pass
        finally:
            self.upstream = None
            if self.stream:
                await self.stream.close()

    async def handle(self, message: dict[str, Any]) -> None:
        event_type = message.get("type")
        if event_type == "session.update":
            await self.handle_session_update(message)
        elif event_type == "input_audio_buffer.append":
            # Audio terminates here and is streamed continuously to Volcengine.
            # We don't run a local VAD or force-finalize noisy chunks; only native
            # Volcengine utterances marked definite become user turns.
            await self.handle_audio_append(message)
        elif event_type == "input_audio_buffer.commit":
            LOG.debug("Ignoring local commit; native Volcengine VAD owns segmentation")
        elif event_type == "input_audio_buffer.clear":
            await self.clear_audio()
        else:
            await self._send_upstream(message)

    async def handle_session_update(self, message: dict[str, Any]) -> None:
        incoming = message.get("session") or {}
        self.session.update(incoming)
        audio = incoming.setdefault("audio", {})
        input_audio = audio.setdefault("input", {})
        transcription = input_audio.setdefault("transcription", {})
        transcription["language"] = "zh-CN"
        self.language = "zh-CN"
        turn_detection = input_audio.get("turn_detection")
        if isinstance(turn_detection, dict):
            self.turn_detection = turn_detection
        await self._send_upstream(message)
        LOG.info("Session forwarded: language=zh-CN turn_detection=%s", self.turn_detection.get("type"))

    async def handle_audio_append(self, message: dict[str, Any]) -> None:
        encoded = message.get("audio")
        if not isinstance(encoded, str):
            await self.error("invalid_request_error", "input_audio_buffer.append.audio must be base64")
            return
        try:
            pcm = base64.b64decode(encoded, validate=True)
        except Exception:
            await self.error("invalid_request_error", "Invalid base64 audio")
            return
        if not pcm:
            return
        if len(pcm) % 2:
            pcm = pcm[:-1]
        if not pcm:
            return

        if self.stream is None:
            stream_id = f"stream_{uuid.uuid4().hex}"
            self.item_id = stream_id
            self.stream = VolcengineStream(
                self.settings,
                self.language,
                stream_id,
                self._on_native_utterance,
            )
            try:
                await self.stream.start(pcm)
            except Exception:
                await self.stream.close()
                self.stream = None
                self.item_id = None
                raise
            LOG.info("[%s] continuous Volcengine stream started (native VAD)", stream_id)
            return

        await self.stream.send_audio(pcm)

    async def _on_native_utterance(
        self, transcript: str, speaker_id: str | None = None
    ) -> None:
        transcript = transcript.strip()
        if not transcript:
            return

        if self.settings.keyword_gate_enabled:
            now = time.time()
            # Prune expired speaker windows first.
            expired = [sid for sid, exp in self.keyword_gate_speakers.items() if exp <= now]
            for sid in expired:
                LOG.info("Keyword gate expired for speaker_id=%s", sid)
                self.keyword_gate_speakers.pop(sid, None)

            transcript_normalized = transcript.lower().strip()
            transcript_head = transcript_normalized.lstrip("，。！？、,.!?~ ")
            has_keyword = any(
                keyword.lower().strip() in transcript_normalized
                for keyword in KEYWORD_GATE_WORDS
            ) or any(
                transcript_head.startswith(prefix)
                for prefix in self.settings.keyword_gate_prefix_words
            )

            speaker_key = speaker_id if speaker_id is not None else "__unknown__"

            if has_keyword:
                new_activation = speaker_key not in self.keyword_gate_speakers
                self.keyword_gate_speakers[speaker_key] = now + KEYWORD_GATE_WINDOW_S
                LOG.info(
                    "Keyword gate %s for %.0fs: speaker_id=%s active=%s",
                    "opened" if new_activation else "refreshed",
                    KEYWORD_GATE_WINDOW_S,
                    speaker_id,
                    sorted(self.keyword_gate_speakers.keys()),
                )
            else:
                if speaker_key not in self.keyword_gate_speakers:
                    LOG.info(
                        "Keyword gate suppressed transcript: speaker_id=%s not in active set %s",
                        speaker_id,
                        sorted(self.keyword_gate_speakers.keys()),
                    )
                    return
                # Active speaker keeps talking — extend their own window only.
                self.keyword_gate_speakers[speaker_key] = now + KEYWORD_GATE_WINDOW_S

        item_id = f"item_{uuid.uuid4().hex}"
        if self.upstream_response_active:
            await self._send_upstream(
                {"type": "response.cancel", "event_id": f"event_{uuid.uuid4().hex}"}
            )
            self.upstream_response_active = False

        # Reachy only sees speech_started after Volcengine native VAD has marked
        # a non-empty utterance definite. Mechanical noise with no definite text
        # therefore cannot switch the UI/state machine into a user-speech turn.
        await self.emit("input_audio_buffer.speech_started", audio_start_ms=0, item_id=item_id)
        await self.emit(
            "conversation.item.input_audio_transcription.delta",
            item_id=item_id,
            content_index=0,
            delta=transcript,
        )
        await self.emit("input_audio_buffer.speech_stopped", audio_end_ms=0, item_id=item_id)
        await self.emit(
            "conversation.item.input_audio_transcription.completed",
            item_id=item_id,
            content_index=0,
            transcript=transcript,
        )
        await self._send_upstream(
            {
                "type": "conversation.item.create",
                "event_id": f"event_{uuid.uuid4().hex}",
                "item": {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": transcript}],
                },
            }
        )
        await self._send_upstream(
            {"type": "response.create", "event_id": f"event_{uuid.uuid4().hex}"}
        )
        LOG.info("[%s] native VAD transcript injected upstream: %s", item_id, transcript)

    async def clear_audio(self, *, emit_confirmation: bool = True) -> None:
        if self.stream:
            await self.stream.close()
        self.stream = None
        self.item_id = None
        if emit_confirmation:
            await self.emit("input_audio_buffer.cleared")

    async def error(self, code: str, message: str) -> None:
        await self.emit("error", error={"type": "invalid_request_error", "code": code, "message": message})


async def websocket_handler(websocket: ServerConnection, settings: Settings) -> None:
    request_path = getattr(getattr(websocket, "request", None), "path", "")
    path = request_path.split("?", 1)[0]
    if path != settings.path:
        await websocket.close(code=1008, reason=f"Expected path {settings.path}")
        return
    peer = getattr(websocket, "remote_address", None)
    LOG.info("Reachy connected: %s path=%s", peer, request_path)
    try:
        await RealtimeAdapterConnection(websocket, settings).run()
    except Exception:
        LOG.exception("Connection handler failed for %s", peer)
        with contextlib.suppress(Exception):
            await websocket.close(code=1011, reason="Adapter error")
    finally:
        LOG.info("Reachy disconnected: %s", peer)


async def async_main() -> None:
    settings = Settings.from_environment()
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        with contextlib.suppress(NotImplementedError):
            loop.add_signal_handler(sig, stop.set)

    LOG.info(
        "Starting adapter on ws://%s:%d%s; Volcengine=%s resource_id=%s",
        settings.host,
        settings.port,
        settings.path,
        settings.volc_url,
        settings.resource_id,
    )
    async with serve(
        lambda ws: websocket_handler(ws, settings),
        settings.host,
        settings.port,
        max_size=4 * 1024 * 1024,
        ping_interval=20,
        ping_timeout=20,
    ):
        await stop.wait()


def main() -> None:
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "DEBUG").upper(),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    try:
        asyncio.run(async_main())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
