#!/usr/bin/env python3
"""OpenAI Realtime-compatible STT adapter backed by Volcengine streaming ASR."""

from __future__ import annotations

import asyncio
import base64
import contextlib
import gzip
import json
import logging
import math
import os
import signal
import struct
import time
import uuid
import wave
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Any, Awaitable, Callable

import aiohttp
from aiohttp import web
from audio_gate import AudioGate, GateMode, Utterance, WakeMarker
from reachy_speaker import (
    ConversationSayClient,
    DaemonSoundClient,
    ReachySpeaker,
    VolcengineTtsClient,
    create_speak_app,
)
from wake_word import SAMPLE_RATE, WakeEvent, WakeWordDetector
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
    app_key: str = field(repr=False)
    access_key: str = field(repr=False)
    resource_id: str
    sample_rate: int
    volc_timeout_s: float
    boosting_table_id: str
    kws_mode: GateMode
    kws_model_path: str
    kws_threshold: float
    kws_preroll_seconds: float
    kws_trigger_timeout_seconds: float
    kws_speaker_window_seconds: float
    kws_match_tolerance_ms: float
    kws_queue_frames: int
    upstream_mode: str
    upstream_url: str
    upstream_session_url: str
    hf_token: str
    reachy_daemon_url: str
    upstream_open_timeout_s: float
    speak_http_host: str
    speak_http_port: int
    speak_api_token: str = field(repr=False)
    reachy_conversation_rpc_url: str
    volc_tts_url: str
    volc_tts_resource_id: str
    volc_tts_voice: str
    volc_tts_timeout_s: float
    speak_total_timeout_s: float
    volc_tts_max_audio_bytes: int
    daemon_sound_timeout_s: float
    daemon_sound_cleanup_delay_s: float

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

        def number(name: str, default: str, *, minimum: float, maximum: float | None = None) -> float:
            try:
                value = float(os.getenv(name, default))
            except ValueError as exc:
                raise RuntimeError(f"{name} must be a number") from exc
            if not math.isfinite(value) or value < minimum or (maximum is not None and value > maximum):
                limit = f"[{minimum}, {maximum}]" if maximum is not None else f">= {minimum}"
                raise RuntimeError(f"{name} must be a finite number in {limit}")
            return value

        def integer(name: str, default: str, *, minimum: int, maximum: int | None = None) -> int:
            value = number(name, default, minimum=minimum, maximum=maximum)
            if not value.is_integer():
                raise RuntimeError(f"{name} must be an integer")
            return int(value)

        try:
            kws_mode = GateMode(os.getenv("KWS_MODE", "shadow").strip().lower())
        except ValueError as exc:
            raise RuntimeError("KWS_MODE must be one of: off, shadow, enforce") from exc
        sample_rate = integer("AUDIO_SAMPLE_RATE", "16000", minimum=1)
        if sample_rate != SAMPLE_RATE:
            raise RuntimeError("AUDIO_SAMPLE_RATE must be 16000 when using this adapter")
        queue_frames = integer("KWS_QUEUE_FRAMES", "32", minimum=1)

        return cls(
            host=os.getenv("ADAPTER_HOST", "0.0.0.0"),
            port=integer("ADAPTER_PORT", "8765", minimum=1, maximum=65535),
            path=os.getenv("ADAPTER_PATH", "/v1/realtime"),
            volc_url=volc_url or "wss://openspeech.bytedance.com/api/v3/sauc/bigmodel_async",
            app_key=app_key,
            access_key=access_key,
            resource_id=resource_id,
            sample_rate=sample_rate,
            volc_timeout_s=number("VOLC_TIMEOUT_SECONDS", "30", minimum=0.1),
            boosting_table_id=os.getenv("VOLC_BOOSTING_TABLE_ID", "").strip(),
            kws_mode=kws_mode,
            kws_model_path=os.getenv(
                "KWS_MODEL_PATH",
                str(Path(__file__).with_name("models") / "reechy-spk150-steps100k-acc98.15-rec97.50.onnx"),
            ).strip(),
            kws_threshold=number("KWS_THRESHOLD", "0.5", minimum=0.0, maximum=1.0),
            kws_preroll_seconds=number("KWS_PREROLL_SECONDS", "1.5", minimum=0.0),
            kws_trigger_timeout_seconds=number("KWS_TRIGGER_TIMEOUT_SECONDS", "3", minimum=0.001),
            kws_speaker_window_seconds=number("KWS_SPEAKER_WINDOW_SECONDS", "30", minimum=0.001),
            kws_match_tolerance_ms=number("KWS_MATCH_TOLERANCE_MS", "400", minimum=0.0),
            kws_queue_frames=queue_frames,
            upstream_mode=os.getenv("UPSTREAM_MODE", "allocator").strip().lower(),
            upstream_url=os.getenv("UPSTREAM_REALTIME_URL", "").strip(),
            upstream_session_url=os.getenv(
                "UPSTREAM_SESSION_URL",
                "https://pollen-robotics-reachy-mini-realtime-url.hf.space/session",
            ).strip(),
            hf_token=os.getenv("HF_TOKEN", "").strip(),
            reachy_daemon_url=os.getenv("REACHY_DAEMON_URL", "http://192.168.31.94:8000").rstrip("/"),
            upstream_open_timeout_s=number("UPSTREAM_OPEN_TIMEOUT_SECONDS", "20", minimum=0.1),
            speak_http_host=os.getenv("SPEAK_HTTP_HOST", "0.0.0.0"),
            speak_http_port=integer("SPEAK_HTTP_PORT", "8766", minimum=1, maximum=65535),
            speak_api_token=os.getenv("SPEAK_API_TOKEN", ""),
            reachy_conversation_rpc_url=os.getenv(
                "REACHY_CONVERSATION_RPC_URL", "ws://192.168.31.94:7860/rpc"
            ).strip(),
            volc_tts_url=os.getenv(
                "VOLC_TTS_URL", "wss://openspeech.bytedance.com/api/v3/tts/bidirection"
            ).strip(),
            volc_tts_resource_id=os.getenv("VOLC_TTS_RESOURCE_ID", "seed-tts-2.0").strip(),
            volc_tts_voice=os.getenv("VOLC_TTS_VOICE", "zh_female_vv_uranus_bigtts").strip(),
            volc_tts_timeout_s=number("VOLC_TTS_TIMEOUT_SECONDS", "30", minimum=0.1),
            speak_total_timeout_s=number("SPEAK_TOTAL_TIMEOUT_SECONDS", "45", minimum=0.1),
            volc_tts_max_audio_bytes=integer(
                "VOLC_TTS_MAX_AUDIO_BYTES", str(16 * 1024 * 1024), minimum=1
            ),
            daemon_sound_timeout_s=number(
                "DAEMON_SOUND_TIMEOUT_SECONDS", "10", minimum=0.1
            ),
            daemon_sound_cleanup_delay_s=number(
                "DAEMON_SOUND_CLEANUP_DELAY_SECONDS", "300", minimum=0.0
            ),
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


def utterance_times_ms(utterance: dict[str, Any]) -> tuple[float | None, float | None]:
    """Read common Volc utterance timeline fields, conservatively.

    Volc responses observed in the wild use start_time/end_time, while some
    variants nest the same names in additions. Unknown or malformed timelines
    are returned as missing instead of guessing and potentially binding a speaker.
    """
    additions = utterance.get("additions")
    sources = (utterance, additions) if isinstance(additions, dict) else (utterance,)
    start = end = None
    for source in sources:
        if start is None:
            start = source.get("start_time", source.get("start_ms"))
        if end is None:
            end = source.get("end_time", source.get("end_ms"))
    try:
        start_ms = float(start) if start is not None else None
        end_ms = float(end) if end is not None else None
    except (TypeError, ValueError):
        return None, None
    if start_ms is None or end_ms is None or start_ms < 0 or end_ms < start_ms:
        return None, None
    return start_ms, end_ms


class VolcengineStream:
    # Max transient reconnect attempts within RECONNECT_WINDOW_S before giving up.
    # 5xxxxxxx 错误码 = 火山服务端 5xx（例如 55000000 grpc RST_STREAM、55000031
    # 服务繁忙），视为可恢复。4xxxxxxx / 鉴权 / 参数错误一律 fail-fast。
    RECONNECT_MAX_ATTEMPTS = 3
    RECONNECT_WINDOW_S = 60.0
    RECONNECT_BACKOFF_S = (0.5, 1.5, 3.0)

    def __init__(
        self,
        settings: Settings,
        language: str,
        stream_id: str,
        on_definite: Callable[[Utterance], Awaitable[None]],
        on_generation_reset: Callable[[int, int], None],
        *,
        timeline_origin_sample: int,
    ):
        self.settings = settings
        self.language = language
        self.item_id = stream_id
        self.on_definite = on_definite
        self.on_generation_reset = on_generation_reset
        self.generation = 0
        self.timeline_origin_sample = timeline_origin_sample
        # Every accepted local PCM16 sample is accounted exactly once as sent
        # or unsent. Unsent samples are not replayed, but still advance time.
        self.sent_samples = 0
        self.unsent_samples = 0
        self.http: aiohttp.ClientSession | None = None
        self.ws: aiohttp.ClientWebSocketResponse | None = None
        self.sequence = 1
        self.receiver: asyncio.Task[None] | None = None
        self.latest_text = ""
        self.last_emitted_text = ""
        self.error: Exception | None = None
        self._closed = False
        # Guard against send_audio racing with reconnect swap of self.ws.
        self._ws_lock = asyncio.Lock()
        # Timestamps of recent successful reconnects for rate limiting.
        self._reconnect_history: list[float] = []

    async def start(self, initial_pcm: bytes) -> None:
        await self._connect_upstream()
        self.receiver = asyncio.create_task(self._receive_loop(), name=f"volc-recv-{self.item_id}")
        if initial_pcm:
            sent = await self._send(
                CLIENT_AUDIO_ONLY_REQUEST,
                POS_SEQUENCE,
                wav_chunk(initial_pcm, self.settings.sample_rate),
            )
            samples = len(initial_pcm) // 2
            if sent:
                self.sent_samples += samples
            else:
                self.unsent_samples += samples

    async def _connect_upstream(self) -> None:
        """(Re)connect to Volcengine and send the initial full_request handshake.

        On reconnect we intentionally reuse the same item_id / uid so downstream
        consumers still see a single logical stream, but reset the binary
        protocol sequence counter (火山 sequence 从 1 开始递增，重连算新会话).
        """
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
        # Close any stale session/ws from a previous connection attempt.
        if self.ws and not self.ws.closed:
            with contextlib.suppress(Exception):
                await self.ws.close()
        if self.http and not self.http.closed:
            with contextlib.suppress(Exception):
                await self.http.close()
        self._begin_generation()
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
        # Handshake: send inline (bypass _send lock — the lock is held by the
        # caller during reconnect, and start() runs single-threaded before the
        # receive loop is spawned).
        payload = json.dumps(request).encode()
        frame = encode_request(CLIENT_FULL_REQUEST, POS_SEQUENCE, self.sequence, payload, serialization=SERIALIZATION_JSON)
        self.sequence += 1
        await self.ws.send_bytes(frame)
        acknowledgement = await self._receive_one()
        self._raise_for_error(acknowledgement)

    def _begin_generation(self) -> None:
        """Advance origin by all local input consumed by the prior generation."""
        self.timeline_origin_sample += self.sent_samples + self.unsent_samples
        self.sent_samples = 0
        self.unsent_samples = 0
        self.generation += 1
        self.on_generation_reset(self.generation, self.timeline_origin_sample)
        self.sequence = 1

    def _record_reconnect(self) -> bool:
        """Return True if we may attempt another reconnect, respecting the rate limit."""
        now = time.time()
        self._reconnect_history = [
            t for t in self._reconnect_history if now - t <= self.RECONNECT_WINDOW_S
        ]
        if len(self._reconnect_history) >= self.RECONNECT_MAX_ATTEMPTS:
            return False
        self._reconnect_history.append(now)
        return True

    async def _reconnect(self, reason: str) -> bool:
        """Best-effort reconnect after a transient upstream error.

        Returns True on success, False if we exceed the reconnect budget or the
        reconnect itself keeps failing. Caller is responsible for stopping the
        receive loop on False. On True, `self.ws` is a fresh WebSocket bound to
        the same logical item_id and ready to accept audio frames.
        """
        if self._closed:
            return False
        if not self._record_reconnect():
            LOG.error(
                "[%s] Upstream reconnect budget exhausted (%d attempts / %.0fs); giving up. reason=%s",
                self.item_id,
                self.RECONNECT_MAX_ATTEMPTS,
                self.RECONNECT_WINDOW_S,
                reason,
            )
            return False
        attempt = len(self._reconnect_history)
        backoff = self.RECONNECT_BACKOFF_S[min(attempt - 1, len(self.RECONNECT_BACKOFF_S) - 1)]
        LOG.warning(
            "[%s] Upstream reconnect attempt %d/%d in %.1fs: reason=%s",
            self.item_id,
            attempt,
            self.RECONNECT_MAX_ATTEMPTS,
            backoff,
            reason,
        )
        await asyncio.sleep(backoff)
        try:
            async with self._ws_lock:
                await self._connect_upstream()
        except Exception as exc:
            LOG.warning("[%s] Upstream reconnect failed: %s", self.item_id, exc)
            return False
        LOG.info("[%s] Upstream reconnect succeeded", self.item_id)
        return True

    async def _send(
        self,
        message_type: int,
        flags: int,
        payload: bytes,
        serialization: int = 0,
    ) -> bool:
        # Serialize sends against reconnect swaps of self.ws / self.sequence.
        # send_audio() 与 _reconnect() 都会持锁；接收方（_receive_loop）单独
        # 读 self.ws，不走这把锁，重连时会自然拿到新 ws。
        async with self._ws_lock:
            if not self.ws:
                raise RuntimeError("Volcengine stream is not connected")
            frame = encode_request(message_type, flags, self.sequence, payload, serialization=serialization)
            self.sequence += 1
            try:
                await self.ws.send_bytes(frame)
                return True
            except (ConnectionClosed, ConnectionResetError, aiohttp.ClientError) as exc:
                # 上游 ws 已经断了，_receive_loop 会看到 CLOSED 类消息并触发重连。
                # 这里静默丢弃当前帧，让下一帧写入新连接。上游少一小段音频比
                # 让整个 stream 崩掉更能保住体感。
                LOG.debug("[%s] send_bytes on closed ws (%s), dropping frame", self.item_id, exc)
                return False

    async def send_audio(self, pcm: bytes) -> None:
        if self._closed:
            return
        samples = len(pcm) // 2
        if await self._send(CLIENT_AUDIO_ONLY_REQUEST, POS_SEQUENCE, pcm):
            self.sent_samples += samples
        else:
            self.unsent_samples += samples

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

    @staticmethod
    def _is_transient_error(exc: BaseException) -> bool:
        """5xxxxxxx = 火山服务端 5xx（RST_STREAM / 服务繁忙 / gRPC INTERNAL 等），
        以及 aiohttp/websocket 层的连接断开，都视为暂态可恢复。
        4xxxxxxx（鉴权/参数/资源）一律 fail-fast，不重连。"""
        if isinstance(exc, (ConnectionClosed, ConnectionResetError, aiohttp.ClientError)):
            return True
        msg = str(exc)
        # RuntimeError 走 code=5xxxxxxx 匹配。RST_STREAM 关键字兜底。
        if "code=5" in msg and "code=5xx" not in msg:
            return True
        if "RST_STREAM" in msg:
            return True
        if "Unexpected Volcengine WebSocket message" in msg:
            # e.g. server sent CLOSED before we asked to close.
            return True
        return False

    async def _receive_loop(self) -> None:
        while True:
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
                                relative_start_ms, relative_end_ms = utterance_times_ms(utterance)
                                origin_ms = self.timeline_origin_sample * 1000.0 / self.settings.sample_rate
                                start_ms = (
                                    origin_ms + relative_start_ms
                                    if relative_start_ms is not None
                                    else None
                                )
                                end_ms = (
                                    origin_ms + relative_end_ms
                                    if relative_end_ms is not None
                                    else None
                                )
                                LOG.info(
                                    "[%s] native VAD definite: generation=%d speaker_id=%s timeline=%s..%s text=%s",
                                    self.item_id,
                                    self.generation,
                                    speaker_id,
                                    start_ms,
                                    end_ms,
                                    utterance_text,
                                )
                                await self.on_definite(
                                    Utterance(
                                        text=utterance_text,
                                        speaker_id=speaker_id,
                                        stream_generation=self.generation,
                                        start_ms=start_ms,
                                        end_ms=end_ms,
                                    )
                                )
                    if response["is_last"]:
                        return
            except asyncio.CancelledError:
                raise
            except ConnectionClosed as exc:
                # Upstream socket dropped. Try to reconnect; if we can't, exit
                # cleanly (mirror the pre-reconnect behavior of returning on
                # ConnectionClosed rather than raising to caller).
                if self._closed:
                    return
                if not await self._reconnect(f"ConnectionClosed: {exc}"):
                    return
                continue
            except Exception as exc:
                if self._closed:
                    return
                if self._is_transient_error(exc):
                    LOG.warning("[%s] Transient upstream error, attempting reconnect: %s", self.item_id, exc)
                    if not await self._reconnect(str(exc)):
                        self.error = exc
                        return
                    continue
                self.error = exc
                LOG.exception("[%s] Volcengine receive failed (non-transient, giving up)", self.item_id)
                return


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
        self.kws_mode = settings.kws_mode
        self.gate = AudioGate(
            mode=self.kws_mode,
            trigger_timeout_s=settings.kws_trigger_timeout_seconds,
            speaker_window_s=settings.kws_speaker_window_seconds,
            match_tolerance_ms=settings.kws_match_tolerance_ms,
        )
        self.detector: WakeWordDetector | None = None
        self._pending_wake: WakeEvent | None = None
        self._stream_starting = False
        self._pending_stream_audio = bytearray()
        self._pending_stream_audio_lock = asyncio.Lock()
        self._stream_lock = asyncio.Lock()
        self._stream_close_lock = asyncio.Lock()
        self._lifecycle_lock = asyncio.Lock()
        self._starting_stream: VolcengineStream | None = None
        self._lifecycle_generation = 0
        self._pcm_leftover = b""

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
        if self.kws_mode is not GateMode.OFF:
            detector = WakeWordDetector(
                self.settings.kws_model_path,
                threshold=self.settings.kws_threshold,
                preroll_seconds=self.settings.kws_preroll_seconds,
                queue_frames=self.settings.kws_queue_frames,
                on_wake=self._on_wake,
            )
            try:
                await detector.start()
            except Exception:
                LOG.exception("KWS model load failed; connection explicitly downgraded to off")
                self.kws_mode = GateMode.OFF
                self.gate = AudioGate(mode=GateMode.OFF)
                await detector.close()
            else:
                self.detector = detector
                LOG.info("KWS detector ready: mode=%s threshold=%.3f", self.kws_mode.value, self.settings.kws_threshold)
        try:
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
            await self.clear_audio(emit_confirmation=False)
            if self.detector:
                await self.detector.close()

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
        pcm = self._pcm_leftover + pcm
        even_length = len(pcm) & ~1
        self._pcm_leftover = pcm[even_length:]
        pcm = pcm[:even_length]
        if not pcm:
            return

        if self.kws_mode is GateMode.ENFORCE:
            async with self._pending_stream_audio_lock:
                stream_starting = self._stream_starting
                if stream_starting:
                    self._pending_stream_audio.extend(pcm)
            if stream_starting:
                if self.detector is not None:
                    self.detector.append(pcm)
                return
        if self.kws_mode is GateMode.ENFORCE and self.stream is None:
            # Before wake, audio is intentionally local-only. The detector's
            # sample-indexed preroll is supplied when a wake event arrives.
            if self.detector is not None:
                self.detector.append(pcm)
            return
        if self.stream is None:
            await self._start_stream(pcm, timeline_origin_sample=0)
        else:
            await self.stream.send_audio(pcm)
        if self.detector is not None:
            self.detector.append(pcm)

    async def _start_stream(self, initial_pcm: bytes, *, timeline_origin_sample: int) -> None:
        async with self._stream_lock:
            if self.stream is not None:
                existing_stream = self.stream
                starting_stream = None
            else:
                existing_stream = None
                stream_id = f"stream_{uuid.uuid4().hex}"
                starting_stream = VolcengineStream(
                    self.settings,
                    self.language,
                    stream_id,
                    self._on_native_utterance,
                    self._on_stream_generation,
                    timeline_origin_sample=timeline_origin_sample,
                )
                lifecycle_generation = self._lifecycle_generation
                self._starting_stream = starting_stream
                self.item_id = stream_id
        if existing_stream is not None:
            if initial_pcm:
                await existing_stream.send_audio(initial_pcm)
            return

        assert starting_stream is not None
        try:
            await starting_stream.start(initial_pcm)
        except BaseException:
            await self._close_stream_safely(starting_stream)
            async with self._stream_lock:
                if self._starting_stream is starting_stream:
                    self._starting_stream = None
                if self.item_id == starting_stream.item_id:
                    self.item_id = None
            self.gate.reset()
            raise

        async with self._stream_lock:
            publish = (
                self._starting_stream is starting_stream
                and self._lifecycle_generation == lifecycle_generation
            )
            if publish:
                self.stream = starting_stream
                self._starting_stream = None
            elif self._starting_stream is starting_stream:
                self._starting_stream = None
        if not publish:
            # clear may have closed before start() actually acquired resources;
            # close again after start returns. The close lock serializes overlap.
            await self._close_stream_safely(starting_stream)
            return
        LOG.info(
            "[%s] Volcengine stream started: generation=%d origin_sample=%d mode=%s",
            starting_stream.item_id,
            starting_stream.generation,
            starting_stream.timeline_origin_sample,
            self.kws_mode.value,
        )

    async def _close_stream_safely(self, stream: VolcengineStream) -> None:
        """Serialize close and finish it even if the caller is cancelled."""
        async with self._stream_close_lock:
            close_task = asyncio.create_task(stream.close())
            try:
                await asyncio.shield(close_task)
            except asyncio.CancelledError:
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await close_task
            except Exception:
                LOG.exception("Failed to close Volcengine stream %s", stream.item_id)

    def _on_stream_generation(self, generation: int, timeline_origin_sample: int) -> None:
        self.gate.reset(generation)
        LOG.info(
            "Volcengine generation reset: generation=%d origin_sample=%d; authorization revoked",
            generation,
            timeline_origin_sample,
        )
        # During enforce startup the wake precedes generation creation. Reapply
        # it only to this newly created generation; reconnect has no pending wake.
        if self._pending_wake is not None:
            event = self._pending_wake
            self._pending_wake = None
            self.gate.on_wake(
                WakeMarker(generation, event.sample_index, event.timestamp_ms)
            )

    async def _on_wake(self, event: WakeEvent) -> None:
        if self.detector is not None and event.detector_generation != self.detector.generation:
            LOG.debug("Ignoring stale KWS event from detector generation %d", event.detector_generation)
            return
        LOG.info(
            "KWS wake: score=%.3f sample_index=%d timestamp_ms=%.1f mode=%s",
            event.score,
            event.sample_index,
            event.timestamp_ms,
            self.kws_mode.value,
        )
        if self.kws_mode is GateMode.ENFORCE and self.stream is None:
            await self._start_after_wake(event)
            return
        generation = self.stream.generation if self.stream is not None else 0
        self.gate.on_wake(WakeMarker(generation, event.sample_index, event.timestamp_ms))

    async def _start_after_wake(self, event: WakeEvent) -> None:
        if self.stream is not None:
            self.gate.on_wake(
                WakeMarker(self.stream.generation, event.sample_index, event.timestamp_ms)
            )
            return
        self._pending_wake = event
        async with self._pending_stream_audio_lock:
            self._stream_starting = True
        try:
            origin = max(0, event.sample_index - len(event.preroll_pcm) // 2)
            # Startup is intentionally outside _lifecycle_lock so clear/finally
            # can detach and close a stream whose start() is blocked.
            await self._start_stream(event.preroll_pcm, timeline_origin_sample=origin)
            async with self._lifecycle_lock:
                if self.stream is None:
                    return
                while True:
                    async with self._pending_stream_audio_lock:
                        if not self._pending_stream_audio:
                            self._stream_starting = False
                            break
                        buffered = bytes(self._pending_stream_audio)
                        self._pending_stream_audio = bytearray()
                    await self.stream.send_audio(buffered)
        finally:
            self._pending_wake = None
            async with self._pending_stream_audio_lock:
                self._stream_starting = False
                self._pending_stream_audio.clear()

    async def _on_native_utterance(self, utterance: Utterance) -> None:
        transcript = utterance.text.strip()
        if not transcript:
            return
        decision = self.gate.decide(utterance)
        LOG.info(
            "KWS gate decision: mode=%s state=%s allow=%s enforce_allow=%s reason=%s generation=%d speaker_id=%s text=%s",
            self.kws_mode.value,
            decision.state.value,
            decision.allow,
            decision.enforce_allow,
            decision.reason,
            utterance.stream_generation,
            utterance.speaker_id,
            transcript,
        )
        if decision.reason == "trigger_utterance_missing_timeline":
            LOG.warning(
                "Wake-matched first definite utterance has no usable timeline; allowing once without speaker authorization"
            )
        elif decision.reason == "trigger_utterance_missing_speaker":
            LOG.warning(
                "Wake-matched first definite utterance has no speaker; allowing once without speaker authorization"
            )
        if not decision.allow:
            return

        item_id = f"item_{uuid.uuid4().hex}"
        if self.upstream_response_active:
            await self._send_upstream(
                {"type": "response.cancel", "event_id": f"event_{uuid.uuid4().hex}"}
            )
            self.upstream_response_active = False

        start_ms = int(utterance.start_ms or 0)
        end_ms = int(utterance.end_ms or start_ms)
        await self.emit("input_audio_buffer.speech_started", audio_start_ms=start_ms, item_id=item_id)
        await self.emit(
            "conversation.item.input_audio_transcription.delta",
            item_id=item_id,
            content_index=0,
            delta=transcript,
        )
        await self.emit("input_audio_buffer.speech_stopped", audio_end_ms=end_ms, item_id=item_id)
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
        async with self._lifecycle_lock:
            async with self._stream_lock:
                self._lifecycle_generation += 1
                stream = self.stream
                starting_stream = self._starting_stream
                self.stream = None
                self._starting_stream = None
                self.item_id = None
            self._pcm_leftover = b""
            self._pending_wake = None
            async with self._pending_stream_audio_lock:
                self._stream_starting = False
                self._pending_stream_audio.clear()
            self.gate.clear()
            if self.detector:
                await self.detector.reset()
        # Never hold lifecycle/stream locks while close waits on receiver/ws.
        for candidate in (stream, starting_stream):
            if candidate is not None:
                await self._close_stream_safely(candidate)
        LOG.info("Audio cleared; stream, detector timeline, and KWS authorization reset")
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


async def run_servers(settings: Settings, stop: asyncio.Event) -> None:
    """Run STT WebSocket and active-speech HTTP listeners with shared clients."""
    timeout = aiohttp.ClientTimeout(total=None, sock_connect=10, sock_read=None)
    async with aiohttp.ClientSession(timeout=timeout) as http:
        conversation = ConversationSayClient(
            http, settings.reachy_conversation_rpc_url,
            timeout_s=min(5.0, settings.speak_total_timeout_s),
        )
        tts = VolcengineTtsClient(
            http,
            url=settings.volc_tts_url,
            app_id=settings.app_key,
            access_key=settings.access_key,
            resource_id=settings.volc_tts_resource_id,
            voice=settings.volc_tts_voice,
            timeout_s=settings.volc_tts_timeout_s,
            max_audio_bytes=settings.volc_tts_max_audio_bytes,
        )
        daemon = DaemonSoundClient(
            http,
            settings.reachy_daemon_url,
            cleanup_delay_s=settings.daemon_sound_cleanup_delay_s,
            timeout_s=settings.daemon_sound_timeout_s,
        )
        speaker = ReachySpeaker(conversation, tts, daemon)
        app = create_speak_app(
            speaker,
            token=settings.speak_api_token,
            total_timeout_s=settings.speak_total_timeout_s,
        )
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, settings.speak_http_host, settings.speak_http_port)
        try:
            await site.start()
            LOG.info(
                "Active speech HTTP listening on http://%s:%d",
                settings.speak_http_host,
                settings.speak_http_port,
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
        finally:
            # Stop accepting HTTP work before cancelling delayed cleanup tasks
            # and closing the shared ClientSession.
            await runner.cleanup()
            await speaker.close()


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
    await run_servers(settings, stop)


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
