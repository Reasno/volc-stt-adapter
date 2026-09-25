from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from audio_gate import GateMode
from volc_stt_adapter import AllocatorRateLimitedError, RealtimeAdapterConnection


class _Response:
    def __init__(self, status, *, headers=None, payload=None):
        self.status = status
        self.headers = headers or {}
        self._payload = payload or {}

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    def raise_for_status(self):
        if self.status >= 400:
            raise AssertionError(f"unexpected HTTP {self.status}")

    async def json(self):
        return self._payload


class _Session:
    def __init__(self, responses):
        self.responses = list(responses)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    def post(self, *args, **kwargs):
        return self.responses.pop(0)


def _settings():
    return SimpleNamespace(
        kws_mode=GateMode.ENFORCE,
        kws_trigger_timeout_seconds=3,
        kws_speaker_window_seconds=30,
        kws_match_tolerance_ms=100,
        upstream_mode="allocator",
        upstream_session_url="https://allocator/session",
        upstream_open_timeout_s=10,
        reachy_daemon_url="",
        hf_token="",
        stream_idle_timeout_seconds=30,
    )


class AllocatorRetryTest(unittest.IsolatedAsyncioTestCase):
    def _connection(self):
        return RealtimeAdapterConnection(SimpleNamespace(), _settings())

    async def test_retry_after_seconds_is_honored(self):
        session = _Session([
            _Response(429, headers={"Retry-After": "2"}),
            _Response(200, payload={"connect_url": "wss://upstream"}),
        ])
        sleep = AsyncMock()
        with patch("volc_stt_adapter.aiohttp.ClientSession", return_value=session), patch(
            "volc_stt_adapter.asyncio.sleep", sleep
        ):
            self.assertEqual(await self._connection()._resolve_upstream_url(), "wss://upstream")
        sleep.assert_awaited_once_with(2.0)

    def test_retry_after_http_date_is_supported_and_capped(self):
        value = format_datetime(datetime.now(timezone.utc) + timedelta(minutes=1))
        self.assertEqual(
            RealtimeAdapterConnection._retry_after_seconds(value),
            RealtimeAdapterConnection.ALLOCATOR_BACKOFF_CAP_S,
        )

    async def test_invalid_retry_after_uses_deterministic_jittered_backoff(self):
        session = _Session([
            _Response(429, headers={"Retry-After": "invalid"}),
            _Response(429),
            _Response(200, payload={"connect_url": "wss://upstream"}),
        ])
        sleep = AsyncMock()
        with patch("volc_stt_adapter.aiohttp.ClientSession", return_value=session), patch(
            "volc_stt_adapter.asyncio.sleep", sleep
        ), patch("volc_stt_adapter.random.random", side_effect=[0.0, 1.0]):
            await self._connection()._resolve_upstream_url()
        self.assertEqual([call.args[0] for call in sleep.await_args_list], [0.25, 1.0])

    async def test_retry_budget_exhaustion_has_semantic_error(self):
        session = _Session([_Response(429) for _ in range(4)])
        with patch("volc_stt_adapter.aiohttp.ClientSession", return_value=session), patch(
            "volc_stt_adapter.asyncio.sleep", AsyncMock()
        ), patch("volc_stt_adapter.random.random", return_value=0.0):
            with self.assertRaisesRegex(AllocatorRateLimitedError, "^Allocator rate limited"):
                await self._connection()._resolve_upstream_url()


if __name__ == "__main__":
    unittest.main()
