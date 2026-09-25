"""Wake/proactive lifecycle markers without utterance admission policy.

Text admission belongs exclusively to ``RealtimeAdapterConnection``'s global
``_conversation_gate_open`` boolean. ``speaker_id`` remains on ``Utterance``
only so downstream speaker-context prompting can consume ASR metadata.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class GateMode(str, Enum):
    OFF = "off"
    SHADOW = "shadow"
    ENFORCE = "enforce"


@dataclass(frozen=True)
class WakeMarker:
    stream_generation: int
    sample_index: int
    timestamp_ms: float


@dataclass(frozen=True)
class Utterance:
    text: str
    speaker_id: str | None
    stream_generation: int
    start_ms: float | None
    end_ms: float | None


class AudioGate:
    """Track wake/proactive lifecycle state, never decide utterance admission."""

    def __init__(
        self,
        *,
        mode: GateMode | str = GateMode.SHADOW,
        trigger_timeout_s: float = 3.0,
        speaker_window_s: float = 30.0,
        match_tolerance_ms: float = 400.0,
        closing_seconds: float = 0.25,
        clock=None,
    ) -> None:
        self.mode = GateMode(mode)
        if trigger_timeout_s <= 0 or speaker_window_s <= 0:
            raise ValueError("gate timeouts must be positive")
        if match_tolerance_ms < 0 or closing_seconds <= 0:
            raise ValueError("gate tolerance must be non-negative and closing_seconds positive")
        # Keep constructor compatibility while deliberately carrying no
        # speaker, generation-window, or utterance-admission state.
        self.stream_generation: int | None = None
        self.wake: WakeMarker | None = None
        self.proactive_reply_armed = False

    def reset(self, stream_generation: int | None = None) -> None:
        self.stream_generation = stream_generation
        self.wake = None
        self.proactive_reply_armed = False

    clear = reset

    def arm_for_reply(self, stream_generation: int) -> bool:
        """Record a proactive arm for the current live stream."""
        if self.mode is not GateMode.ENFORCE or self.stream_generation != stream_generation:
            return False
        self.wake = None
        self.proactive_reply_armed = True
        return True

    def on_wake(self, marker: WakeMarker) -> None:
        """Record the latest wake marker for startup/lifecycle diagnostics."""
        if self.mode is GateMode.OFF:
            return
        self.stream_generation = marker.stream_generation
        self.wake = marker
        self.proactive_reply_armed = False
