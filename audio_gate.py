"""Pure wake-word/speaker authorization state machine.

This module deliberately has no network, audio-model, or asyncio dependencies.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum
from typing import Callable


class GateMode(str, Enum):
    OFF = "off"
    SHADOW = "shadow"
    ENFORCE = "enforce"


class GateState(str, Enum):
    SLEEPING = "sleeping"
    TRIGGERED = "triggered"
    ACTIVE = "active"
    CLOSING = "closing"


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


@dataclass(frozen=True)
class Decision:
    allow: bool
    enforce_allow: bool
    reason: str
    state: GateState
    identity: tuple[int, str] | None = None
    authorization_created: bool = False
    authorization_refreshed: bool = False


class AudioGate:
    """Deterministic wake/speaker gate.

    A wake marker is bound only by a definite utterance whose interval contains
    the marker (within tolerance).  Authorization is scoped to the Volc stream
    generation so reconnects cannot reuse a speaker id.
    """

    def __init__(
        self,
        *,
        mode: GateMode | str = GateMode.SHADOW,
        trigger_timeout_s: float = 3.0,
        speaker_window_s: float = 30.0,
        match_tolerance_ms: float = 400.0,
        closing_seconds: float = 0.25,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.mode = GateMode(mode)
        if trigger_timeout_s <= 0 or speaker_window_s <= 0:
            raise ValueError("gate timeouts must be positive")
        if match_tolerance_ms < 0 or closing_seconds <= 0:
            raise ValueError("gate tolerance must be non-negative and closing_seconds positive")
        self.trigger_timeout_s = trigger_timeout_s
        self.speaker_window_s = speaker_window_s
        self.match_tolerance_ms = match_tolerance_ms
        self.closing_seconds = closing_seconds
        self._clock = clock
        self.state = GateState.SLEEPING
        self.stream_generation: int | None = None
        self.wake: WakeMarker | None = None
        self.identity: tuple[int, str] | None = None
        self._deadline: float | None = None
        # A proactive reply starts without SSD identity, but the wildcard is
        # consumable by exactly one definite utterance.
        self._proactive_reply_pending = False

    def _advance(self) -> None:
        now = self._clock()
        while self._deadline is not None and now >= self._deadline:
            expired_at = self._deadline
            if self.state in {GateState.TRIGGERED, GateState.ACTIVE}:
                self.state = GateState.CLOSING
                self._deadline = expired_at + self.closing_seconds
                self.wake = None
                self.identity = None
                self._proactive_reply_pending = False
            elif self.state is GateState.CLOSING:
                self.state = GateState.SLEEPING
                self._deadline = None

    def tick(self) -> GateState:
        self._advance()
        return self.state

    def reset(self, stream_generation: int | None = None) -> None:
        self.stream_generation = stream_generation
        self.state = GateState.SLEEPING
        self.wake = None
        self.identity = None
        self._deadline = None
        self._proactive_reply_pending = False

    clear = reset

    def arm_for_reply(self, stream_generation: int) -> bool:
        """Authorize one proactive reply, binding its first SSD identity.

        The generation must already be the gate's current stream generation;
        this prevents a delayed HTTP request from authorizing a reconnected or
        cleared stream.
        """
        self._advance()
        if (
            self.mode is not GateMode.ENFORCE
            or self.stream_generation != stream_generation
        ):
            return False
        self.wake = None
        self.identity = None
        self._proactive_reply_pending = True
        self.state = GateState.ACTIVE
        self._deadline = self._clock() + self.speaker_window_s
        return True

    def on_wake(self, marker: WakeMarker) -> None:
        if self.mode is GateMode.OFF:
            return
        self.stream_generation = marker.stream_generation
        self.wake = marker
        self.identity = None
        self._proactive_reply_pending = False
        self.state = GateState.TRIGGERED
        self._deadline = self._clock() + self.trigger_timeout_s

    def _result(
        self,
        enforce_allow: bool,
        reason: str,
        *,
        created: bool = False,
        refreshed: bool = False,
    ) -> Decision:
        return Decision(
            allow=enforce_allow if self.mode is GateMode.ENFORCE else True,
            enforce_allow=enforce_allow,
            reason=reason,
            state=self.state,
            identity=self.identity,
            authorization_created=created,
            authorization_refreshed=refreshed,
        )

    def decide(self, utterance: Utterance) -> Decision:
        if self.mode is GateMode.OFF:
            return self._result(True, "mode_off")
        self._advance()
        if utterance.stream_generation != self.stream_generation:
            return self._result(False, "stream_generation_mismatch")

        if self.state is GateState.TRIGGERED and self.wake is not None:
            if utterance.start_ms is None or utterance.end_ms is None:
                self._enter_closing()
                return self._result(True, "trigger_utterance_missing_timeline")
            matched = (
                utterance.start_ms - self.match_tolerance_ms
                <= self.wake.timestamp_ms
                <= utterance.end_ms + self.match_tolerance_ms
            )
            if not matched:
                return self._result(False, "utterance_does_not_match_wake")
            if not utterance.speaker_id:
                self._enter_closing()
                return self._result(True, "trigger_utterance_missing_speaker")
            self.identity = (utterance.stream_generation, utterance.speaker_id)
            self.wake = None
            self.state = GateState.ACTIVE
            self._deadline = self._clock() + self.speaker_window_s
            return self._result(True, "wake_bound_to_speaker", created=True)

        if self.state is GateState.ACTIVE:
            if self._proactive_reply_pending:
                self._proactive_reply_pending = False
                if not utterance.speaker_id:
                    self._enter_closing()
                    return self._result(True, "proactive_reply_missing_speaker")
                self.identity = (utterance.stream_generation, utterance.speaker_id)
                self._deadline = self._clock() + self.speaker_window_s
                return self._result(True, "proactive_reply_bound_to_speaker", created=True)
            identity = (
                (utterance.stream_generation, utterance.speaker_id)
                if utterance.speaker_id
                else None
            )
            if identity != self.identity:
                return self._result(False, "speaker_not_authorized")
            self._deadline = self._clock() + self.speaker_window_s
            return self._result(True, "authorized_speaker", refreshed=True)

        if self.state is GateState.CLOSING:
            return self._result(False, "gate_closing")
        return self._result(False, "gate_sleeping")

    def _enter_closing(self) -> None:
        self.state = GateState.CLOSING
        self.wake = None
        self.identity = None
        self._proactive_reply_pending = False
        self._deadline = self._clock() + self.closing_seconds
