from __future__ import annotations

import unittest

from audio_gate import AudioGate, GateMode, GateState, Utterance, WakeMarker


class Clock:
    def __init__(self): self.now = 0.0
    def __call__(self): return self.now
    def advance(self, seconds): self.now += seconds


def utterance(speaker="alice", generation=1, start=900.0, end=1300.0, text="瑞奇你好"):
    return Utterance(text, speaker, generation, start, end)


class AudioGateTest(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        self.gate = AudioGate(mode="enforce", clock=self.clock, match_tolerance_ms=100)
        self.gate.reset(1)

    def wake(self):
        self.gate.on_wake(WakeMarker(1, 16000, 1000.0))

    def test_all_states_and_trigger_timeout(self):
        self.assertEqual(self.gate.state, GateState.SLEEPING)
        self.wake()
        self.assertEqual(self.gate.state, GateState.TRIGGERED)
        self.clock.advance(3.0)
        self.assertEqual(self.gate.tick(), GateState.CLOSING)
        self.clock.advance(0.25)
        self.assertEqual(self.gate.tick(), GateState.SLEEPING)

    def test_matching_wake_binds_and_30_second_boundary(self):
        self.wake()
        decision = self.gate.decide(utterance())
        self.assertTrue(decision.allow)
        self.assertTrue(decision.authorization_created)
        self.assertEqual(decision.identity, (1, "alice"))
        self.clock.advance(29.999)
        self.assertTrue(self.gate.decide(utterance(text="具体文本A")).allow)
        self.clock.advance(30.0)
        self.assertFalse(self.gate.decide(utterance(text="具体文本B")).allow)
        self.assertEqual(self.gate.state, GateState.CLOSING)

    def test_same_speaker_refreshes_other_and_missing_speaker_drop(self):
        self.wake(); self.gate.decide(utterance())
        self.clock.advance(20)
        decision = self.gate.decide(utterance(text="续窗文本"))
        self.assertTrue(decision.allow)
        self.assertTrue(decision.authorization_refreshed)
        self.clock.advance(20)
        self.assertFalse(self.gate.decide(utterance(speaker="bob", text="其他人文本")).allow)
        self.assertFalse(self.gate.decide(utterance(speaker=None, text="未知人文本")).allow)
        self.assertTrue(self.gate.decide(utterance(text="续窗后文本")).allow)

    def test_missing_timeline_passes_once_without_binding(self):
        self.wake()
        decision = self.gate.decide(utterance(start=None, end=None, text="无时间轴首句"))
        self.assertTrue(decision.allow)
        self.assertFalse(decision.authorization_created)
        self.assertIsNone(decision.identity)
        self.assertEqual(decision.state, GateState.CLOSING)
        self.assertFalse(self.gate.decide(utterance(text="不得续开")).allow)

    def test_nonmatching_interval_does_not_bind(self):
        self.wake()
        decision = self.gate.decide(utterance(start=2000, end=2500))
        self.assertFalse(decision.allow)
        self.assertIsNone(decision.identity)
        self.assertEqual(decision.state, GateState.TRIGGERED)

    def test_missing_speaker_does_not_bind(self):
        self.wake()
        decision = self.gate.decide(utterance(speaker=None))
        self.assertTrue(decision.allow)
        self.assertIsNone(decision.identity)
        self.assertEqual(decision.state, GateState.CLOSING)

    def test_generation_reset_and_clear_revoke(self):
        self.wake(); self.gate.decide(utterance())
        self.gate.reset(2)
        self.assertFalse(self.gate.decide(utterance(generation=1, text="旧代文本")).allow)
        self.gate.clear()
        self.assertFalse(self.gate.decide(utterance(generation=2, text="清理后文本")).allow)

    def test_shadow_and_off_actual_allow_with_metadata(self):
        shadow = AudioGate(mode=GateMode.SHADOW, clock=self.clock)
        shadow.reset(1)
        decision = shadow.decide(utterance(text="shadow文本"))
        self.assertTrue(decision.allow)
        self.assertFalse(decision.enforce_allow)
        off = AudioGate(mode=GateMode.OFF, clock=self.clock)
        decision = off.decide(utterance(text="off文本"))
        self.assertTrue(decision.allow)
        self.assertTrue(decision.enforce_allow)
        self.assertEqual(decision.reason, "mode_off")

    def test_proactive_reply_binds_first_speaker_and_refreshes(self):
        self.assertTrue(self.gate.arm_for_reply(1))
        first = self.gate.decide(utterance(text="主动回复"))
        self.assertTrue(first.allow)
        self.assertEqual(first.reason, "proactive_reply_bound_to_speaker")
        self.assertEqual(first.identity, (1, "alice"))
        self.assertFalse(self.gate.decide(utterance(speaker="bob")).allow)
        self.assertTrue(self.gate.decide(utterance()).authorization_refreshed)

    def test_proactive_reply_missing_speaker_fails_closed(self):
        self.assertTrue(self.gate.arm_for_reply(1))
        first = self.gate.decide(utterance(speaker=None))
        self.assertTrue(first.allow)
        self.assertEqual(first.reason, "proactive_reply_missing_speaker")
        self.assertEqual(first.state, GateState.CLOSING)
        self.assertFalse(self.gate.decide(utterance()).allow)

    def test_proactive_reply_expires_and_generation_reset_revokes(self):
        self.assertTrue(self.gate.arm_for_reply(1))
        self.clock.advance(30)
        self.assertFalse(self.gate.decide(utterance()).allow)
        self.gate.reset(2)
        self.assertFalse(self.gate.arm_for_reply(1))
        self.assertTrue(self.gate.arm_for_reply(2))
        self.assertFalse(self.gate.decide(utterance(generation=1)).allow)

    def test_instances_do_not_share_authorization(self):
        other = AudioGate(mode="enforce", clock=self.clock); other.reset(1)
        self.wake(); self.assertTrue(self.gate.decide(utterance(text="gate1唤醒")).allow)
        self.assertFalse(other.decide(utterance(text="gate2未唤醒")).allow)


if __name__ == "__main__":
    unittest.main()
