from __future__ import annotations

import unittest

from audio_gate import AudioGate, GateMode, Utterance, WakeMarker


class AudioGateLifecycleTest(unittest.TestCase):
    def test_wake_records_marker_without_speaker_authorization(self):
        gate = AudioGate(mode="enforce")
        marker = WakeMarker(7, 16000, 1000.0)

        gate.on_wake(marker)

        self.assertEqual(gate.stream_generation, 7)
        self.assertEqual(gate.wake, marker)
        self.assertFalse(gate.proactive_reply_armed)

    def test_proactive_arm_tracks_only_current_generation(self):
        gate = AudioGate(mode="enforce")
        gate.reset(3)

        self.assertFalse(gate.arm_for_reply(2))
        self.assertTrue(gate.arm_for_reply(3))
        self.assertTrue(gate.proactive_reply_armed)

    def test_reset_and_clear_remove_lifecycle_markers(self):
        gate = AudioGate(mode="enforce")
        gate.on_wake(WakeMarker(1, 10, 1.0))
        gate.reset(2)
        self.assertEqual(gate.stream_generation, 2)
        self.assertIsNone(gate.wake)

        self.assertTrue(gate.arm_for_reply(2))
        gate.clear()
        self.assertIsNone(gate.stream_generation)
        self.assertIsNone(gate.wake)
        self.assertFalse(gate.proactive_reply_armed)

    def test_off_mode_ignores_wake_and_cannot_arm(self):
        gate = AudioGate(mode=GateMode.OFF)
        gate.on_wake(WakeMarker(1, 10, 1.0))

        self.assertIsNone(gate.wake)
        self.assertFalse(gate.arm_for_reply(1))

    def test_utterance_keeps_speaker_metadata_as_data_only(self):
        utterance = Utterance("你好", "speaker-3", 9, 100.0, 300.0)

        self.assertEqual(utterance.speaker_id, "speaker-3")
        self.assertEqual(utterance.stream_generation, 9)

    def test_constructor_validation_is_preserved(self):
        with self.assertRaises(ValueError):
            AudioGate(trigger_timeout_s=0)
        with self.assertRaises(ValueError):
            AudioGate(speaker_window_s=0)
        with self.assertRaises(ValueError):
            AudioGate(match_tolerance_ms=-1)
        with self.assertRaises(ValueError):
            AudioGate(closing_seconds=0)


if __name__ == "__main__":
    unittest.main()
