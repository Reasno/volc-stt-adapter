from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from audio_gate import GateMode
from volc_stt_adapter import Settings, utterance_times_ms


BASE_ENV = {
    "ENV_FILE": "",
    "VOLC_APP_KEY": "test-app",
    "VOLC_RESOURCE_ID": "test-resource",
}


class SettingsTest(unittest.TestCase):
    def test_kws_defaults_to_shadow(self):
        with patch.dict(os.environ, BASE_ENV, clear=True):
            settings = Settings.from_environment()
        self.assertEqual(settings.kws_mode, GateMode.SHADOW)
        self.assertEqual(settings.kws_threshold, 0.5)
        self.assertEqual(settings.kws_preroll_seconds, 1.5)
        self.assertEqual(settings.volc_tts_resource_id, "seed-tts-2.0")
        self.assertEqual(settings.volc_tts_voice, "zh_female_vv_uranus_bigtts")
        self.assertEqual(settings.volc_tts_cache_entries, 100)

    def test_invalid_mode_and_fractional_queue_are_rejected(self):
        with patch.dict(os.environ, {**BASE_ENV, "KWS_MODE": "enabled"}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "KWS_MODE"):
                Settings.from_environment()
        with patch.dict(os.environ, {**BASE_ENV, "KWS_QUEUE_FRAMES": "2.5"}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "integer"):
                Settings.from_environment()

    def test_tts_cache_entries_must_be_non_negative_integer(self):
        for value in ("-1", "1.5", "invalid"):
            with self.subTest(value=value), patch.dict(
                os.environ, {**BASE_ENV, "VOLC_TTS_CACHE_ENTRIES": value}, clear=True
            ):
                with self.assertRaisesRegex(RuntimeError, "VOLC_TTS_CACHE_ENTRIES"):
                    Settings.from_environment()
        with patch.dict(
            os.environ, {**BASE_ENV, "VOLC_TTS_CACHE_ENTRIES": "0"}, clear=True
        ):
            self.assertEqual(Settings.from_environment().volc_tts_cache_entries, 0)

    def test_sensitive_values_are_not_in_repr(self):
        with patch.dict(
            os.environ,
            {**BASE_ENV, "VOLC_ACCESS_KEY": "secret-access"},
            clear=True,
        ):
            settings = Settings.from_environment()
        rendered = repr(settings)
        self.assertNotIn("test-app", rendered)
        self.assertNotIn("secret-access", rendered)

    def test_common_utterance_timeline_fields(self):
        self.assertEqual(utterance_times_ms({"start_time": 12, "end_time": "34"}), (12.0, 34.0))
        self.assertEqual(
            utterance_times_ms({"additions": {"start_ms": 56, "end_ms": 78}}),
            (56.0, 78.0),
        )
        self.assertEqual(utterance_times_ms({"start_time": 10}), (None, None))


if __name__ == "__main__":
    unittest.main()
