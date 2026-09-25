import asyncio
import os
import shutil
import tempfile
import unittest
import wave
import time
import sys
from pathlib import Path

# Add parent dir to path to import volc_stt_adapter
sys.path.append(str(Path(__file__).parent.parent))

from volc_stt_adapter import WakeRecorder, cleanup_wake_audio

class WakeAudioTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.directory = Path(self.test_dir)

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    async def test_wake_recorder_lifecycle(self):
        preroll = b"\x01\x00" * 16000  # 1s at 16k
        recorder = WakeRecorder(
            self.directory,
            score=0.987,
            preroll=preroll,
            postroll_samples=16000  # 1s post-roll
        )
        
        # Initial state
        self.assertEqual(len(recorder.pcm), 32000)
        self.assertFalse(recorder.finished)
        
        # Append some data
        postroll_data = b"\x02\x00" * 8000
        recorder.append(postroll_data)
        self.assertFalse(recorder.finished)
        
        # Complete post-roll
        recorder.append(postroll_data)
        self.assertTrue(recorder.finished)
        self.assertEqual(len(recorder.pcm), 64000)
        
        # Save
        await recorder.save(sample_rate=16000, max_age_days=1, max_size_mb=10)
        
        # Verify file exists
        files = list(self.directory.glob("false_wake_*.wav"))
        self.assertEqual(len(files), 1)
        self.assertIn("0.987", files[0].name)
        
        # Verify content
        with wave.open(str(files[0]), "rb") as f:
            self.assertEqual(f.getnframes(), 32000)
            self.assertEqual(f.getnchannels(), 1)
            self.assertEqual(f.getsampwidth(), 2)

    async def test_cleanup_logic(self):
        # Create 3 files with different ages
        files = []
        now = time.time()
        for i in range(3):
            path = self.directory / f"false_wake_20260925_00000{i}_0.900.wav"
            with wave.open(str(path), "wb") as f:
                f.setnchannels(1)
                f.setsampwidth(2)
                f.setframerate(16000)
                f.writeframes(b"\x00" * 1024)
            # Set mtime back: 0 -> 40 days ago, 1 -> 10 days ago, 2 -> now
            mtime = now - (40 - i * 20) * 86400
            os.utime(path, (mtime, mtime))
            files.append(path)

        # Cleanup: 30 days limit
        await cleanup_wake_audio(str(self.directory), max_age_days=30, max_size_mb=500)
        
        self.assertFalse(files[0].exists())
        self.assertTrue(files[1].exists())
        self.assertTrue(files[2].exists())

    async def test_size_cleanup(self):
        # Each file is ~1MB. Set limit to 2MB.
        now = time.time()
        for i in range(5):
            path = self.directory / f"false_wake_20260925_00000{i}_0.900.wav"
            with wave.open(str(path), "wb") as f:
                f.setnchannels(1)
                f.setsampwidth(2)
                f.setframerate(16000)
                f.writeframes(b"\x00" * 500000) # 1,000,000 bytes
            # Oldest first
            mtime = now - (10 - i) * 100
            os.utime(path, (mtime, mtime))
            
        # Limit to 2 MB
        await cleanup_wake_audio(str(self.directory), max_age_days=30, max_size_mb=2)
        
        remaining = list(self.directory.glob("*.wav"))
        self.assertLess(len(remaining), 5)
        # Should keep the newest one
        newest = max(remaining, key=lambda p: p.stat().st_mtime)
        self.assertIn("00004", newest.name)

if __name__ == "__main__":
    unittest.main()
