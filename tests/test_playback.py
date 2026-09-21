"""Headless playback integration tests, using generated WAVs."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np
import soundfile as sf
from PySide6.QtCore import QPoint, Qt
from PySide6.QtMultimedia import QMediaPlayer
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication
from app import Window, STYLE
from engine import STEMS


class PlaybackTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        cls.app.setStyle("Fusion")
        cls.app.setStyleSheet(STYLE)

    def test_synchronized_mix_playback(self):
        class FakeStream:
            def __init__(self, **kwargs):
                self.callback = kwargs["callback"]
                self.active = False
            def start(self):
                self.active = True
            def stop(self):
                self.active = False
            def close(self):
                self.active = False

        with tempfile.TemporaryDirectory() as directory, patch("app.sd.OutputStream", FakeStream):
            folder = Path(directory)
            for stem in STEMS["bs_sw"]:
                sf.write(folder / (stem + ".wav"), np.zeros((44100 * 8, 2)), 44100)
            window = Window()
            window.worker = SimpleNamespace(count="bs_sw", isRunning=lambda: False)
            window.on_result(folder)
            window.show()
            QTest.qWait(100)

            vocals = folder / "vocals.wav"
            drums = folder / "drums.wav"
            # The default is silent until the user selects a part.
            self.assertFalse(any(box.isChecked() for box in window.mix_checks.values()))
            window.mix_checks[str(vocals)].setChecked(True)
            slider, label = window.seek_controls[str(vocals)]
            slider.setValue(3000)
            self.assertEqual(label.text(), "0:03 / 0:08")

            window.toggle_mix_playback()
            self.assertTrue(window.mix_realtime_playing)
            self.assertEqual(window.mix_selected_paths, {str(vocals)})
            # The callback owns a shared sample clock, independent of QMediaPlayer.
            out = np.zeros((441, 2), dtype=np.float32)
            window._mix_audio_callback(out, len(out), None, None)
            window.refresh_playback_position()
            self.assertGreaterEqual(window.smooth_position, 3000)

            # Changing a checkbox changes only the callback selection; it never
            # rewinds or rebuilds the transport.
            before = window.mix_frame
            window.mix_checks[str(drums)].setChecked(True)
            self.assertEqual(window.mix_selected_paths, {str(vocals), str(drums)})
            self.assertEqual(window.mix_frame, before)
            slider.setValue(2000)
            self.assertAlmostEqual(window.mix_frame, 88200, delta=2)

            window.stop_mix_playback()
            self.assertFalse(window.mix_realtime_playing)
            window.clear_results()
            window.close()
            QTest.qWait(100)

if __name__ == "__main__":
    unittest.main()
