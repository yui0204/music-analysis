"""Headless playback integration tests, using generated WAVs."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import tempfile
import unittest
from pathlib import Path
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

    def test_seek_and_switch(self):
        with tempfile.TemporaryDirectory() as directory:
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
            slider, label = window.seek_controls[str(vocals)]
            self.assertEqual(slider.maximum(), 8000)
            slider.setValue(3000)
            self.assertEqual(label.text(), "0:03 / 0:08")
            self.assertTrue(window.player.source().isEmpty())
            window.play(vocals)
            QTest.qWait(350)
            self.assertGreaterEqual(window.player.position(), 3000)
            self.assertLess(window.player.position(), 4000)
            window.play(vocals)
            self.assertEqual(window.player.playbackState(), QMediaPlayer.PausedState)
            slider.setValue(5000)
            self.assertEqual(window.player.position(), 5000)
            self.assertEqual(window.player.playbackState(), QMediaPlayer.PausedState)
            window.play(vocals)
            slider.setValue(2000)
            self.assertEqual(window.player.playbackState(), QMediaPlayer.PlayingState)
            self.assertAlmostEqual(window.player.position(), 2000, delta=100)
            other, _ = window.seek_controls[str(drums)]
            other.setValue(4000)
            self.assertEqual(window.player.source().toLocalFile(), str(vocals))
            window.play(drums)
            QTest.qWait(350)
            self.assertGreaterEqual(window.player.position(), 4000)
            self.assertLess(window.player.position(), 5000)
            self.assertEqual(window.play_buttons[str(vocals)].text(), "▶ 試聴")
            self.assertEqual(window.play_buttons[str(drums)].text(), "Ⅱ 一時停止")
            window.play(drums)
            QTest.mouseClick(slider, Qt.LeftButton, pos=QPoint(slider.width() // 2, slider.height() // 2))
            self.assertAlmostEqual(slider.value(), 4000, delta=150)
            self.assertFalse(slider.isSliderDown())
            QTest.mousePress(slider, Qt.LeftButton, pos=QPoint(slider.width() // 2, 10))
            QTest.mouseMove(slider, QPoint(slider.width() * 3 // 4, 10))
            QTest.mouseRelease(slider, Qt.LeftButton, pos=QPoint(slider.width() * 3 // 4, 10))
            self.assertAlmostEqual(slider.value(), 6000, delta=200)
            self.assertFalse(slider.isSliderDown())
            QTest.keyClick(slider, Qt.Key_Left)
            self.assertAlmostEqual(slider.value(), 5000, delta=200)
            window.play(vocals)
            QTest.qWait(250)
            self.assertGreaterEqual(window.player.position(), 4800)
            self.assertLess(window.player.position(), 5600)
            window.play(vocals)
            window.resize(820, 1150)
            QTest.qWait(100)
            window.grab().save("/tmp/stem-seek.png")
            window.close()


if __name__ == "__main__":
    unittest.main()
