"""Real checkpoint integration tests with generated audio; no user files."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("NUMBA_CACHE_DIR", os.path.join(os.path.dirname(__file__), ".cache", "numba"))
from pathlib import Path
import sys
import tempfile
import time

import numpy as np
import soundfile as sf
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication
from app import Window, STYLE
from engine import SPECIALISTS, STEMS


def main():
    app = QApplication([])
    app.setStyle("Fusion")
    app.setStyleSheet(STYLE)
    kinds = sys.argv[1:] or list(SPECIALISTS)
    with tempfile.TemporaryDirectory(prefix="specialist-test-") as directory:
        root = Path(directory)
        rate = 44100
        t = np.arange(rate * 2) / rate
        mix = np.stack([0.08 * np.sin(2 * np.pi * 440 * t), 0.07 * np.sin(2 * np.pi * 110 * t)], axis=1).astype("float32")
        source = root / "日本語 バンド.wav"
        sf.write(source, mix, rate, subtype="FLOAT")
        window = Window()
        window.output = root / "results"
        window.update_destination()
        window.select_source(str(source))
        window.show()
        for kind in kinds:
            window.specialist_model.setCurrentIndex(window.specialist_model.findData(kind))
            print(f"TEST: {kind}", flush=True)
            window.specialist_start.click()
            assert window.worker.count == kind
            assert not window.specialist_start.isEnabled()
            assert not window.specialist_model.isEnabled()
            errors, outputs = [], []
            window.worker.failed.connect(errors.append)
            window.worker.result.connect(outputs.append)
            window.worker.log.connect(lambda line: print(line, flush=True))
            deadline = time.monotonic() + 900
            while window.worker.isRunning() and time.monotonic() < deadline:
                QTest.qWait(100)
            if window.worker.isRunning():
                window.worker.runner.cancel()
                window.worker.wait(5000)
                raise AssertionError("Inference timed out")
            QTest.qWait(100)
            assert not errors, errors
            assert outputs and outputs[0]
            folder = outputs[0]
            assert {p.stem for p in folder.glob("*.wav")} == set(STEMS[kind])
            assert len(window.play_buttons) == len(STEMS[kind])
            for stem in STEMS[kind]:
                data, sr = sf.read(folder / f"{stem}.wav", dtype="float32")
                info = sf.info(folder / f"{stem}.wav")
                assert sr == rate and data.shape == mix.shape and np.isfinite(data).all()
                assert info.subtype == "PCM_24"
            assert window.specialist_start.isEnabled()
            assert window.specialist_start.isEnabled()
            window.resize(840, 1200)
            QTest.qWait(100)
            window.grab().save(f"/tmp/stem-{kind}.png")
            print(f"PASS: {kind} real inference, exact target stems, format, duration, UI", flush=True)
        window.close()


if __name__ == "__main__":
    main()
