"""Cancel an already-running child and confirm no success is reported."""
import sys
import tempfile
import threading
from pathlib import Path
from unittest.mock import patch
from engine import Separation

with tempfile.TemporaryDirectory() as directory:
    source = Path(directory) / "input.wav"
    source.touch()
    ready = threading.Event()
    result = []
    runner = Separation()
    with patch("engine.command", return_value=[sys.executable, "-u", "-c",
               "import time; print('READY', flush=True); time.sleep(30)"]):
        worker = threading.Thread(target=lambda: result.append(runner.run(
            source, Path(directory) / "out", "bs_sw", lambda _: ready.set(), lambda _: None)))
        worker.start()
        assert ready.wait(5)
        runner.cancel()
        worker.join(5)
        assert not worker.is_alive()
        assert result == [None]
        assert runner.process.poll() is not None
print("PASS: in-flight cancellation terminates child and reports cancellation")
