"""Separation process runner; independent of the UI so it can be tested headlessly."""
import os
import re
import signal
import subprocess
import sys
import threading
from pathlib import Path

SPECIALISTS = {
    "bs_sw": ("BS RoFormer SW · 6 stem", "bs_roformer_sw", ("vocals", "drums", "bass", "guitar", "piano", "other")),
    "bs_sw_acoustic_guitar": ("BS RoFormer SW → Mega53 · アコギ", "bs_roformer_sw_acoustic_guitar",
                              ("vocals", "drums", "bass", "guitar", "piano", "other", "acoustic-guitar", "guitar-other")),
}
STEMS = {key: spec[2] for key, spec in SPECIALISTS.items()}
EXTENSIONS = {".wav", ".mp3", ".flac", ".m4a", ".aiff", ".aif", ".ogg", ".aac", ".mp4"}


def command(source, destination, count, output_format="int24"):
    source, destination = Path(source), Path(destination)
    if output_format != "int24":
        raise ValueError("保存形式は24-bit WAVのみ対応しています。")
    if count not in SPECIALISTS:
        raise ValueError("対応していないモデルです。")
    return [sys.executable, str(Path(__file__).with_name("bs_roformer_worker.py")), count,
            str(source), str(destination), output_format]


def complete_output(folder, count):
    return all((Path(folder) / (stem + ".wav")).is_file() for stem in STEMS[count])


class Separation:
    def __init__(self):
        self.cancelled = threading.Event()
        self.process = None

    def cancel(self):
        self.cancelled.set()
        process = self.process
        if process and process.poll() is None:
            try:
                if os.name == "posix":
                    os.killpg(process.pid, signal.SIGTERM)
                else:
                    process.terminate()
            except ProcessLookupError:
                pass

    def run(self, source, destination, count, log, progress, output_format="int24"):
        source, destination = Path(source).resolve(), Path(destination).resolve()
        if not source.is_file() or source.suffix.lower() not in EXTENSIONS:
            raise ValueError("対応する音声ファイルを選択してください。")
        destination = destination / source.stem
        if count not in SPECIALISTS:
            raise ValueError("対応していないモデルです。")
        folder = destination
        if complete_output(folder, count):
            log(f"既存結果を使用します: {folder}")
            progress(100)
            return folder
        folder.parent.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env["PATH"] = "/opt/homebrew/bin:/usr/local/bin:" + env.get("PATH", "")
        # Keep downloaded weights with the app, rather than modifying user caches.
        env["NUMBA_CACHE_DIR"] = str(Path(__file__).parent / ".cache" / "numba")
        env["BS_ROFORMER_MODELS_PATH"] = str(Path(__file__).parent / ".cache" / "bs_roformer")
        env["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        self.process = subprocess.Popen(command(source, folder, count, output_format),
                                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                        text=True, bufsize=1, env=env, start_new_session=True)
        if self.cancelled.is_set():
            self.cancel()
        tail = []
        try:
            for line in self.process.stdout:
                line = line.strip()
                if not line:
                    continue
                tail = (tail + [line])[-15:]
                log(line)
                if line.startswith("STEM_PROGRESS "):
                    progress(min(99, int(line.split()[1])))
                # tqdm model downloads also contain percentages. Only report
                # separation progress when the unit is audio seconds.
                match = re.search(r"(\d+)%.*\[.*(?:s/s|s/it)", line)
                if match:
                    progress(min(99, int(match.group(1))))
            code = self.process.wait()
        finally:
            self.process.stdout.close()
        if self.cancelled.is_set():
            return None
        if code:
            raise RuntimeError("分離に失敗しました。詳細ログを確認してください。\n" + "\n".join(tail[-5:]))
        if not complete_output(folder, count):
            raise RuntimeError("分離したファイルを確認できませんでした。")
        progress(100)
        return folder
