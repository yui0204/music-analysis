"""Vocal melody transcription via Basic Pitch for a synchronized vocal roll view."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pretty_midi

VOCAL_LOW = 48   # C3
VOCAL_HIGH = 84  # C6


def _empty_events() -> dict:
    return {"_model": "Basic Pitch vocal melody", "notes": []}


def dominant_monophonic(notes: list[dict], step: float = 0.04, min_length: float = 0.06) -> list[dict]:
    if not notes:
        return []
    clean = []
    for note in notes:
        try:
            start = float(note.get("start", 0))
            end = float(note.get("end", start))
            pitch = int(note.get("pitch", 0))
            velocity = int(note.get("velocity", 0))
        except (TypeError, ValueError):
            continue
        if end <= start or pitch < VOCAL_LOW or pitch > VOCAL_HIGH:
            continue
        clean.append({"pitch": pitch, "start": start, "end": end, "velocity": velocity})
    if not clean:
        return []

    first = min(note["start"] for note in clean)
    last = max(note["end"] for note in clean)
    frames = []
    t = first
    while t < last:
        mid = t + step / 2
        active = [note for note in clean if note["start"] <= mid < note["end"]]
        if active:
            best = max(active, key=lambda n: (n["velocity"], -abs(n["pitch"] - 64)))
            frames.append((round(t, 3), round(t + step, 3), best))
        t += step

    merged = []
    for start, end, note in frames:
        pitch = int(note["pitch"])
        velocity = int(note["velocity"])
        if merged and merged[-1]["pitch"] == pitch and start <= merged[-1]["end"] + step * 1.5:
            merged[-1]["end"] = end
            merged[-1]["velocity"] = max(merged[-1]["velocity"], velocity)
        else:
            merged.append({"pitch": pitch, "start": start, "end": end, "velocity": velocity})

    return [
        {
            "pitch": int(note["pitch"]),
            "start": round(float(note["start"]), 3),
            "end": round(float(note["end"]), 3),
            "velocity": int(note["velocity"]),
        }
        for note in merged
        if note["end"] - note["start"] >= min_length
    ]


def midi_to_events(midi_path: str | Path) -> dict:
    midi = pretty_midi.PrettyMIDI(str(midi_path))
    notes = []
    for instrument in midi.instruments:
        if instrument.is_drum:
            continue
        for note in instrument.notes:
            pitch = int(note.pitch)
            if pitch < VOCAL_LOW or pitch > VOCAL_HIGH:
                continue
            notes.append({
                "pitch": pitch,
                "start": round(float(note.start), 3),
                "end": round(float(note.end), 3),
                "velocity": int(note.velocity),
            })
    notes.sort(key=lambda item: (item["start"], item["pitch"], item["end"]))
    mono = dominant_monophonic(notes)
    return {"_model": "Basic Pitch vocal melody", "notes": mono, "_raw_notes": len(notes)}


def _find_midi(output_dir: Path, source: Path) -> Path:
    candidates = [output_dir / f"{source.stem}_basic_pitch.mid", output_dir / f"{source.stem}.mid"]
    candidates.extend(sorted(output_dir.glob("*.mid"), key=lambda p: p.stat().st_mtime, reverse=True))
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise RuntimeError("Basic PitchのMIDI出力が見つかりませんでした。")


def estimate_vocal_events(path: str | Path, folder: str | Path) -> dict:
    path = Path(path)
    folder = Path(folder)
    out_dir = folder / ".basic-pitch-vocals"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = out_dir / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    old_tmpdir = os.environ.get("TMPDIR")
    os.environ["TMPDIR"] = str(tmp_dir)
    try:
        from basic_pitch.inference import ICASSP_2022_MODEL_PATH, predict_and_save
        predict_and_save(
            [str(path)],
            str(out_dir),
            save_midi=True,
            sonify_midi=False,
            save_model_outputs=False,
            save_notes=False,
            model_or_model_path=ICASSP_2022_MODEL_PATH,
            minimum_frequency=120.0,
            maximum_frequency=1050.0,
        )
    except Exception as api_error:
        command = [sys.executable, "-m", "basic_pitch", str(out_dir), str(path)]
        env = dict(os.environ)
        env["TMPDIR"] = str(tmp_dir)
        try:
            subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
        except Exception as cli_error:
            raise RuntimeError(f"Basic Pitchを実行できませんでした: {api_error}; CLI: {cli_error}") from cli_error
    finally:
        if old_tmpdir is None:
            os.environ.pop("TMPDIR", None)
        else:
            os.environ["TMPDIR"] = old_tmpdir

    return midi_to_events(_find_midi(out_dir, path))


def load_or_estimate(folder: str | Path, force: bool = False) -> dict:
    folder = Path(folder)
    cache = folder / "vocals-midi.json"
    if cache.is_file() and not force:
        data = json.loads(cache.read_text(encoding="utf-8"))
        data.setdefault("_model", "Basic Pitch vocal melody")
        data.setdefault("notes", [])
        if "vocal melody" not in data.get("_model", "").lower():
            raw_count = len(data.get("notes", []))
            data["notes"] = dominant_monophonic(data.get("notes", []))
            data["_model"] = "Basic Pitch vocal melody"
            data["_raw_notes"] = raw_count
        return data
    vocals = folder / "vocals.wav"
    if not vocals.is_file():
        return _empty_events()
    events = estimate_vocal_events(vocals, folder)
    cache.write_text(json.dumps(events, ensure_ascii=False, indent=2), encoding="utf-8")
    return events
