"""Piano transcription via Transkun V2 for the synchronized piano roll."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pretty_midi


def _empty_events() -> dict:
    return {"_model": "Transkun V2", "notes": []}


def midi_to_events(midi_path: str | Path) -> dict:
    midi = pretty_midi.PrettyMIDI(str(midi_path))
    notes = []
    for instrument in midi.instruments:
        if instrument.is_drum:
            continue
        for note in instrument.notes:
            notes.append({
                "pitch": int(note.pitch),
                "start": round(float(note.start), 3),
                "end": round(float(note.end), 3),
                "velocity": int(note.velocity),
            })
    notes.sort(key=lambda item: (item["start"], item["pitch"], item["end"]))
    return {"_model": "Transkun V2", "notes": notes}


def estimate_piano_events(path: str | Path, folder: str | Path) -> dict:
    path = Path(path)
    folder = Path(folder)
    midi_path = folder / "piano-transkun.mid"
    command = [sys.executable, "-m", "transkun.transcribe", str(path), str(midi_path), "--device", "cpu"]
    subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    return midi_to_events(midi_path)


def load_or_estimate(folder: str | Path, force: bool = False) -> dict:
    folder = Path(folder)
    cache = folder / "piano-midi.json"
    if cache.is_file() and not force:
        data = json.loads(cache.read_text(encoding="utf-8"))
        data.setdefault("_model", "Transkun V2")
        data.setdefault("notes", [])
        return data
    piano = folder / "piano.wav"
    if not piano.is_file():
        return _empty_events()
    events = estimate_piano_events(piano, folder)
    cache.write_text(json.dumps(events, ensure_ascii=False, indent=2), encoding="utf-8")
    return events
