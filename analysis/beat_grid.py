"""Beat/downbeat estimation and drum quantization helpers."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def _empty_grid() -> dict:
    return {"_model": "Beat This!", "beats": [], "downbeats": [], "bpm": None}


def estimate_beats(audio_path: str | Path) -> dict:
    from beat_this.inference import File2Beats
    tracker = File2Beats(checkpoint_path="final0", device="cpu", dbn=False)
    beats, downbeats = tracker(str(audio_path))
    beats = sorted(round(float(x), 4) for x in beats)
    downbeats = sorted(round(float(x), 4) for x in downbeats)
    intervals = np.diff(beats)
    intervals = intervals[(intervals > 0.2) & (intervals < 2.5)]
    bpm = None
    if intervals.size:
        bpm = round(float(60.0 / np.median(intervals)), 2)
    return {"_model": "Beat This!", "beats": beats, "downbeats": downbeats, "bpm": bpm}


def load_or_estimate(folder: str | Path, force: bool = False) -> dict:
    folder = Path(folder)
    cache = folder / "beat-grid.json"
    if cache.is_file() and not force:
        data = json.loads(cache.read_text(encoding="utf-8"))
        data.setdefault("beats", [])
        data.setdefault("downbeats", [])
        data.setdefault("bpm", None)
        data.setdefault("_model", "Beat This!")
        return data
    audio = folder / "drums.wav"
    if not audio.is_file():
        audio = folder / "piano.wav"
    if not audio.is_file():
        return _empty_grid()
    grid = estimate_beats(audio)
    cache.write_text(json.dumps(grid, ensure_ascii=False, indent=2), encoding="utf-8")
    return grid


def quantize_drum_events(events: dict, grid: dict) -> dict:
    beats = [float(x) for x in grid.get("beats", [])]
    if not beats:
        return events
    grid_points = []
    for left, right in zip(beats, beats[1:]):
        span = right - left
        for divisions in (4, 3):
            grid_points.extend(left + span * index / divisions for index in range(divisions))
    grid_points.extend(beats)
    quantized = dict(events)
    for lane, times in events.items():
        if lane.startswith("_"):
            continue
        snapped = []
        for time in times:
            nearest = min(grid_points, key=lambda beat: abs(beat - float(time)))
            snapped.append(round(nearest, 4))
        # Do not deduplicate: repeated/nearby hits should keep their visual dots.
        quantized[lane] = sorted(snapped)
    quantized["_quantized_to"] = "Beat This! 16th grid"
    return quantized


def quantize_note_events(events: dict, grid: dict, preserve_end: bool = False) -> dict:
    """Snap note timing to nearby 16th/triplet subdivisions.

    Piano can preserve note ends so sustained chord voicings retain their
    original duration while only the onset is aligned.
    """
    beats = [float(x) for x in grid.get("beats", [])]
    if len(beats) < 2:
        return events
    points = []
    for left, right in zip(beats, beats[1:]):
        span = right - left
        for divisions in (4, 3):
            points.extend(left + span * index / divisions for index in range(divisions))
    points.extend(beats)
    quantized = dict(events)
    notes = []
    for note in events.get("notes", []):
        item = dict(note)
        start = float(item.get("start", 0.0))
        end = float(item.get("end", start))
        item["start"] = round(min(points, key=lambda point: abs(point - start)), 4)
        if not preserve_end:
            item["end"] = round(max(item["start"], min(points, key=lambda point: abs(point - end))), 4)
        notes.append(item)
    quantized["notes"] = notes
    quantized["_quantized_to"] = "Beat This! 16th/triplet grid"
    return quantized
