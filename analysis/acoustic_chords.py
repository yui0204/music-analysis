"""Offline Solitito inference for acoustic-guitar.wav.

DSP ported from greblus/solitito src/audio.rs (MIT; vendor/solitito/LICENSE),
commit e6a38fbabb9d4a6267a5a966d2707fe22c2cf8d9. Centered windows and
three-frame voting are this app's offline adaptation, not Solitito's live UI.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import urllib.request

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent
MODEL_REVISION = "96f63770aea422a5aa2cf6dc775f0375f5866ef4"
MODEL_FILE = "best_model_v2_take6_onset.onnx"
CACHE_FILE = "acoustic-guitar-chords.json"
ASSETS = {
    MODEL_FILE: "0c30d2c904f46a246fcd957da5eca5701a7b6e1f7839bc0ee58bc06b2ca02545",
    "dsp_weights.json": "26fd0135195a4e55cf2791f4d9e6275f1ba2dc36e1b0d2d5121836ca4633a823",
}
PITCHES = ("C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B")
# Preserve the trained taxonomy: 'sus' does not distinguish sus2 from sus4.
QUALITIES = ("", "m", "maj7", "7", "m7", "m7b5", "dim7", "aug", "sus")
QUALITY_INTERVALS = ((0, 4, 7), (0, 3, 7), (0, 4, 7, 11), (0, 4, 7, 10),
                     (0, 3, 7, 10), (0, 3, 6, 10), (0, 3, 6, 9),
                     (0, 4, 8), (0, 5, 7))
SR, FFT_SIZE, HOP, CONTEXT, STRIDE = 16000, 8192, 256, 48, 8


def ensure_assets() -> Path:
    directory = ROOT / ".cache" / "solitito"
    directory.mkdir(parents=True, exist_ok=True)
    for name, digest in ASSETS.items():
        path = directory / name
        if path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == digest:
            continue
        url = f"https://huggingface.co/greblus/solitito-ai/resolve/{MODEL_REVISION}/{name}"
        with tempfile.NamedTemporaryFile(dir=directory, suffix=".part", delete=False) as target:
            temporary = Path(target.name)
            try:
                with urllib.request.urlopen(url, timeout=30) as response:
                    while chunk := response.read(1024 * 1024):
                        target.write(chunk)
                target.flush()
                if hashlib.sha256(temporary.read_bytes()).hexdigest() != digest:
                    raise ValueError(f"Solititoのモデル検証に失敗しました: {name}")
                os.replace(temporary, path)
            finally:
                temporary.unlink(missing_ok=True)
    return directory


class FeatureExtractor:
    def __init__(self, path: Path):
        from scipy.sparse import csr_matrix
        data = json.loads(path.read_text())
        if (data.get("format"), data.get("sr"), data.get("fft_size"), data.get("n_bins")) != ("sparse-csr-v1", SR, FFT_SIZE, 144):
            raise ValueError("Solitito DSP weights are incompatible")
        kernel = np.asarray(data["cqt_re"], dtype=np.float32) + 1j * np.asarray(data["cqt_im"], dtype=np.float32)
        self.kernel = csr_matrix((kernel, data["cqt_fft_idx"], data["cqt_offsets"]), shape=(144, 4097))
        self.chroma = np.asarray(data["chroma_weights"], dtype=np.float32).reshape(144, 12)
        self.window = np.hanning(FFT_SIZE).astype(np.float32)

    def transform(self, chunks: np.ndarray) -> np.ndarray:
        from scipy.fft import rfft
        spectrum = rfft(chunks * self.window, axis=-1)
        magnitude = np.abs(self.kernel @ spectrum.T).T
        # Match Solitito's file-input DSP (bass boost on, gain 5, first 36 bins).
        magnitude[:, :36] *= 5.0
        reference = np.maximum(magnitude.max(axis=1, keepdims=True), 0.005)
        cqt = np.clip((20 * np.log10(np.maximum(magnitude, 1e-9) / reference) + 80) / 80, 0, 1)
        chroma = cqt @ self.chroma
        chroma /= np.maximum(chroma.max(axis=1, keepdims=True), 1e-9)
        bass = cqt[:, :24].reshape(-1, 12, 2).mean(axis=2)
        return np.concatenate((cqt, chroma, bass), axis=1).astype(np.float32)


def _softmax(logits):
    value = np.exp(logits - np.max(logits))
    return value / value.sum()


def _prediction(root_logits, quality_logits):
    roots, qualities = _softmax(root_logits), _softmax(quality_logits)
    root, quality = int(roots.argmax()), int(qualities.argmax())
    if root >= 12 or quality >= 9:
        return "N.C.", 0.0
    return PITCHES[root] + QUALITIES[quality], float(roots[root] * qualities[quality])


def _midi_guide(folder: Path):
    """Load sustained MIDI evidence used as a weak prior for Solitito frames."""
    try:
        grid = json.loads((folder / "beat-grid.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        grid = {}
    beats = sorted(float(t) for t in grid.get("beats", []) if np.isfinite(float(t)))
    intervals = np.diff(beats) if len(beats) >= 2 else np.array([])
    valid = intervals[(intervals > 0.2) & (intervals < 2.5)]
    quarter = float(np.median(valid)) if len(valid) else 60.0 / max(30.0, min(300.0, float(grid.get("bpm") or 120)))
    result = {"piano": [], "bass": [], "quarter": quarter, "grid": grid}
    for kind, filename in (("piano", "piano-midi.json"), ("bass", "bass-midi.json")):
        try:
            notes = json.loads((folder / filename).read_text(encoding="utf-8")).get("notes", [])
        except (OSError, ValueError):
            notes = []
        eighth = quarter / 2
        if kind == "bass":
            result[kind] = [dict(note) for note in notes
                            if float(note.get("end", 0)) - float(note.get("start", 0)) >= eighth - 1e-4]
            continue
        merged = []
        for note in sorted((dict(note) for note in notes),
                           key=lambda item: (int(item.get("pitch", 0)), float(item.get("start", 0)))):
            if (merged and int(merged[-1].get("pitch", 0)) == int(note.get("pitch", 0)) and
                    float(note.get("start", 0)) - float(merged[-1].get("end", 0)) <= eighth * 0.35):
                merged[-1]["end"] = max(float(merged[-1].get("end", 0)), float(note.get("end", 0)))
            else:
                merged.append(note)
        result[kind] = [note for note in merged
                        if (float(note.get("end", 0)) - float(note.get("start", 0)) >= eighth - 1e-4 or
                            sum(abs(float(other.get("start", 0)) - float(note.get("start", 0))) <= 0.06
                                for other in merged) >= 2)]
    result["enabled"] = bool(result["piano"] or result["bass"])
    return result


def _midi_bias(guide: dict, start: float, end: float):
    """Return small root/quality priors; acoustic logits remain dominant."""
    if not guide.get("enabled"):
        return np.zeros(12, dtype=np.float32), np.zeros(len(QUALITIES), dtype=np.float32)
    piano = np.zeros(12, dtype=np.float32)
    bass = np.zeros(12, dtype=np.float32)
    for kind, target in (("piano", piano), ("bass", bass)):
        for note in guide[kind]:
            overlap = max(0.0, min(end, float(note.get("end", 0))) - max(start, float(note.get("start", 0))))
            if overlap <= 0:
                continue
            velocity = max(0.1, min(1.0, float(note.get("velocity", 80)) / 127.0))
            target[int(note.get("pitch", 0)) % 12] += overlap * velocity
    total = float(piano.sum() + bass.sum())
    if total <= 0:
        return np.zeros(12, dtype=np.float32), np.zeros(len(QUALITIES), dtype=np.float32)
    root = 1.05 * (bass / max(1e-6, float(bass.sum()))) + 0.28 * (piano / max(1e-6, float(piano.sum()))) if bass.sum() else 0.28 * piano / max(1e-6, float(piano.sum()))
    present = {index for index, value in enumerate(piano + bass) if value > 0}
    quality = np.zeros(len(QUALITIES), dtype=np.float32)
    for index, intervals in enumerate(QUALITY_INTERVALS):
        scores = []
        for candidate_root in range(12):
            pcs = {(candidate_root + interval) % 12 for interval in intervals}
            scores.append(len(present & pcs) / len(pcs) - 0.12 * len(present - pcs))
        quality[index] = max(scores)
    return root.astype(np.float32), quality


def _segments(predictions, duration):
    if not predictions:
        return []
    labels = [item[0] for item in predictions]
    smoothed = labels[:]
    for i in range(1, len(labels) - 1):
        if labels[i - 1] == labels[i + 1] and labels[i] != "N.C.":
            smoothed[i] = labels[i - 1]
    segments = []
    step = STRIDE * HOP / SR
    for i, (name, (_, confidence)) in enumerate(zip(smoothed, predictions)):
        start = 0.0 if i == 0 else (i - 0.5) * step
        end = duration if i == len(labels) - 1 else min(duration, (i + 0.5) * step)
        if end <= start:
            continue
        if name != labels[i]:
            confidence = 0.0  # Do not assign a different chord's confidence.
        if segments and segments[-1]["chord"] == name:
            last = segments[-1]
            length = last["end"] - last["start"]
            last["confidence"] = (last["confidence"] * length + confidence * (end - start)) / (end - last["start"])
            last["end"] = end
        else:
            segments.append(dict(start=start, end=end, chord=name, confidence=confidence))
    return [{k: round(v, 4) if isinstance(v, float) else v for k, v in segment.items()} for segment in segments]


def _quarter_segments(predictions, duration: float, grid: dict):
    """Vote Solitito frames directly into quarter-note analysis cells."""
    beats = sorted({float(t) for t in grid.get("beats", [])
                    if np.isfinite(float(t)) and 0 < float(t) < duration})
    if beats:
        edges = [0.0] + beats + [duration]
    else:
        try:
            bpm = float(grid.get("bpm", 120))
        except (TypeError, ValueError):
            bpm = 120.0
        if not np.isfinite(bpm) or bpm <= 0:
            bpm = 120.0
        step = 60.0 / bpm
        edges = [index * step for index in range(math.ceil(duration / step) + 1)]
        edges[-1] = duration
    frame_step = STRIDE * HOP / SR
    cells = []
    for start, end in zip(edges, edges[1:]):
        members = [item for index, item in enumerate(predictions)
                   if start <= index * frame_step < end]
        if not members:
            midpoint = (start + end) / 2
            members = [predictions[min(len(predictions) - 1, max(0, round(midpoint / frame_step)))]]
        votes = {}
        confidences = {}
        for name, confidence in members:
            votes[name] = votes.get(name, 0.0) + 0.08 + float(confidence)
            confidences.setdefault(name, []).append(float(confidence))
        name = max(votes, key=votes.get)
        confidence = float(np.mean(confidences[name]))
        if cells and cells[-1]["chord"] == name:
            cells[-1]["end"] = end
            cells[-1]["confidence"] = (cells[-1]["confidence"] + confidence) / 2
        else:
            cells.append({"start": start, "end": end, "chord": name, "confidence": confidence})
    return [{key: round(value, 4) if isinstance(value, float) else value
             for key, value in cell.items()} for cell in cells]


def filter_short_chords(data: dict, grid: dict) -> dict:
    """Replace short chord runs with the best supported adjacent chord.

    Keep explicit silence and pre-existing gaps as boundaries.
    Retain raw segments so changing the beat grid never compounds filtering.
    """
    beats = np.asarray(sorted({float(t) for t in grid.get("beats", [])
                               if np.isfinite(float(t)) and float(t) >= 0}))
    downbeats = np.asarray(sorted({float(t) for t in grid.get("downbeats", [])
                                   if np.isfinite(float(t)) and float(t) >= 0}))
    downbeats = np.asarray(sorted({float(t) for t in grid.get("downbeats", [])
                                   if np.isfinite(float(t)) and float(t) >= 0}))
    bpm = grid.get("bpm")
    try:
        bpm = float(bpm)
    except (ValueError, TypeError):
        bpm = 120.0
    if not np.isfinite(bpm) or bpm <= 0:
        bpm = 120.0

    def beat_position(t):
        if len(beats) < 2:
            return t * bpm / 60.0
        # Interpolate in musical time, including tempo changes within a chord.
        index = int(np.clip(np.searchsorted(beats, t, side="right") - 1, 0, len(beats) - 2))
        return index + (t - beats[index]) / (beats[index + 1] - beats[index])

    raw = data.get("_raw_chords", data.get("chords", []))
    kept = []
    for chord in raw:
        start, end = float(chord["start"]), float(chord["end"])
        duration_beats = beat_position(end) - beat_position(start)
        weak_short = ("confidence" in chord and
                      float(chord.get("confidence", 0.5)) < 0.70 and
                      duration_beats <= 1.0 + 1e-8)
        kept.append(end > start and (chord["chord"] == "N.C." or
                                    (duration_beats >= 1.0 - 1e-8 and not weak_short)))

    def adjacent(a, b):
        return abs(float(a["end"]) - float(b["start"])) < 1e-7

    def preceding_barline(t):
        if not len(downbeats):
            return None
        candidates = downbeats[downbeats <= float(t) + 1e-7]
        return float(candidates[-1]) if len(candidates) else None

    def support(chord):
        length = beat_position(float(chord["end"])) - beat_position(float(chord["start"]))
        # Cap duration influence: a very long but uncertain chord should not
        # automatically defeat a clearly recognized neighbouring chord.
        confidence = float(chord.get("confidence", 0.5))
        return (0.25 + 0.75 * confidence) * min(4.0, length)

    repaired = []
    i = 0
    while i < len(raw):
        if kept[i]:
            repaired.append(dict(raw[i]))
            i += 1
            continue
        first = i
        i += 1
        while i < len(raw) and not kept[i] and adjacent(raw[i - 1], raw[i]):
            i += 1
        candidates = []
        if first > 0 and kept[first - 1] and adjacent(raw[first - 1], raw[first]):
            candidates.append(raw[first - 1])
        if i < len(raw) and kept[i] and adjacent(raw[i - 1], raw[i]):
            candidates.append(raw[i])
        candidates = [c for c in candidates if c["chord"] != "N.C."]
        weak_short = ("confidence" in raw[first] and
                      float(raw[first].get("confidence", 0.5)) < 0.70)
        # A low-confidence one-beat ornament at a bar-end should extend the
        # established preceding chord instead of pulling the next chord early.
        if weak_short and first > 0 and kept[first - 1] and candidates:
            chosen = raw[first - 1]
        else:
            chosen = max(candidates, key=support) if candidates else None
        same_bridge = (len(candidates) == 2 and
                       candidates[0].get("chord") == candidates[1].get("chord"))
        repaired.append({"start": raw[first]["start"], "end": raw[i - 1]["end"],
                         "chord": chosen["chord"] if chosen else "N.C.",
                         "confidence": 0.0, "interpolated": True,
                         "same_chord_bridge": same_bridge})

    filtered = []
    for chord in repaired:
        if filtered and filtered[-1]["chord"] == chord["chord"] and adjacent(filtered[-1], chord):
            previous = filtered[-1]
            length = float(previous["end"]) - float(previous["start"])
            extra = float(chord["end"]) - float(chord["start"])
            previous["confidence"] = round((previous.get("confidence", 0) * length +
                                             chord.get("confidence", 0) * extra) / (length + extra), 4)
            previous["end"] = chord["end"]
            if chord.get("interpolated"):
                previous["interpolated"] = True
            if chord.get("same_chord_bridge"):
                barline = preceding_barline(previous["start"])
                if barline is not None and barline < previous["start"]:
                    previous["start"] = barline
                previous["same_chord_bridge"] = True
        else:
            filtered.append(dict(chord))
    # Solitito can occasionally emit no segment between two valid predictions
    # (especially around onset/context windows). Treat a short missing span as
    # an interpolation gap, while preserving explicit N.C. predictions.
    if filtered:
        beat_length = 60.0 / bpm
        if len(beats) >= 2:
            intervals = np.diff(beats)
            valid = intervals[(intervals > 0.2) & (intervals < 2.5)]
            if len(valid):
                beat_length = float(np.median(valid))
        gap_limit = beat_length + 1e-7
        for left, right in zip(filtered, filtered[1:]):
            gap = float(right["start"]) - float(left["end"])
            if (gap > 1e-7 and gap <= gap_limit and
                    left.get("chord") != "N.C." and right.get("chord") != "N.C."):
                if support(left) >= support(right):
                    left["end"] = right["start"]
                    left["interpolated"] = True
                else:
                    right["start"] = left["end"]
                    right["interpolated"] = True
        # A full-beat N.C. island between two detected chords is usually a
        # missed onset/context window rather than intentional silence. Keep
        # very short explicit silences intact, but fill these larger islands.
        for index in range(1, len(filtered) - 1):
            item = filtered[index]
            length = float(item["end"]) - float(item["start"])
            left, right = filtered[index - 1], filtered[index + 1]
            if (item.get("chord") == "N.C." and length >= beat_length - 1e-7 and
                    left.get("chord") != "N.C." and right.get("chord") != "N.C."):
                chosen = left if support(left) >= support(right) else right
                item["chord"] = chosen["chord"]
                item["confidence"] = 0.0
                item["interpolated"] = True
        # A barline extension can move a bridge start earlier than the prior
        # segment. Re-establish one ordered, non-overlapping timeline before
        # quantization so the view never contains stacked cards.
        normalized = []
        for item in filtered:
            current = dict(item)
            if normalized and float(current["start"]) < float(normalized[-1]["end"]):
                current["start"] = normalized[-1]["end"]
            if float(current["end"]) > float(current["start"]):
                normalized.append(current)
        filtered = normalized
    return {**data, "_raw_chords": [dict(c) for c in raw], "chords": filtered,
            "_duration_filter": "quarter-note", "_gap_fill": "adjacent-support-v1",
            "_same_chord_start": "preceding-downbeat",
            "_filtered_count": kept.count(False),
            "_filter_tempo_source": "beats" if len(beats) >= 2 else f"bpm:{bpm:g}"}


def quantize_chords(data: dict, grid: dict) -> dict:
    """Snap shared boundaries to quarter-note beat positions."""
    beats = np.asarray(sorted({float(t) for t in grid.get("beats", [])
                               if np.isfinite(float(t)) and float(t) >= 0}))
    downbeats = np.asarray(sorted({float(t) for t in grid.get("downbeats", [])
                                   if np.isfinite(float(t)) and float(t) >= 0}))
    try:
        bpm = float(grid.get("bpm", 120))
    except (TypeError, ValueError):
        bpm = 120.0
    if not np.isfinite(bpm) or bpm <= 0:
        bpm = 120.0
    chords = data.get("chords", [])
    duration = float(data.get("duration", max((c["end"] for c in chords), default=0)))

    def snap(t):
        if t <= 0 or t >= duration:
            return max(0.0, min(duration, t))
        if len(beats) < 2:
            origin = float(beats[0]) if len(beats) else 0.0
            step = 60.0 / bpm
            result = origin + np.floor((t - origin) / step + 0.5) * step
        else:
            i = int(np.clip(np.searchsorted(beats, t, side="right") - 1, 0, len(beats) - 2))
            left, right = beats[i], beats[i + 1]
            result = left if t - left < right - t else right
            # Acoustic onset context often labels the final beat of a bar as
            # the following bar's chord. Prefer the next downbeat for that
            # boundary so the new chord starts on the musical barline.
            following = downbeats[downbeats >= result - 1e-6]
            if len(following) and following[0] > result and following[0] - result <= (right - left) * 1.1:
                result = following[0]
        return float(np.clip(result, 0, duration))

    quantized = []
    for original in chords:
        chord = {**original, "start": snap(float(original["start"])), "end": snap(float(original["end"]))}
        if chord["end"] <= chord["start"]:
            continue
        if quantized and quantized[-1]["chord"] == chord["chord"] and abs(quantized[-1]["end"] - chord["start"]) < 1e-8:
            previous = quantized[-1]
            a = previous["end"] - previous["start"]
            b = chord["end"] - chord["start"]
            previous["confidence"] = (previous.get("confidence", 0) * a + chord.get("confidence", 0) * b) / (a + b)
            previous["end"] = chord["end"]
            if chord.get("interpolated"):
                previous["interpolated"] = True
        else:
            quantized.append(chord)
    return {**data, "chords": quantized, "_quantized_to": "quarter-note",
            "_quantize_tempo_source": "beats" if len(beats) >= 2 else f"bpm:{bpm:g}"}


def prepare_chords(data: dict, grid: dict) -> dict:
    # Always restart from raw evidence when the grid or UI is refreshed.
    return quantize_chords(filter_short_chords(data, grid), grid)


def estimate_chords(audio_path: str | Path) -> dict:
    import onnxruntime as ort
    import soxr
    ort.disable_telemetry_events()
    audio_path = Path(audio_path)
    audio, sample_rate = sf.read(audio_path, dtype="float32", always_2d=True)
    duration = len(audio) / sample_rate
    result = {"_model": "Solitito v2 take6 onset", "_model_revision": MODEL_REVISION,
              "_source": audio_path.name, "_version": 1, "duration": duration, "chords": []}
    if not len(audio):
        return result
    mono = audio.mean(axis=1)
    if not np.isfinite(mono).all():
        raise ValueError("音声に不正なサンプル値があります。")
    if sample_rate != SR:
        mono = soxr.resample(mono, sample_rate, SR).astype(np.float32)
    assets = ensure_assets()
    extractor = FeatureExtractor(assets / "dsp_weights.json")
    # Center each FFT frame on its timestamp. Context center, rather than its
    # last frame, determines the output timestamp (no live-inference latency).
    padded = np.pad(mono, (FFT_SIZE // 2, FFT_SIZE // 2))
    windows = np.lib.stride_tricks.sliding_window_view(padded, FFT_SIZE)[::HOP]
    features = np.concatenate([extractor.transform(windows[i:i + 256]) for i in range(0, len(windows), 256)])
    features = np.pad(features, ((CONTEXT // 2, CONTEXT // 2), (0, 0)), mode="edge")
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(assets / MODEL_FILE), sess_options=options, providers=["CPUExecutionProvider"])
    predictions = []
    guide = _midi_guide(audio_path.parent)
    guided_frames = 0
    # Gate only near-silence; separated stems can be much quieter than a mic.
    gate = max(1e-5, float(np.max(np.abs(mono))) * 0.001)
    for frame in range(0, int(np.ceil(len(mono) / HOP)), STRIDE):
        sample = frame * HOP
        local = mono[max(0, sample - STRIDE * HOP // 2):min(len(mono), sample + STRIDE * HOP // 2)]
        if not len(local) or np.sqrt(np.mean(local * local)) < gate:
            predictions.append(("N.C.", 0.0))
            continue
        context = features[frame:frame + CONTEXT][None, :, :]
        root, quality = session.run(["root_logits", "quality_logits"], {"features": context})
        root_logits = root.reshape(-1).astype(np.float32)
        quality_logits = quality.reshape(-1).astype(np.float32)
        frame_start = max(0.0, (sample - STRIDE * HOP // 2) / SR)
        frame_end = min(duration, (sample + STRIDE * HOP // 2) / SR)
        root_bias, quality_bias = _midi_bias(guide, frame_start, frame_end)
        if np.any(root_bias) or np.any(quality_bias):
            # The acoustic model remains the primary vote. MIDI only nudges
            # the N-best ranking, with bass stronger for roots and piano
            # weaker for quality so passing voicings do not dominate.
            root_logits[:12] += 0.34 * root_bias
            quality_logits[:len(QUALITIES)] += 0.16 * quality_bias
            guided_frames += 1
        predictions.append(_prediction(root_logits, quality_logits))
    result["chords"] = _quarter_segments(predictions, duration, guide.get("grid", {}))
    result["_initial_grid"] = "quarter-note"
    result["_midi_guidance"] = "sustained-bass-piano-joint-prior" if guide.get("enabled") else "none"
    result["_midi_guided_frames"] = guided_frames
    return result


def load_or_estimate(folder: str | Path, force: bool = False) -> dict:
    folder = Path(folder)
    cache = folder / CACHE_FILE
    grid_path = folder / "beat-grid.json"
    grid = json.loads(grid_path.read_text(encoding="utf-8")) if grid_path.is_file() else {}
    if cache.is_file() and not force:
        return prepare_chords(json.loads(cache.read_text(encoding="utf-8")), grid)
    data = prepare_chords(estimate_chords(folder / "acoustic-guitar.wav"), grid)
    with tempfile.NamedTemporaryFile(mode="w", dir=folder, suffix=".json.tmp", encoding="utf-8", delete=False) as output:
        temporary = Path(output.name)
        try:
            json.dump(data, output, ensure_ascii=False, indent=2)
            output.flush()
            os.replace(temporary, cache)
        finally:
            temporary.unlink(missing_ok=True)
    return data


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=Path)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    print(json.dumps(load_or_estimate(args.folder, args.force), ensure_ascii=False))
