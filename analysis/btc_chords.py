"""CPU, bounded-memory BTC large-vocabulary inference and MIDI-assisted decoding."""
from __future__ import annotations

from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import tempfile
import urllib.request

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
REVISION = "2682317be668032e6e4b269ded36adaa2ad57df0"
SHA256 = "1673d23f8f9a55ae7f9e8b80a51da616debb22675b8d8b67ea6ce0ef37b0ab51"
MODEL_FILE = "btc_model_large_voca.pt"
MODEL = "BTC large-vocabulary + MIDI evidence v1"
PITCHES = ("C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B")
# EXACT official checkpoint order; X and N are separate output classes.
QUALITIES = ("m", "", "dim", "aug", "m6", "6", "m7", "mMaj7", "maj7", "7", "dim7", "m7b5", "sus2", "sus4")
INTERVALS = ((0,3,7), (0,4,7), (0,3,6), (0,4,8), (0,3,7,9),
             (0,4,7,9), (0,3,7,10), (0,3,7,11), (0,4,7,11),
             (0,4,7,10), (0,3,6,9), (0,3,6,10), (0,2,7), (0,5,7))
LABELS = tuple(root + quality for root in PITCHES for quality in QUALITIES) + ("X", "N.C.")
PCS = [tuple((root + tone) % 12 for tone in intervals)
       for root in range(12) for intervals in INTERVALS]


def audio_paths(folder):
    """Reconstruct the original mix once; never double-count acoustic subdivisions."""
    folder = Path(folder)
    for name in ("mix.wav", "original.wav"):
        if (folder / name).is_file():
            return [folder / name]
    paths = [folder / (name + ".wav") for name in
             ("vocals", "drums", "bass", "guitar", "piano", "other")
             if (folder / (name + ".wav")).is_file()]
    if not (folder / "guitar.wav").is_file():
        paths += [folder / (name + ".wav") for name in ("acoustic-guitar", "guitar-other")
                  if (folder / (name + ".wav")).is_file()]
    return paths


def ensure_model():
    directory = ROOT / ".cache" / "btc"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / MODEL_FILE
    if path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == SHA256:
        return path
    url = f"https://raw.githubusercontent.com/jayg996/BTC-ISMIR19/{REVISION}/test/{MODEL_FILE}"
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=directory, suffix=".part", delete=False) as out:
            temporary = Path(out.name)
            with urllib.request.urlopen(url, timeout=60) as response:
                while block := response.read(1024 * 1024):
                    out.write(block)
        if hashlib.sha256(temporary.read_bytes()).hexdigest() != SHA256:
            raise ValueError("BTCモデルのSHA-256検証に失敗しました")
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return path


def load_model():
    import torch
    from vendor.btc.model import BTC_model
    config = dict(feature_size=144, timestep=108, num_chords=170,
                  hidden_size=128, num_layers=8, num_heads=4, total_key_depth=128,
                  total_value_depth=128, filter_size=128, input_dropout=0.,
                  layer_dropout=0., attention_dropout=0., relu_dropout=0., probs_out=True)
    # The official checkpoint contains numpy normalization scalars.
    with torch.serialization.safe_globals([np.core.multiarray.scalar, np.dtype, type(np.dtype("float64"))]):
        checkpoint = torch.load(ensure_model(), map_location="cpu", weights_only=True)
    model = BTC_model(config).eval()
    model.load_state_dict(checkpoint["model"], strict=True)
    return model, float(checkpoint["mean"]), float(checkpoint["std"])


def infer_audio(paths):
    """Read only ten seconds of audio at a time, including when summing stems."""
    import librosa
    import torch
    model, mean, std = load_model()
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(min(4, previous_threads))
    probabilities, bounds = [], []
    try:
        with ExitStack() as stack, torch.inference_mode():
            streams = [stack.enter_context(sf.SoundFile(path)) for path in paths]
            duration = max(len(stream) / stream.samplerate for stream in streams)
            for start in np.arange(0., duration, 10.):
                length = min(10., duration - start)
                count = max(1, int(round(length * 22050)))
                mix = np.zeros(count, dtype=np.float32)
                for stream in streams:
                    position = int(round(start * stream.samplerate))
                    if position >= len(stream):
                        continue
                    stream.seek(position)
                    audio = stream.read(int(round(length * stream.samplerate)), dtype="float32", always_2d=True).mean(axis=1)
                    if stream.samplerate != 22050:
                        audio = librosa.resample(audio, orig_sr=stream.samplerate, target_sr=22050)
                    mix[:min(count, len(audio))] += audio[:count]
                # Same 10-second CQT/log normalization as official test.py.
                feature = librosa.cqt(np.pad(mix, (0, max(0, 44100 - count))), sr=22050,
                                      n_bins=144, bins_per_octave=24, hop_length=2048)
                feature = ((np.log(np.abs(feature) + 1e-6).T - mean) / std).astype(np.float32)
                valid = min(len(feature), int(np.ceil(length * 22050 / 2048)))
                padded = np.zeros((108, 144), dtype=np.float32)
                padded[:min(108, len(feature))] = feature[:108]
                hidden, _ = model.self_attn_layers(torch.from_numpy(padded).unsqueeze(0))
                probs = torch.softmax(model.output_layer(hidden), dim=-1)[0, :valid].numpy().copy()
                if np.max(np.abs(mix)) < 1e-5:
                    probs[:] = 0.
                    probs[:, 169] = 1.
                probabilities.append(probs)
                # Use actual CQT hop within each block, with no accumulated drift.
                for i in range(valid):
                    bounds.append((float(start + i * 2048 / 22050),
                                   float(min(start + (i + 1) * 2048 / 22050, start + length))))
    finally:
        torch.set_num_threads(previous_threads)
    if not probabilities:
        return np.empty((0, 170), dtype=np.float32), [], 0.
    return np.concatenate(probabilities), bounds, duration


def decode(probabilities, bounds, piano_notes=(), bass_notes=(), acoustic=None):
    """O(frames * 170) Viterbi. Audio likelihood dominates weak MIDI priors."""
    from analysis.chord_estimator import _frame_weights, _acoustic_at
    if not len(bounds):
        return []
    piano = _frame_weights(list(piano_notes), bounds)
    bass = _frame_weights(list(bass_notes), bounds)
    emissions = np.log(np.maximum(probabilities, 1e-7))
    for t, ((start, end), p, b) in enumerate(zip(bounds, piano, bass)):
        bass_pc = int(np.argmax(b)) if sum(b) and max(b) / sum(b) >= .7 else None
        for k, pcs in enumerate(PCS):
            if sum(p):
                emissions[t, k] += .25 * (2 * sum(p[pc] for pc in pcs) / sum(p) - 1)
            if bass_pc is not None:
                emissions[t, k] += .2 if bass_pc == k // 14 else (0. if bass_pc in pcs else -.2)
        name, confidence = _acoustic_at(acoustic or {}, start, end)
        if name in LABELS[:168]:
            emissions[t, LABELS.index(name)] += .1 * confidence
    previous = emissions[0].copy()
    back = np.empty((len(bounds), 170), dtype=np.int16)
    indices = np.arange(170)
    for t in range(1, len(bounds)):
        winner = int(previous.argmax())
        change = previous[winner] - 1.2
        stay = previous >= change
        back[t] = np.where(stay, indices, winner)
        previous = emissions[t] + np.maximum(previous, change)
    path = np.empty(len(bounds), dtype=np.int16)
    path[-1] = int(previous.argmax())
    for t in range(len(bounds) - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    segments = []
    for t, ((start, end), k) in enumerate(zip(bounds, path)):
        name = LABELS[k]
        # X stays explicitly unknown; do not invent a major triad or mark silence.
        confidence = float(probabilities[t, k])
        if segments and segments[-1]["chord"] == name:
            item = segments[-1]
            total = end - item["start"]
            item["confidence"] = (item["confidence"] * (start - item["start"]) + confidence * (end - start)) / total
            item["end"] = end
        else:
            segments.append(dict(start=start, end=end, chord=name, confidence=confidence))
    # Aggregate bass over a complete decoded chord, so individual bass attacks
    # cannot split one chord into many fleeting inversions.
    segment_bass = _frame_weights(list(bass_notes), [(s["start"], s["end"]) for s in segments])
    for item, weights in zip(segments, segment_bass):
        k = LABELS.index(item["chord"])
        if k < 168 and sum(weights) and max(weights) / sum(weights) >= .7:
            bass_pc = int(np.argmax(weights))
            if bass_pc != k // 14 and bass_pc in PCS[k]:
                item["chord"] += "/" + PITCHES[bass_pc]
        for key in ("start", "end", "confidence"):
            item[key] = round(item[key], 4)
    return segments


def estimate(folder, acoustic=None, acoustic_only=False):
    from analysis.chord_estimator import _read_json, _sustained_notes
    folder = Path(folder)
    paths = ([folder / "acoustic-guitar.wav"] if acoustic_only else audio_paths(folder))
    if not paths:
        raise FileNotFoundError("BTC解析に必要な音源.wavがありません")
    probabilities, bounds, duration = infer_audio(paths)
    grid = _read_json(folder / "beat-grid.json")
    piano = _sustained_notes(_read_json(folder / "piano-midi.json").get("notes", []), grid, "piano")
    bass = _sustained_notes(_read_json(folder / "bass-midi.json").get("notes", []), grid, "bass")
    acoustic = acoustic if acoustic is not None else _read_json(folder / "acoustic-guitar-chords.json")
    if acoustic_only or acoustic.get("_model") == MODEL:
        acoustic = {}  # Never feed a previous BTC result back into itself.
    return {"_model": MODEL, "_model_revision": REVISION, "_device": "cpu",
            "_audio_sources": [path.name for path in paths],
            "_pipeline": "BTC audio likelihood -> weak MIDI/Solitito evidence -> Viterbi -> bass inversion",
            "_timing": "audio frames (~93 ms); no quarter-note restriction",
            "_vocabulary": "168 chords + X (unknown) + N.C.; MIDI bass inversions",
            "duration": duration, "chords": decode(probabilities, bounds, piano, bass, acoustic)}


def load_acoustic(folder, force=False):
    folder = Path(folder)
    cache = folder / "acoustic-guitar-chords.json"
    if cache.is_file() and not force:
        return json.loads(cache.read_text(encoding="utf-8"))
    data = estimate(folder, acoustic_only=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=folder, suffix=".tmp", encoding="utf-8", delete=False) as out:
        temporary = Path(out.name)
        try:
            json.dump(data, out, ensure_ascii=False, indent=2)
            out.flush()
            temporary.replace(cache)
        finally:
            temporary.unlink(missing_ok=True)
    return data


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=Path)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    print(json.dumps(load_acoustic(args.folder, args.force), ensure_ascii=False))
