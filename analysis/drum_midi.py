"""Drum transcription helpers for the synchronized drum view."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf

LANES = ("kick", "snare", "tom", "hihat", "cymbal")
PITCH_TO_LANE = {35: "kick", 36: "kick", 38: "snare", 40: "snare", 47: "tom", 45: "tom", 43: "tom", 41: "tom", 42: "hihat", 44: "hihat", 46: "hihat", 49: "cymbal", 51: "cymbal", 57: "cymbal"}


def _empty_events() -> dict:
    return {lane: [] for lane in LANES}


def _moving_average(values: np.ndarray, width: int) -> np.ndarray:
    width = max(1, int(width))
    if width <= 1:
        return values
    kernel = np.ones(width, dtype=np.float32) / width
    return np.convolve(values, kernel, mode="same")


def _pick_peaks(score: np.ndarray, rate: float, threshold: float, min_gap_seconds: float) -> list[float]:
    if score.size < 3 or float(score.max(initial=0)) <= 0:
        return []
    limit = max(threshold, float(np.quantile(score, 0.88)))
    min_gap = max(1, int(min_gap_seconds * rate))
    candidates = np.flatnonzero((score[1:-1] >= score[:-2]) & (score[1:-1] > score[2:]) & (score[1:-1] >= limit)) + 1
    selected: list[int] = []
    for index in candidates[np.argsort(score[candidates])[::-1]]:
        if all(abs(index - other) >= min_gap for other in selected):
            selected.append(int(index))
    return [round(index / rate, 3) for index in sorted(selected)]


def estimate_drum_events_fallback(path: str | Path) -> dict:
    audio, sample_rate = sf.read(str(path), always_2d=True, dtype="float32")
    mono = audio.mean(axis=1)
    if mono.size == 0:
        return _empty_events()

    frame = 2048
    hop = 512
    if mono.size < frame:
        mono = np.pad(mono, (0, frame - mono.size))
    frames = np.lib.stride_tricks.sliding_window_view(mono, frame)[::hop]
    window = np.hanning(frame).astype(np.float32)
    spectrum = np.abs(np.fft.rfft(frames * window, axis=1)).astype(np.float32)
    freqs = np.fft.rfftfreq(frame, 1 / sample_rate)

    def band(low: float, high: float) -> np.ndarray:
        mask = (freqs >= low) & (freqs < high)
        if not np.any(mask):
            return np.zeros(spectrum.shape[0], dtype=np.float32)
        return spectrum[:, mask].mean(axis=1)

    low = band(35, 150)
    mid = band(150, 2600)
    high = band(5000, 16000)
    flux = np.maximum(0, np.diff(spectrum, axis=0, prepend=spectrum[:1])).mean(axis=1)

    def onset_curve(values: np.ndarray) -> np.ndarray:
        values = np.maximum(0, values - _moving_average(values, 16))
        values += 0.35 * flux
        peak = float(values.max(initial=0))
        return values / peak if peak > 0 else values

    low_o = onset_curve(low)
    mid_o = onset_curve(mid)
    high_o = onset_curve(high)
    frame_rate = sample_rate / hop

    events = _empty_events()
    events["kick"] = _pick_peaks(np.maximum(0, low_o - 0.30 * high_o), frame_rate, 0.34, 0.12)
    events["snare"] = _pick_peaks(np.maximum(0, mid_o - 0.25 * low_o), frame_rate, 0.38, 0.12)
    events["hihat"] = _pick_peaks(np.maximum(0, high_o - 0.15 * low_o), frame_rate, 0.30, 0.06)
    return events


def estimate_drum_events_adtof(path: str | Path) -> dict:
    import torch
    from adtof_pytorch import (
        FRAME_RNN_THRESHOLDS,
        LABELS_5,
        PeakPicker,
        calculate_n_bins,
        create_frame_rnn_model,
        get_default_weights_path,
        load_audio_for_model,
        load_pytorch_weights,
    )

    device = "cpu"
    n_bins = calculate_n_bins()
    model = create_frame_rnn_model(n_bins)
    weights = get_default_weights_path()
    if weights:
        model = load_pytorch_weights(model, weights, strict=False)
    model.eval().to(device)

    x = load_audio_for_model(str(path)).to(device)
    with torch.no_grad():
        pred = model(x).cpu().numpy()

    picker = PeakPicker(thresholds=FRAME_RNN_THRESHOLDS, fps=100)
    peaks = picker.pick(pred, labels=LABELS_5, label_offset=0)[0]
    events = _empty_events()
    for pitch, times in peaks.items():
        lane = PITCH_TO_LANE.get(int(pitch))
        if lane:
            events[lane].extend(round(float(time), 3) for time in times)
    for lane in events:
        events[lane] = sorted(set(events[lane]))
    return events


def estimate_drum_events(path: str | Path) -> dict:
    try:
        events = estimate_drum_events_adtof(path)
        events["_model"] = "ADTOF-pytorch"
        return events
    except Exception as error:
        events = estimate_drum_events_fallback(path)
        events["_model"] = "fallback-onset"
        events["_warning"] = str(error)
        return events


def load_or_estimate(folder: str | Path, force: bool = False) -> dict:
    folder = Path(folder)
    cache = folder / "drums-midi.json"
    if cache.is_file() and not force:
        data = json.loads(cache.read_text(encoding="utf-8"))
        for lane in LANES:
            data.setdefault(lane, [])
        return data
    drums = folder / "drums.wav"
    if not drums.is_file():
        return _empty_events()
    events = estimate_drum_events(drums)
    cache.write_text(json.dumps(events, ensure_ascii=False, indent=2), encoding="utf-8")
    return events
