"""Beat-level MIDI chord estimation with normalized evidence and sequence decoding."""
from __future__ import annotations

import json
import math
import numpy as np
from pathlib import Path

PITCH_NAMES = ("C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B")
MODEL = "Downbeat-grid MIDI + two-pass Solitito reranker v8"
# Tuneable inference parameters. Keep algorithmic weights here so experiments
# do not require searching through the decoding code.
PARAMS = {
    # Drum onset search radius, measured as a fraction of one quarter note.
    "drum_radius_quarter": 0.15,
    # Lane reliability for moving the fixed beat lattice. Kick/cymbal are
    # strongest; hi-hat is deliberately weak because it often leads/lag beats.
    "drum_kick_strength": 1.0,
    "drum_cymbal_strength": 1.0,
    "drum_snare_strength": 0.65,
    "drum_tom_strength": 0.65,
    "drum_hihat_strength": 0.30,
    # Harmony evidence: piano describes chord quality, bass describes root.
    "piano_weight": 0.55,
    "bass_weight": 0.45,
    # A bass note that equals the candidate root disambiguates equal-pitch-set
    # labels (for example Dm7 versus F6). A mere chord-tone bass can still
    # represent an inversion, but must not tie the actual root.
    "bass_root_score": 1.0,
    "bass_chord_tone_score": 0.35,
    "bass_non_chord_score": 0.0,
    # Evidence used to detect a possible chord boundary. Drum is grid-only.
    "boundary_bass": 0.50,
    "boundary_piano": 0.35,
    "boundary_meter": 0.15,
    # Sequence transition scores. Increasing change_transition suppresses
    # unnecessary changes; boundary_transition lets harmonic evidence win.
    "same_chord_transition": 0.12,
    "change_transition": -0.30,
    "boundary_transition": 0.35,
    # Adaptive half-note prior: changes on beats 2/4 need stronger evidence.
    "offbeat_change_penalty": -0.50,
    # One-bar diversity prior, applied only when a new candidate is introduced.
    "bar_third_chord_penalty": -0.15,
    "bar_fourth_chord_penalty": -0.35,
    # Solitito is only allowed to rerank ambiguous MIDI cells.
    "acoustic_margin": 0.10,
    "acoustic_bonus": 0.15,
}
TEMPLATES = [
    ("", {0, 4, 7}), ("m", {0, 3, 7}),
    ("maj7", {0, 4, 7, 11}), ("m7", {0, 3, 7, 10}),
    ("7", {0, 4, 7, 10}), ("m7b5", {0, 3, 6, 10}),
    ("dim", {0, 3, 6}), ("sus4", {0, 5, 7}), ("sus2", {0, 2, 7}),
    ("aug", {0, 4, 8}), ("6", {0, 4, 7, 9}), ("m6", {0, 3, 7, 9}),
    ("add9", {0, 4, 7, 2}), ("madd9", {0, 3, 7, 2}),
    ("7sus4", {0, 5, 7, 10}),
    # Extensions remain candidates only when their extra tones are actually
    # sustained in piano MIDI; the coverage/missing-tone score rejects them
    # for ordinary triads and passing notes.
    ("9", {0, 2, 4, 7, 10}), ("maj9", {0, 2, 4, 7, 11}),
    ("m9", {0, 2, 3, 7, 10}), ("11", {0, 2, 4, 5, 7, 10}),
    ("m11", {0, 2, 3, 5, 7, 10}), ("13", {0, 2, 4, 7, 9, 10}),
    ("m13", {0, 2, 3, 7, 9, 10}), ("6/9", {0, 2, 4, 7, 9}),
]
CANDIDATES = [(root, quality, {(root + i) % 12 for i in intervals})
              for root in range(12) for quality, intervals in TEMPLATES]
EXTENDED_QUALITIES = {"9", "maj9", "m9", "11", "m11", "13", "m13", "6/9"}


def _extended_quality_is_explicit(root: int, quality: str, piano_chroma: list[float], bass_pc: int | None) -> bool:
    """Allow MIDI-only extensions only with sustained defining piano tones.

    Solitito has no 9/11/13 output labels, so these must never win merely from
    a partial voicing or a passing piano note.  A matching bass root plus the
    quality-defining tones is deliberately stricter than the basic vocabulary.
    """
    if quality not in EXTENDED_QUALITIES or bass_pc != root:
        return quality not in EXTENDED_QUALITIES
    present = lambda interval: piano_chroma[(root + interval) % 12] >= 0.10
    required = {
        "9": (2, 10), "maj9": (2, 11), "m9": (2, 3, 10),
        "11": (2, 5, 10), "m11": (2, 3, 5, 10),
        "13": (9, 10), "m13": (3, 9, 10), "6/9": (2, 9),
    }
    return all(present(interval) for interval in required[quality])


def _read_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _note_weight(note: dict, start: float, end: float) -> float:
    note_start = float(note.get("start", 0))
    note_end = float(note.get("end", note_start))
    overlap = max(0.0, min(end, note_end) - max(start, note_start))
    velocity = max(0.0, min(127.0, float(note.get("velocity", 80)))) / 127
    return overlap * velocity


def _segment_bounds(grid: dict, duration: float) -> list[tuple[float, float]]:
    if not math.isfinite(duration) or duration <= 0:
        return []
    def valid(values):
        return sorted({float(x) for x in values if math.isfinite(float(x)) and 0 < float(x) < duration})
    beats = valid(grid.get("beats", []))
    downbeats = valid(grid.get("downbeats", []))
    if beats:
        points = beats + downbeats
    elif downbeats:
        points = downbeats + [float((a + b) / 2) for a, b in zip(downbeats, downbeats[1:])]
    else:
        bpm = grid.get("bpm")
        step = 60 / float(bpm) if bpm and math.isfinite(float(bpm)) and 30 <= float(bpm) <= 300 else 0.5
        points = [i * step for i in range(1, math.ceil(duration / step))]
    # Coalesce tracker jitter without dropping short intro/outro intervals.
    points = sorted(set([0.0, duration] + points))
    cleaned = [0.0]
    for point in points[1:-1]:
        if point - cleaned[-1] >= 0.05 and duration - point >= 0.05:
            cleaned.append(point)
    cleaned.append(duration)
    return list(zip(cleaned, cleaned[1:]))


def _scores(weights: list[float], bass_pc: int | None,
            acoustic_name: str | None = None, anchor_strength: float = 0.0) -> list[float]:
    total = sum(weights)
    if total <= 0:
        return []
    chroma = [w / total for w in weights]
    # Soft coverage penalizes invented chord tones, including unsupported sevenths.
    support = [min(1.0, value / 0.12) for value in chroma]
    scores = []
    for root, quality, pcs in CANDIDATES:
        hit = sum(chroma[pc] for pc in pcs)
        missing = sum(1 - support[pc] for pc in pcs) / len(pcs)
        score = 1.6 * hit - 0.8 * (1 - hit) - 0.65 * missing
        score += 0.06 * support[root] - 0.025 * (len(pcs) - 3)
        if bass_pc is not None:
            score += 0.34 if bass_pc == root else (0.0 if bass_pc in pcs else -0.08)
        if acoustic_name and acoustic_name != "N.C.":
            candidate = PITCH_NAMES[root] + quality
            anchor_root = acoustic_name.split("/", 1)[0]
            if candidate == acoustic_name:
                score += 0.55 * anchor_strength
            elif anchor_root == PITCH_NAMES[root]:
                score += 0.16 * anchor_strength
            else:
                score -= 0.08 * anchor_strength
        scores.append(score)
    return scores


def _label(index: int, weights: list[float], bass_pc: int | None,
           scores: list[float]) -> tuple[str, float]:
    root, quality, pcs = CANDIDATES[index]
    total = sum(weights)
    hit = sum(weights[pc] for pc in pcs) / total
    runner_up = max(score for i, score in enumerate(scores) if i != index)
    margin = max(0.0, scores[index] - runner_up)
    coverage = sum(min(1.0, weights[pc] / total / 0.12) for pc in pcs) / len(pcs)
    # A heuristic evidence score, not a calibrated probability.
    confidence = hit * coverage * (0.35 + 0.65 * min(1.0, margin / 0.25))
    name = PITCH_NAMES[root] + quality
    if bass_pc is not None and bass_pc != root and bass_pc in pcs:
        name += "/" + PITCH_NAMES[bass_pc]
    return name, round(confidence, 3)


def _estimate_one(pc_weights: list[float], bass_pc: int | None) -> tuple[str, float]:
    if sum(w > sum(pc_weights) * 0.06 for w in pc_weights) < 2:
        return "N.C.", 0.0
    scores = _scores(pc_weights, bass_pc)
    index = max(range(len(scores)), key=scores.__getitem__)
    return _label(index, pc_weights, bass_pc, scores)


def _decode(frames: list[dict]) -> list[int | None]:
    """Viterbi with a fixed change cost; silence breaks harmonic continuity.

    Uniform change costs permit O(frames * candidates) decoding.
    """
    result = [None] * len(frames)
    position = 0
    while position < len(frames):
        if not frames[position]["scores"]:
            position += 1
            continue
        start = position
        previous = frames[position]["scores"][:]
        back = []
        position += 1
        while position < len(frames) and frames[position]["scores"]:
            winner = max(range(len(previous)), key=previous.__getitem__)
            changed = previous[winner] - 0.18
            parents = [i if previous[i] >= changed else winner for i in range(len(previous))]
            previous = [score + max(previous[i], changed)
                        for i, score in enumerate(frames[position]["scores"])]
            back.append(parents)
            position += 1
        winner = max(range(len(previous)), key=previous.__getitem__)
        result[position - 1] = winner
        for offset in range(len(back) - 1, -1, -1):
            winner = back[offset][winner]
            result[start + offset] = winner
    return result


def _frame_weights(notes: list[dict], bounds: list[tuple[float, float]]) -> list[list[float]]:
    # Sweep active notes instead of scanning the full MIDI for every beat.
    notes = sorted(notes, key=lambda n: float(n.get("start", 0)))
    cursor = 0
    active = []
    frames = []
    for start, end in bounds:
        active = [n for n in active if float(n.get("end", n.get("start", 0))) > start]
        while cursor < len(notes) and float(notes[cursor].get("start", 0)) < end:
            active.append(notes[cursor])
            cursor += 1
        weights = [0.0] * 12
        for note in active:
            weights[int(note.get("pitch", 0)) % 12] += _note_weight(note, start, end)
        frames.append(weights)
    return frames


def _quarter_note_duration(grid: dict) -> float:
    beats = sorted(float(t) for t in grid.get("beats", []) if math.isfinite(float(t)))
    intervals = np.diff(beats) if len(beats) >= 2 else []
    valid = [float(value) for value in intervals if 0.2 < float(value) < 2.5]
    if valid:
        return float(np.median(valid))
    bpm = grid.get("bpm")
    if bpm and 30 <= float(bpm) <= 300:
        return 60.0 / float(bpm)
    return 0.5


def _sustained_notes(notes: list[dict], grid: dict, kind: str = "piano") -> list[dict]:
    """Suppress ornaments without discarding articulated chords."""
    eighth = _quarter_note_duration(grid) / 2
    if kind == "bass":
        # Bass timing is an anchor: never manufacture sustain by joining
        # repeated pitches across an intervening note or rest.
        return [dict(note) for note in notes
                if float(note.get("end", 0)) - float(note.get("start", 0)) >= eighth - 1e-4]
    merged = []
    for note in sorted((dict(note) for note in notes),
                       key=lambda item: (int(item.get("pitch", 0)), float(item.get("start", 0)))):
        if (merged and int(merged[-1].get("pitch", 0)) == int(note.get("pitch", 0)) and
                float(note.get("start", 0)) - float(merged[-1].get("end", 0)) <= eighth * 0.35):
            merged[-1]["end"] = max(float(merged[-1].get("end", 0)), float(note.get("end", 0)))
            merged[-1]["velocity"] = max(float(merged[-1].get("velocity", 0)), float(note.get("velocity", 0)))
        else:
            merged.append(note)
    # Short piano notes are useful when they form a simultaneous voicing. A
    # lone short note is much more likely to be an ornament or passing tone.
    chord_onsets = []
    for note in merged:
        onset = float(note.get("start", 0))
        chord_onsets.append(sum(abs(float(other.get("start", 0)) - onset) <= 0.06
                                for other in merged) >= 2)
    return [note for note, simultaneous in zip(merged, chord_onsets)
            if simultaneous or float(note.get("end", 0)) - float(note.get("start", 0)) >= eighth - 1e-4]


def _metrical_bonus(time: float, grid: dict) -> float:
    """Prefer chord changes on beats 1 and 3, without banning beats 2/4."""
    quarter = _quarter_note_duration(grid)
    beats = sorted(float(t) for t in grid.get("beats", []) if math.isfinite(float(t)))
    downbeats = sorted(float(t) for t in grid.get("downbeats", []) if math.isfinite(float(t)))
    if downbeats and min(abs(time - point) for point in downbeats) <= quarter * 0.12:
        return 1.0
    if beats:
        nearest = min(range(len(beats)), key=lambda index: abs(beats[index] - time))
        if abs(beats[nearest] - time) <= quarter * 0.12:
            anchor = max((i for i, beat in enumerate(beats)
                          if any(abs(beat - downbeat) <= quarter * 0.12 for downbeat in downbeats)),
                         default=0)
            phase = (nearest - anchor) % 4
            return 0.72 if phase == 2 else 0.22
    return 0.08


def _acoustic_at(acoustic: dict | None, start: float, end: float) -> tuple[str | None, float]:
    if not acoustic:
        return None, 0.0
    midpoint = (start + end) / 2
    for item in acoustic.get("chords", []):
        if float(item.get("start", 0)) <= midpoint < float(item.get("end", 0)):
            return str(item.get("chord", "N.C.")), float(item.get("confidence", 0.0))
    return None, 0.0


def _root_of(name: str | None) -> str | None:
    if not name or name == "N.C.":
        return None
    root = name[:2] if len(name) > 1 and name[1] in "#b" else name[:1]
    return root if root in PITCH_NAMES else {"Db": "C#", "D#": "Eb", "Gb": "F#", "G#": "Ab", "A#": "Bb"}.get(root, root)


def _midi_candidate_for_acoustic(name: str | None) -> int | None:
    """Find the template index for an acoustic label, including slash basses."""
    if not name or name == "N.C.":
        return None
    plain = name.split("/", 1)[0]
    root = _root_of(plain)
    if root is None:
        return None
    quality = plain[len(plain[:2]) if len(plain) > 1 and plain[1] in "#b" else 1:]
    for index, (candidate_root, candidate_quality, _) in enumerate(CANDIDATES):
        if PITCH_NAMES[candidate_root] == root and candidate_quality == quality:
            return index
    return None


def _fixed_drum_grid(grid: dict, drums: dict, duration: float) -> list[tuple[float, float]]:
    """Build Beat This! quarter cells, then anchor each beat once to drums."""
    quarter = _quarter_note_duration(grid)
    detected = sorted({float(t) for t in grid.get("beats", [])
                       if math.isfinite(float(t)) and 0 < float(t) < duration})
    downbeats = sorted({float(t) for t in grid.get("downbeats", [])
                        if math.isfinite(float(t)) and 0 <= float(t) < duration})
    beats = []
    if downbeats:
        # Downbeats define absolute bar phase. Fill beats 2/3/4 from each bar,
        # using Beat This! detections when present and BPM spacing otherwise.
        beats.extend(t for t in detected if t < downbeats[0] - quarter * 0.25)
        for bar_index, downbeat in enumerate(downbeats):
            next_downbeat = (downbeats[bar_index + 1] if bar_index + 1 < len(downbeats)
                             else min(duration, downbeat + quarter * 4))
            local_quarter = ((next_downbeat - downbeat) / 4
                             if next_downbeat - downbeat >= quarter * 2 else quarter)
            for phase in range(4):
                expected = downbeat + phase * local_quarter
                if not 0 < expected < duration:
                    continue
                if phase == 0:
                    point = downbeat
                else:
                    nearby = [beat for beat in detected
                              if abs(beat - expected) <= local_quarter * 0.35]
                    point = min(nearby, key=lambda beat: abs(beat - expected)) if nearby else expected
                beats.append(point)
    else:
        beats = detected
    if not beats:
        bpm = grid.get("bpm")
        step = 60.0 / float(bpm) if bpm and 30 <= float(bpm) <= 300 else quarter
        beats = [index * step for index in range(1, math.ceil(duration / step))]
    beats = sorted(set(beats))
    lane_strength = {"kick": PARAMS["drum_kick_strength"], "cymbal": PARAMS["drum_cymbal_strength"],
                     "snare": PARAMS["drum_snare_strength"], "tom": PARAMS["drum_tom_strength"],
                     "hihat": PARAMS["drum_hihat_strength"]}
    anchored = []
    radius = quarter * PARAMS["drum_radius_quarter"]
    for beat in beats:
        choices = []
        for lane, times in drums.items():
            if not isinstance(times, list):
                continue
            strength = lane_strength.get(lane, 0.2)
            for onset in times:
                if isinstance(onset, (int, float)) and abs(float(onset) - beat) <= radius:
                    choices.append((strength, -abs(float(onset) - beat), float(onset)))
        is_downbeat = any(abs(beat - downbeat) <= quarter * 0.12 for downbeat in downbeats)
        if choices and not is_downbeat:
            # Prefer an onset that is both a strong lane and close to the
            # expected beat; a distant kick must not beat a nearby snare.
            strength, _, onset = max(choices, key=lambda item: (
                item[0] * max(0.0, 1.0 - abs(item[2] - beat) / radius),
                -abs(item[2] - beat)))
            proximity = max(0.0, 1.0 - abs(onset - beat) / radius)
            beat = beat + (strength * proximity) * (onset - beat)
        if not anchored or beat - anchored[-1] >= quarter * 0.5:
            anchored.append(round(beat, 4))
    points = [0.0] + anchored + [duration]
    return [(left, right) for left, right in zip(points, points[1:]) if right > left]


def _grid_first_estimate(acoustic: dict, grid: dict, piano_notes: list[dict],
                         bass_notes: list[dict], drums: dict, duration: float) -> dict:
    """Fix quarter-note timing first, then rerank MIDI N-best with Solitito."""
    bounds = _fixed_drum_grid(grid, drums, duration)
    piano_frames = _frame_weights(piano_notes, bounds)
    bass_frames = _frame_weights(bass_notes, bounds)
    drum_times = {lane: [float(time) for time in times if isinstance(time, (int, float))]
                  for lane, times in drums.items() if isinstance(times, list)}
    lane_weights = {"kick": 1.0, "cymbal": 0.75, "snare": 0.5,
                    "tom": 0.4, "hihat": 0.25}

    frames = []
    previous_bass_pc = None
    first_downbeat = min((float(value) for value in grid.get("downbeats", [])
                          if isinstance(value, (int, float))), default=None)
    for frame_index, ((start, end), piano_weights, bass_weights) in enumerate(
            zip(bounds, piano_frames, bass_frames)):
        piano_total, bass_total = sum(piano_weights), sum(bass_weights)
        bass_pc = max(range(12), key=bass_weights.__getitem__) if bass_total else None
        if bass_pc is not None and bass_weights[bass_pc] / bass_total < 0.55:
            bass_pc = None
        acoustic_name, acoustic_confidence = _acoustic_at(acoustic, start, end)
        acoustic_name = acoustic_name or "N.C."
        piano_chroma = ([weight / piano_total for weight in piano_weights]
                        if piano_total else [0.0] * 12)
        raw_scores = []
        if piano_total or bass_total:
            for root, quality, pcs in CANDIDATES:
                if not _extended_quality_is_explicit(root, quality, piano_chroma, bass_pc):
                    raw_scores.append(float("-inf"))
                    continue
                if piano_total:
                    hit = sum(piano_chroma[pc] for pc in pcs)
                    color_pcs = pcs - {root, (root + 7) % 12}
                    color = (sum(min(1.0, piano_chroma[pc] / 0.12) for pc in color_pcs) /
                             max(1, len(color_pcs)))
                    extras = sum(piano_chroma[pc] for pc in range(12) if pc not in pcs)
                    piano_score = max(0.0, min(1.0, 0.58 * hit + 0.42 * color - 0.25 * extras))
                else:
                    piano_score = 0.5
                if bass_pc is None:
                    bass_score = 0.5
                elif bass_pc == root:
                    bass_score = PARAMS["bass_root_score"]
                elif bass_pc in pcs:
                    bass_score = PARAMS["bass_chord_tone_score"]
                else:
                    bass_score = PARAMS["bass_non_chord_score"]
                raw_scores.append(PARAMS["piano_weight"] * piano_score + PARAMS["bass_weight"] * bass_score)
        emissions = {}
        # Keep the pickup/silence before the first downbeat out of harmonic
        # inference. Otherwise the first real chord is pulled back to t=0.
        pre_downbeat = first_downbeat is not None and end <= first_downbeat + 1e-4
        if pre_downbeat:
            emissions[None] = 0.0
            ranked = []
        elif raw_scores:
            ranked = [candidate for candidate in sorted(range(len(raw_scores)), key=raw_scores.__getitem__, reverse=True)
                      if math.isfinite(raw_scores[candidate])][:8]
            for candidate in ranked:
                emissions[candidate] = raw_scores[candidate]
        else:
            ranked = []
            emissions[None] = 0.0

        boundary = 0.0
        if frame_index > 0:
            previous_piano = piano_frames[frame_index - 1]
            previous_total = sum(previous_piano)
            piano_change = (sum(abs(a / previous_total - b / piano_total)
                                for a, b in zip(previous_piano, piano_weights)) / 2
                            if previous_total and piano_total else 0.0)
            bass_change = 1.0 if (bass_pc is not None and previous_bass_pc is not None and
                                  bass_pc != previous_bass_pc) else 0.0
            boundary = (PARAMS["boundary_bass"] * bass_change + PARAMS["boundary_piano"] * piano_change +
                        PARAMS["boundary_meter"] * _metrical_bonus(start, grid))
        frames.append({"emissions": emissions, "boundary": boundary,
                       "bass": bass_pc, "acoustic": acoustic_name,
                       "acoustic_confidence": acoustic_confidence,
                       "midi_ranked": ranked, "midi_scores": raw_scores,
                       "pre_downbeat": pre_downbeat})
        previous_bass_pc = bass_pc

    def viterbi(emission_key):
        paths = []
        previous = {}
        downbeats = sorted(float(value) for value in grid.get("downbeats", [])
                           if isinstance(value, (int, float)))
        bar_ids = [sum(1 for downbeat in downbeats if downbeat <= start)
                   for start, _ in bounds]
        beat_times = sorted(float(value) for value in grid.get("beats", [])
                            if isinstance(value, (int, float)))
        beat_phases = []
        for start, _ in bounds:
            prior_downbeat = max((value for value in downbeats if value <= start), default=None)
            if prior_downbeat is None:
                beat_phases.append(0)
            else:
                beat_phases.append(sum(1 for value in beat_times
                                       if prior_downbeat < value < start + 1e-5) % 4)
        for frame_index, frame in enumerate(frames):
            current, parents = {}, {}
            for candidate, emission in frame[emission_key].items():
                root = CANDIDATES[candidate][0] if candidate is not None else None
                if frame_index == 0:
                    used = frozenset(() if candidate is None else (candidate,))
                    state = (candidate, used)
                    current[state], parents[state] = emission, None
                    continue
                choices = []
                for prior_state, score in previous.items():
                    prior, used_roots = prior_state
                    families = used_roots if bar_ids[frame_index] == bar_ids[frame_index - 1] else frozenset()
                    # Count full chord candidates, not just roots: C, Cmaj7,
                    # and C/E must consume separate slots in the bar budget.
                    identity = candidate if candidate is not None else None
                    is_new = identity is not None and identity not in families
                    new_roots = families if not is_new else families | {identity}
                    count = len(new_roots)
                    variety_penalty = (0.0 if not is_new or count <= 2
                                       else (PARAMS["bar_third_chord_penalty"] if count == 3 else PARAMS["bar_fourth_chord_penalty"]))
                    if prior == candidate:
                        transition = PARAMS["same_chord_transition"]
                    else:
                        # Prefer changes on beats 1/3. Changes on beats 2/4
                        # remain possible when MIDI evidence is strong.
                        offbeat_penalty = PARAMS["offbeat_change_penalty"] if beat_phases[frame_index] in (1, 3) else 0.0
                        transition = (PARAMS["change_transition"] +
                                      PARAMS["boundary_transition"] * frame["boundary"] + offbeat_penalty)
                    new_score = emission + score + transition + variety_penalty
                    state = (candidate, new_roots)
                    if state not in current or new_score > current[state]:
                        current[state] = new_score
                        parents[state] = prior_state
            previous = current
            paths.append(parents)
        decoded = [None] * len(frames)
        if previous:
            state = max(previous, key=previous.get)
            for index in range(len(frames) - 1, -1, -1):
                decoded[index] = state[0]
                state = paths[index].get(state)
        return decoded

    midi_decoded = viterbi("emissions")

    # Add confidence-scaled Solitito bonuses only to ambiguous MIDI frames.
    # Neighbouring acoustic cells are soft evidence for a possible one-beat
    # offset; the fixed timing grid itself never moves.
    for index, frame in enumerate(frames):
        rerank = dict(frame["emissions"])
        if frame.get("pre_downbeat"):
            frame["rerank_emissions"] = rerank
            continue
        ranked = frame["midi_ranked"]
        scores = frame["midi_scores"]
        margin = (scores[ranked[0]] - scores[ranked[1]]
                  if len(ranked) > 1 else (0.0 if not ranked else 1.0))
        if margin < PARAMS["acoustic_margin"] or not ranked:
            for offset, weight in ((0, 1.0), (-1, 0.5), (1, 0.5)):
                source_index = index + offset
                if not 0 <= source_index < len(frames):
                    continue
                source = frames[source_index]
                acoustic_index = _midi_candidate_for_acoustic(source["acoustic"])
                if acoustic_index is None or (ranked and acoustic_index not in ranked[:3]):
                    continue
                if acoustic_index not in rerank:
                    rerank[acoustic_index] = 0.0
                rerank[acoustic_index] += PARAMS["acoustic_bonus"] * float(source["acoustic_confidence"]) * weight
        frame["rerank_emissions"] = rerank
    reranked = viterbi("rerank_emissions")
    acoustic_reranks = {index for index, (before, after) in enumerate(zip(midi_decoded, reranked))
                        if before != after}

    segments = []
    for frame_index, ((start, end), frame, candidate) in enumerate(zip(bounds, frames, reranked)):
        if candidate is None:
            name, confidence = "N.C.", 0.0
        else:
            root, quality, pcs = CANDIDATES[candidate]
            name = PITCH_NAMES[root] + quality
            bass_pc = frame["bass"]
            if bass_pc is not None and bass_pc != root and bass_pc in pcs:
                name += "/" + PITCH_NAMES[bass_pc]
            confidence = min(1.0, max(0.0, frame["rerank_emissions"].get(candidate, 0.0)))
        item = {"start": round(start, 3), "end": round(end, 3), "chord": name,
                "confidence": round(confidence, 3), "_boundary_score": round(frame["boundary"], 3)}
        if frame_index in acoustic_reranks:
            item["_solitito_rerank"] = True
        if name.split("/", 1)[0] != frame["acoustic"].split("/", 1)[0]:
            item["_midi_correction"] = True
        if segments and segments[-1]["chord"] == name:
            segments[-1]["end"] = round(end, 3)
            segments[-1]["confidence"] = round(max(segments[-1]["confidence"], confidence), 3)
        else:
            segments.append(item)
    return {"_model": MODEL, "_anchor": "acoustic-guitar-chords.json",
            "_pipeline": "downbeats/beats -> drum anchors -> fixed grid -> MIDI Viterbi -> Solitito bonus -> second Viterbi",
            "_timing": "beat-grid quarter-note",
            "_timing_source": "drum MIDI + BPM + downbeats",
            "_label_policy": "MIDI-only first pass; confidence-weighted Solitito top-3 bonus; second Viterbi",
            "_boundary_weights": "bass=.50,piano-chroma=.35,meter=.15; drum=grid-anchor-only",
            "_midi_note_filter": "bass>=eighth; piano>=eighth-or-simultaneous-chord",
            "chords": segments}


def estimate_chords(folder: str | Path, acoustic: dict | None = None,
                    drums: dict | None = None) -> dict:
    folder = Path(folder)
    grid = _read_json(folder / "beat-grid.json")
    piano_notes = _read_json(folder / "piano-midi.json").get("notes", [])
    bass_notes = _read_json(folder / "bass-midi.json").get("notes", [])
    acoustic = acoustic or _read_json(folder / "acoustic-guitar-chords.json")
    drums = drums or _read_json(folder / "drums-midi.json")
    duration = max((float(n.get("end", n.get("start", 0))) for n in piano_notes + bass_notes), default=0.0)
    duration = max(duration, float(acoustic.get("duration", 0.0)))
    piano_notes = _sustained_notes(piano_notes, grid, "piano")
    bass_notes = _sustained_notes(bass_notes, grid, "bass")
    if acoustic.get("chords"):
        return _grid_first_estimate(acoustic, grid, piano_notes, bass_notes, drums, duration)
    bounds = _segment_bounds(grid, duration)
    piano = _frame_weights(piano_notes, bounds)
    bass = _frame_weights(bass_notes, bounds)
    frames = []
    drum_times = sorted(float(time) for values in drums.values() if isinstance(values, list)
                        for time in values if isinstance(time, (int, float)))
    for (start, end), piano_weights, bass_weights in zip(bounds, piano, bass):
        bass_total = sum(bass_weights)
        piano_total = sum(piano_weights)
        bass_pc = max(range(12), key=bass_weights.__getitem__) if bass_total else None
        if bass_pc is not None and bass_weights[bass_pc] / bass_total < 0.65:
            bass_pc = None  # Walking/passing bass is not reliable inversion evidence.
        # Cap bass influence so a loud bass cannot overwhelm the harmony.
        bass_scale = min(1.0, 0.3 * piano_total / bass_total) if bass_total and piano_total else 1.0
        weights = [p + b * bass_scale for p, b in zip(piano_weights, bass_weights)]
        acoustic_name, acoustic_confidence = _acoustic_at(acoustic, start, end)
        drum_activity = sum(start <= time < end for time in drum_times)
        anchor_strength = min(1.0, acoustic_confidence + (0.12 if drum_activity else 0.0))
        scores = (_scores(weights, bass_pc, acoustic_name, anchor_strength)
                  if sum(w > sum(weights) * 0.06 for w in weights) >= 2 else [])
        frames.append({"weights": weights, "bass": bass_pc, "scores": scores,
                       "acoustic": acoustic_name})
    segments = []
    for (start, end), frame, index in zip(bounds, frames, _decode(frames)):
        chord, confidence = ("N.C.", 0.0) if index is None else _label(index, frame["weights"], frame["bass"], frame["scores"])
        if segments and segments[-1]["chord"] == chord:
            previous = segments[-1]
            length = previous["end"] - previous["start"]
            previous["confidence"] = (previous["confidence"] * length + confidence * (end - start)) / (length + end - start)
            previous["end"] = end
        else:
            segments.append({"start": start, "end": end, "chord": chord, "confidence": confidence})
    for segment in segments:
        for key in ("start", "end", "confidence"):
            segment[key] = round(segment[key], 3)
    return {"_model": MODEL, "_anchor": "acoustic-guitar-chords.json",
            "_timing": "beat-grid quarter-note", "_midi_note_filter": "bass>=eighth; piano>=eighth-or-simultaneous-chord",
            "chords": segments}


def load_or_estimate(folder: str | Path, force: bool = False,
                     acoustic: dict | None = None, drums: dict | None = None) -> dict:
    folder = Path(folder)
    cache = folder / "chords.json"
    # Preserve existing caches, including manual edits, until explicit reanalysis.
    if cache.is_file() and not force:
        data = _read_json(cache)
        data.setdefault("_model", "MIDI beat-grid chord estimator")
        data.setdefault("chords", [])
        return data
    data = estimate_chords(folder, acoustic=acoustic, drums=drums)
    cache.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return data
