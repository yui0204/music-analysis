"""Deterministic musical regression cases; no model downloads or user-file writes."""
import json
from pathlib import Path
import tempfile
import unittest

from analysis.chord_estimator import (_decode, _estimate_one, _fixed_drum_grid,
                             _metrical_bonus, _segment_bounds, _sustained_notes,
                             estimate_chords, load_or_estimate)


def notes(pitches, start=0, end=1, velocity=80):
    return [dict(pitch=p, start=start, end=end, velocity=velocity) for p in pitches]


def weights(pitches):
    result = [0.0] * 12
    for pitch in pitches:
        result[pitch % 12] += 1
    return result


class ChordTests(unittest.TestCase):
    def estimate(self, piano, bass=(), beats=(0, 0.5, 1, 1.5, 2)):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for filename, data in [("piano-midi.json", {"notes": piano}),
                                   ("bass-midi.json", {"notes": list(bass)}),
                                   ("beat-grid.json", {"beats": beats, "downbeats": [0, 2]})]:
                (root / filename).write_text(json.dumps(data))
            return estimate_chords(root)["chords"]

    def test_chord_vocabulary_all_roots(self):
        cases = [("", [0, 4, 7]), ("m", [0, 3, 7]), ("7", [0, 4, 7, 10]),
                 ("maj7", [0, 4, 7, 11]), ("m7", [0, 3, 7, 10]),
                 ("m7b5", [0, 3, 6, 10]), ("add9", [0, 4, 7, 2])]
        from analysis.chord_estimator import PITCH_NAMES
        for root in range(12):
            for quality, intervals in cases:
                with self.subTest(root=root, quality=quality):
                    self.assertEqual(_estimate_one(weights([root + i for i in intervals]), root)[0], PITCH_NAMES[root] + quality)

    def test_scale_invariance(self):
        chroma = weights([60, 64, 67, 71])
        self.assertEqual(_estimate_one(chroma, 0), _estimate_one([x * 0.001 for x in chroma], 0))

    def test_single_note_is_unknown(self):
        self.assertEqual(_estimate_one(weights([36]), 0), ("N.C.", 0.0))
        self.assertEqual(self.estimate([], notes([36]))[0]["chord"], "N.C.")

    def test_half_bar_change(self):
        result = self.estimate(notes([60, 64, 67], 0, 1) + notes([65, 69, 72], 1, 2))
        self.assertEqual([(s["start"], s["end"], s["chord"]) for s in result], [(0, 1, "C"), (1, 2, "F")])

    def test_late_acoustic_rerank_uses_quarter_timing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "beat-grid.json").write_text(json.dumps({"beats": [0, .5, 1, 1.5, 2], "downbeats": [0, 2]}))
            (root / "piano-midi.json").write_text(json.dumps({"notes": notes([60, 64, 67], 0, 2)}))
            (root / "bass-midi.json").write_text(json.dumps({"notes": notes([36], 0, 2)}))
            (root / "drums-midi.json").write_text(json.dumps({"kick": [0, .5, 1, 1.5]}))
            acoustic = {"duration": 2, "chords": [{"start": 0, "end": 2, "chord": "C", "confidence": .95}]}
            result = estimate_chords(root, acoustic=acoustic, drums={"kick": [0, .5, 1, 1.5]})
            self.assertEqual([s["chord"] for s in result["chords"]], ["C"])
            self.assertEqual(result["_anchor"], "acoustic-guitar-chords.json")
            self.assertEqual(result["_timing"], "beat-grid quarter-note")
            self.assertEqual(result["_pipeline"], "downbeats/beats -> drum anchors -> fixed grid -> MIDI Viterbi -> Solitito bonus -> second Viterbi")

    def test_drum_anchors_adjust_grid_once_by_lane_strength(self):
        grid = {"beats": [0, .5, 1], "bpm": 120}
        kick = _fixed_drum_grid(grid, {"kick": [.53]}, 1.5)
        self.assertEqual(kick[0][1], .518)
        hihat = _fixed_drum_grid(grid, {"hihat": [.53]}, 1.5)
        self.assertAlmostEqual(hihat[0][1], .505, places=3)

    def test_relaxed_midi_filter_and_metrical_accents(self):
        grid = {"beats": [0, .5, 1, 1.5, 2], "downbeats": [0, 2], "bpm": 120}
        bass = notes([36], 0, .26) + notes([38], .5, .6)
        self.assertEqual([n["pitch"] for n in _sustained_notes(bass, grid, "bass")], [36])
        bass_line = (notes([36], 0, .3) + notes([38], .3, .6) + notes([36], .6, .9))
        self.assertEqual(len(_sustained_notes(bass_line, grid, "bass")), 3)
        piano = notes([60, 64, 67], 0, .1) + notes([66], .5, .6)
        self.assertEqual(len(_sustained_notes(piano, grid, "piano")), 3)
        self.assertGreater(_metrical_bonus(0, grid), _metrical_bonus(.5, grid))
        self.assertGreater(_metrical_bonus(1, grid), _metrical_bonus(.5, grid))

    def test_inversion(self):
        self.assertEqual(self.estimate(notes([60, 64, 67]), notes([40]))[0]["chord"], "C/E")

    def test_bass_root_resolves_equal_pitch_set_names(self):
        # Dm7 and F6 share D/F/A/C. The bass root must decide the label.
        voicing = notes([62, 65, 69, 72])
        self.assertEqual(self.estimate(voicing, notes([38]))[0]["chord"], "Dm7")
        self.assertEqual(self.estimate(voicing, notes([41]))[0]["chord"], "F6")

    def test_sustained_extension_tones_produce_extended_labels(self):
        self.assertEqual(_estimate_one(weights([0, 2, 4, 7, 10]), 0)[0], "C9")
        self.assertEqual(_estimate_one(weights([0, 2, 4, 7, 11]), 0)[0], "Cmaj9")
        self.assertEqual(_estimate_one(weights([0, 2, 3, 7, 10]), 0)[0], "Cm9")

    def test_passing_note(self):
        result = self.estimate(notes([60, 64, 67], 0, 2) + notes([66], 0.6, 0.68))
        self.assertEqual([s["chord"] for s in result], ["C"])

    def test_sequence_smooths_ambiguity_but_allows_clear_change(self):
        ambiguous = [{"scores": s} for s in ([1, 0], [0.9, 1], [1, 0])]
        self.assertEqual(_decode(ambiguous), [0, 0, 0])
        clear = [{"scores": s} for s in ([1, 0], [0, 1], [0, 1])]
        self.assertEqual(_decode(clear), [0, 1, 1])

    def test_silence_breaks_sequence(self):
        result = self.estimate(notes([60, 64, 67], 0, 0.5) + notes([65, 69, 72], 1.5, 2))
        self.assertEqual([s["chord"] for s in result], ["C", "N.C.", "F"])

    def test_zero_velocity(self):
        self.assertEqual(self.estimate(notes([60, 64, 67], velocity=0))[0]["chord"], "N.C.")

    def test_bounds_cover_tail_and_reject_outside(self):
        bounds = _segment_bounds({"beats": [-1, 0, 0.5, 0.51, 1, 99], "downbeats": [0, 100]}, 1.03)
        self.assertEqual(bounds[0][0], 0)
        self.assertEqual(bounds[-1][1], 1.03)
        self.assertTrue(all(a[1] == b[0] for a, b in zip(bounds, bounds[1:])))
        self.assertEqual(_segment_bounds({}, 0), [])
        self.assertEqual(_segment_bounds({}, 0.1), [(0, 0.1)])

    def test_cache_preserves_manual_edits(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / "chords.json"
            data = {"_model": "old", "chords": [{"chord": "Dm", "edited": True}]}
            cache.write_text(json.dumps(data))
            self.assertEqual(load_or_estimate(directory), data)
            self.assertEqual(load_or_estimate(directory, force=True)["chords"], [])


if __name__ == "__main__":
    unittest.main()
