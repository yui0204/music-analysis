import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf

from analysis.btc_chords import LABELS, ROOT, MODEL_FILE, audio_paths, decode, infer_audio
from analysis.chord_estimator import load_or_estimate


class BTCChordTests(unittest.TestCase):
    def test_taxonomy_preserves_rare_qualities_and_unknown(self):
        self.assertEqual(len(LABELS), 170)
        self.assertEqual(LABELS[7], "CmMaj7")
        self.assertEqual(LABELS[10:14], ("Cdim7", "Cm7b5", "Csus2", "Csus4"))
        self.assertEqual(LABELS[168:], ("X", "N.C."))

    def test_audio_dominates_conflicting_midi_and_keeps_fast_change(self):
        bounds = [(i * .1, (i + 1) * .1) for i in range(4)]
        probs = np.full((4, 170), 1e-8)
        probs[:2, 12] = .999
        probs[2:, 7] = .999
        piano = [dict(pitch=p, start=0, end=.4, velocity=100) for p in (62, 66, 69)]
        result = decode(probs, bounds, piano)
        self.assertEqual([s["chord"] for s in result], ["Csus2", "CmMaj7"])
        self.assertEqual(result[0]["end"], .2)

    def test_unknown_is_not_silence(self):
        probs = np.zeros((2, 170))
        probs[0, 168] = 1
        probs[1, 169] = 1
        self.assertEqual([s["chord"] for s in decode(probs, [(0,1),(1,2)])], ["X", "N.C."])

    def test_sustained_bass_adds_inversion_without_fragmenting(self):
        probs = np.zeros((10,170))
        probs[:,1] = 1
        bounds = [(i / 10, (i + 1) / 10) for i in range(10)]
        bass = [dict(pitch=40, start=.1, end=1, velocity=80)]
        result = decode(probs, bounds, bass_notes=bass)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["chord"], "C/E")

    def test_no_double_counting_guitar_and_cached_edits_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            for name in ("guitar", "acoustic-guitar", "guitar-other", "bass"):
                sf.write(folder / (name + ".wav"), np.zeros(100), 22050)
            self.assertEqual({p.stem for p in audio_paths(folder)}, {"guitar", "bass"})
            saved = {"chords": [{"chord": "Cadd9", "start": 0, "end": 1}]}
            (folder / "chords.json").write_text(json.dumps(saved))
            with patch("analysis.btc_chords.estimate", return_value={"chords": []}) as estimate:
                self.assertEqual(load_or_estimate(folder)["chords"], saved["chords"])
                estimate.assert_not_called()
                load_or_estimate(folder, force=True)
                estimate.assert_called_once()

    @unittest.skipUnless((ROOT / ".cache" / "btc" / MODEL_FILE).is_file(), "BTC checkpoint not cached")
    def test_real_checkpoint_short_silence_and_chunk_timing(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mix.wav"
            for duration in (.05, 20.3):
                sf.write(path, np.zeros(round(duration * 22050), dtype=np.float32), 22050)
                probabilities, bounds, actual = infer_audio([path])
                self.assertAlmostEqual(actual, duration, places=4)
                self.assertEqual(bounds[0][0], 0)
                self.assertAlmostEqual(bounds[-1][1], duration, places=4)
                self.assertTrue(all(a[1] == b[0] for a,b in zip(bounds, bounds[1:])))
                self.assertTrue(all(a < b for a,b in bounds))
                self.assertTrue(np.isfinite(probabilities).all())
                self.assertTrue((probabilities.argmax(axis=1) == 169).all())


if __name__ == "__main__":
    unittest.main()
