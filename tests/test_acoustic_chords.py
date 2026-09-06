"""Solitito DSP, output and Qt integration tests using generated audio."""
import json
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf
from PySide6.QtWidgets import QApplication
from PySide6.QtTest import QTest

from analysis.acoustic_chords import (CACHE_FILE, ROOT, FeatureExtractor, _prediction, _quarter_segments, _segments,
                             filter_short_chords, load_or_estimate, prepare_chords, quantize_chords)
from app import Window, STYLE, chord_edit_suggestions, chord_shape, USER_CHORD_SHAPES, USER_CHORD_SHAPES_PATH
from app import ACOUSTIC_CHORD_CACHE, ChordAnalysisWorker, FullAnalysisWorker


class AcousticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        cls.app.setStyle("Fusion")
        cls.app.setStyleSheet(STYLE)

    def test_head_mapping_and_non_chord(self):
        roots = np.full(13, -20.0)
        roots[0] = 20
        for index, expected in [(0, "C"), (2, "Cmaj7"), (5, "Cm7b5"), (6, "Cdim7"), (8, "Csus"), (9, "N.C."), (10, "N.C.")]:
            quality = np.full(11, -20.0)
            quality[index] = 20
            self.assertEqual(_prediction(roots, quality)[0], expected)
        roots[12] = 40
        self.assertEqual(_prediction(roots, np.zeros(11))[0], "N.C.")

    def test_shapes_only_come_from_json(self):
        self.assertEqual(USER_CHORD_SHAPES_PATH.name, "chord_shape.json")
        stored = json.loads(USER_CHORD_SHAPES_PATH.read_text())
        self.assertEqual(USER_CHORD_SHAPES, stored)
        self.assertEqual(chord_shape("Cadd9"), stored["Cadd9"])
        with patch.dict(USER_CHORD_SHAPES, {"Cm": stored["Cm"]}, clear=True):
            self.assertIsNone(chord_shape("G"))
            self.assertIsNone(chord_shape("Cm/Eb"))
            self.assertEqual(chord_shape("Cmin"), stored["Cm"])

    def test_edit_candidates_stay_in_the_same_harmonic_family(self):
        self.assertEqual(chord_edit_suggestions("C"), ["Cadd9", "Csus4", "C6", "Cmaj7", "C7"])
        self.assertNotIn("Cm", chord_edit_suggestions("C"))
        self.assertNotIn("C", chord_edit_suggestions("Cm"))
        self.assertTrue(all(chord_shape(name) is not None for name in chord_edit_suggestions("C7")))

    def test_quarter_grid_quantization_with_tempo_changes(self):
        grid = {"beats": [0, .6, 1.2, 2, 2.8]}
        data = {"duration": 2.91, "chords": [dict(start=0, end=.47, chord="C"),
                    dict(start=.47, end=1.43, chord="G"), dict(start=1.43, end=2.91, chord="F")]}
        result = quantize_chords(data, grid)["chords"]
        self.assertEqual([c["chord"] for c in result], ["C", "G", "F"])
        np.testing.assert_allclose([c["start"] for c in result], [0, .6, 1.2])
        np.testing.assert_allclose([c["end"] for c in result], [.6, 1.2, 2.91])
        self.assertEqual(data["chords"][0]["end"], .47)

    def test_quantization_collapses_zero_length_and_recomputes_from_raw(self):
        raw = {"duration": 2, "chords": [dict(start=0, end=.51, chord="C"),
                    dict(start=.51, end=.54, chord="D"), dict(start=.54, end=2, chord="C")]}
        self.assertEqual(len(quantize_chords(raw, {"bpm": 120})["chords"]), 1)
        prepared = prepare_chords(raw, {"beats": [0, .5, 1, 1.5, 2]})
        self.assertEqual(prepared, prepare_chords(prepared, {"beats": [0, .5, 1, 1.5, 2]}))
        self.assertEqual(prepare_chords(prepared, {"bpm": 90}), prepare_chords(raw, {"bpm": 90}))

    def test_quarter_note_filter_boundary_and_raw_preservation(self):
        data = {"chords": [dict(start=0, end=.4999, chord="C"),
                           dict(start=.4999, end=.9999, chord="G"),
                           dict(start=1, end=1.01, chord="N.C.")]}
        filtered = filter_short_chords(data, {"bpm": 120})
        self.assertEqual([c["chord"] for c in filtered["chords"]], ["G", "N.C."])
        self.assertEqual(len(data["chords"]), 3)
        self.assertEqual(filtered["_raw_chords"], data["chords"])
        self.assertEqual(filter_short_chords(filtered, {"bpm": 120}), filtered)
        self.assertEqual(filter_short_chords(filtered, {"bpm": 240})["chords"], data["chords"])

    def test_filter_uses_local_beats_across_tempo_change(self):
        data = {"chords": [dict(start=.25, end=1, chord="C"),
                           dict(start=1, end=1.75, chord="G")]}
        filtered = filter_short_chords(data, {"beats": [0, .5, 1.5, 2.5], "bpm": 120})
        self.assertEqual([c["chord"] for c in filtered["chords"]], ["C"])
        for grid in ({}, {"bpm": None}, {"bpm": 0}, {"bpm": "bad"}):
            self.assertEqual(len(filter_short_chords(data, grid)["chords"]), 2)

    def test_existing_cache_filtered_without_inference_or_rewrite(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            data = {"chords": [dict(start=0, end=.1, chord="C"), dict(start=.1, end=1, chord="G")]}
            cache = folder / CACHE_FILE
            cache.write_text(json.dumps(data))
            self.assertEqual([c["chord"] for c in load_or_estimate(folder)["chords"]], ["G"])
            self.assertEqual(json.loads(cache.read_text()), data)

    def test_analysis_buttons_have_separate_cache_and_force_policies(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            for name in ("vocals.wav", "drums.wav", "piano.wav", "bass.wav", "acoustic-guitar.wav"):
                (folder / name).write_bytes(b"wav")
            calls = []
            import app
            with patch.object(app, "load_or_estimate_beats", side_effect=lambda p, force=False: calls.append(("beats", force)) or {"beats": []}), \
                 patch.object(app, "load_or_estimate_vocal", side_effect=lambda p, force=False: calls.append(("vocal", force)) or {"notes": []}), \
                 patch.object(app, "load_or_estimate", side_effect=lambda p, force=False: calls.append(("drum", force)) or {"kick": []}), \
                 patch.object(app, "load_or_estimate_piano", side_effect=lambda p, force=False: calls.append(("piano", force)) or {"notes": []}), \
                 patch.object(app, "load_or_estimate_bass", side_effect=lambda p, force=False: calls.append(("bass", force)) or {"notes": []}), \
                 patch.object(app, "load_or_estimate_acoustic", side_effect=lambda p, force=False: calls.append(("acoustic", force)) or {"chords": []}), \
                 patch.object(app, "load_or_estimate_chords", side_effect=lambda *args, **kwargs: calls.append(("chords", kwargs.get("force"))) or {"chords": []}):
                worker = FullAnalysisWorker(folder, force=False)
                worker.run()
            self.assertEqual(calls, [("beats", False), ("vocal", False), ("drum", False), ("piano", False), ("bass", False)])
            (folder / ACOUSTIC_CHORD_CACHE).write_text(json.dumps({"chords": []}))
            (folder / "drums-midi.json").write_text(json.dumps({"kick": []}))
            calls = []
            with patch.object(app, "load_or_estimate_acoustic", side_effect=lambda *args, **kwargs: calls.append(kwargs.get("force")) or {"chords": []}), \
                 patch.object(app, "load_or_estimate_chords", side_effect=lambda *args, **kwargs: calls.append(kwargs.get("force")) or {"chords": []}):
                worker = ChordAnalysisWorker(folder)
                worker.run()
            self.assertEqual(calls, [True, True])

    def test_short_run_fills_from_stronger_neighbour(self):
        for left_conf, right_conf, expected in [(.9, .4, 1.2), (.4, .9, 1.0)]:
            data = {"chords": [dict(start=0, end=1, chord="C", confidence=left_conf),
                               dict(start=1, end=1.1, chord="D", confidence=.9),
                               dict(start=1.1, end=1.2, chord="E", confidence=.9),
                               dict(start=1.2, end=2.2, chord="G", confidence=right_conf)]}
            result = filter_short_chords(data, {"bpm": 120})
            self.assertEqual([c["chord"] for c in result["chords"]], ["C", "G"])
            self.assertEqual(result["chords"][0]["end"], expected)
            self.assertEqual(result["chords"][1]["start"], expected)
            self.assertEqual(filter_short_chords(result, {"bpm": 120}), result)

    def test_matching_neighbours_merge_without_crossing_silence(self):
        data = {"chords": [dict(start=0, end=1, chord="C", confidence=.9),
                           dict(start=1, end=1.2, chord="D", confidence=.9),
                           dict(start=1.2, end=2, chord="C", confidence=.9),
                           dict(start=2, end=2.1, chord="N.C.", confidence=0),
                           dict(start=2.1, end=3, chord="C", confidence=.9)]}
        result = filter_short_chords(data, {"bpm": 120})["chords"]
        self.assertEqual([c["chord"] for c in result], ["C", "N.C.", "C"])
        self.assertEqual(result[0]["end"], 2)
        self.assertLess(result[0]["confidence"], .9)

    def test_no_neighbour_stays_unknown_and_existing_gap_is_preserved(self):
        result = filter_short_chords({"chords": [dict(start=0, end=.2, chord="C"),
                                                dict(start=1, end=2, chord="G")]}, {"bpm": 120})["chords"]
        self.assertEqual(result[0]["chord"], "N.C.")
        self.assertEqual(result[0]["end"], .2)
        self.assertEqual(result[1]["start"], 1)

    def test_same_chord_bridge_starts_at_preceding_downbeat(self):
        data = {"chords": [dict(start=.5, end=1, chord="C", confidence=.9),
                           dict(start=1, end=1.1, chord="D", confidence=.9),
                           dict(start=1.1, end=2, chord="C", confidence=.9),
                           dict(start=2, end=3, chord="G", confidence=.9)]}
        result = filter_short_chords(data, {"beats": [0, .5, 1, 1.5, 2, 2.5, 3],
                                            "downbeats": [0, 2], "bpm": 120})["chords"]
        self.assertEqual([c["chord"] for c in result], ["C", "G"])
        self.assertEqual(result[0]["start"], 0)
        self.assertEqual(result[0]["end"], 2)
        self.assertTrue(result[0]["same_chord_bridge"])

    def test_different_neighbours_do_not_move_to_barline(self):
        data = {"chords": [dict(start=.5, end=1, chord="C", confidence=.9),
                           dict(start=1, end=1.1, chord="D", confidence=.9),
                           dict(start=1.1, end=2, chord="G", confidence=.9)]}
        result = filter_short_chords(data, {"downbeats": [0, 2], "bpm": 120})["chords"]
        self.assertEqual(result[0]["start"], .5)
        self.assertEqual(result[0]["end"], 1)

    def test_segments_contiguous_and_keep_silence(self):
        result = _segments([("C", .9), ("G", .8), ("C", .7), ("N.C.", 0), ("C", .9)], .6)
        self.assertEqual([s["chord"] for s in result], ["C", "N.C.", "C"])
        self.assertEqual(result[0]["start"], 0)
        self.assertEqual(result[-1]["end"], .6)
        self.assertTrue(all(a["end"] == b["start"] for a, b in zip(result, result[1:])))
        self.assertEqual(_segments([("C", .9)], .03)[0]["end"], .03)

    def test_initial_solitito_votes_are_quarter_note_cells(self):
        predictions = [("C", .8)] * 4 + [("G", .8)] * 4
        result = _quarter_segments(predictions, 1.0, {"beats": [0, .5, 1]})
        self.assertEqual([(item["start"], item["end"]) for item in result], [(0.0, .5), (.5, 1.0)])

    def test_dsp_matches_scalar_reference(self):
        path = ROOT / ".cache/solitito/dsp_weights.json"
        if not path.exists():
            self.skipTest("Download official assets to run DSP parity")
        data = json.loads(path.read_text())
        samples = np.random.default_rng(4).normal(0, .1, 8192).astype(np.float32)
        spectrum = np.fft.rfft(samples * np.hanning(8192))
        magnitudes = []
        for i in range(144):
            first, last = data["cqt_offsets"][i:i + 2]
            value = sum(spectrum[data["cqt_fft_idx"][j]] * complex(data["cqt_re"][j], data["cqt_im"][j]) for j in range(first, last))
            magnitudes.append(abs(value) * (5 if i < 36 else 1))
        mag = np.asarray(magnitudes)
        cqt = np.clip((20 * np.log10(np.maximum(mag, 1e-9) / max(mag.max(), .005)) + 80) / 80, 0, 1)
        chroma = cqt @ np.array(data["chroma_weights"]).reshape(144, 12)
        chroma /= chroma.max()
        expected = np.concatenate([cqt, chroma, cqt[:24].reshape(12, 2).mean(axis=1)])
        extractor = FeatureExtractor(path)
        np.testing.assert_allclose(extractor.transform(samples[None])[0], expected, atol=2e-5)
        self.assertTrue(np.isfinite(extractor.transform(np.zeros((1, 8192), dtype=np.float32))).all())

    def test_cached_view_and_folder_switch(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            sf.write(folder / "acoustic-guitar.wav", np.zeros(16000), 16000)
            data = {"chords": [{"start": 0, "end": .5, "chord": "C"}, {"start": .5, "end": 1, "chord": "G7"}]}
            (folder / CACHE_FILE).write_text(json.dumps(data))
            window = Window()
            try:
                window.render_results(folder, ["acoustic-guitar"])
                window.show()
                QTest.qWait(50)
                self.assertTrue(window.acoustic_chord_view.isVisible())
                self.assertIsNone(window.chord_view)
                window.render_playback_position(750)
                self.assertEqual(window.acoustic_chord_view.current_chord(), "G7")
                window.grab().save("/tmp/stem-acoustic-chords-ui.png")
                window.load_acoustic_chords()
                window.clear_results()
                self.assertIsNone(window.acoustic_process)
                self.assertIsNone(window.acoustic_chord_view)
                self.assertEqual(json.loads((folder / CACHE_FILE).read_text()), data)
            finally:
                window.close()

    def test_real_model_via_ui_and_cache(self):
        if not (ROOT / ".cache/solitito/best_model_v2_take6_onset.onnx").exists():
            self.skipTest("Download official model to run inference")
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            t = np.arange(32000) / 16000
            audio = sum(.05 * np.sin(2 * np.pi * freq * t) for freq in (130.81, 164.81, 196.0))
            sf.write(folder / "acoustic-guitar.wav", np.column_stack([audio, audio]), 16000)
            window = Window()
            try:
                window.render_results(folder, ["acoustic-guitar"])
                window.load_acoustic_chords()
                deadline = time.monotonic() + 30
                while window.acoustic_process is not None and time.monotonic() < deadline:
                    QTest.qWait(50)
                self.assertIsNone(window.acoustic_process)
                self.assertTrue((folder / CACHE_FILE).exists(), window.acoustic_chord_note.text())
                data = json.loads((folder / CACHE_FILE).read_text())
                self.assertEqual(data["chords"], window.acoustic_chord_view.chords)
                self.assertEqual(data["_raw_chords"][-1]["end"], 2)
                self.assertTrue(all(c["chord"] == "N.C." or c["end"] - c["start"] >= .5 - 1e-8
                                    for c in data["chords"]))
                self.assertFalse((folder / "chords.json").exists())
            finally:
                window.close()


if __name__ == "__main__":
    unittest.main()
