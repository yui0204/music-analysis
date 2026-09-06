Solitito attribution
===================

Upstream: https://github.com/greblus/solitito
DSP source: src/audio.rs and dist/gen_weights.py
Source revision: e6a38fbabb9d4a6267a5a966d2707fe22c2cf8d9
License: MIT (see LICENSE in this directory).

acoustic_chords.py adapts the upstream sparse pseudo-CQT, log normalization,
chroma and bass features for offline Python inference. The file-input bass
boost is preserved. Offline centered context and three-frame voting are
STEM Studio additions. Label mapping preserves dim7 and leaves sus unspecified.

Model: https://huggingface.co/greblus/solitito-ai
Model revision: 96f63770aea422a5aa2cf6dc775f0375f5866ef4
Files: best_model_v2_take6_onset.onnx and dsp_weights.json (MIT).
Downloads are pinned and checked against SHA-256 in acoustic_chords.py.

Upstream training data includes GuitarSet (CC BY 4.0): Qingyang Xi,
Rachel M. Bittner, Johan Pauwels, Xuzhou Ye & Juan P. Bello.
https://guitarset.weebly.com/
