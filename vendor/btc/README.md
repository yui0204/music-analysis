# BTC attribution

Source: https://github.com/jayg996/BTC-ISMIR19
Revision: 2682317be668032e6e4b269ded36adaa2ad57df0
License: MIT, see LICENSE (Jonggwon Park, 2019).

model.py and transformer_modules.py are the official architecture, with relative
imports and NumPy compatibility fixes. The official large-vocabulary checkpoint
is downloaded to .cache/btc, revision-pinned and SHA-256 verified before loading.
The adapter uses checkpoint normalization and 10-second CQT chunks, float32 CPU
inference, and at most four PyTorch threads. No training or CUDA is required.

The taxonomy is 14 qualities x 12 roots, plus X (unknown) and N (no chord).
It does not directly predict add9/9/11/13 or inversions. The application adds
inversions only when supported by sustained bass MIDI. Accuracy superiority
over other models has not been established on this project's songs.
