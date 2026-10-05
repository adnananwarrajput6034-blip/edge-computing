"""
Training-settings sweep on the REAL live Pi dataset
===================================================
The live demo trained with a single setting. This script asks which setting
works best on the real enrolled data (Pi microphone audio, YOLO labels):

    learning rate   x   warm start (UrbanSound8K conv stack) vs from scratch

For each combination it trains for 40 epochs, records test accuracy after
every epoch, and repeats with 3 seeds. Batch size 8, as on the Pi.

Output: data/pi_captures/results/live/training_sweep.json + printed table.

Usage:
    .venv/bin/python scripts/sweep_live_training.py
"""

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.edge.processing.stft import STFTProcessor  # noqa: E402
from src.experiments.pi_live import build_live_model, load_dataset  # noqa: E402

DATASET = REPO_ROOT / "data/pi_captures/results/live/live_dataset.npz"
OUT = REPO_ROOT / "data/pi_captures/results/live/training_sweep.json"
LRS = [1e-3, 1e-4]
EPOCHS = 40
REPORT_AT = [1, 3, 5, 10, 20, 40]
SEEDS = [0, 1, 2]
BATCH = 8


def to_spec(audio, stft):
    """Same conversion the live servers use (strategy_a_server --live-dataset)."""
    return stft.process_batch(audio)[..., np.newaxis].astype(np.float32)


class EpochAccuracy(tf.keras.callbacks.Callback):
    def __init__(self, X, y):
        super().__init__()
        self.X, self.y, self.acc = X, y, []

    def on_epoch_end(self, epoch, logs=None):
        p = self.model.predict(self.X, verbose=0).argmax(axis=1)
        self.acc.append(float((p == self.y).mean()))


def main() -> int:
    d = load_dataset(DATASET)
    stft = STFTProcessor(sample_rate=16000, n_fft=512, hop_length=160,
                         n_mels=128, window_length=400, normalize=True)
    Xtr, Xte = to_spec(d["audio_train"], stft), to_spec(d["audio_test"], stft)
    ytr, yte = np.asarray(d["y_train"]), np.asarray(d["y_test"])
    n_cls = len(d["classes"])
    print(f"live dataset: {len(ytr)} train / {len(yte)} test, classes {d['classes']} "
          f"(chance {1 / n_cls:.3f})\n")

    results = {}
    for warm in (True, False):
        for lr in LRS:
            key = f"{'warm' if warm else 'scratch'}_lr{lr:g}"
            curves = []
            for seed in SEEDS:
                tf.keras.utils.set_random_seed(seed)
                model, _ = build_live_model(n_cls, learning_rate=lr, verbose=False,
                                         warm_start=None if not warm else
                                         REPO_ROOT / "models/audio_cnn_urbansound8k.keras")
                cb = EpochAccuracy(Xte, yte)
                model.fit(Xtr, ytr, epochs=EPOCHS, batch_size=BATCH, verbose=0,
                          callbacks=[cb], shuffle=True)
                curves.append(cb.acc)
            curves = np.array(curves)
            results[key] = {
                "warm_start": warm, "lr": lr,
                "mean_per_epoch": curves.mean(axis=0).round(4).tolist(),
                "std_per_epoch": curves.std(axis=0).round(4).tolist(),
            }
            row = "  ".join(f"ep{e}={curves[:, e - 1].mean():.3f}±{curves[:, e - 1].std():.3f}"
                            for e in REPORT_AT)
            print(f"{key:18s} {row}")

    OUT.write_text(json.dumps({
        "classes": d["classes"], "n_train": int(len(ytr)), "n_test": int(len(yte)),
        "chance": round(1 / n_cls, 4), "batch_size": BATCH, "seeds": SEEDS,
        "epochs": EPOCHS, "results": results}, indent=2))
    print(f"\nSaved: {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
