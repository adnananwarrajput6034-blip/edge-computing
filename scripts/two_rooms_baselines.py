"""Reference points for the two-room federated run (5 Nov 2026).

Trains the same warm-started AudioCNN as the Pi clients (LR 1e-3, batch 8,
40 epochs = 5 rounds x 8 local epochs) on:
  - room 1 only  (what Pi 1 would learn alone)
  - room 2 only  (what Pi 2 would learn alone)
  - both rooms   (centralized, like Strategies A/B: the upper bound)
and evaluates every model, plus the federated model, on the shared test set,
overall, per room and per class.

  .venv/bin/python scripts/two_rooms_baselines.py \
      --union data/pi_captures/two_rooms_union.npz \
      --federated results/two_pi_20261105/c_two_rooms/trained.keras \
      --out results/two_pi_20261105/c_two_rooms/baselines.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import tensorflow as tf  # noqa: E402

from src.edge.processing.stft import STFTProcessor  # noqa: E402
from src.experiments.pi_live import build_live_model  # noqa: E402


def report(model, X, y, room, classes) -> dict:
    pred = np.argmax(model.predict(X, verbose=0), axis=1)
    out = {"overall": float(np.mean(pred == y))}
    for r in np.unique(room):
        m = room == r
        out[f"room{r + 1}"] = float(np.mean(pred[m] == y[m]))
    out["per_class"] = {classes[c]: float(np.mean(pred[y == c] == c)) for c in np.unique(y)}
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--union", required=True)
    p.add_argument("--federated", default=None)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None)
    p.add_argument("--model", action="append", default=[],
                   help="NAME=PATH of an extra saved model to evaluate (repeatable)")
    p.add_argument("--skip-train", action="store_true",
                   help="Only evaluate --federated / --model, train no references")
    a = p.parse_args()

    z = np.load(a.union, allow_pickle=True)
    classes = [str(c) for c in z["classes"]]
    stft = STFTProcessor(sample_rate=16000, n_fft=512, hop_length=160,
                         n_mels=128, window_length=400, normalize=True)
    feat = lambda audio: stft.process_batch(audio)[..., np.newaxis].astype(np.float32)  # noqa: E731
    Xtr, ytr, rtr = feat(z["audio_train"]), z["y_train"].astype(np.int32), z["room_train"]
    Xte, yte, rte = feat(z["audio_test"]), z["y_test"].astype(np.int32), z["room_test"]

    results = {"classes": classes, "test_clips": int(len(yte)),
               "majority_baseline": float(np.bincount(yte).max() / len(yte)),
               "epochs": a.epochs, "seed": a.seed}
    refs = [] if a.skip_train else [("room1_only", rtr == 0), ("room2_only", rtr == 1),
                                    ("centralized_both_rooms", np.ones_like(rtr, dtype=bool))]
    for name, mask in refs:
        tf.keras.utils.set_random_seed(a.seed)
        model, _ = build_live_model(len(classes), learning_rate=1e-3)
        model.fit(Xtr[mask], ytr[mask], epochs=a.epochs, batch_size=8, verbose=0)
        results[name] = report(model, Xte, yte, rte, classes)
        print(f"{name:24s} {results[name]['overall']:.3f}  room1 {results[name]['room1']:.3f}  "
              f"room2 {results[name]['room2']:.3f}", flush=True)
    if a.federated:
        model = tf.keras.models.load_model(a.federated)
        results["federated_two_pis"] = report(model, Xte, yte, rte, classes)
        f = results["federated_two_pis"]
        print(f"{'federated_two_pis':24s} {f['overall']:.3f}  room1 {f['room1']:.3f}  room2 {f['room2']:.3f}")
    for spec in a.model:
        name, path = spec.split("=", 1)
        results[name] = report(tf.keras.models.load_model(path), Xte, yte, rte, classes)
        f = results[name]
        print(f"{name:24s} {f['overall']:.3f}  room1 {f['room1']:.3f}  room2 {f['room2']:.3f}")
    for k in ["room1_only", "room2_only", "centralized_both_rooms", "federated_two_pis"] + [m.split("=", 1)[0] for m in a.model]:
        if k in results:
            print(f"  {k:24s} per class: " + ", ".join(f"{c} {v:.2f}" for c, v in results[k]["per_class"].items()))
    if a.out:
        Path(a.out).write_text(json.dumps(results, indent=2))
        print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
