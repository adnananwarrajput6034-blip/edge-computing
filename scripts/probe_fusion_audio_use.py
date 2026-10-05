"""
Probe: does a fusion model actually use its audio input?
========================================================
Two checks per fusion model on the VGGSound test split:

1. Blackout diversity — with every image black, how many DIFFERENT classes
   does the model predict? A model that ignores audio predicts one constant
   class (the audio is the only input that still varies).
2. Audio swap — keep the real images, shuffle the audio between test clips,
   and count how many predictions change. 0% means audio has no influence.

Output: data/vggsound/fusion_audio_probe.json + printed table.

Usage:
    .venv/bin/python scripts/probe_fusion_audio_use.py
"""

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
CACHE = REPO_ROOT / "data/vggsound/paired_cache.npz"
MODELS_DIR = REPO_ROOT / "models"
OUT = REPO_ROOT / "data/vggsound/fusion_audio_probe.json"
SEED = 42


def main() -> int:
    d = np.load(CACHE, allow_pickle=True)
    X_img = d["images"][d["img_idx_test"]].astype(np.float32)
    X_aud, y = d["X_aud_test"], d["y_test"]
    X_blk = np.zeros_like(X_img)
    perm = np.random.default_rng(SEED).permutation(len(y))

    def predict(model, img, aud):
        return model.predict({"vision_input": img, "audio_input": aud},
                             batch_size=32, verbose=0).argmax(axis=1)

    results = {}
    for path in sorted(MODELS_DIR.glob("fusion_fusion*.keras")):
        name = path.stem.removeprefix("fusion_")
        model = tf.keras.models.load_model(path)
        blk = predict(model, X_blk, X_aud)
        clean = predict(model, X_img, X_aud)
        swapped = predict(model, X_img, X_aud[perm])
        results[name] = {
            "blackout_distinct_classes": int(len(np.unique(blk))),
            "audio_swap_changed": int((clean != swapped).sum()),
            "audio_swap_total": int(len(y)),
        }
        r = results[name]
        print(f"{name:18s} blackout distinct classes={r['blackout_distinct_classes']}  "
              f"audio swap changed {r['audio_swap_changed']}/{r['audio_swap_total']}")

    OUT.write_text(json.dumps(results, indent=2))
    print(f"\nSaved: {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
