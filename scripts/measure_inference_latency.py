"""
Inference latency of the trained models
=======================================
Times one prediction (batch size 1, as on a device that hears one clip at a
time) for each step of the pipeline:

    STFT              8,000 samples -> 51 x 128 spectrogram
    AudioCNN          spectrogram -> class
    MobileNetV3       224 x 224 image -> 576-number feature vector
    FV fusion head    feature vector + spectrogram -> class

Each step is warmed up, then timed 200 times; the median and the 95th
percentile are reported. Run on whatever machine executes the script.

Output: results/latency/inference_latency_<hostname>.json + printed table.

Usage:
    .venv/bin/python scripts/measure_inference_latency.py
"""

import json
import os
import platform
import socket
import sys
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.edge.processing.stft import STFTProcessor  # noqa: E402

RUNS, WARMUP = 200, 20
OUT_DIR = REPO_ROOT / "results/latency"


def timeit(fn):
    for _ in range(WARMUP):
        fn()
    t = []
    for _ in range(RUNS):
        t0 = time.perf_counter()
        fn()
        t.append((time.perf_counter() - t0) * 1000)
    return {"median_ms": round(float(np.median(t)), 2),
            "p95_ms": round(float(np.percentile(t, 95)), 2)}


def main() -> int:
    rng = np.random.default_rng(0)
    audio = (rng.standard_normal(8000) * 3000).astype(np.int16)
    stft = STFTProcessor(sample_rate=16000, n_fft=512, hop_length=160,
                         n_mels=128, window_length=400, normalize=True)
    spec = stft.process_batch(audio[None])[..., np.newaxis].astype(np.float32)
    image = rng.random((1, 224, 224, 3), dtype=np.float32) * 255

    audio_cnn = tf.keras.models.load_model(REPO_ROOT / "models/audio_cnn_urbansound8k.keras")
    backbone = tf.keras.applications.MobileNetV3Small(
        input_shape=(224, 224, 3), include_top=False, pooling="avg", weights="imagenet")
    fv = backbone(image, training=False).numpy()
    fv_fusion = tf.keras.models.load_model(REPO_ROOT / "models/fv_fusion.keras")
    fusion_in = {"vision_fv_input": fv, "audio_input": spec}

    results = {
        "stft": timeit(lambda: stft.process_batch(audio[None])),
        "audio_cnn": timeit(lambda: audio_cnn(spec, training=False)),
        "mobilenet_v3_small": timeit(lambda: backbone(image, training=False)),
        "fv_fusion_head": timeit(lambda: fv_fusion(fusion_in, training=False)),
    }
    host = socket.gethostname().split(".")[0]
    meta = {"host": host, "machine": platform.machine(), "system": platform.system(),
            "tensorflow": tf.__version__, "runs": RUNS, "batch_size": 1}
    for k, v in results.items():
        print(f"{k:20s} median {v['median_ms']:7.2f} ms   p95 {v['p95_ms']:7.2f} ms")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"inference_latency_{host}.json"
    out.write_text(json.dumps({"meta": meta, "results": results}, indent=2))
    print(f"\nSaved: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
