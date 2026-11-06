"""Put several live datasets on ONE shared class list (the label map).

FedAvg averages the model's outputs by position, so every client must number
its classes the same way. Each room's build_dataset.py output has its own
vocabulary (room 1: background, cell phone, person; room 2: background,
bottle, laptop). This script:

  1. takes the sorted union of all class names as the shared list,
  2. re-numbers every dataset's labels by NAME onto that list
     (a room simply has 0 clips of the classes it never saw),
  3. writes one file per input plus a union file for the server's test set
     (with `room_train` / `room_test` arrays saying where each clip came from).

Usage:
  .venv/bin/python scripts/merge_vocab.py \
      data/pi_captures/live_dataset_NOV5.npz:room1 \
      data/pi_captures/room2_dataset.npz:room2 \
      --out-dir data/pi_captures --union two_rooms_union.npz
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def load(path: Path) -> dict:
    z = np.load(path, allow_pickle=True)
    return {k: z[k] for k in z.files}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("inputs", nargs="+", help="dataset.npz:name pairs")
    p.add_argument("--out-dir", default="data/pi_captures")
    p.add_argument("--union", default="two_rooms_union.npz")
    args = p.parse_args()

    named = []
    for item in args.inputs:
        path, name = item.rsplit(":", 1)
        named.append((name, load(Path(path))))
    shared = sorted({str(c) for _, d in named for c in d["classes"]})
    print(f"shared class list: {shared}")

    out_dir = Path(args.out_dir)
    parts = {"audio_train": [], "y_train": [], "audio_test": [], "y_test": [],
             "room_train": [], "room_test": []}
    for r, (name, d) in enumerate(named):
        own = [str(c) for c in d["classes"]]
        remap = np.array([shared.index(c) for c in own], dtype=np.int64)
        y_tr, y_te = remap[d["y_train"]], remap[d["y_test"]]
        out = out_dir / f"{name}_shared.npz"
        np.savez_compressed(out, audio_train=d["audio_train"], y_train=y_tr,
                            audio_test=d["audio_test"], y_test=y_te,
                            classes=np.array(shared, dtype=object))
        counts = {shared[i]: int(n) for i, n in enumerate(np.bincount(y_tr, minlength=len(shared))) if n}
        print(f"{name}: own {own} -> {out} | train {len(y_tr)} {counts} | test {len(y_te)}")
        parts["audio_train"].append(d["audio_train"]); parts["y_train"].append(y_tr)
        parts["audio_test"].append(d["audio_test"]); parts["y_test"].append(y_te)
        parts["room_train"].append(np.full(len(y_tr), r)); parts["room_test"].append(np.full(len(y_te), r))

    union = {k: np.concatenate(v) for k, v in parts.items()}
    out = out_dir / args.union
    np.savez_compressed(out, classes=np.array(shared, dtype=object),
                        rooms=np.array([n for n, _ in named], dtype=object), **union)
    yt = union["y_test"]
    maj = np.bincount(yt).max() / len(yt)
    print(f"union -> {out} | train {len(union['y_train'])} | test {len(yt)} "
          f"{ {shared[i]: int(n) for i, n in enumerate(np.bincount(yt, minlength=len(shared)))} } "
          f"| majority baseline {maj:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
