"""Build a training dataset from one or more recorded sessions
(src/experiments/pi_record.py), choosing the labelling rules AFTER recording:
record everything first, decide later.

Added after the thesis submission (branch demo-improvements). The output has the
same format as pi_enroll.py, so strategies A, B and C work with it unchanged.

1. See what is in the recording:
     .venv/bin/python scripts/build_dataset.py data/pi_captures/session_X --summary

2. Optional human check: open the folder in Finder, look at the photos, press the
   spacebar on a .wav to listen, and DELETE ticks you don't trust (deleted ticks are
   skipped). To give a tick a name yourself, add a line to relabel.csv in the
   folder, e.g.   tick0123,pen     (your names win over YOLO and skip all filters)

3. Build the dataset:
     .venv/bin/python scripts/build_dataset.py data/pi_captures/session_X \
         --classes "cell phone,person" --min-conf 0.5 \
         --min-samples 20 --background --out data/pi_captures/live_dataset.npz

"With sound" is automatic: each recording uses the threshold pi_record measured
in the room before it started (room.json, 1.5x the room's noise). Recordings
without room.json use 1.5x their quietest 10% of ticks. --min-rms 0.012 still
sets one fixed number for all recordings.

Several recordings can be combined: give more than one folder
(session_X session_Y ...). Each folder keeps its own relabel.csv.

--classes is a PRIORITY list: when several are visible in one frame, the first one
wins ("cell phone,person" = the phone wins over you holding it). If a class earlier
in the list is only half-visible (recorded, but below --min-conf), the tick is
unclear and skipped, so a half-seen ringing phone is not called "person".
--background adds a "background" class from quiet ticks where none of the classes
is visible, so the model also learns what "nothing" sounds like.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.experiments.pi_live import save_dataset  # noqa: E402

BACKGROUND = "background"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("sessions", nargs="+",
                   help="one or more session folders copied from the Pi "
                        "(several recordings are combined)")
    p.add_argument("--summary", action="store_true",
                   help="only show what is in the recording")
    p.add_argument("--classes", default="",
                   help="comma-separated labels in priority order")
    p.add_argument("--min-conf", type=float, default=0.5)
    p.add_argument("--min-rms", default="auto",
                   help="loudness gate for YOLO-labelled ticks: 'auto' = each "
                        "recording's own threshold (see above), or a number "
                        "(0 = off)")
    p.add_argument("--min-samples", type=int, default=20)
    p.add_argument("--background", action="store_true",
                   help="add a 'background' class from quiet ticks where "
                        "none of the classes is visible")
    p.add_argument("--background-max-rms", default="auto",
                   help="background ticks must be quieter than this "
                        "('auto' = the same per-recording threshold)")
    p.add_argument("--test-fraction", type=float, default=0.25)
    p.add_argument("--out", default="data/pi_captures/live_dataset.npz")
    return p.parse_args()


def best_conf(row: dict) -> dict:
    """Highest confidence per class in one tick."""
    out: dict = {}
    for d in row["detections"]:
        out[d["class"]] = max(out.get(d["class"], 0.0), d["conf"])
    return out


RMS_FACTOR = 1.5   # same rule as pi_record: 'with sound' = 1.5x the room's noise


def auto_threshold(session: Path, rows: list[dict]) -> tuple[float, str]:
    """A recording's own 'with sound' threshold, and where it came from."""
    room = session / "room.json"
    if room.exists():
        info = json.loads(room.read_text())
        if info.get("threshold") is not None:
            if info.get("noise_rms") is not None:
                return float(info["threshold"]), "room measured at the start"
            return float(info["threshold"]), "number given when recording"
    quiet = float(np.percentile([r["rms"] for r in rows], 10)) if rows else 0.008
    return round(RMS_FACTOR * quiet, 4), f"1.5x its quietest ticks ({quiet:.4f})"


def summary(rows: list[dict], deleted: int, thr: dict) -> None:
    n = len(rows)
    rms = np.array([r["rms"] for r in rows]) if rows else np.zeros(0)
    print(f"ticks: {n} kept, {deleted} deleted by you")
    if n:
        loud = sum(r["rms"] >= thr[r["dir"]] for r in rows)
        print(f"loudness: median {np.median(rms):.4f}, with sound: {loud}, "
              f"clear (>= 0.03): {int((rms >= 0.03).sum())}")
    stats: dict = {}
    for r in rows:
        for c, conf in best_conf(r).items():
            s = stats.setdefault(c, [0, 0, 0])
            s[0] += 1
            if conf >= 0.5:
                s[1] += 1
                if r["rms"] >= thr[r["dir"]]:
                    s[2] += 1
    print(f"\n{'label':<16s}{'seen':>7s}{'>=0.5':>8s}{'>=0.5 + sound':>15s}")
    for c, (a, b, s) in sorted(stats.items(), key=lambda kv: -kv[1][2]):
        print(f"{c:<16s}{a:7d}{b:8d}{s:15d}")
    print("\n'>=0.5 + sound' is what a class could get with --min-conf 0.5 and "
          "the loudness gate. Pick classes with 20 or more.")


def main() -> int:
    args = parse_args()
    rows_all: list[dict] = []
    relabel: dict = {}                     # (session dir, tick) -> your name
    for s in args.sessions:
        session = Path(s)
        log = session / "ticks.jsonl"
        if not log.exists():
            print(f"ERROR: {log} not found (is this a pi_record session?)",
                  file=sys.stderr)
            return 1
        for line in log.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                row["dir"] = str(session)
                rows_all.append(row)
        rl = session / "relabel.csv"
        if rl.exists():
            for line in rl.read_text().splitlines():
                if "," in line:
                    k, v = line.split(",", 1)
                    relabel[(str(session), k.strip())] = v.strip()
    rows = [r for r in rows_all if (Path(r["dir"]) / f"{r['file']}.wav").exists()]
    deleted = len(rows_all) - len(rows)

    min_thr: dict = {}                     # session dir -> 'with sound' threshold
    bg_thr: dict = {}                      # session dir -> background ceiling
    for s in args.sessions:
        key = str(Path(s))
        auto, how = auto_threshold(Path(s), [r for r in rows_all if r["dir"] == key])
        min_thr[key] = auto if args.min_rms == "auto" else float(args.min_rms)
        bg_thr[key] = (auto if args.background_max_rms == "auto"
                       else float(args.background_max_rms))
        print(f"{Path(s).name}: 'with sound' = rms >= {min_thr[key]:.4f} "
              f"({how if args.min_rms == 'auto' else 'given'})")

    if args.summary:
        summary(rows, deleted, min_thr)
        return 0

    priority = [c.strip() for c in args.classes.split(",") if c.strip()]
    if not priority and not relabel:
        print("ERROR: give --classes (or a relabel.csv)", file=sys.stderr)
        return 1

    labels: list[str] = []
    files: list[Path] = []
    why: Counter = Counter()
    for r in rows:
        wav = Path(r["dir"]) / f"{r['file']}.wav"
        if (r["dir"], r["file"]) in relabel:         # your decision wins
            labels.append(relabel[(r["dir"], r["file"])])
            files.append(wav)
            why["named by you"] += 1
            continue
        conf = best_conf(r)
        label = next((c for c in priority if conf.get(c, 0.0) >= args.min_conf),
                     None)
        if label is None:
            if args.background and r["rms"] < bg_thr[r["dir"]]:
                labels.append(BACKGROUND)
                files.append(wav)
                why["background"] += 1
            else:
                why["no chosen class visible"] += 1
            continue
        if any(c in conf for c in priority[:priority.index(label)]):
            why["unclear (half-visible)"] += 1
            continue
        if r["rms"] < min_thr[r["dir"]]:
            why["too quiet"] += 1
            continue
        labels.append(label)
        files.append(wav)
        why["labelled by YOLO"] += 1

    counts = Counter(labels)
    if BACKGROUND in counts:                         # keep classes balanced
        others = [n for c, n in counts.items() if c != BACKGROUND]
        cap = max(others) if others else counts[BACKGROUND]
        if counts[BACKGROUND] > cap:
            keep_idx = [i for i, l in enumerate(labels) if l == BACKGROUND]
            rng = np.random.default_rng(0)
            drop = set(rng.choice(keep_idx, counts[BACKGROUND] - cap,
                                  replace=False).tolist())
            labels = [l for i, l in enumerate(labels) if i not in drop]
            files = [f for i, f in enumerate(files) if i not in drop]
            counts = Counter(labels)

    classes = sorted(c for c, n in counts.items() if n >= args.min_samples)
    print("--- build summary ---")
    for k, v in why.most_common():
        print(f"  {k:<26s} {v:5d} ticks")
    for c, n in counts.most_common():
        print(f"  {'KEPT   ' if c in classes else 'dropped'} {c:<16s} {n:4d} samples")
    if len(classes) < 2:
        print(f"\nERROR: need >=2 classes with >={args.min_samples} samples, "
              f"got {len(classes)}. Change the rules or record more.",
              file=sys.stderr)
        return 1

    import soundfile as sf
    keep = [i for i, l in enumerate(labels) if l in classes]
    audio = np.stack([sf.read(files[i], dtype="int16")[0] for i in keep])
    info = save_dataset(Path(args.out), audio, [labels[i] for i in keep],
                        classes, test_fraction=args.test_fraction)
    report = {"sessions": args.sessions, "rules": vars(args),
              "sound_thresholds": min_thr, "reasons": dict(why), **info}
    Path(args.out).with_suffix(".build.json").write_text(json.dumps(report, indent=2))
    print(f"\nvocabulary: {classes}")
    print(f"train {info['n_train']} / test {info['n_test']} samples")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
