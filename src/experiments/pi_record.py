"""Record EVERYTHING: one tick = 500 ms of audio + one camera frame + every
YOLO detection + the loudness. Nothing is filtered or thrown away here.

Added after the thesis submission (branch demo-improvements). The thesis
enrollment (pi_enroll.py) labels while it records. This tool separates the
two steps: record once, then choose the labelling rules afterwards with
scripts/build_dataset.py, and look at / listen to every tick in between.

On the Pi:
    .venv/bin/python -m src.experiments.pi_record --audio-device 2
    (Ctrl-C to stop)

    With --target "cell phone" the live line says GOOD for every tick the build
    can use (seen at >= 0.5 confidence, with sound) and counts them, so you see
    at once whether the pose and loudness work and when you have enough.

    "With sound" is measured, not guessed: before recording starts the Pi
    listens to the quiet room for 3 seconds (stay quiet, no sound playing) and
    sets the threshold to 1.5x the room's noise (about +3.5 dB). Start your
    sound only when it says RECORDING.

Output: results/live/session_<time>/
    tick0001.wav  tick0001.jpg  ...   every tick (16 kHz int16 audio, frame)
    ticks.jsonl                       per tick: loudness + all detections
    room.json                         room noise + the 'with sound' threshold,
                                      reused by scripts/build_dataset.py
"""
from __future__ import annotations

import argparse
import json
import shutil
import socket
import signal
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.experiments.pi_live import MicRecorder, TARGET_SR  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--audio-device", type=int, default=None,
                   help="sounddevice input index of the USB mic")
    p.add_argument("--camera-index", type=int, default=0)
    p.add_argument("--model", default="yolov8n.pt")
    p.add_argument("--min-conf", type=float, default=0.25,
                   help="Store YOLO detections at or above this confidence. "
                        "The default 0.25 keeps almost everything; the real "
                        "threshold is chosen later in build_dataset.py.")
    p.add_argument("--tick-ms", type=int, default=1000)
    p.add_argument("--duration-min", type=float, default=None,
                   help="Stop after N minutes (default: run until Ctrl-C)")
    p.add_argument("--out-dir", default="results/live")
    p.add_argument("--min-rms", default="auto",
                   help="Loudness a tick needs to count as 'with sound'. "
                        "'auto' (default): listen to the quiet room for 3 s "
                        "at the start and use 1.5x its noise. Or a number, "
                        "e.g. 0.012.")
    p.add_argument("--target", default="",
                   help="Comma-separated labels you want to collect, e.g. "
                        "'cell phone'. The live line then shows GOOD for each "
                        "usable tick (seen at >= 0.5 with sound) and counts "
                        "them.")
    return p.parse_args()


def open_camera(index: int):
    """Same camera mode as scripts/pi_detect_check.sh (V4L2 + MJPG)."""
    import cv2
    cam = cv2.VideoCapture(index, cv2.CAP_V4L2)
    if cam.isOpened():
        cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        return cam
    return cv2.VideoCapture(index)


GOOD_CONF = 0.5    # same as '>=0.5 + sound' in build_dataset --summary
RMS_FACTOR = 1.5   # 'with sound' = at least 1.5x the room's noise (~ +3.5 dB)


def chunk_rms(audio: np.ndarray) -> float:
    """Loudness of one int16 chunk on a 0..1 scale."""
    return float(np.sqrt(np.mean((audio.astype(np.float32) / 32768.0) ** 2)))


def measure_room(mic, seconds: float = 3.0) -> float:
    """Typical loudness of the quiet room: the median of 500 ms chunks."""
    levels = [chunk_rms(mic.finish(mic.start_chunk()))
              for _ in range(max(1, round(seconds * 2)))]
    return float(np.median(levels))


def usable_target(row: dict, targets: list[str], min_rms: float) -> str | None:
    """The first target seen at >= GOOD_CONF in a tick with sound, else None."""
    if row["rms"] < min_rms:
        return None
    best: dict = {}
    for d in row["detections"]:
        best[d["class"]] = max(best.get(d["class"], 0.0), d["conf"])
    return next((t for t in targets if best.get(t, 0.0) >= GOOD_CONF), None)


def status_line(row: dict, hit: str | None, targets: list[str],
                good: Counter, min_rms: float) -> str:
    """One live line per tick, cut to the terminal width so it never wraps."""
    top = ", ".join(f"{d['class']} {d['conf']:.2f}"
                    for d in row["detections"][:3]) or "nothing"
    line = f"[record] tick {row['tick']}: rms {row['rms']:.3f} | {top}"
    if targets:
        so_far = ", ".join(f"{t} {good[t]}" for t in targets)
        verdict = f"GOOD {hit}" if hit else "not usable"
        line = (f"[record] tick {row['tick']}: {verdict} "
                f"(good so far: {so_far}) | rms {row['rms']:.3f} "
                f"(need {min_rms:.3f}) | {top}")
    width = shutil.get_terminal_size((100, 24)).columns - 1
    return line[:width].ljust(width)


def main() -> int:
    args = parse_args()
    import cv2
    import soundfile as sf
    from src.edge.processing.vision import VisionProcessor

    session = Path(args.out_dir) / f"session_{time.strftime('%Y%m%d_%H%M%S')}"
    session.mkdir(parents=True)
    stop = {"flag": False}
    signal.signal(signal.SIGINT, lambda *_: stop.update(flag=True))
    signal.signal(signal.SIGTERM, lambda *_: stop.update(flag=True))

    print(f"[record] saving EVERY tick to {session}", flush=True)
    cam = open_camera(args.camera_index)
    if not cam.isOpened():
        print(f"ERROR: cannot open camera {args.camera_index}", file=sys.stderr)
        return 1
    mic = MicRecorder(args.audio_device)
    print(f"[record] mic: {mic.name} @ {mic.native_sr} Hz -> {TARGET_SR} Hz",
          flush=True)
    print(f"[record] loading YOLO ({args.model})...", flush=True)
    vision = VisionProcessor(model_path=args.model,
                             confidence_threshold=args.min_conf)
    ok, warm = cam.read()
    if ok:
        vision.detect(cv2.cvtColor(warm, cv2.COLOR_BGR2RGB))

    if args.min_rms == "auto":
        print("[record] measuring the room for 3 s: stay quiet, no sound "
              "playing...", flush=True)
        noise = measure_room(mic)
        min_rms = round(max(RMS_FACTOR * noise, 0.002), 4)
        print(f"[record] room noise rms {noise:.4f} -> 'with sound' means "
              f"rms >= {min_rms:.4f} ({RMS_FACTOR}x the room)", flush=True)
        if noise > 0.02:
            print("[record] WARNING: the room was loud while measuring. If a "
                  "sound was playing, press Ctrl-C and start again.",
                  flush=True)
    else:
        noise, min_rms = None, float(args.min_rms)
    (session / "room.json").write_text(json.dumps({
        "noise_rms": None if noise is None else round(noise, 5),
        "threshold": min_rms,
        "factor": RMS_FACTOR if noise is not None else None}))

    print("[record] RECORDING. Show things, make sounds, talk: everything is "
          "kept. Ctrl-C to stop.\n", flush=True)

    t_start = time.perf_counter()
    deadline = t_start + args.duration_min * 60 if args.duration_min else None
    ticks, with_sound = 0, 0
    seen: Counter = Counter()
    targets = [t.strip() for t in args.target.split(",") if t.strip()]
    good: Counter = Counter()

    with open(session / "ticks.jsonl", "w") as log:
        while not stop["flag"]:
            if deadline and time.perf_counter() > deadline:
                break
            tick_t0 = time.perf_counter()
            rec = mic.start_chunk()          # 500 ms window opens
            ok, frame = cam.read()           # frame lands inside that window
            audio = mic.finish(rec)
            if not ok:
                print("\n[record] WARN camera read failed", flush=True)
                continue
            ticks += 1

            dets = vision.detect(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            dets = sorted(dets, key=lambda d: -float(d["confidence"]))
            rms = chunk_rms(audio)

            stem = f"tick{ticks:04d}"
            cv2.imwrite(str(session / f"{stem}.jpg"), frame)
            sf.write(str(session / f"{stem}.wav"), audio, TARGET_SR)
            row = {
                "tick": ticks, "file": stem, "time": round(time.time(), 3),
                "rms": round(rms, 4),
                "detections": [{"class": d["class"],
                                "conf": round(float(d["confidence"]), 3)}
                               for d in dets],
            }
            log.write(json.dumps(row) + "\n")
            log.flush()

            for c in {d["class"] for d in row["detections"]}:
                seen[c] += 1
            with_sound += rms >= min_rms
            hit = usable_target(row, targets, min_rms) if targets else None
            if hit:
                good[hit] += 1
            print("\r" + status_line(row, hit, targets, good, min_rms),
                  end="", flush=True)

            delay = args.tick_ms / 1000.0 - (time.perf_counter() - tick_t0)
            if delay > 0:
                time.sleep(delay)

    cam.release()
    print(f"\n\n--- recording summary ---\n"
          f"ticks {ticks} in {time.perf_counter() - t_start:.0f} s, "
          f"saved to {session}\n"
          f"ticks with sound (rms >= {min_rms:.4f}): {with_sound}")
    if targets:
        print("usable for the build (seen >= 0.5 with sound): "
              + ", ".join(f"{t} {good[t]}" for t in targets))
    print(f"what YOLO saw (ticks per label, confidence >= {args.min_conf}):")
    for c, n in seen.most_common(12):
        print(f"  {c:<16s} {n:4d}")
    print("\nNext, on the laptop:\n"
          f"  rsync -az {socket.gethostname()}:~/thesis/edge/edgeComputing/{session}/ "
          f"data/pi_captures/{session.name}/\n"
          f"  .venv/bin/python scripts/build_dataset.py "
          f"data/pi_captures/{session.name} --summary")
    return 0


if __name__ == "__main__":
    sys.exit(main())
