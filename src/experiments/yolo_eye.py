"""
YOLO as a separate helper process (6 Nov 2026, post-submission)
==============================================================

On the Pi, TensorFlow (the ear) and PyTorch (YOLO) crash when loaded into one
process (an OpenMP clash on aarch64; see pi_infer.py). So YOLO runs here, in its
own process: it owns the camera, runs YOLOv8n on the newest frame about three
times a second, and publishes the result in memory (/dev/shm):

    <out-dir>/frame_0.jpg, frame_1.jpg   the newest photo (two files, written in turn)
    <out-dir>/latest.json                {"time", "frame": "frame_1.jpg", "dets": [...]}

latest.json is replaced atomically and always names a complete photo, so a reader
never sees half a file. pi_teach.py --yolo starts this helper by itself.

    .venv/bin/python -m src.experiments.yolo_eye --out-dir /dev/shm/yolo_eye
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import time
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--camera-index", type=int, default=0)
    p.add_argument("--model", default="yolov8n.pt")
    p.add_argument("--conf", type=float, default=0.25, help="Report detections from this confidence")
    p.add_argument("--out-dir", default="/dev/shm/yolo_eye")
    a = p.parse_args()

    import cv2
    from ultralytics import YOLO

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cam = cv2.VideoCapture(a.camera_index, cv2.CAP_V4L2)
    if not cam.isOpened():
        cam = cv2.VideoCapture(a.camera_index)
    cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    model = YOLO(a.model)
    stop = {"flag": False}
    signal.signal(signal.SIGTERM, lambda *_: stop.update(flag=True))
    signal.signal(signal.SIGINT, lambda *_: stop.update(flag=True))

    k = 0
    while not stop["flag"]:
        cam.grab()
        ok, frame = cam.read()
        if not ok:
            time.sleep(0.1)
            continue
        res = model(frame, verbose=False, conf=a.conf)[0]
        dets = [{"class": res.names[int(b.cls)], "conf": round(float(b.conf), 3)} for b in res.boxes]
        name = f"frame_{k % 2}.jpg"
        cv2.imwrite(str(out / name), frame)
        tmp = out / "latest.tmp"
        tmp.write_text(json.dumps({"time": time.time(), "frame": name, "dets": dets}))
        os.replace(tmp, out / "latest.json")
        k += 1
    cam.release()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
