"""
Teach the device new things, live: ear + eye + human (6 Nov 2026, post-submission)
=================================================================================

THE EAR  = the AudioCNN (sound -> name), as everywhere in the thesis.
THE EYE  = the camera + MobileNetV3-Small (the vision part of the thesis fusion
           model). It turns each photo into a "fingerprint" of 576 numbers and
           compares it with the fingerprints of things you have named. So it can
           learn objects YOLO does not know (YOLO has no class "key").
THE HUMAN names a sound once; after that the eye can teach the ear by itself.

--mode quiz   (default)
    Waits for a sound, listens as long as it lasts (up to --max-event ticks),
    then asks ONCE:   "Ear: bottle 89%  ·  Eye: key 0.91  ->  Enter = key"
      Enter       = accept the suggestion       a number = that class
      a new name  = new class (asks y/n), then records --want loud examples
      s = skip    ·    q = quit
--mode listen
    Says what it hears every second. Press Enter when it is wrong; it asks about
    the last sound. With --auto it never asks: when the eye is sure and the ear
    disagrees, the eye's name becomes the label (the camera teaches the microphone,
    as in the thesis, but now also for classes YOLO does not have).

Learning: a new name grows the ear's output layer by one class (old outputs kept;
class-incremental learning); every named sound is replayed with this Pi's stored
clips (--data) and the ear retrains its last two Dense layers. The eye stores the
fingerprints of every named photo. Clips, photos, answers (teach_log.jsonl) and
every version (models/) are saved in --run-dir.

    .venv/bin/python -m src.experiments.pi_teach \\
        --model results/live/teach/start.keras \\
        --data results/live/room2_shared.npz --audio-device 2

Keep the camera view EMPTY for the first 3 seconds: that is what "nothing" looks like.
"""
from __future__ import annotations

import argparse
import json
import os
import select
import sys
import time
from collections import Counter, deque
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.experiments.pi_live import MicRecorder, load_dataset  # noqa: E402
from src.experiments.pi_record import chunk_rms  # noqa: E402

RMS_FACTOR = 1.5
NOTHING = "(nothing)"
EYE_SRC = {"v": ""}  # where the eye's last answer came from: "taught" or "YOLO"
DARK = 30          # mean pixel value (0-255) below which the camera "cannot see"
QUIT = {"q", "quit", "exit"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help=".keras with a .classes.json beside it")
    p.add_argument("--data", required=True,
                   help="This Pi's own dataset (.npz), replayed when retraining")
    p.add_argument("--audio-device", type=int, default=None)
    p.add_argument("--mode", choices=["quiz", "listen"], default="quiz")
    p.add_argument("--auto", action="store_true",
                   help="listen mode: when the eye is sure, use its name as the label (no human)")
    p.add_argument("--no-camera", action="store_true", help="ear only")
    p.add_argument("--show-first", action="store_true",
                   help="Before listening: teach the EYE silently. Show one thing at a time "
                        "and type its name; the camera remembers it.")
    p.add_argument("--camera-index", type=int, default=0)
    p.add_argument("--yolo", action="store_true",
                   help="Also run YOLO live: started as a separate helper process "
                        "(yolo_eye.py) that owns the camera, because TensorFlow and PyTorch "
                        "crash in one process on the Pi")
    p.add_argument("--yolo-dir", default="/dev/shm/yolo_eye")
    p.add_argument("--yolo-conf", type=float, default=0.5,
                   help="YOLO confidence needed for the eye to name an object")
    p.add_argument("--threshold", type=float, default=0.60,
                   help="Ear: below this probability the answer is 'unknown'")
    p.add_argument("--eye-threshold", type=float, default=0.80,
                   help="Eye: fingerprint similarity needed to name a thing")
    p.add_argument("--eye-margin", type=float, default=0.04,
                   help="Eye: how much the best match must beat the second best")
    p.add_argument("--min-event-seconds", type=float, default=3.0,
                   help="quiz: after a sound starts, keep listening at least this long")
    p.add_argument("--max-event-seconds", type=float, default=8.0,
                   help="quiz: longest sound event before it asks")
    p.add_argument("--novelty-ticks", type=int, default=4,
                   help="listen: loud unknown ticks in a row before it asks")
    p.add_argument("--want", type=int, default=12,
                   help="Loud examples to collect after you name a new sound")
    p.add_argument("--max-ticks", type=int, default=45,
                   help="Give up collecting a new sound after this many ticks")
    p.add_argument("--retrain-every", type=int, default=6,
                   help="Learn again after this many named sounds of known classes")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--train", choices=["head", "all"], default="head",
                   help="head: only the last two Dense layers learn (less forgetting)")
    p.add_argument("--tick-ms", type=int, default=1000)
    p.add_argument("--run-dir", default="results/live/teach")
    return p.parse_args()


# ------------------------------------------------------------------ the ear

def features(stft, audio: np.ndarray) -> np.ndarray:
    return stft.process_batch(np.atleast_2d(audio))[..., np.newaxis].astype(np.float32)


def grow_head(model, old_classes: list, new_name: str):
    """Same network with one more output. Every old weight is copied, so the
    old classes keep their meaning (outputs are averaged by position later)."""
    from src.server.training.models.audio_cnn import AudioCNN
    bigger = AudioCNN(num_classes=len(old_classes) + 1)
    for dst, src in zip(bigger.layers[:-1], model.layers[:-1]):
        if src.weights:
            dst.set_weights(src.get_weights())
    k_old, b_old = model.layers[-1].get_weights()
    k_new, b_new = bigger.layers[-1].get_weights()          # fresh random column
    k_new[:, :k_old.shape[1]] = k_old
    b_new[:b_old.shape[0]] = b_old
    # Start the new class "shy": small weights and a low bias, so it does not
    # out-vote the trained classes before it has seen a single example.
    k_new[:, -1] *= 0.1
    b_new[-1] = float(np.min(b_old)) - 1.0
    bigger.layers[-1].set_weights([k_new, b_new])
    return bigger, list(old_classes) + [new_name]


def retrain(model, X, y, epochs: int, train: str, lr: float = 1e-3) -> float:
    import tensorflow as tf
    for layer in model.layers:
        layer.trainable = (train == "all") or isinstance(layer, tf.keras.layers.Dense)
    model.compile(optimizer=tf.keras.optimizers.Adam(lr),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    # Class weighting: every class counts the same, so a new sound with 10 clips
    # is not drowned by 60 stored clips of an old one.
    counts = np.bincount(y, minlength=int(y.max()) + 1)
    weights = {c: len(y) / (len(counts) * n) for c, n in enumerate(counts) if n}
    h = model.fit(X, y, epochs=epochs, batch_size=8, verbose=0, shuffle=True,
                  class_weight=weights)
    for layer in model.layers:
        layer.trainable = True
    return float(h.history["accuracy"][-1])


def predict(model, X) -> np.ndarray:
    return model(X, training=False).numpy()


def accuracy(model, X, y) -> float:
    if len(y) == 0:
        return float("nan")
    return float(np.mean(np.argmax(predict(model, X), axis=1) == y))


# ------------------------------------------------------------------ the eye

class Eye:
    """Learns things by example: a photo's fingerprint is compared with the
    average fingerprint (the "prototype") of every named thing."""

    def __init__(self, index: int, shared_dir: str | None = None):
        import cv2
        import tensorflow as tf
        from collections import OrderedDict
        self.cv2 = cv2
        self.shared = Path(shared_dir) if shared_dir else None
        self.dets: OrderedDict = OrderedDict()         # id(photo) -> YOLO detections
        if self.shared is None:
            self.cam = cv2.VideoCapture(index, cv2.CAP_V4L2)
            if not self.cam.isOpened():
                self.cam = cv2.VideoCapture(index)
            self.cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.net = tf.keras.applications.MobileNetV3Small(
            input_shape=(224, 224, 3), include_top=False, pooling="avg",
            weights="imagenet", include_preprocessing=True)
        self.examples: dict = {}                       # name -> list of fingerprints
        self.photo()                                   # warm up the camera and the network

    def photo(self):
        if self.shared is not None:                    # the YOLO helper's newest photo
            try:
                info = json.loads((self.shared / "latest.json").read_text())
            except (OSError, ValueError):
                return None
            if time.time() - info["time"] > 3.0:       # helper stopped: no photo
                return None
            frame = self.cv2.imread(str(self.shared / info["frame"]))
            if frame is not None:
                self.dets[id(frame)] = info["dets"]
                while len(self.dets) > 300:
                    self.dets.popitem(last=False)
            return frame
        self.cam.grab()                                # drop the buffered old frame
        ok, frame = self.cam.read()
        return frame if ok else None

    def yolo_guess(self, frames: list, min_conf: float):
        """YOLO's best name over these photos. Real objects win over 'person',
        like the priority rule used when the recordings were labelled."""
        best: dict = {}
        for f in frames:
            for d in self.dets.get(id(f), []):
                best[d["class"]] = max(best.get(d["class"], 0.0), d["conf"])
        sure = {c: v for c, v in best.items() if v >= min_conf}
        if not sure:
            return None, 0.0
        things = {c: v for c, v in sure.items() if c != "person"}
        pick = max(things or sure, key=lambda c: (things or sure)[c])
        return pick, (things or sure)[pick]

    def fingerprint(self, frame) -> np.ndarray:
        rgb = self.cv2.resize(self.cv2.cvtColor(frame, self.cv2.COLOR_BGR2RGB), (224, 224))
        v = self.net(rgb.astype(np.float32)[None], training=False).numpy()[0]
        return v / (np.linalg.norm(v) + 1e-9)

    def learn(self, name: str, frames: list) -> None:
        fps = [self.fingerprint(f) for f in frames if not self.too_dark(f)]   # never learn from darkness
        self.examples.setdefault(name, []).extend(fps)

    @staticmethod
    def too_dark(frame) -> bool:
        """Lights off or lens covered: the eye cannot see, so it must not guess."""
        return frame is None or float(frame.mean()) < DARK

    def guess(self, frames: list):
        """(name, similarity, margin) of the best match, or (None, 0, 0)."""
        frames = [f for f in frames if not self.too_dark(f)]
        if not frames:
            return "(cannot see)", 0.0, 0.0
        fps = [self.fingerprint(f) for f in frames if f is not None]
        if not fps or not self.examples:
            return None, 0.0, 0.0
        q = np.mean(fps, axis=0)
        q /= np.linalg.norm(q) + 1e-9
        sims = {}
        for name, ex in self.examples.items():
            proto = np.mean(ex, axis=0)
            sims[name] = float(q @ (proto / (np.linalg.norm(proto) + 1e-9)))
        ranked = sorted(sims.items(), key=lambda kv: -kv[1])
        best, s1 = ranked[0]
        s2 = ranked[1][1] if len(ranked) > 1 else 0.0
        return best, s1, s1 - s2

    def save(self, path: Path) -> None:
        np.savez_compressed(path, **{k.replace(" ", "_"): np.array(v) for k, v in self.examples.items()})


# ------------------------------------------------------------------ keyboard helpers

def drain_keyboard() -> None:
    """Throw away anything typed while it was busy (recording, training)."""
    while select.select([sys.stdin], [], [], 0)[0]:
        if not sys.stdin.readline():
            break


def enter_pressed() -> bool:
    if select.select([sys.stdin], [], [], 0)[0]:
        sys.stdin.readline()
        return True
    return False


def clean_name(text: str) -> str:
    return " ".join(text.strip().lower().split())


def read(prompt: str) -> str:
    ans = clean_name(input(prompt))
    if ans in QUIT:
        raise KeyboardInterrupt
    return ans


def ask_label(ear: str, ear_conf: float, eye: str | None, eye_sim: float,
              suggestion: str | None, classes: list) -> tuple[str, str]:
    """Ask about one sound event. Returns (action, name); action is 'ok', 'fix', 'new' or 'skip'."""
    drain_keyboard()
    menu = "  ".join(f"{i + 1}={c}" for i, c in enumerate(classes))
    ear_txt = f"{ear} {ear_conf:.0%}" if ear != "unknown" else f"not sure ({ear_conf:.0%})"
    eye_txt = (f"   ·   Eye: {eye} {eye_sim:.2f} ({EYE_SRC['v']})" if eye else
               "   ·   Eye: cannot see (dark), the ear decides" if eye_sim < 0 else "")
    hint = f"Enter = {suggestion}" if suggestion else "Enter = skip"
    print(f"\n  Ear: {ear_txt}{eye_txt}")
    print(f"  {hint}   ·   {menu}   ·   or a new name   ·   s = skip   ·   q = quit")
    while True:
        ans = read("  your answer > ")
        if ans == "":
            return (("ok" if suggestion == ear else "fix"), suggestion) if suggestion else ("skip", "")
        if ans == "s":
            return ("skip", "")
        if ans.isdigit():
            i = int(ans) - 1
            if 0 <= i < len(classes):
                return ("ok" if classes[i] == ear else "fix", classes[i])
            print(f"  (no class number {ans}; use 1–{len(classes)})")
            continue
        if ans in classes:
            return ("ok" if ans == ear else "fix", ans)
        if len(ans) > 30 or not any(ch.isalpha() for ch in ans):
            print("  (please type a short name, e.g. key)")
            continue
        if read(f"  Create a NEW class '{ans}'? y/n > ") in ("y", "yes", "j", "ja"):
            return ("new", ans)
        print("  OK, not created. Answer again:")


# ------------------------------------------------------------------ main

def main() -> int:
    args = parse_args()
    run = Path(args.run_dir)
    for sub in ("clips", "photos", "models"):
        (run / sub).mkdir(parents=True, exist_ok=True)
    log = open(run / "teach_log.jsonl", "a", buffering=1)

    import soundfile as sf
    import tensorflow as tf
    from src.edge.processing.stft import STFTProcessor

    model = tf.keras.models.load_model(args.model, compile=False)
    classes = json.loads(Path(args.model).with_suffix(".classes.json").read_text())["classes"]
    stft = STFTProcessor(sample_rate=16000, n_fft=512, hop_length=160,
                         n_mels=128, window_length=400, normalize=True)
    ds = load_dataset(Path(args.data))
    data_classes = list(ds["classes"])
    if list(classes[:len(data_classes)]) != data_classes:
        print(f"ERROR: --data classes {data_classes} must be the first classes of the model "
              f"{classes}", file=sys.stderr)
        return 1
    X_rep, y_rep = features(stft, ds["audio_train"]), np.asarray(ds["y_train"])
    # Clips named in earlier sessions (clips/<time>_<name>_<nn>.wav) are replayed too,
    # so a class taught yesterday is not forgotten when the ear learns again today.
    prev_a, prev_y = [], []
    for wav in sorted((run / "clips").glob("*.wav")):
        name = " ".join(wav.stem.split("_")[1:-1])
        if name in classes:
            audio, _ = sf.read(wav, dtype="int16")
            prev_a.append(audio[:8000]); prev_y.append(classes.index(name))
    if prev_a:
        X_rep = np.concatenate([X_rep, features(stft, np.stack(prev_a))])
        y_rep = np.concatenate([y_rep, prev_y])
        print(f"[teach] also replaying {len(prev_a)} clips named in earlier sessions: "
              f"{dict(Counter(classes[i] for i in prev_y))}")
    X_tst, y_tst = features(stft, ds["audio_test"]), np.asarray(ds["y_test"])
    taught: list = []            # (clip, name) named during this session, trained on
    held: list = []              # (clip, name) of new sounds, kept back to test them
    pending: list = []           # named sounds of known classes, not trained on yet

    mic = MicRecorder(args.audio_device)
    helper = None
    if args.yolo and not args.no_camera:
        import shutil
        import subprocess
        shutil.rmtree(args.yolo_dir, ignore_errors=True)
        print("[teach] starting the YOLO helper (its own process) ...", flush=True)
        helper = subprocess.Popen(
            [sys.executable, "-m", "src.experiments.yolo_eye", "--out-dir", args.yolo_dir,
             "--camera-index", str(args.camera_index)],
            cwd=str(REPO_ROOT), stdout=subprocess.DEVNULL,
            stderr=open(run / "yolo_eye.log", "w"))
        for _ in range(90):
            if (Path(args.yolo_dir) / "latest.json").exists():
                break
            time.sleep(0.5)
        else:
            print("[teach] the YOLO helper did not start; see yolo_eye.log", file=sys.stderr)
            helper.terminate()
            return 1
    eye = None if args.no_camera else Eye(args.camera_index, args.yolo_dir if helper else None)
    print(f"[teach] Ear: {len(y_rep)} stored clips of this Pi are replayed when it learns.")
    print(f"[teach] Eye: {'off' if eye is None else 'camera on, MobileNetV3 fingerprints'}"
          f"{' + live YOLO (helper process)' if helper else ''}.")
    print("[teach] 3 seconds: stay QUIET and keep the camera view EMPTY ...", flush=True)
    levels, frames0 = [], []
    for _ in range(6):
        rec = mic.start_chunk()
        if eye:
            frames0.append(eye.photo())
        levels.append(chunk_rms(mic.finish(rec)))
    gate = RMS_FACTOR * float(np.median(levels))
    if eye:
        eye.learn(NOTHING, frames0)
    print(f"[teach] a sound counts when it is louder than rms {gate:.4f}")
    print(f"[teach] mode: {args.mode}{' + auto' if args.auto else ''}. I know: {classes}")
    print("[teach] quiz: make a sound, I ask once when it stops. listen: Enter = 'that was wrong'.\n",
          flush=True)

    recent: deque = deque(maxlen=args.novelty_ticks)
    streak, version, heard, answers = 0, 0, Counter(), Counter()

    if args.show_first and eye:                          # step 1: teach the eye, silently
        print("[teach] EYE TEACHING. Hold ONE thing in front of the camera and press Enter.\n"
              "        I take 6 photos and tell you what I see. Type 'done' when finished.\n")
        try:
            while True:
                if read("  Hold the thing in view, then Enter (or type done) > ") == "done":
                    break
                shots = []
                for _ in range(6):                       # six photos over about two seconds
                    shots.append(eye.photo()); time.sleep(0.3)
                g, sim, _ = eye.guess(shots)
                yname, yconf = (eye.yolo_guess(shots, args.yolo_conf) if eye.shared is not None
                                else (None, 0.0))
                print(f"    taught fingerprints say: '{g}' (match {sim:.2f})"
                      + (f"   ·   YOLO says: '{yname}' ({yconf:.2f})" if yname else
                         "   ·   YOLO: no object it knows" if eye.shared is not None else ""))
                hint = f"Enter = '{yname}'" if yname else "type its name"
                name = read(f"  What is it? ({hint}, or s = skip) > ") or (yname or "")
                if name in ("", "s"):
                    continue
                eye.learn(NOTHING if name == "nothing" else name, shots)
                for i, f in enumerate(shots[:3]):
                    if f is not None:
                        eye.cv2.imwrite(str(run / "photos" / f"show_{time.strftime('%H%M%S')}_{name}_{i}.jpg"), f)
                g, sim, _ = eye.guess([eye.photo()])
                print(f"    learned. Now the eye sees: '{g}' (match {sim:.2f})")
        except KeyboardInterrupt:
            pass
        print(f"[teach] The eye knows: {sorted(eye.examples)}. Now the ear's turn.\n", flush=True)

    def tick():
        """Half a second of sound (and one photo taken inside it)."""
        rec = mic.start_chunk()
        frame = eye.photo() if eye else None
        a = mic.finish(rec)
        return a, chunk_rms(a), frame

    def ear_says(clips: list):
        p = predict(model, features(stft, np.stack(clips))).mean(axis=0)
        top = int(p.argmax())
        conf = float(p[top])
        return (classes[top] if conf >= args.threshold else "unknown"), classes[top], conf

    def known(name) -> bool:
        """A name the ear already has, or one a human taught the eye.
        Only these may label a sound without a human typing it."""
        return bool(name) and (name in classes or (eye is not None and name in eye.examples))

    def eye_says(frames: list):
        EYE_SRC["v"] = ""
        if not eye:
            return None, 0.0
        name, sim, margin = eye.guess(frames)
        if name == "(cannot see)":
            return None, -1.0                            # dark: the ear decides alone
        if name and name != NOTHING and sim >= args.eye_threshold and margin >= args.eye_margin:
            EYE_SRC["v"] = "taught"                      # something you showed it today
            return name, sim
        if eye.shared is not None:                       # YOLO's 80 known objects
            yname, yconf = eye.yolo_guess(frames, args.yolo_conf)
            if yname:
                EYE_SRC["v"] = "YOLO"
                return yname, yconf
        return None, sim

    def collect(name: str, first: list, first_frames: list):
        clips, frames = list(first), list(first_frames)
        print(f"\n  >>> Make the sound '{name}' now, again and again, close to the mic"
              f"{' and IN FRONT OF THE CAMERA' if eye else ''}. I need {args.want} loud examples. <<<",
              flush=True)
        time.sleep(1.0)
        for _ in range(args.max_ticks):
            if len(clips) >= args.want:
                break
            a, r, f = tick()
            if r >= gate:
                clips.append(a); frames.append(f)
                print(f"    ● {len(clips):2d}/{args.want}   rms {r:.3f}", flush=True)
            else:
                print(f"    ·  quiet (rms {r:.3f}) — keep going", flush=True)
        drain_keyboard()
        if len(clips) < args.want:
            print(f"  (got {len(clips)} of {args.want}; I will learn from those)")
        return clips, frames

    def learn(examples: list, reason: str) -> None:
        """examples: list of (clip, name). New names grow the ear's output layer first."""
        nonlocal model, classes, version
        if not examples:
            return
        before_own = accuracy(model, X_tst, y_tst)
        new_names = [n for n in dict.fromkeys(n for _, n in examples) if n not in classes]
        for n in new_names:
            model, classes = grow_head(model, classes, n)
            mine = [e for e in examples if e[1] == n]
            k = len(mine) // 4 if len(mine) >= 8 else 0      # keep a quarter back to test it
            if k:
                held.extend(mine[-k:])
                examples = [e for e in examples if not any(e is h for h in mine[-k:])]
        stamp = time.strftime("%H%M%S")
        for i, (a, n) in enumerate(examples):
            sf.write(run / "clips" / f"{stamp}_{n.replace(' ', '_')}_{i:02d}.wav", a, 16000)
        taught.extend(examples)
        X = np.concatenate([X_rep, features(stft, np.stack([a for a, _ in taught]))])
        y = np.concatenate([y_rep, [classes.index(n) for _, n in taught]])
        print(f"\n[teach] the ear is learning: {len(y)} clips ({len(taught)} named today), "
              f"{args.epochs} epochs ...", flush=True)
        t0 = time.perf_counter()
        train_acc = retrain(model, X, y, args.epochs, args.train)
        secs = time.perf_counter() - t0
        after_own = accuracy(model, X_tst, y_tst)
        if held:
            hp = np.argmax(predict(model, features(stft, np.stack([a for a, _ in held]))), axis=1)
            right = int(sum(classes[i] == n for i, (_, n) in zip(hp, held)))
            held_txt = f"{right} of {len(held)}"
        else:
            held_txt = "-"
        version += 1
        out = run / "models" / f"model_v{version}.keras"
        model.save(out)
        out.with_suffix(".classes.json").write_text(json.dumps({"classes": classes}, indent=2))
        if eye:
            eye.save(run / "models" / f"eye_v{version}.npz")
        print(f"[teach] learned in {secs:.0f} s.\n"
              f"        Old sounds (this Pi's test clips): {before_own:.2f} -> {after_own:.2f}"
              f"   (did it forget?)\n"
              f"        New sounds, clips it never trained on: {held_txt} right   (did it learn?)\n"
              f"[teach] I now know: {classes}\n", flush=True)
        log.write(json.dumps({"time": time.time(), "event": "learned", "reason": reason,
                              "new_classes": new_names, "examples": len(examples),
                              "seconds": round(secs, 1), "train_accuracy": train_acc,
                              "old_test_before": before_own, "old_test_after": after_own,
                              "held_back": held_txt, "classes": classes, "model": str(out)}) + "\n")
        drain_keyboard()

    def handle(answer: tuple, clips: list, frames: list, ear: str, conf: float, by: str) -> None:
        action, name = answer
        answers[action] += 1
        log.write(json.dumps({"time": time.time(), "event": "answer", "by": by, "action": action,
                              "name": name, "ear": ear, "conf": conf, "ticks": len(clips)}) + "\n")
        if action == "skip":
            return
        if action == "new":
            clips, frames = collect(name, clips, frames)
            if eye:
                eye.learn(name, frames)
            for i, f in enumerate(frames[:6]):
                if f is not None and eye:
                    eye.cv2.imwrite(str(run / "photos" / f"{time.strftime('%H%M%S')}_{name}_{i}.jpg"), f)
            learn([(c, name) for c in clips], "new class")
            return
        if eye:
            eye.learn(name, frames)                       # the eye learns from every named photo
        pending.extend((c, name) for c in clips)
        if action == "fix":
            print(f"  noted: that was '{name}', not '{ear}'.")
        if len(pending) >= args.retrain_every:
            learn(list(pending), "named sounds")
            pending.clear()

    try:
        while True:
            t0 = time.perf_counter()
            a, r, f = tick()
            loud = r >= gate
            if args.mode == "quiz":
                if not loud:
                    print(".", end="", flush=True)
                else:
                    # listen at least --min-event-seconds, then until 2 quiet ticks in a row
                    clips, frames, quiet, t_start = [a], [f], 0, time.perf_counter()
                    while True:
                        held_for = time.perf_counter() - t_start
                        if held_for >= args.max_event_seconds:
                            break
                        if held_for >= args.min_event_seconds and quiet >= 2:
                            break
                        a2, r2, f2 = tick()
                        if r2 >= gate:
                            clips.append(a2); frames.append(f2); quiet = 0
                        else:
                            quiet += 1
                    ear, top, conf = ear_says(clips)
                    seen, sim = eye_says(frames)
                    heard[ear] += 1
                    if seen and seen != ear and known(seen):
                        suggestion = seen                       # the eye corrects the ear
                    else:
                        suggestion = ear if ear != "unknown" else None
                    print(f"\n  [listened {time.perf_counter() - t_start:.1f} s, "
                          f"{len(clips)} loud half-seconds]", end="")
                    handle(ask_label(ear, conf, seen, sim, suggestion, classes),
                           clips, frames, ear, conf, "human")
            else:
                ear, top, conf = ear_says([a])
                seen, sim = eye_says([f])
                heard[ear] += 1
                eye_txt = (f"   eye: {seen} {sim:.2f} ({EYE_SRC['v']})" if seen else
                           "   eye: cannot see" if sim < 0 else "")
                print(f"{ear:<12} {conf:.2f} {'#' * int(conf * 20):<20} rms {r:.3f}{eye_txt}", flush=True)
                if loud:
                    recent.append((a, f, ear, conf))
                if loud and ear == "unknown":
                    streak += 1
                elif ear != "unknown":
                    streak = 0
                # the eye teaches the ear. Never with YOLO's "person": a person is in view
                # whenever someone makes a sound, so it would mislabel the sound of the
                # thing in their hand (the ringtone-labelled-"person" lesson of 5 Nov).
                yolo_person = EYE_SRC["v"] == "YOLO" and seen == "person"
                if args.auto and loud and seen and seen != ear and not yolo_person and not known(seen):
                    print(f"  (the camera sees '{seen}', but that is not one of my sounds: ignored)")
                elif args.auto and loud and seen and seen != ear and not yolo_person:
                    print(f"  -> the camera sees '{seen}': I use that as the label (no human).")
                    handle(("fix", seen), [a], [f], ear, conf, "eye")
                elif enter_pressed() and recent:                     # "that was wrong"
                    clip, frame, g, c = recent[-1]
                    print("\n[teach] Paused. About the last loud sound:")
                    handle(ask_label(g, c, *eye_says([frame]), None, classes), [clip], [frame], g, c, "human")
                    recent.clear(); streak = 0
                elif streak >= args.novelty_ticks and recent and not args.auto:
                    clip, frame, g, c = recent[-1]
                    print("\n[teach] I keep hearing a sound I don't know.")
                    handle(ask_label(g, c, *eye_says([frame]), None, classes), [clip], [frame], g, c, "human")
                    recent.clear(); streak = 0
            log.write(json.dumps({"time": time.time(), "event": "tick", "rms": r}) + "\n")
            time.sleep(max(0.0, args.tick_ms / 1000.0 - (time.perf_counter() - t0)))
    except (KeyboardInterrupt, EOFError):
        print("\n[teach] stopping.", flush=True)

    if pending:
        print(f"[teach] learning your last {len(pending)} named sounds before I stop "
              f"(Ctrl-C again to skip) ...", flush=True)
        try:
            learn(list(pending), "named sounds at the end")
        except KeyboardInterrupt:
            print("[teach] skipped.", flush=True)
    if helper:
        helper.terminate()
    summary = {"heard": dict(heard), "answers": dict(answers), "final_classes": classes,
               "eye_knows": sorted(eye.examples) if eye else [], "versions": version}
    (run / "teach_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n[teach] stopped. Heard: {dict(heard)}. Answers: {dict(answers)}.\n"
          f"        Ear knows: {classes}\n        Eye knows: {summary['eye_knows']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
