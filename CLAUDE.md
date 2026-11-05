# CLAUDE.md - AI Assistant Context File

> **Purpose:** This file provides context for Claude (or other AI assistants) to understand
> this project quickly in future sessions. Read this first before helping with the project.
>
> **Last updated:** 5 November 2026 (after submission, during defense preparation).

---

## Project Summary (TL;DR)

This is a **Master's thesis** (M.Sc. ICE, BHT Berlin) on **cross-modal federated learning** at
the edge. A camera labels sounds, so a device can learn to recognise objects by their **sound**
alone. That helps when vision fails, for example in fog or darkness.

- **Submitted:** 5 October 2026. The submitted PDF is
  `docs/thesis_draft/Adnan_Anwar_Rajput_931043_Master_Thesis_FBVII.pdf` (81 pages).
- **Now:** live demo on 11 Nov 2026; defense around 18 Nov 2026 (date to be confirmed).
- **Branches:**
  - `main` is the code **as submitted**. Do not change it.
  - `demo-improvements` (public) holds the post-submission work: new recording tools and the
    live run of 5 Nov.

**There are three models:**
1. **YOLOv8n** (fixed, pre-trained): the *teacher*. It looks at the camera frame and gives a
   label ("car", "cell phone", "person"). It is never trained here.
2. **AudioCNN** (trained): the *student*. It learns to recognise the object from a 500 ms
   log-mel spectrogram. The A/B/C comparison uses this model.
3. **Fusion model** (trained): frozen MobileNetV3-Small for the image plus the audio branch. It is
   used for the fusion experiments (blackout test) and for the federated fusion variants FB/FC.

---

## Critical Concepts

### 1. The labelling assumption (weak supervision)
Camera sees a phone → YOLOv8 says "cell phone" → the 500 ms of audio from the same moment is
labelled "cell phone". The labels are noisy. In the thesis, 85.1% of YOLO's detections were
correct (40 of 47 on VGGSound thumbnails).

### 2. Three strategies (plus two fusion variants)

| Strategy | On the device | Sent to the server | Server does |
|---|---|---|---|
| A: Centralized | (YOLO labels) | Raw audio + labels (zlib) | All training |
| B: Hybrid | STFT | Spectrograms + labels (zlib) | All training |
| C: Federated | STFT + local training | Model weights only | FedAvg (weighted average) |
| FB / FC | Fusion versions of B / C | Spectrogram + image feature vector / weights | Training / FedAvg |

### 3. How it really works (do not trust older notes)
- **Communication:** MQTT over a Mosquitto broker. QoS 1; the global model is a **retained**
  message.
  - **FedAvg is our own code** (`src/server/aggregation/fedavg.py`); **Flower is NOT used**. The
    old egg-info still lists `flwr`; ignore it.
  - **No TLS:** the LAN broker (`configs/mosquitto_lan.conf`) is anonymous, so only run it while
    training.
- **Topics:** `thesis/edge/{node_id}/data` (A/B), `thesis/edge/{node_id}/weights` (C),
  `thesis/server/model/global` (retained). Defined in `src/common/constants.py`.
- **Batching:** clients send batches (500 samples per message in the replay runs). The A/B servers
  retrain when the pool grows by 1,000 samples. On the live Pi the batches are small, around 30.
- **Model freshness (replay, Table 5.13):** A/B produce a new model about every 250 s
  (waiting for data); C about every 24 s. Ignore the old "12–48 hours" claim.

---

## Key Technical Details

### Audio processing (`src/edge/processing/stft.py`, same code on laptop and Pi)
- 16,000 Hz, 500 ms clip = 8,000 samples
- Window 25 ms (400 samples), hop 10 ms (160), FFT 512 → 257 bins (31.25 Hz apart)
- 128 mel bands, log scale (top_db 80), each clip scaled to 0–1
- Output: **(51, 128, 1)** with librosa's centre padding (1 + ⌊8000/160⌋ = 51)

### AudioCNN (`src/server/training/models/audio_cnn.py`)
- 3 × [Conv2D (32/64/128) + BatchNorm + ReLU + MaxPool] → GlobalAveragePooling (128) → Dense 128 → softmax
- **111,370 parameters** (110,080 + 129 × classes for 10 classes)
- Baseline on UrbanSound8K (train folds 1–8, val 9, test 10): **64.9%** accuracy
  (the old "68.6%" is outdated)

### Fusion model (`src/server/training/models/fusion_model.py`)
- Vision: MobileNetV3-Small (ImageNet, frozen), 576 → Dense 256
- Audio: the same CNN branch → 128
- Head: Concat (384) → Dense 256 → Dropout → Dense 128 → Dropout → softmax
- About 1.33 M parameters, 389,445 trainable

### Hardware
- Edge: **Raspberry Pi 5, 8 GB** (Pi 1, user `adi6034`). The camera is a USB webcam, and its
  microphone is broken. Use the USB microphone (sounddevice index 2). Pi 2 is being set up.
- Server: MacBook Air M1, 8 GB (not AWS).
- Software: Python 3.11/3.13, TensorFlow 2.21, librosa 0.11, Ultralytics 8.4, paho-mqtt 2.1,
  Mosquitto 2.1.2.

---

## Results to quote (from the submitted thesis)

| What | A | B | C |
|---|---|---|---|
| Replay upload per node (10 rounds) | 69.2 MB | 51.0 MB (−26.2%) | 4.18 MB |
| Replay final accuracy (start 0.649) | 0.661 | 0.688 | 0.676 (local LR 5e-5) |
| Device compute | 0 | 11.3 s | 233 s |
| Live Pi upload (Table 5.7) | 1,337 KB | 979 KB | 1,245 KB |
| Live Pi accuracy (32 test clips) | 0.531 | 0.531 | 0.563 |

- **Break-even for C:** about 30 samples per round. Below that, sending weights costs more than
  sending data.
- **Fusion, blackout test:** recipe model 0.718 clean / 0.398 with the camera blacked out.
  FB final 0.845 / 0.583; FC final 0.854 / 0.573.

**Units:** the thesis uses 1 KB = 1,000 bytes. The Pi scripts print KiB (1,024 bytes).

### Known issues in the submitted text (prepare answers, do not hide them)
- The A/B servers resumed a saved Adam optimiser at learning rate 1.25e-4, not 1e-3 as
  Section 5.4 says.
- The thesis live run compares with chance (0.33), but the majority baseline was 0.44.
- Frames without padding: the thesis says 48; librosa with `center=False` gives 47.
- FC is sometimes quoted with its best round (0.883 / 0.670) instead of the final one.

### Outdated documents
`docs/RESULTS.md`, `REVISION.md`, `PRIMER.md`, `DEFENSE_QA.md`, `FUSION.md`, `PRESENTATION.md`
and `THESIS_BLUEPRINT.md` contain older numbers (68.6%, −36%, "minutes vs hours"). Quote only
the thesis PDF or this file.

---

## Post-submission work (branch `demo-improvements`)

- `src/experiments/pi_record.py`:
  - records every tick (wav + jpg + `ticks.jsonl` with all detections and loudness);
  - `--target` shows a live GOOD counter;
  - `--min-rms auto` measures the room for 3 s and sets the loudness gate to 1.5 × the room
    noise (saved in `room.json`).
- `scripts/build_dataset.py`:
  - chooses the labels after recording, from one or more sessions;
  - uses a priority list and skips ticks where an earlier class is only half visible;
  - has an automatic loudness gate and a background class;
  - per-session `relabel.csv` files hold the human check.
- **Live run of 5 Nov 2026** (one Pi):
  - Dataset: 205 clips (cell phone 79, person 74, background 52), 154 train / 51 test;
    majority baseline 0.39.
  - Accuracy: A 0.941, B 0.922, C 0.941.
  - Upload: 1,930 / 1,615 / 1,245 KB (thesis units).
  - Live listening with model C: background 75 / cell phone 35 / person 6 / unknown 2 of 118 ticks.
  - Recordings, photos and results are **public on purpose**: `data/pi_captures/`
    (see its README) and `data/pi_captures/results/live_nov5/`.
  - New figures: `docs/figures/20_live_signals_nov5.png` and `21_live_bursts_nov5.png`
    (`scripts/make_signals_figures.py`).

### Next steps
1. Set up Pi 2 (Wi-Fi, venv, `PI_HOST=pi2 ./scripts/sync_to_pi.sh`, copy the model and the
   dataset).
2. Even split (IID): `--shard 0/2` and `--shard 1/2`; the C server with `--num-clients 2`.
3. Uneven split (non-IID): needs a **new class-skewed shard option** in the Pi clients
   (not written yet).

---

## Project Structure (actual)

```
src/experiments/    # What actually runs: strategy_{a,b,c,fb,fc}_{client,server}.py (replay),
                    #   strategy_{a,b,c}_pi_client.py (live Pi), pi_record / pi_enroll / pi_live / pi_infer
src/edge/           # processing/stft.py, processing/vision.py (YOLO), sensors/, communication/mqtt_client.py
src/server/         # training/models/{audio_cnn,fusion_model}.py, aggregation/fedavg.py
src/common/         # constants.py (MQTT topics, audio settings), config.py, utils.py
scripts/            # data prep, training, build_dataset.py, make_thesis_figures.py, sync_to_pi.sh, test_*.py
configs/            # mosquitto_lan.conf (LAN broker), edge/server/strategies YAML
models/             # audio_cnn_urbansound8k.keras (baseline), fusion models, yolov8n.pt
results/            # run outputs; results/demo/ = the THESIS September models: never overwrite
data/pi_captures/   # live recordings of 5 Nov 2026 (public)
docs/thesis_draft/  # thesis chapters (markdown) and the submitted PDF
docs/figures/       # thesis figures (PNG); diagram sources in docs/figures/src/*.mmd
```

---

## How to run a live round (start order: broker → server → Pi)

```bash
# Mac, terminal 1: broker (anonymous; only while training)
mosquitto -c configs/mosquitto_lan.conf
# Mac, terminal 2: server (example: Strategy C, two Pis)
.venv/bin/python -m src.experiments.strategy_c_server --broker localhost --num-rounds 3 \
    --num-clients 2 --epochs-per-round 8 --live-dataset data/pi_captures/live_dataset.npz \
    --save-model results/demo_day/trained.keras --run-dir results/demo_day/strategy_c
# Pi: client
.venv/bin/python -m src.experiments.strategy_c_pi_client --node-id A --broker <LAPTOP_IP> \
    --dataset results/live/live_dataset.npz --shard 0/2 --run-dir results/live/strategy_c
```

- "Connection refused" means the broker is not running.
- Save demo runs in `results/demo_day/`, never in `results/demo/`.

---

## Testing

- The real checks are the scripts: `scripts/test_stft.py`, `test_audio_cnn.py`,
  `test_fusion_model.py`, `test_yolo.py`.
- `tests/test_basic.py` is old scaffolding. Five of its ten tests fail because they expect
  classes that no longer exist and 49 frames.
- To run it anyway: `.venv/bin/python -m pytest -o addopts="" tests`. pytest-cov is not
  installed, so the coverage options in `pyproject.toml` must be overridden.

---

## Debugging Tips

- Pi camera: USB webcam (not `libcamera`). Microphone: `arecord -l`; in Python, sounddevice
  index 2.
- MQTT traffic: `mosquitto_sub -h <LAPTOP_IP> -t "thesis/#" -v`
- A quiet lab has an rms of about 0.007–0.009. The automatic gate sets the threshold at 1.5× the
  room noise.
- `*.wav` and `data/pi_captures/results/*` are git-ignored by default (with exceptions for the
  5 Nov data). Run `git check-ignore` before claiming that a file is committed.

---

## Important Constraints

1. **Raspberry Pi 5 (8 GB):** full TensorFlow runs. YOLO takes 318 ms per frame on the Pi, and
   training C on the Pi takes about 17–28 s per round.
2. **Privacy is the core concern:** in C the audio never leaves the device; in A raw audio does.
3. **Honesty:** the thesis (Section 3.9) states that AI tools were used for programming.

---

## Contact / Supervision

- **Author:** Adnan Anwar Rajput
- **First Supervisor:** Prof. Dr. André Jakob
- **Second Supervisor:** Prof. Dr. Marcus Purat
- **Institution:** Berliner Hochschule Für Technik, Department VII
