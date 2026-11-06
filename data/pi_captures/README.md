# Pi Captures

Scratch area for images and audio pulled off the Raspberry Pi 5 edge node (`pi1`)
during hardware bring-up and sensor testing.

The raw dumps (`images/`, `audio/`, `evidence/`) are **not tracked in git** — they
are regenerable at any time by re-running a capture against the Pi. The live
experiment of 5 Nov 2026 (last section) **is** tracked, on purpose.

```
images/   webcam frames (JPEG, /dev/video0, Sonix 1080P FHD)
audio/    microphone recordings (WAV 16 kHz mono, plughw:2,0, Generalplus USB)
```

## Hardware

| | |
|---|---|
| Host | `pi1` (alias in `~/.ssh/config`, key auth, user `adi6034`) |
| Camera | `/dev/video0` — MJPG/YUYV at 1920x1080, 1280x720, 640x480 |
| Microphone | ALSA card 2, `plughw:2,0` |

The webcam exposes its own onboard mic as ALSA card 3. Always address the real
USB mic explicitly as `plughw:2,0` (or by name, `plughw:Device,0`) — card
numbering can swap across reboots depending on USB enumeration order.

## Capturing

```bash
scripts/pull_from_pi.sh          # one frame + 3 s of audio
scripts/pull_from_pi.sh -d 10    # 10 s of audio
```

Files are timestamped, so repeated runs accumulate rather than overwrite.

## Live experiment of 5 Nov 2026 (post-submission, tracked on purpose)

Recorded with `src/experiments/pi_record.py` (every tick is kept) and turned into a
dataset with `scripts/build_dataset.py`. Each tick is `tickNNNN.wav` (0.5 s, 16 kHz),
`tickNNNN.jpg` (the camera frame) and one line in `ticks.jsonl` (loudness and all
YOLO detections).

| Folder | What it is | In the dataset? |
|---|---|---|
| `session_20261105_002237/` | 1st recording: talking, then the phone (the ringtone was mostly too quiet) | yes |
| `session_20261105_004018/` | 2nd: talking close to the mic (ticks 1–90); phone held too close to the camera (91–176) | ticks 1–90 |
| `session_20261105_011731/` | 3rd: timer alarm; phone sideways and too close ("traffic light", "clock") | no (failed) |
| `session_20261105_013159/` | 4th: alarm not heard; phone sideways again ("tv", "parking meter") | no (failed) |
| `session_20261105_015624/` | 5th: phone on a fixed stand playing ringtones; room noise measured (`room.json`) | yes, except ticks 99–100 |
| `removed_004018/`, `removed_015624/` | audio of the ticks taken out after the human check (their photos stay in the session folders) | no |
| `live_dataset.build.json` | rules and counts of the final dataset: cell phone 79, person 74, background 52 | — |
| `results/live_nov5/` | `laptop/`: server reports and trained models of A, B, C; `pi/`: Pi reports (bytes sent, STFT and training time) and the live inference log | — |

Rebuild the dataset (the removed ticks must stay out of the session folders):

```bash
.venv/bin/python scripts/build_dataset.py data/pi_captures/session_20261105_002237 \
    data/pi_captures/session_20261105_004018 data/pi_captures/session_20261105_015624 \
    --classes "cell phone,person" --min-conf 0.5 --background --min-samples 20
```

Results with one Pi (154 training and 51 test clips): test accuracy A 0.941, B 0.922,
C 0.941; sent by the Pi 1,930 / 1,615 / 1,245 KB (KB = 1,000 bytes, as in the thesis; the
Pi's scripts print KiB = 1,024 bytes, so their output shows 1,885 / 1,577 / 1,216). Live inference with model C (microphone
only, 118 ticks): background 75, cell phone 35, person 6, unknown 2.

## Two Pis and a second room, evening of 5 Nov 2026 (post-submission)

Pi 2 (also a Raspberry Pi 5) joined, and the one working USB microphone was moved to it. The
built-in microphones of both webcams fail on Linux ("cannot get freq"), so recordings are made in
turns; training needs no microphone.

| Folder | What | Used? |
|---|---|---|
| `session_20261105_205119` | Bottle shaken close to the camera; YOLO mostly said vase or person | yes, 53 ticks named **bottle by the author** (`relabel.csv`) |
| `session_20261105_205535` | A still bottle while a **phone video** played bottle sounds | **no**: the visible object was not the sound source |
| `session_20261105_211519` | Bottle swirled and rolled; camera lying flat, YOLO saw nothing or vase | yes, 28 ticks named bottle by the author |
| `session_20261105_212056`, `_212641` | Laptop playing music (YOLO: laptop) | yes, 50 ticks labelled by YOLO |
| `session_20261105_212851` | One minute of a quiet room | background |

Room-2 dataset (rebuild it; `.npz` files are not tracked):

```bash
.venv/bin/python scripts/build_dataset.py data/pi_captures/session_20261105_205119 \
    data/pi_captures/session_20261105_211519 data/pi_captures/session_20261105_212056 \
    data/pi_captures/session_20261105_212641 data/pi_captures/session_20261105_212851 \
    --classes "bottle,laptop" --min-conf 0.5 --background --min-samples 20 \
    --out data/pi_captures/room2_dataset.npz
.venv/bin/python scripts/merge_vocab.py data/pi_captures/live_dataset_NOV5.npz:room1 \
    data/pi_captures/room2_dataset.npz:room2 --out-dir data/pi_captures --union two_rooms_union.npz
```

All two-Pi runs (from the laptop and from each Pi), including one failed and one contaminated
run, are in `results/two_pi_20261105/`; its `laptop/README.md` explains each one.
- **Two rooms, federated:** 0.777 on 103 test clips.
- **Each Pi alone:** 0.670 / 0.602.
- **All data in one place:** 0.845.

On the Pis, the contaminated run and its clean rerun wrote to the same folder, so the Pi-side
files of `c_swapped_unchecked` belong to the clean rerun.
