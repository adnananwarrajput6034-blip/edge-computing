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
C 0.941; sent by the Pi 1,885 / 1,577 / 1,216 KB. Live inference with model C (microphone
only, 118 ticks): background 75, cell phone 35, person 6, unknown 2.
