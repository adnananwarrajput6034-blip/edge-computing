# Live teaching on Pi 2, 6 Nov 2026 (post-submission prototype)

`src/experiments/pi_teach.py`:
- **The ear** (AudioCNN) listens.
- **A human** names sounds it does not know. A new name grows the output layer by one class
  (class-incremental learning).
- **The eye** (MobileNetV3-Small fingerprints, and YOLO in a separate process with `--yolo`)
  can suggest names or, with `--auto`, label sounds itself.
- Retraining replays the Pi's stored clips plus all named clips, and trains the last two
  Dense layers with class weighting.
- Each folder holds:
  - `clips/`: named 0.5 s clips;
  - `photos/`: photos of named things;
  - `models/`: every ear version, plus the eye's fingerprints;
  - `teach_log.jsonl`: every tick, answer and learning step;
  - `teach_summary.json`.

| Folder | Version of the tool | What happened |
|---|---|---|
| `try1_typing` | Typing while it printed every second | Typed letters merged: two junk classes (`kkeyskkkeykekeys`, `keykeyskeys`); knocking keys was loud in only 5 of 20 ticks. Led to quiz mode. |
| `try2_quiz` | Quiz mode, one question per 0.5 s | "key" learned: held-back key clips 1 → 2 → 3 of 3 right; old test clips 0.81 → 0.94. A class `no` was created by accident (typed `exit`, then `no`); `q`/`exit` now quit. |
| `try3_quiz_eye` | Quiz mode with the eye, events of ≥ 3–5 s, 20 examples | Eye: `key 0.83–0.97`. Ear: held-back key clips 4 → 5 of 5 right; old test clips 0.81 → 0.88–0.96. The eye corrected the ear several times (`Ear: bottle 66% · Eye: key 0.91`). Source of `try4`'s starting model (`models/model_v7.keras`). |
| `try4_listen_auto_yolo` | Listen + auto + live YOLO helper, eye taught first | Eye taught `bottle`, `mobile phone` and `suicase` (a typo of suitcase). YOLO mostly saw `person` when the author was in view. One automatic label (key → bottle, the eye's call). Lessons: synonyms (`mobile phone` vs the existing `cell phone`) and typos become separate names, so a real system needs a controlled vocabulary. YOLO's `person` is never used as an automatic label. |

**Lessons for the defense:**
- The eye learns a new object from a few photos; the ear needs more examples (key vs bottle
  sounds alike).
- Retraining on the Pi takes 7–9 s and old sounds were not forgotten.
- A confidently wrong ear (`bottle 0.94` for keys) never asks by itself, so typed corrections
  and the eye are needed.
- When several things are in view, the camera cannot know which one made the sound.
