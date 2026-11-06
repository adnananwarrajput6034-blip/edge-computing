# Experiment day, 6 Nov 2026 (post-submission)

**Setup:**
- Two Raspberry Pi 5 boards with the two-room data of 5 Nov: Pi 1 = room 1 (cell phone, person,
  background); Pi 2 = room 2 (bottle, laptop, background).
- One shared class list.
- Test set: both rooms' 103 test clips; majority baseline 0.32.
- Strategy C (FedAvg). Every run here has **40 local epochs per Pi** in total.
- `laptop/` holds the server side, `pi1/` and `pi2/` the client side.

## Step 1: live graphs
- Each Pi publishes one progress message per local epoch (`thesis/edge/<id>/metrics`).
- The server writes them, plus its own test accuracy after every round, to
  `<run>/tensorboard/` and `<run>/epochs.jsonl`.
- To view: `.venv/bin/tensorboard --logdir data/pi_captures/results/experiments_20261106/laptop`.

## Step 2: repeats (8 local epochs × 5 rounds, learning rate 1e-3)
- **Federated final accuracy:** 0.777 (5 Nov) / 0.748 / 0.728 / 0.408 (`rooms_rep1..3`).
  Mean 0.665, sd 0.17.
- **References** (`baselines_seed0..2.json`, 3 seeds each, 40 epochs on the Mac):
  - Pi 1 alone: 0.667
  - Pi 2 alone: 0.589
  - all data in one place: 0.858
- **Correction:** with one run, federated looked better than each Pi alone. With four runs, its
  mean equals Pi 1 alone and its spread is large. It does learn all five classes, while each
  Pi alone cannot.
- Files: `repeats.png`, `repeats_per_round.csv`, `repeats_summary.csv`
  (`scripts/compare_runs.py`, pandas).

## Step 3: client drift remedies (same 40 local epochs per Pi)
| Setting | Runs | Mean, second half of training | sd | Upload per Pi |
|---|---|---|---|---|
| 8 epochs × 5 rounds, LR 1e-3 | 4 | 0.665 | 0.119 | 2.1 MB |
| 4 epochs × 10 rounds, LR 1e-3 (`rooms_e4_rep1`) | 1 | 0.689 | 0.071 | 4.2 MB |
| 1 epoch × 40 rounds, LR 1e-3 (`rooms_e1_rep1`) | 1 | 0.758 | 0.073 | 16.6 MB |
| 8 epochs × 5 rounds, **LR 1e-4** (`rooms_lr1e4_rep1..2`) | 2 | 0.723 (finals 0.728, 0.816) | 0.050 | 2.1 MB |

- Fewer local epochs (the Pis meet more often) or a smaller local learning rate (smaller
  steps) both reduce client drift.
- The smaller learning rate costs no extra communication.
- Files: `local_epochs.png`, `local_epochs_summary.csv`, `local_epochs_rounds.csv`
  (`scripts/compare_local_epochs.py`).
- **Caveat:** one or two runs per setting.
