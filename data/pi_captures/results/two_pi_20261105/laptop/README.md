# Two-Pi runs, 5 Nov 2026 (after submission)

These runs used Pi 1 (141.64.161.178) and Pi 2 (141.64.161.142), both Raspberry Pi 5 boards on
the BHT "Devices" Wi-Fi. The server was the MacBook Air (141.64.161.205).

**Setup:**
- Strategy C: 3 rounds × 8 local epochs, local LR 1e-3.
- Data: the 5 Nov dataset (`live_dataset_NOV5.npz`), split evenly: Pi 1 `--shard 0/2`, Pi 2
  `--shard 1/2`, 77 clips each with no overlap.
- Test set: 51 clips on the server; majority baseline 0.39.

| Folder | What | Test accuracy per round | Valid? |
|---|---|---|---|
| `c_iid` | Both Pis use the same class order | 0.863 → 0.941 → **0.980** | yes |
| `c_swapped_unchecked` | Pi 2 numbers its classes `background, person, cell phone`; server check off (`--no-class-check`) | 0.706 → 0.529 → **0.647** | yes |
| `c_swapped_checked` | The same swapped Pi 2, server check on | server **refused** Pi 2 in round 1; stopped after 70 s | yes |
| `_failed_quoting_c_swapped_unchecked` | Quote marks reached Pi 2 as part of the class names; Pi 2 rejected its own `--class-order` | no rounds | no (operator error) |
| `_contaminated_two_servers_c_swapped_unchecked` | The server from the failed run was still alive and reconnected to the new broker. Two servers published global models, so the Pis trained some rounds twice. | 0.863 → 0.647 → 0.627 | **no** |

**What the runs show:**
- **Label order is a silent failure.** FedAvg averages output neurons by position. With one Pi
  using a different class order, nothing crashes, but accuracy falls from 0.980 to 0.647.
  - Background has the same position on both Pis, so its 20 test clips stay correct.
  - Cell phone and person (31 clips) get mixed, which predicts about (20 + 31/2) / 51 ≈ 0.70.
- **The fix:**
  - Each client sends its class list with its weights (`classes` in the metadata).
  - The server refuses a mismatch (`c_swapped_checked`).
- **Second fix, from the contaminated run:**
  - The server tags every global model with a `run_id`. A client follows one run and ignores
    any other server.
  - A client also skips a round it has already trained.
- **Bug found while setting up Pi 2:**
  - `scripts/sync_to_pi.sh` excluded every folder named `models/`, including the code folder
    `src/server/training/models/`.
  - Pi 1 only worked because it had an old copy. The excludes are now anchored to the top level.

**Caveat:** one run per condition, 51 test clips. The difference between 0.980 and yesterday's
one-Pi 0.941 is two clips and is not evidence that two Pis are better.

## Two rooms (non-IID), 5 Nov 2026 evening

- **Pi 1 = room 1:** the 5 Nov morning recordings (background 39, cell phone 59, person 56 train clips).
- **Pi 2 = room 2:** new recordings with Pi 2 and the moved USB mic (background 61, bottle 61, laptop 38).
  - bottle: 81 clips **labelled by the author from the photos** (relabel.csv). YOLO could not
    recognise the crumpled, moving, see-through bottle (it said vase, train or nothing).
  - laptop: 50 clips labelled by YOLO.
  - background: 81 clips.
  - Session `session_20261105_205535` was left out: the sound came from a phone video while a
    still bottle stood in view, so the camera's label did not match the sound source.
- **Shared class list** (`scripts/merge_vocab.py`): background, bottle, cell phone, laptop, person.
- **Test set:** both rooms' test clips together, 103 clips; majority baseline 0.32.
- **Strategy C:** 5 rounds × 8 local epochs, LR 1e-3. Each Pi sent 406 KiB per round.

**Federated, per round:** 0.515 → 0.641 → 0.699 → 0.641 → **0.777**. The dip in round 4 is client drift.

**Reference models** (`scripts/two_rooms_baselines.py`, same model and 40 epochs, one seed):

| Model | All 103 | Room 1 | Room 2 | bottle | cell phone | laptop |
|---|---|---|---|---|---|---|
| Pi 1 alone | 0.670 | 0.961 | 0.385 | 0.00 | 0.90 | 0.00 |
| Pi 2 alone | 0.602 | 0.255 | 0.942 | 0.90 | 0.00 | 0.92 |
| **Federated, two Pis** | **0.777** | 0.725 | 0.827 | 0.80 | 0.45 | 0.67 |
| Centralized, all data | 0.845 | 0.902 | 0.788 | 0.75 | 0.80 | 0.50 |

**Reading:**
- Federated learning let each Pi learn the other room's classes without receiving any of its
  audio. Pi 1 alone scores 0.00 on bottle; the shared model scores 0.80.
- It stays below the centralized model (0.777 vs 0.845). That gap is the price of non-IID
  data with no raw-data sharing.
- **Caveat:** one run per model; only 12 laptop and 20 cell phone test clips.
