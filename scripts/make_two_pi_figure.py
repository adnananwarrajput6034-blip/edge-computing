"""Figure 22: two Pis, two rooms - federated vs each Pi alone vs centralized vs relay.

Reads the tracked results in data/pi_captures/results/two_pi_20261105/laptop/ and writes
docs/figures/22_two_rooms_nov5.png (post-submission figure; not in the submitted thesis).

  .venv/bin/python scripts/make_two_pi_figure.py
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1]
RES = REPO / "data/pi_captures/results/two_pi_20261105/laptop"
OUT = REPO / "docs/figures/22_two_rooms_nov5.png"

INK, INK2, GRID, SURF = "#1f2937", "#4b5563", "#e5e7eb", "#ffffff"
FED, RELAY = "#2a78d6", "#eb6834"          # left panel: method identity
ROOM1, ROOM2 = "#008300", "#d55181"        # right panel: room identity (validated pair)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9.5, "axes.edgecolor": INK2,
                     "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "axes.axisbelow": True, "grid.color": GRID, "grid.linewidth": 0.8,
                     "figure.facecolor": SURF, "axes.facecolor": SURF})


def rounds(run):
    return [r["test_accuracy"] for r in json.loads((RES / run / "server.json").read_text())["rounds"]]


fed = rounds("c_two_rooms")
relay = rounds("relay_1_pi1_alone") + rounds("relay_2_pi2_after_pi1")
base = json.loads((RES / "c_two_rooms/baselines.json").read_text())
ev = json.loads((RES / "relay_2_pi2_after_pi1/evaluation.json").read_text())
central, majority = base["centralized_both_rooms"]["overall"], base["majority_baseline"]

fig, (a, b) = plt.subplots(1, 2, figsize=(12.5, 4.7), gridspec_kw={"width_ratios": [1.05, 1]})
fig.suptitle("Two Pis, two rooms (5–6 Nov 2026): accuracy on both rooms' 103 test clips",
             color=INK, fontsize=12, x=0.01, ha="left")

# left: accuracy per round
a.axhline(central, color=INK2, lw=1.2, ls=(0, (5, 3)))
a.text(10.3, central, f"all data on one computer {central:.3f}", color=INK2, va="center", ha="right",
       fontsize=8.5, bbox=dict(fc=SURF, ec="none", pad=1))
a.axhline(majority, color=INK2, lw=1, ls=(0, (1, 2)))
a.text(10.3, majority + 0.025, f"always guess the largest class {majority:.2f}", color=INK2, ha="right", fontsize=8.5)
a.axvline(5.5, color=GRID, lw=6, zorder=0)
a.text(5.5, 0.18, "relay: Pi 1 off,\nPi 2 on", color=INK2, ha="center", fontsize=8.5)
a.plot(range(1, 6), fed, color=FED, lw=2, marker="o", ms=7, mec=SURF, mew=2, zorder=3)
a.plot(range(1, 11), relay, color=RELAY, lw=2, marker="s", ms=6.5, mec=SURF, mew=2, zorder=3)
a.annotate(f"federated (both Pis every round)  {fed[-1]:.3f}", (5, fed[-1]), xytext=(0.9, 0.93),
           textcoords="data", color=INK, fontsize=9, arrowprops=dict(arrowstyle="-", color=FED, lw=1))
a.annotate(f"relay (one Pi at a time)  {relay[-1]:.3f}", (10, relay[-1]), xytext=(6.1, 0.74),
           textcoords="data", color=INK, fontsize=9, arrowprops=dict(arrowstyle="-", color=RELAY, lw=1))
a.set_xlim(0.5, 10.5); a.set_ylim(0.15, 1.0); a.set_xticks(range(1, 11))
a.set_xlabel("training round (8 local epochs per active Pi)"); a.set_ylabel("test accuracy")
a.set_title("Accuracy after every round", color=INK, fontsize=10.5, loc="left", pad=10)

# right: per-room accuracy of the final models
models = [("Pi 1\nalone", base["room1_only"]), ("Pi 2\nalone", base["room2_only"]),
          ("Federated\n(C, both)", base["federated_two_pis"]), ("Relay\n(Pi 1, then Pi 2)", ev["relay_after_pi2"]),
          ("All data on\none computer", base["centralized_both_rooms"])]
w, gap = 0.36, 0.03
for i, (name, m) in enumerate(models):
    for j, (key, col) in enumerate([("room1", ROOM1), ("room2", ROOM2)]):
        x = i + (j - 0.5) * (w + gap)
        b.bar(x, m[key], width=w, color=col, zorder=2)
        b.text(x, m[key] + 0.015, f"{m[key]:.2f}", ha="center", va="bottom", fontsize=8.5, color=INK)
b.set_xticks(range(len(models))); b.set_xticklabels([n for n, _ in models], fontsize=8.8)
b.set_ylim(0, 1.22); b.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0]); b.set_ylabel("test accuracy in that room")
b.bar(0, 0, color=ROOM1, label="room 1 clips (cell phone, person, background)")
b.bar(0, 0, color=ROOM2, label="room 2 clips (bottle, laptop, background)")
b.legend(loc="upper center", frameon=False, fontsize=8.3, ncol=2, bbox_to_anchor=(0.5, 1.0), handlelength=1.2, columnspacing=1.2)
b.set_title("Each final model, per room", color=INK, fontsize=10.5, loc="left", pad=10)
b.grid(axis="x", visible=False)

fig.text(0.01, 0.005, "One run per model. Room 1 = Pi 1's recordings, room 2 = Pi 2's. "
         "Relay forgets room 1: cell phone and person fall from 1.00 to 0.00.", color=INK2, fontsize=8.5)
fig.tight_layout(rect=(0, 0.03, 1, 0.95))
fig.savefig(OUT, dpi=170)
print(f"wrote {OUT}")
