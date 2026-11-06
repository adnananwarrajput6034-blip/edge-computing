"""Step 3 of 6 Nov 2026: same total training (40 local epochs per Pi), split into
5 rounds x 8, 10 rounds x 4 or 40 rounds x 1. Plots test accuracy against the
epochs each Pi has trained, so the runs share one x-axis, and prints a pandas
summary (final round and the mean over the second half of training).

  .venv/bin/python scripts/compare_local_epochs.py --out results/experiments_20261106/local_epochs
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

INK, INK2, GRID = "#1f2937", "#4b5563", "#e5e7eb"
# (label, local epochs per round, local learning rate, colour, runs)
SETS = [
    ("8 epochs × 5 rounds, LR 1e-3", 8, "1e-3", "#2a78d6",
     ["results/two_pi_20261105/c_two_rooms", "results/experiments_20261106/rooms_rep1",
      "results/experiments_20261106/rooms_rep2", "results/experiments_20261106/rooms_rep3"]),
    ("4 epochs × 10 rounds, LR 1e-3", 4, "1e-3", "#eb6834", ["results/experiments_20261106/rooms_e4_rep1"]),
    ("1 epoch × 40 rounds, LR 1e-3", 1, "1e-3", "#1baf7a", ["results/experiments_20261106/rooms_e1_rep1"]),
    ("8 epochs × 5 rounds, LR 1e-4", 8, "1e-4", "#8a3ffc",
     ["results/experiments_20261106/rooms_lr1e4_rep1", "results/experiments_20261106/rooms_lr1e4_rep2"]),
]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    a = p.parse_args()
    rows = []
    for name, e, lr, _, runs in SETS:
        for run in runs:
            path = Path(run) / "server.json"
            if not path.exists():
                continue
            s = json.loads(path.read_text())
            for r in s["rounds"]:
                rows.append({"setting": name, "local_epochs": e, "lr": lr, "run": Path(run).name, "round": r["round"],
                             "epochs_trained": r["round"] * e, "test_accuracy": r["test_accuracy"],
                             "upload_kib_per_pi": r["round"] * 406})
    df = pd.DataFrame(rows)
    second_half = df[df.epochs_trained > 20]
    g = df.groupby("setting")
    summary = pd.DataFrame({
        "runs": g["run"].nunique(),
        "rounds": g["round"].max(),
        "final_mean": df[df.epochs_trained == 40].groupby("setting")["test_accuracy"].mean(),
        "second_half_mean": second_half.groupby("setting")["test_accuracy"].mean(),
        "second_half_sd": second_half.groupby("setting")["test_accuracy"].std(),
        "upload_MB_per_pi": g["round"].max() * 406 * 1024 / 1e6,
    }).round(3).reindex([n for n, *_ in SETS])
    print(summary.to_string())
    summary.to_csv(f"{a.out}_summary.csv")
    df.to_csv(f"{a.out}_rounds.csv", index=False)

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9.5, "axes.edgecolor": INK2,
                         "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                         "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                         "axes.axisbelow": True, "grid.color": GRID, "grid.linewidth": 0.8})
    fig, ax = plt.subplots(figsize=(11, 6.0))
    for name, e, lr, col, runs in SETS:
        d = df[df.setting == name]
        many = d["run"].nunique() > 1
        for run, gr in d.groupby("run"):
            ax.plot(gr.epochs_trained, gr.test_accuracy, color=col, lw=1.1 if many else 1.8,
                    alpha=0.35 if many else 0.9, marker="o", ms=3.5 if e == 1 else 5)
        if many:
            m = d.groupby("epochs_trained")["test_accuracy"].mean()
            ax.plot(m.index, m.values, color=col, lw=2.6, marker="o", ms=7, mec="white", mew=2)
        s_ = summary.loc[name]
        ax.plot([], [], color=col, lw=2.4, label=f"{name} ({int(s_['runs'])} run{'s' if s_['runs'] > 1 else ''}"
                f"{', thick = mean' if many else ''}): 2nd-half mean {s_['second_half_mean']:.2f}, "
                f"upload {s_['upload_MB_per_pi']:.1f} MB per Pi")
    ax.axvspan(20, 40.5, color=GRID, alpha=0.35, zorder=0)
    ax.text(30, 0.925, "second half of training (used for the mean)", ha="center", color=INK2, fontsize=8.5)
    ax.set_xlim(0, 41); ax.set_ylim(0.2, 0.95)
    ax.set_xlabel("epochs trained by each Pi (same total: 40)"); ax.set_ylabel("test accuracy, both rooms (103 clips)")
    ax.set_title("Two ways to reduce client drift: fewer local epochs (meet more often) or a smaller learning rate (smaller steps)",
                 loc="left", color=INK, fontsize=11)
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.13), frameon=False, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(f"{a.out}.png", dpi=170)
    print(f"wrote {a.out}.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
