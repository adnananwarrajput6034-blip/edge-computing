"""Put repeated runs into one pandas table and one chart (6 Nov 2026).

Federated runs: every server.json given with --fed. Reference models: every
baselines_seed*.json given with --base (from scripts/two_rooms_baselines.py).

  .venv/bin/python scripts/compare_runs.py \
      --fed results/two_pi_20261105/c_two_rooms results/experiments_20261106/rooms_rep1 \
            results/experiments_20261106/rooms_rep2 results/experiments_20261106/rooms_rep3 \
      --base results/experiments_20261106/baselines_seed0.json \
             results/experiments_20261106/baselines_seed1.json \
             results/experiments_20261106/baselines_seed2.json \
      --out results/experiments_20261106/repeats
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

INK, INK2, GRID, FED = "#1f2937", "#4b5563", "#e5e7eb", "#2a78d6"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--fed", nargs="+", required=True)
    p.add_argument("--base", nargs="+", default=[])
    p.add_argument("--out", required=True)
    a = p.parse_args()

    # one row per (run, round)
    rows = []
    for run in a.fed:
        s = json.loads((Path(run) / "server.json").read_text())
        for r in s["rounds"]:
            rows.append({"run": Path(run).name, "round": r["round"], "test_accuracy": r["test_accuracy"]})
    rounds = pd.DataFrame(rows)
    per_round = rounds.pivot(index="round", columns="run", values="test_accuracy")

    # one row per (method, seed/run) with the final accuracy
    finals = [{"method": "Federated (both Pis)", "run": k, "final": v.dropna().iloc[-1]}
              for k, v in per_round.items()]
    names = {"room1_only": "Pi 1 alone", "room2_only": "Pi 2 alone",
             "centralized_both_rooms": "All data on one computer"}
    for f in a.base:
        b = json.loads(Path(f).read_text())
        for key, name in names.items():
            finals.append({"method": name, "run": f"seed {b['seed']}", "final": b[key]["overall"]})
    finals = pd.DataFrame(finals)
    summary = finals.groupby("method")["final"].agg(["count", "mean", "std", "min", "max"]).round(3)

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    print("Test accuracy per round (columns = runs):\n", per_round.round(3).to_string(), "\n")
    print("Final accuracy per method:\n", summary.to_string())
    per_round.round(4).to_csv(f"{out}_per_round.csv")
    summary.to_csv(f"{out}_summary.csv")

    # chart
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9.5, "axes.edgecolor": INK2,
                         "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                         "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                         "axes.axisbelow": True, "grid.color": GRID, "grid.linewidth": 0.8})
    fig, (l, r) = plt.subplots(1, 2, figsize=(12.5, 4.6), gridspec_kw={"width_ratios": [1, 1.1]})
    fig.suptitle("Same experiment, repeated: two Pis, two rooms (103 test clips)", color=INK,
                 fontsize=12, x=0.01, ha="left")
    for run, col in per_round.items():
        l.plot(per_round.index, col, color=FED, lw=1.2, alpha=0.45, marker="o", ms=4)
    mean = per_round.mean(axis=1)
    l.plot(mean.index, mean, color=FED, lw=2.6, marker="o", ms=7, mec="white", mew=2,
           label=f"federated, mean of {per_round.shape[1]} runs")
    c = summary.loc["All data on one computer", "mean"]
    l.axhline(c, color=INK2, lw=1.2, ls=(0, (5, 3)))
    l.text(6.25, c + 0.008, f"all data {c:.2f}", va="bottom", ha="right", fontsize=8.5, color=INK2)
    for name in ["Pi 1 alone", "Pi 2 alone"]:
        v = summary.loc[name, "mean"]
        l.axhline(v, color=INK2, lw=1, ls=(0, (1, 2)))
        l.text(6.25, v + 0.006, f"{name.lower()} {v:.2f}", va="bottom", ha="right", fontsize=8.5, color=INK2)
    l.set_xlim(0.7, 6.3); l.set_xticks(per_round.index); l.set_ylim(0.3, 0.95)
    l.set_xlabel("round (each Pi: 8 local epochs per round)"); l.set_ylabel("test accuracy, both rooms")
    l.set_title("Federated, every run (thin) and their mean (thick)", loc="left", fontsize=10.5, color=INK)
    l.legend(loc="lower left", frameon=False, fontsize=8.5)

    order = ["Pi 2 alone", "Pi 1 alone", "Federated (both Pis)", "All data on one computer"]
    for i, m in enumerate(order):
        vals = finals.loc[finals.method == m, "final"].to_list()
        r.scatter([i] * len(vals), vals, s=46, color=FED if m.startswith("Fed") else INK2, zorder=3,
                  edgecolor="white", linewidth=1.5)
        mu = sum(vals) / len(vals)
        r.plot([i - 0.25, i + 0.25], [mu, mu], color=INK, lw=2)
        r.text(i + 0.29, mu, f"{mu:.2f}", va="center", fontsize=9, color=INK)
    r.set_xticks(range(len(order)))
    r.set_xticklabels(["Pi 2\nalone", "Pi 1\nalone", "Federated\n(both Pis)", "All data on\none computer"])
    r.set_ylim(0.3, 0.95); r.set_xlim(-0.5, len(order) - 0.2)
    r.set_ylabel("final test accuracy")
    r.set_title("Final accuracy: each dot is one run, the line is the mean", loc="left", fontsize=10.5, color=INK)
    r.grid(axis="x", visible=False)
    fig.text(0.01, 0.01, "Federated: 4 runs on the Pis (5 rounds × 8 local epochs, LR 1e-3). "
             "References: 3 random starts each, trained on the Mac with the same model and 40 epochs.",
             color=INK2, fontsize=8.5)
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    fig.savefig(f"{out}.png", dpi=170)
    print(f"\nwrote {out}.png, {out}_per_round.csv, {out}_summary.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
