"""
Thesis result charts (Chapter 5)
================================
Reads the run folders under results/ and writes PNG charts to docs/figures/.
Each chart reads only from result files, so the figures always match the data.

Usage:
    .venv/bin/python scripts/make_thesis_figures.py
"""

import glob
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "docs/figures"

# Validated categorical order (light surface): blue, orange, aqua, yellow.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
MARKERS = ["o", "s", "^", "D"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BASELINE_ACC = json.loads(
    (REPO / "data/urbansound8k/training_metrics.json").read_text())["test_accuracy"]

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
    "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.axisbelow": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "figure.facecolor": "white", "axes.facecolor": "white",
})


def latest(strategy):
    """server.json of the newest run of a strategy."""
    runs = sorted(glob.glob(str(REPO / f"results/strategy_{strategy}/run_*")))
    return json.loads(Path(runs[-1], "server.json").read_text())


def _inv_has(run, lr):
    """True if a C run used this local LR on the full 5000-sample setup."""
    inv = Path(run, "invocation.txt").read_text()
    return re.search(rf"local.lr\W+{re.escape(lr)}\b", inv) and re.search(r"max.samples\W+5000", inv)


def save(fig, name):
    fig.savefig(OUT / name, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT / name)


def fig_strategy_a():
    d = latest("a")
    rounds = [r["round"] for r in d["rounds"]]
    acc = [r["test_accuracy"] for r in d["rounds"]]
    tt = [r["train_time_seconds"] for r in d["rounds"]]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.2))
    ax1.plot(rounds, acc, color=SERIES[0], lw=2, marker="o", ms=5)
    ax1.axhline(BASELINE_ACC, color=INK2, lw=1, ls="--")
    ax1.text(3.2, BASELINE_ACC + 0.004, f"starting model {BASELINE_ACC:.3f}",
             ha="left", va="bottom", color=INK2, fontsize=8)
    ax1.set(title="Test accuracy", xlabel="Round", ylim=(0.5, 0.75), xticks=rounds)
    ax2.bar(rounds, tt, color=SERIES[0], width=0.6)
    ax2.set(title="Server training time (s)", xlabel="Round", xticks=rounds)
    ax2.text(rounds[-1], tt[-1] + 0.5, f"{tt[-1]:.1f} s", ha="center", va="bottom",
             color=INK, fontsize=8)
    save(fig, "14_result_strategy_a.png")


def fig_lr_sweep():
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    for i, lr in enumerate(["1e-3", "5e-4", "1e-4", "5e-5"]):
        runs = [r for r in sorted(glob.glob(str(REPO / "results/strategy_c/run_*")))
                if _inv_has(r, lr)]
        d = json.loads(Path(runs[-1], "server.json").read_text())
        rounds = [r["round"] for r in d["rounds"]]
        acc = [r["test_accuracy"] for r in d["rounds"]]
        ax.plot(rounds, acc, color=SERIES[i], lw=2, marker=MARKERS[i], ms=5,
                label=f"local LR {lr}")
        ax.text(rounds[-1] + 0.15, acc[-1], lr, color=INK, fontsize=8, va="center")
    ax.axhline(BASELINE_ACC, color=INK2, lw=1, ls="--")
    ax.text(3.3, BASELINE_ACC - 0.004, f"starting model {BASELINE_ACC:.3f}",
            color=INK2, fontsize=8, va="top")
    ax.set(xlabel="Federated round", ylabel="Test accuracy (fold 10)",
           ylim=(0.38, 0.75), xticks=range(1, 11), xlim=(0.7, 10.9))
    ax.legend(frameon=False, loc="lower right", ncol=2, fontsize=8)
    save(fig, "15_result_lr_sweep.png")


def fig_upload():
    a, b = latest("a"), latest("b")
    c_runs = [r for r in sorted(glob.glob(str(REPO / "results/strategy_c/run_*")))
              if _inv_has(r, "5e-5")]
    c = json.loads(Path(c_runs[-1], "server.json").read_text())
    vals = [sum(x["per_client_bytes_total"].values()) / 2 / 1e6 for x in (a, b, c)]
    names = ["A centralized\n(raw audio)", "B hybrid\n(spectrograms)", "C federated\n(weights)"]
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    bars = ax.bar(names, vals, color=SERIES[:3], width=0.55)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 1, f"{v:.1f} MB",
                ha="center", va="bottom", color=INK, fontsize=9)
    ax.set(ylabel="Upload per node (MB)", ylim=(0, max(vals) * 1.15))
    ax.grid(axis="x", visible=False)
    save(fig, "16_result_upload.png")


def fig_fusion_blackout():
    path = REPO / "data/vggsound/fusion_blackout_results.json"
    if not path.exists():
        print("skip fusion chart (no blackout results yet)")
        return
    rows = json.loads(path.read_text())["results"]
    order = [("vision", "Vision\nonly"), ("audio", "Audio\nonly"),
             ("fusion", "Naive\nfusion"), ("fusion_md0.5", "Fusion +\ndropout 0.5"),
             ("fusion_md0.5_aws", "Fusion +\nwarm start\n+ dropout 0.5")]
    names = [label for key, label in order if key in rows]
    clean = [rows[key]["clean"] for key, _ in order if key in rows]
    black = [rows[key]["blackout"] for key, _ in order if key in rows]
    x = range(len(names))
    fig, ax = plt.subplots(figsize=(8, 3.6))
    w = 0.38
    ax.bar([i - w / 2 - 0.01 for i in x], clean, w, color=SERIES[0], label="clean")
    ax.bar([i + w / 2 + 0.01 for i in x], black, w, color=SERIES[1], label="blackout")
    for i in x:
        ax.text(i - w / 2, clean[i] + 0.01, f"{clean[i]:.2f}", ha="center", fontsize=8)
        ax.text(i + w / 2, black[i] + 0.01, f"{black[i]:.2f}", ha="center", fontsize=8)
    ax.axhline(0.2, color=INK2, lw=1, ls="--", label="chance (0.20)")
    ax.set_xticks(list(x), names, fontsize=8)
    ax.set(ylabel="Test accuracy", ylim=(0, 1))
    ax.legend(frameon=False, loc="upper right", fontsize=8)
    ax.grid(axis="x", visible=False)
    save(fig, "17_result_fusion_blackout.png")


def fig_fusion_strategies():
    fb, fc = latest("fb"), latest("fc")
    blackout = json.loads((REPO / "data/vggsound/fusion_blackout_results.json")
                          .read_text())["results"]
    start = blackout["fusion_md0.5_aws"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
    for ax, key, title in ((ax1, "test_accuracy", "Clean (camera works)"),
                           (ax2, "blackout_accuracy", "Blackout (camera black)")):
        for i, (d, name) in enumerate(((fb, "FB hybrid"), (fc, "FC federated"))):
            rounds = [0] + [r["round"] for r in d["rounds"]]
            vals = [start["clean" if key == "test_accuracy" else "blackout"]] + \
                [r[key] for r in d["rounds"]]
            ax.plot(rounds, vals, color=SERIES[i], lw=2, marker=MARKERS[i], ms=5,
                    label=name)
        ref = blackout["vision"]["clean"] if key == "test_accuracy" else blackout["audio"]["clean"]
        ax.axhline(ref, color=INK2, lw=1, ls="--",
                   label="vision only" if key == "test_accuracy" else "audio only")
        ax.set(title=title, xlabel="Round (0 = starting model)", xticks=range(0, 11))
        ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax1.set(ylabel="Test accuracy", ylim=(0.3, 0.95))
    save(fig, "18_result_fb_fc.png")


def fig_live_pi():
    live = REPO / "data/pi_captures/results/live"
    names = ["A centralized", "B hybrid", "C federated"]
    up, acc = [], []
    for s in "abc":
        up.append(json.loads((live / f"strategy_{s}/pi_client_A.json").read_text())
                  ["bytes_uploaded"] / 1e3)
        acc.append(json.loads((REPO / f"results/demo/strategy_{s}/server.json").read_text())
                   ["rounds"][-1]["test_accuracy"])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.2))
    for ax, vals, title, fmt in ((ax1, up, "Upload from the Pi (KB)", "{:.0f} KB"),
                                 (ax2, acc, "Final test accuracy (32 clips)", "{:.2f}")):
        bars = ax.bar(names, vals, color=SERIES[:3], width=0.55)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v * 1.01, fmt.format(v),
                    ha="center", va="bottom", color=INK, fontsize=9)
        ax.set(title=title, ylim=(0, max(vals) * 1.18))
        ax.grid(axis="x", visible=False)
        ax.tick_params(axis="x", labelsize=8)
    ax2.axhline(1 / 3, color=INK2, lw=1, ls="--", label="chance (0.33)")
    ax2.legend(frameon=False, fontsize=8, loc="upper left")
    save(fig, "19_result_live_pi.png")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    fig_strategy_a()
    fig_lr_sweep()
    fig_upload()
    fig_fusion_blackout()
    fig_fusion_strategies()
    fig_live_pi()
