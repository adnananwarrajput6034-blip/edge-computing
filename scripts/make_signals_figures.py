"""Figures 20-21: the 5 Nov recordings in time domain, frequency domain and as log-mel spectrograms.

Usage: .venv/bin/python scripts/make_signals_figures.py [repo_root]
Writes docs/figures/20_live_signals_nov5.png and docs/figures/21_live_bursts_nov5.png
(post-submission figures; not in the submitted thesis).
"""
import json, sys
from pathlib import Path
import numpy as np, soundfile as sf, librosa
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
REPO = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1]
OUT = REPO / "docs/figures"
sys.path.insert(0, str(REPO))
from src.edge.processing.stft import STFTProcessor
proc = STFTProcessor(sample_rate=16000, n_mels=128)
CAP = REPO / "data/pi_captures"
INK, INK2, GRID = "#1f2937", "#4b5563", "#e5e7eb"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9.5, "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.axisbelow": True, "grid.color": GRID, "grid.linewidth": 0.8, "figure.facecolor": "white"})
def rows(s): return [json.loads(l) for l in open(CAP / s / "ticks.jsonl") if l.strip()]
def best(r, c): return max([d["conf"] for d in r["detections"] if d["class"] == c] or [0.0])
def load(s, r):
    x, _ = sf.read(CAP / s / f"{r['file']}.wav", dtype="int16"); return x.astype(np.float32) / 32768.0
def dom(x):
    f = np.abs(np.fft.rfft(x * np.hanning(len(x)))); fr = np.fft.rfftfreq(len(x), 1 / 16000); m = fr > 80
    return float(fr[m][np.argmax(f[m])])
def clip(x): return float(np.mean(np.abs(x) >= 0.976)) * 100
ring_s, speech_s, quiet_s = "session_20261105_015624", "session_20261105_004018", "session_20261105_002237"
ring = [r for r in rows(ring_s) if best(r, "cell phone") >= 0.5 and r["rms"] >= 0.03 and (CAP / ring_s / f"{r['file']}.wav").exists()]
ring = [r for r in ring if clip(load(ring_s, r)) < 0.5 and dom(load(ring_s, r)) >= 400]
r_ring = max(ring, key=lambda r: r["rms"])
speech = [r for r in rows(speech_s) if r["tick"] <= 90 and best(r, "person") >= 0.5 and 0.05 <= r["rms"] <= 0.25 and (CAP / speech_s / f"{r['file']}.wav").exists()]
speech = [r for r in speech if clip(load(speech_s, r)) < 0.5 and dom(load(speech_s, r)) < 400]
r_speech = sorted(speech, key=lambda r: r["rms"])[len(speech) // 2]
quiet = [r for r in rows(quiet_s) if r["rms"] < 0.0113 and best(r, "person") < 0.5 and best(r, "cell phone") < 0.5]
med = float(np.median([r["rms"] for r in quiet])); r_quiet = min(quiet, key=lambda r: abs(r["rms"] - med))
picks = [("Ringtone", "cell phone", ring_s, r_ring), ("Speech", "person", speech_s, r_speech), ("Silence", "background", quiet_s, r_quiet)]
xs = [load(s, r) for _, _, s, r in picks]
mels = librosa.mel_frequencies(n_mels=128, fmin=0.0, fmax=8000.0)
hz_ticks = [250, 500, 1000, 2000, 4000, 8000]; band_ticks = [int(np.argmin(np.abs(mels - h))) for h in hz_ticks]
# ---------- Figure 20 ----------
recno = {"session_20261105_002237": 1, "session_20261105_004018": 2, "session_20261105_015624": 5}
fig = plt.figure(figsize=(12.5, 9.6), constrained_layout=True)
gs = fig.add_gridspec(3, 4, width_ratios=[1, 1, 1, 0.035], height_ratios=[1, 1, 1.25])
ax = np.array([[fig.add_subplot(gs[i, j]) for j in range(3)] for i in range(3)]); cax = fig.add_subplot(gs[2, 3])
peak = max(np.abs(x).max() for x in xs) * 1.1
specs = [20 * np.log10(np.abs(np.fft.rfft(x * np.hanning(len(x)))) + 1e-6) for x in xs]
top_db = max(s_.max() for s_ in specs); fr = np.fft.rfftfreq(8000, 1 / 16000)
t = np.arange(8000) / 16000 * 1000; colours = ["#2563eb", "#ea580c", "#6b7280"]
ylab = ["Time domain\namplitude (\u22121 \u2026 1)", "Frequency domain\nlevel (dB below the loudest)", "Spectrogram\nfrequency (Hz, mel scale)"]
for j, ((name, label, sess, r), x, spec) in enumerate(zip(picks, xs, specs)):
    a = ax[0, j]; a.plot(t, x, lw=0.5, color=colours[j]); a.set_ylim(-peak, peak); a.set_xlim(0, 500); a.set_xlabel("Time (ms)")
    a.set_title(f"{name} \u2192 label \"{label}\"\nrecording {recno[sess]}, tick {r['tick']}, rms {r['rms']:.3f}", fontsize=10, color=INK)
    b = ax[1, j]; b.plot(fr, spec - top_db, lw=0.6, color=colours[j]); b.set_xlim(0, 8000); b.set_ylim(-110, 5); b.set_xlabel("Frequency (Hz)")
    if j < 2:
        d = dom(x); dy = (spec - top_db)[np.argmin(np.abs(fr - d))]
        b.annotate(f"strongest: {d:,.0f} Hz", xy=(d, dy), xytext=(d + 1400, -22), fontsize=8.5, color=INK, arrowprops=dict(arrowstyle="-", color=INK2, lw=0.7))
    c = ax[2, j]; S = proc.process(x)
    im = c.imshow(S.T, origin="lower", aspect="auto", cmap="magma", vmin=0, vmax=1, extent=[0, 500, 0, 128])
    c.set_yticks([b_ + 0.5 for b_ in band_ticks]); c.set_yticklabels([f"{h:,}" for h in hz_ticks]); c.grid(False)
    c.set_xlabel(f"Time (ms): {S.shape[0]} frames \u00d7 {S.shape[1]} mel bands")
for i in range(3): ax[i, 0].set_ylabel(ylab[i])
cb = fig.colorbar(im, cax=cax); cb.set_label("0 = quiet \u2026 1 = loud", fontsize=8.5)
fig.suptitle("One 0.5 s tick, three ways: time domain, frequency domain and the spectrogram the AudioCNN sees (5 Nov 2026)", fontsize=11.5, color=INK)
fig.savefig(OUT / "20_live_signals_nov5.png", dpi=170); plt.close(fig)
# ---------- Figure 21 ----------
def strip(s, first, n=10):
    rs = [r for r in rows(s) if first <= r["tick"] < first + n]
    S = np.concatenate([proc.process(load(s, r)) for r in rs], axis=0)
    return rs, S
ring_rs, ring_S = strip(ring_s, 44); sp_rs, sp_S = strip(speech_s, 40)
fig, ax = plt.subplots(2, 1, figsize=(12, 6.4), sharex=True)
for a, (name, s, rs, S) in zip(ax, [("Ringtone (stage recording)", ring_s, ring_rs, ring_S), ("Speech (talking close to the mic)", speech_s, sp_rs, sp_S)]):
    a.imshow(S.T, origin="lower", aspect="auto", cmap="magma", vmin=0, vmax=1, extent=[0, len(rs), 0, 128])
    a.set_yticks([b_ + 0.5 for b_ in band_ticks]); a.set_yticklabels([f"{h:,}" for h in hz_ticks]); a.grid(False)
    a.set_ylabel("Hz (mel scale)"); a.set_title(pad=30, label=f"{name}: {s.replace('session_', 'session ')}, ticks {rs[0]['tick']}–{rs[-1]['tick']}", fontsize=10, loc="left", color=INK)
    for k, r in enumerate(rs):
        a.axvline(k, color="white", lw=0.8, alpha=0.6)
        lab = max(r["detections"], key=lambda d: d["conf"])["class"] if r["detections"] else "nothing"
        a.text(k + 0.5, 131, f"{lab}\nrms {r['rms']:.3f}", ha="center", va="bottom", fontsize=7, color=INK2, clip_on=False)
ax[1].set_xticks([k + 0.5 for k in range(10)]); ax[1].set_xticklabels([f"tick {k + 1}" for k in range(10)])
ax[1].set_xlabel("10 ticks side by side (each 0.5 s; the 0.5 s between two ticks is not recorded)")
fig.suptitle("Energy bursts: a ringtone is horizontal lines that switch on and off; speech is short stacks of low harmonics", fontsize=11, color=INK, y=0.995)
fig.tight_layout(rect=(0, 0, 1, 0.95), h_pad=3.2); fig.savefig(OUT / "21_live_bursts_nov5.png", dpi=180); plt.close(fig)
for name, label, s, r in picks:
    print(f"{name}: {s} tick {r['tick']} rms {r['rms']:.3f} strongest {dom(load(s, r)):.0f} Hz, detections {[(d['class'], d['conf']) for d in r['detections'][:2]]}")
print("ringtone strip ticks", [r["tick"] for r in ring_rs], "speech strip ticks", [r["tick"] for r in sp_rs])
