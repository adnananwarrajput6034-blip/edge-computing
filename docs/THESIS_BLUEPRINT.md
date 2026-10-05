# Thesis Blueprint — how to write it in 3 days

This is your map, not your text. **You write every sentence yourself.** Each
section tells you: what it is for, what goes in, where the material already is
in this repo, what figure/table to put there, and which papers to cite.

Read Part 0 today. Then follow the 3-day plan in Part 7, one block at a time.
When a block is done, tick it. Don't read ahead.

---

# Part 0 — Three things to do before writing a single line

## 0.1 Ask about AI use (5 minutes, today)
You signed nothing yet, but you will: the declaration says you wrote it
"without assistance from external sources other than those indicated". BHT's
good-practice rules (Amtliche Mitteilung 02/2023, §1 and §11) treat
undisclosed help as misconduct. So:
- Email Prof. Jakob today: "I used an AI assistant for coding support and for
  structuring my writing. How should I declare this?" — one line.
- Whatever the answer, add one honest sentence to the declaration or the
  methodology. This protects you completely.

## 0.2 The raw result files for the main experiments are not on this laptop
`docs/RESULTS.md` cites `results/strategy_a/run_*/server.json`,
`results/strategy_c/...` (LR sweep) and `data/vggsound/fusion_blackout_results.json`.
**None of these exist on disk right now.** Only `results/demo/` (live demo
runs) and `data/pi_captures/` exist.

An examiner may ask "show me the data" (BHT §8: results must be documented
and archived). Two options:
- They're on another machine / backup → copy them back into the repo.
- They're gone → re-run them. The data and scripts are here
  (`scripts/run_strategy.py`, `docs/EXPERIMENTS.md`). I can run A, B, C and
  the fusion evaluation **in the background while you write**. Just say "run
  them".

## 0.3 Number contradictions — full repo audit (2026-10-01)
**Rule for the thesis: only write a number you can point to a file for.**

| Topic | What the docs say | What the files on disk say | Do |
|---|---|---|---|
| AudioCNN baseline | 70.97% (RESULTS, PRIMER, ARCHITECTURE), 68.6% (CLAUDE.md, PRESENTATION) | `models/audio_cnn_urbansound8k.keras` re-evaluated = **64.87%** (`data/urbansound8k/training_metrics.json`, 837 fold‑10 clips, macro‑F1 0.649). 70.97% is in no file here. | Find the 70.97% model (other machine?) or re-train / use 64.87%. |
| Fusion data | 973 pairs, 200/class | `data/vggsound/pairs_manifest.json`: 792 kept, **bird only 3** | Fusion numbers (0.847 / 0.551, FB/FC) need their files from the machine that ran them, or a re-run. |
| Fusion models | trained | no `fusion_*.keras` / `fv_fusion.keras` here | same |
| Live demo bytes | 2726 / 1737 / 1215 KB (old rehearsal in PRESENTATION Acts 3–5) | JSONs: **1305 / 956 / 1216 KB**, B = −26.7% | Use JSON numbers. |
| Live enrollment | "95% detection" | `enroll_report.json`: 247 ticks, 160 labelled = **64.8%**, YOLO 317.6 ms/frame, classes car 40 / cell phone 33 / person 55 | Use JSON numbers. |
| YOLO on Pi | ~430 ms | measured 315–318 ms | Use measured. |
| Pi clients | "only Strategy A" (CODE_FLOW, PRIMER, ARCHITECTURE, README) | A, B, C Pi clients all exist and ran | Write the newer truth. |
| Flower | exposé, CLAUDE.md, configs | custom FedAvg over MQTT (`src/server/aggregation/fedavg.py`); Flower only in an older harness (`logs/`) | One honest sentence in Methodology explaining the switch. |
| Hardware | Pi 4, TLS, TFLite, I2S mic, Camera Module 3 | Pi 5, USB webcam + USB mic, PyTorch YOLO, no TLS, QoS 1 everywhere | Describe what you actually used. |
| Frame count | 49 (stft.py docstring, test) | 51 (librosa center=True); 48 without centering | Use 51, explain. |
| Fusion head | Dense 256 → N | code: 256 → 128 → N, ~389K trainable | Use code. |
| Figures | 02 shows 256‑dim FV, "C unchanged"; 05 shows only Strategy A, 430 ms | actual: 576‑float FV, FB/FC; enroll → A/B/C replay → infer | Redraw 02 and 05 (Mermaid source in DIAGRAMS.md). |

---

# Part 1 — What the colleague theses teach you

You read two passed BHT Dept. VII theses. Lessons:

| | Naveed-Ul-Haq (2024) | Muhammad Ali (2025) | **You** |
|---|---|---|---|
| Pages | ~45 | ~92 | **aim 55–65** |
| Reference style | IEEE numeric [1] | APA (Author, Year) | **pick IEEE numeric** — normal for engineering, shorter in text |
| Own experiment | 6 audio files into 2 web tools | 1 ICMP flood, 3 Mininet hosts | 3 strategies × real MQTT × Pi hardware × fusion model |
| Structure | Intro / LitReview / Method / Results / Conclusion | Intro / LitReview / Method / Design & Impl. / Results / Conclusion | same 6–7 chapter skeleton |

**What this means for you:** the bar that passed is reachable. Your
experimental material is already larger than both. Your risk is not "too
little content" — it's *not finishing* and *inconsistent numbers*.

Things to copy:
- Every chapter starts with 2–3 lines saying what the chapter does, and ends
  with a short summary that hands over to the next chapter (Ali does this well).
- Every figure has a numbered caption and "Source: own work" or the citation.
- A "Limitations" section written honestly.

Things NOT to copy:
- Ali's Chapter 6 says "Chapter 4 presented…" for results that are in Chapter 5,
  and his results table numbers (1 Gb/s, 85–90% CPU) aren't traceable to any
  log. Examiners notice this. Every number you write must point to a file.
- Duplicate references in the list (both theses have them).
- Claims like "perfect accuracy" or "no false positives". You say what you
  measured, with its limits.

Gender‑sensitive language (BHT note): a technical thesis rarely names people.
When it does: "the user", "the operator", "participants", "they". Never a
default "he".

---

# Part 2 — The spine: title, one-sentence claim, questions

Write these on a sticky note. Every chapter serves them.

**Title (from your exposé, keep it):**
*Integrated Edge-to-Cloud Systems: Federated Learning for Sensor Fusion and
Object Recognition*

**One-sentence claim:**
In a self-labelling audio-visual edge system, moving training from the server
to the device cuts upload bandwidth by about 16× and makes server load
constant, with no accuracy loss — but only if the local learning rate is
tuned; and a fused audio-visual model keeps working when the camera fails.

**Research questions (3 — not more):**
- **RQ1** How do centralized (A), hybrid (B) and federated (C) training placement
  compare in accuracy, bandwidth, device compute, server compute and privacy?
- **RQ2** Under what conditions does FedAvg match centralized training on
  edge devices?
- **RQ3** Can an audio-visual fusion model keep recognizing objects when
  vision fails, and does that survive hybrid and federated training?

**Note on the exposé:** it promised "select one strategy for final
implementation". You did better: you implemented all three plus fusion. In the
conclusion you *recommend* a placement per situation. Say that explicitly
once, so the examiner sees the promise was kept.

**What the exposé promised that you did NOT do — name it in Limitations:**
power in watts, inference latency per strategy, memory per strategy, model
freshness, non-IID. Don't hide them. One table row each: "planned / not
measured / why / future work". Honest beats missing.

---

# Part 3 — Chapter plan (≈60 pages)

Pages are targets, not rules. Writing order is different from reading order —
see Part 7.

## Front matter (not counted)
Cover page (BHT format) · Declaration · Abstract (½ page, written LAST) ·
Table of contents · List of figures · List of tables · Abbreviations
(FL, FedAvg, IID, MQTT, QoS, STFT, CNN, YOLO, FV, LR, Pi).

---

## Chapter 1 — Introduction (5 pages)

**Purpose:** why this matters, what you ask, what you deliver.

| Section | Content | Your material |
|---|---|---|
| 1.1 Motivation | Robots see with cameras; fog/darkness/smoke blind them; sound doesn't need light; humans do the same. | Exposé §1, REVISION.md Part 1 |
| 1.2 Problem | Learning sounds needs labels; hand-labelling is infeasible → let vision label audio. Then: *where* should continuous training run? Privacy vs bandwidth vs accuracy vs device load. | REVISION.md 1.2–1.4 |
| 1.3 Research questions & objectives | RQ1–RQ3 from Part 2. Objectives as a numbered list. | Part 2 above |
| 1.4 Contributions | 4–5 bullets: the measured A/B/C comparison; B's real saving is 26% not 66%; naive FedAvg destabilizes a converged model, LR fix; fusion recipe + blackout result; FL cost scales with model size. | REVISION.md Part 8 "main contribution" |
| 1.5 Structure | One sentence per chapter. | — |

**Figure 1.1:** one simple picture of the idea: camera → YOLO → label →
attached to microphone clip → audio model. (Draw it yourself in draw.io or
on paper and scan — that's fine.)

**Cite:** [Aytar SoundNet] for "vision teaches sound", [Owens], [Arandjelović].

---

## Chapter 2 — Background and Related Work (13 pages)

**Purpose:** teach the examiner just enough theory to understand Ch. 3–5, and
show who did similar things. Don't write a textbook. Every subsection ends with
one sentence: "In this thesis, X is used for Y."

### 2.1 Audio as a signal and the spectrogram (3 p)
- Sampling (16 kHz), short-time Fourier transform, window/hop, mel scale.
- **Formulas to include** (see Part 4): STFT, mel conversion, frame count.
- **Figure:** `docs/study/waveform-vs-spectrogram.png`.
- Material: `docs/study/audio-params.html`, `docs/PRIMER.md`.
- Cite: [Hershey CNN audio], [Salamon UrbanSound8K], [Piczak ESC-50], [McFee librosa].

### 2.2 CNNs for sound classification (1.5 p)
- Spectrogram = image → CNN. Conv, pooling, softmax, cross-entropy (formula).
- Cite: [Hershey], [LeCun] or [Goodfellow book].

### 2.3 Object detection as a teacher: YOLO (1 p)
- One-stage detector, COCO classes, confidence threshold. Why pretrained & frozen.
- Cite: [Redmon YOLO], [Ultralytics YOLOv8].

### 2.4 Cross-modal / weak supervision (1.5 p)
- Labels from another sensor. Noisy labels. Why deep nets tolerate noise.
- Cite: [Aytar], [Owens], [Arandjelović], [Rolnick label noise].

### 2.5 Federated learning (3 p) ← **your professor asked for the maths here**
- The idea, the round, FedAvg, why shared initialization matters (your
  exposé §3.2 already explains permutation symmetry — reuse your own words).
- Non-IID and client drift, FedProx (only as related work).
- Privacy limits: gradient inversion, secure aggregation, DP (one paragraph).
- **Formulas:** all of Part 4.2.
- **Figure:** `docs/figures/04_sequence_strategy_c_fedavg.png`.
- Cite: [McMahan FedAvg], [Kairouz open problems], [Zhao non-IID], [Li FedProx],
  [Zhu deep leakage], [Bonawitz secure aggregation], [Beutel Flower].

### 2.6 Edge computing and MQTT (1 p)
- Publish/subscribe, broker, topics, QoS 1, retained messages.
- Cite: [MQTT v5 OASIS], [Ridolfi FL on RPi].

### 2.7 Multimodal fusion and modality imbalance (1.5 p)
- Early vs late fusion. The known problem: the strong modality dominates
  training ("modality imbalance") — this is exactly what you found. Modality
  dropout as a remedy.
- Cite: [Wang "what makes multimodal hard"], [Peng OGM-GE], [Neverova ModDrop],
  [Howard MobileNetV3].

### 2.8 Research gap (½ p)
Most FL work compares accuracy on benchmark data; few measure *real network
bytes* for a *self-labelling* edge system, and few test whether fusion
robustness survives federated training. That's your gap. 4–5 sentences.

---

## Chapter 3 — System Design and Methodology (12 pages)

**Purpose:** what you built and how you measured, so someone could repeat it.

| Section | Content | Material |
|---|---|---|
| 3.1 Research approach | Comparative experimental study; one variable at a time (audio-only first, then fusion). | REVISION 2.6 last para |
| 3.2 System architecture | Edge (Pi), broker, server. The tick loop. Batching (500 samples). | REVISION Part 2, `docs/ARCHITECTURE.md`; **Fig:** `01_architecture_current_audiocnn.png` |
| 3.3 Communication design | Topics table, QoS 1, retained global model, round numbers, binary + zlib serialization, why batching. | REVISION 2.3–2.5 |
| 3.4 The three strategies | Table: client does / wire carries / server does. Sequence diagrams. | REVISION 2.6, **Figs** `03_…ab.png`, `04_…fedavg.png` |
| 3.5 Datasets | UrbanSound8K (folds 1–8/9/10), VGGSound subset (5 classes, 973 pairs, split by clip — leakage explanation). | REVISION Part 4 step 1, Part 6 step 2 |
| 3.6 Models | AudioCNN (111K params) and FusionModel (layer table, 1.33M / ~390K trainable). | FUSION.md 1.2, `src/server/training/models/` |
| 3.7 Experimental protocol | 2 nodes, disjoint halves, 10 rounds, 5,000 samples/node, same warm start, bytes counted at sender AND receiver. Why replay not live (3 reasons). | REVISION 5.0, Part 3 |
| 3.8 Metrics | Accuracy, upload bytes, device compute time, server compute time, data leaving device, blackout accuracy. Define each in one line. | — |

**Most important paragraph of the whole thesis:** 3.7 "why replay". Examiners
WILL ask. Your answer is in REVISION Part 3 — rewrite it in your words.

---

## Chapter 4 — Implementation (9 pages)

**Purpose:** how the design became code and hardware. Engineering decisions,
not code dumps.

| Section | Content | Material |
|---|---|---|
| 4.1 Hardware & software | Pi 5, USB webcam, USB mic, laptop; Python 3.13, TF 2.21, librosa, ultralytics, paho-mqtt, mosquitto. Table. | CLAUDE.md, PRESENTATION.md appendix |
| 4.2 Audio pipeline | STFT params and why each value. | `src/edge/processing/stft.py`, audio-params.html |
| 4.3 Labelling pipeline | YOLO on frame, confidence threshold, frame discarded. Latency 47 ms Mac / 430 ms Pi. | RESULTS §6 |
| 4.4 Client & server code | Short pseudocode (not Python) for: client loop, server queue/train loop, FedAvg aggregation. ~10 lines each. | `docs/CODE_FLOW.md`, `scripts/run_strategy.py` |
| 4.5 Fusion training recipe | Warm-start audio branch, then modality dropout 0.5. Feature-vector wire format (576 floats) + equivalence check. | FUSION.md |
| 4.6 Live Pi deployment | 2 config changes, no code changes; enrollment of live classes; B and C Pi clients. | REVISION 5.5, commit `4d07e6a` |
| 4.7 Problems and how they were solved | Broken webcam mic (RMS = max → constant). LR instability. YouTube format change in VGGSound download. Dead audio branch. | REVISION, git log |

**Fig:** `05_sequence_live_pi_capture.png`, a photo of your Pi setup, one
frame from the evidence gallery.

Section 4.7 is gold — examiners love "what went wrong and how you found it".

---

## Chapter 5 — Results (13 pages)

**Purpose:** numbers only, with short factual comments. Interpretation goes
mostly into Ch. 6. Each subsection: table/figure → 3–6 sentences describing it.

| Section | Table / figure | Source |
|---|---|---|
| 5.1 Baseline AudioCNN | accuracy 70.97%, per-class F1 (siren weakest) | RESULTS §5 |
| 5.2 Strategy A | round / pool / accuracy / train time table; **line plot** train time vs round | RESULTS §2 |
| 5.3 Strategy B | upload vs A; 26% vs predicted 66% — explain compression | RESULTS §3 |
| 5.4 Strategy C + LR sweep | LR table; **plot** accuracy per round for 4 LRs | RESULTS §4 |
| 5.5 A vs B vs C | the headline table; **bar chart** upload MB per strategy | RESULTS §1 |
| 5.6 Live Pi validation | 119 ticks, 26% labelled, 452 KB, byte match; live A/B/C bytes (2726 / 1737 / 1215 KB) | RESULTS §7, commit 4d07e6a |
| 5.7 Fusion & blackout | clean vs blackout table; **grouped bar chart** | RESULTS §8 |
| 5.8 FB / FC | table; blackout accuracy per round plot | RESULTS §8 F6 |

**Plots:** matplotlib from the JSON files (this is why 0.2 matters). I can
write one `scripts/make_thesis_figures.py` that produces all of them — ask.

---

## Chapter 6 — Discussion (6 pages)

**Purpose:** what the numbers mean, answered per research question.

- 6.1 RQ1 — the trade-off triangle: bandwidth/privacy vs device compute.
  Why accuracy is flat (correct, not broken).
- 6.2 RQ2 — why FedAvg needed a smaller LR on a converged model (the valley
  picture). Link to FedProx literature.
- 6.3 RQ3 — modality imbalance, the recipe, graceful degradation; FC scaling
  with model size.
- 6.4 Privacy, honestly: "raw audio never leaves" ≠ "zero information leaves".
- 6.5 Threats to validity / limitations: single fold, 98-clip fusion test
  (±5%), IID only, VGGSound frames are vision-friendly, black ≠ real fog,
  2 clients, power/latency/memory/freshness not measured.
- 6.6 Comparison with related work: 1 short table — your result vs
  [Ridolfi], [McMahan], [Wang multimodal].

---

## Chapter 7 — Conclusion and Future Work (3 pages)

- 7.1 Summary: answer RQ1–3 in one paragraph each.
- 7.2 Recommendation per situation (this fulfils the exposé's "select one"):
  - bandwidth/privacy-critical, device can compute → **C**
  - weak device, good network → **A**
  - B only if bandwidth matters a little and compute a little.
- 7.3 Future work (your priority list): model freshness experiment, non-IID +
  FedProx, power/latency/memory on Pi, fusion training on Pi, 10-fold CV,
  intermediate fog curve.

---

## Appendix (not counted)
- A: Configuration files and commands to reproduce (from EXPERIMENTS.md)
- B: Full per-round tables
- C: Evidence gallery samples
- D: Repository link

---

# Part 3b — "Which method did you try, and which one won?"

Your professor's point: the examiners want to see **comparison, observation,
decision** — not just the final number. You already did this many times. Each
row below becomes one short subsection in Ch. 5 (what you measured) and one
sentence in Ch. 6 (why the winner won). Same pattern every time:

> **Problem → options tried → metric → observation → decision.**

| # | Problem | Options tried | Metric | Observation | Decision |
|---|---|---|---|---|---|
| 1 | Where to train? (the core RQ) | A centralized / B hybrid / C federated | upload MB, accuracy, device s, server s | C: 16× less upload, same accuracy, more device compute | depends on situation → Ch. 7 recommendation |
| 2 | How much does B save? | size formula vs measured bytes | upload MB | formula 18.4%, measured 26.2% (spectrograms compress better) | B saves little; see thesis 5.10 |
| 3 | FedAvg unstable | local LR 1e-3 / 5e-4 / 1e-4 / 5e-5 | accuracy per round | oscillates at high LR, stable at 5e-5 | 5e-5 |
| 4 | Fusion worse than vision alone | naive / modality dropout 0.3 / 0.5 / audio warm-start + dropout 0.5 | clean + blackout accuracy | only warm-start + dropout uses audio | warm-start then dropout |
| 5 | How to send vision to the server (FB) | JPEG frame (30–50 KB) vs 576-float feature vector (~1 KB) | bytes, prediction agreement | FV 100% agreement, ~30× smaller, not invertible | feature vector |
| 6 | FL framework | Flower (early harness, see `logs/`) vs own FedAvg over MQTT | byte comparability | same transport for A/B/C makes bytes comparable | own FedAvg over MQTT |
| 7 | Live labels didn't match dataset classes | fixed UrbanSound8K classes vs runtime enrollment by YOLO | usable samples | fixed classes → 0 usable samples; enrollment → 128 | enrollment |
| 8 | Pi microphone | webcam built-in mic vs USB mic | RMS of signal | webcam mic = constant full-scale (dead) | USB mic |
| 9 | Spectrogram frames | numpy STFT (48 frames) vs librosa centered (51) | shape | — | librosa, 51×128 |

Rows 1–5 are the main results. Rows 6–9 go into Ch. 4.7 "Problems and how
they were solved".

---

# Part 4 — The mathematics (put this in 2.1, 2.2, 2.5)

Type these in your editor's equation tool. Explain every symbol in one line
under the equation — that is what makes it clear.

## 4.1 Audio
**Frame count** (librosa, `center=True`):
  T = 1 + ⌊N / H⌋,  N = 16000 Hz × 0.5 s = 8000 samples, H = 160 (10 ms)
  → T = 1 + 50 = **51 frames**. (Without centering: 1 + ⌊(8000−400)/160⌋ = 48.
  This is why your shape is 51 × 128 — good examiner detail.)

**STFT:**
  X(m, k) = Σₙ x(n + mH) · w(n) · e^(−j2πkn/N_FFT)
  m = frame index, k = frequency bin, w = window (400 samples = 25 ms),
  N_FFT = 512 → 257 frequency bins.

**Mel scale:**  mel(f) = 2595 · log₁₀(1 + f/700)  → 128 mel bands.

**Log-mel:**  S(m, b) = 10 · log₁₀(Σₖ |X(m,k)|² · M_b(k))
  M_b = triangular mel filter b.

## 4.2 Federated learning (your professor's request)
**Global objective** — K clients, client k holds n_k samples, n = Σ n_k:
  F(w) = Σₖ (n_k / n) · F_k(w),   F_k(w) = (1/n_k) Σ_{i∈D_k} ℓ(w; x_i, y_i)

**Local update** on client k, starting from the global model w_t:
  w ← w − η · ∇ℓ(w; batch)    (repeated for E local epochs)  → result w_{t+1}^k

**FedAvg aggregation:**
  w_{t+1} = Σₖ (n_k / n) · w_{t+1}^k

With your 2 nodes: w_{t+1} = (n_A/n)·w_A + (n_B/n)·w_B.

**Why the LR mattered (write this in words + this inequality):**
  ‖w_{t+1}^k − w_t‖ grows with η · E · (number of batches).
  If the start w_t is already near a minimum, large local steps move both
  clients far up opposite slopes; their average is not at the minimum.
  Small η keeps ‖w^k − w_t‖ small, so the average stays near the minimum.
  (Cite [McMahan], [Li FedProx] — FedProx adds μ/2·‖w − w_t‖² to stop exactly this drift.)

**Upload cost per round (bandwidth model):**
  B_C ≈ P × 4 bytes (float32), P = trainable parameters
  Check: 111K × 4 ≈ 444 KB ≈ your measured ~400 KB/round (zlib saves a bit).
  B_A ≈ samples × bytes per clip  → grows with data; B_C does not.
  **This one formula explains your "FL cost scales with model size, not data" finding.**

## 4.3 Training
**Softmax:**  p_c = e^{z_c} / Σⱼ e^{z_j}
**Cross-entropy:**  ℓ = −Σ_c y_c · log p_c
**Accuracy:**  correct predictions / total test samples
**Modality dropout:** for each training sample, with probability p = 0.5,
  replace the image input with zeros (black).

---

# Part 5 — References (real papers, with links)

Put each into Zotero (free: zotero.org — the browser button imports a paper in
one click and formats IEEE automatically). Read only the **abstract +
conclusion** of each. You do not need to read them fully in 3 days.

**Federated learning**
1. McMahan et al., "Communication-Efficient Learning of Deep Networks from Decentralized Data," AISTATS 2017. https://arxiv.org/abs/1602.05629 ← FedAvg, **must cite**
2. Kairouz et al., "Advances and Open Problems in Federated Learning," 2021. https://arxiv.org/abs/1912.04977
3. Zhao et al., "Federated Learning with Non-IID Data," 2018. https://arxiv.org/abs/1806.00582
4. Li et al., "Federated Optimization in Heterogeneous Networks (FedProx)," MLSys 2020. https://arxiv.org/abs/1812.06127
5. Beutel et al., "Flower: A Friendly Federated Learning Framework," 2020. https://arxiv.org/abs/2007.14390
6. Ridolfi et al., "Implementation and Evaluation of a FL Framework on Raspberry PI Platforms for IoT 6G Applications," Future Internet 2023. https://www.mdpi.com/1999-5903/15/11/358
7. Zhu, Liu, Han, "Deep Leakage from Gradients," NeurIPS 2019. https://arxiv.org/abs/1906.08935
8. Bonawitz et al., "Practical Secure Aggregation for Privacy-Preserving ML," CCS 2017. https://eprint.iacr.org/2017/281

**Audio**
9. Salamon, Jacoby, Bello, "A Dataset and Taxonomy for Urban Sound Research," ACM MM 2014. https://urbansounddataset.weebly.com/urbansound8k.html
10. Piczak, "ESC: Dataset for Environmental Sound Classification," ACM MM 2015. https://github.com/karolpiczak/ESC-50
11. Hershey et al., "CNN Architectures for Large-Scale Audio Classification," ICASSP 2017. https://arxiv.org/abs/1609.09430
12. McFee et al., "librosa: Audio and Music Signal Analysis in Python," SciPy 2015. https://librosa.org
13. Chen et al., "VGGSound: A Large-Scale Audio-Visual Dataset," ICASSP 2020. https://arxiv.org/abs/2004.14368

**Vision teaches sound / weak supervision**
14. Aytar, Vondrick, Torralba, "SoundNet: Learning Sound Representations from Unlabeled Video," NeurIPS 2016. https://arxiv.org/abs/1610.09932 ← closest to your idea, **must cite**
15. Owens et al., "Ambient Sound Provides Supervision for Visual Learning," ECCV 2016. https://arxiv.org/abs/1608.07017
16. Arandjelović, Zisserman, "Look, Listen and Learn," ICCV 2017. https://arxiv.org/abs/1705.08168
17. Rolnick et al., "Deep Learning is Robust to Massive Label Noise," 2017. https://arxiv.org/abs/1705.10694

**Vision models**
18. Redmon et al., "You Only Look Once," CVPR 2016. https://arxiv.org/abs/1506.02640
19. Jocher et al., "Ultralytics YOLOv8," 2023. https://github.com/ultralytics/ultralytics
20. Howard et al., "Searching for MobileNetV3," ICCV 2019. https://arxiv.org/abs/1905.02244

**Multimodal fusion**
21. Wang, Tran, Feiszli, "What Makes Training Multi-Modal Classification Networks Hard?," CVPR 2020. https://arxiv.org/abs/1905.12681 ← explains your "naive fusion worse than vision", **must cite**
22. Peng et al., "Balanced Multimodal Learning via On-the-fly Gradient Modulation," CVPR 2022. https://arxiv.org/abs/2203.15580
23. Neverova et al., "ModDrop: Adaptive Multi-Modal Gesture Recognition," TPAMI 2016. https://arxiv.org/abs/1501.00102 ← modality dropout origin

**Communication**
24. OASIS, "MQTT Version 5.0," 2019. https://docs.oasis-open.org/mqtt/mqtt/v5.0/mqtt-v5.0.html
25. Eclipse Mosquitto. https://mosquitto.org

Also cite your own exposé's references [Frontiers 2012 neuromorphic audio-visual
robot] — it shows continuity.

**How to cite (IEEE):** in text "FedAvg [1] averages…", "as shown in [3], [4]".
Reference list numbered in order of first appearance. Zotero does the
numbering if you use its Word/LibreOffice plugin.

**Rule:** if a sentence contains a fact you didn't measure yourself, it needs a [n].

---

# Part 6 — Writing rules for you (ADHD-friendly)

1. **Ugly first.** Write the raw version without fixing anything. The handout
   calls it "Zero Draft". Fix language on Day 3 only.
2. **One section = one 45‑minute block.** Timer on. Phone in another room.
   15 min break. Never "write chapter 3" — always "write 3.4".
3. **Start from tables, not from words.** Paste the table, then write 3
   sentences describing it. Results chapters write themselves this way.
4. **Talk first if stuck.** Record yourself explaining the section for 2
   minutes (the handout suggests this), then type what you said.
5. **When stuck >10 min, skip** and leave `[TODO: …]`. Come back later.
6. **Ask me per section**: "explain 2.5 again", "check my 3.7 paragraph" —
   I'll review what *you* wrote, point out gaps, and check numbers against the
   repo. I won't write your paragraphs; that keeps the declaration true.

---

# Part 7 — The 3-day plan

Blocks are 45 min. 8 blocks a day is a strong day; 6 is fine.

## Day 1 — the concrete chapters (Results + Implementation)
| # | Block | Done |
|---|---|---|
| 1 | Part 0: email professor; tell me whether to re-run experiments; get BHT formatting rules | ☐ |
| 2 | Set up document: BHT template, headings for all chapters (copy Part 3 headings) | ☐ |
| 3 | 5.1 + 5.2 | ☐ |
| 4 | 5.3 + 5.4 | ☐ |
| 5 | 5.5 + 5.6 | ☐ |
| 6 | 5.7 + 5.8 | ☐ |
| 7 | 4.1 + 4.2 + 4.3 | ☐ |
| 8 | 4.4 + 4.5 + 4.6 + 4.7 | ☐ |

## Day 2 — explain how and why (Methodology + Background + Discussion)
| # | Block | Done |
|---|---|---|
| 1 | 3.1 + 3.2 + 3.3 | ☐ |
| 2 | 3.4 + 3.5 + 3.6 | ☐ |
| 3 | 3.7 + 3.8 | ☐ |
| 4 | Zotero: import refs 1–25 (30 min) | ☐ |
| 5 | 2.1 + 2.2 (with formulas 4.1, 4.3) | ☐ |
| 6 | 2.5 (with formulas 4.2) | ☐ |
| 7 | 2.3 + 2.4 + 2.6 + 2.7 + 2.8 | ☐ |
| 8 | Chapter 6 | ☐ |

## Day 3 — frame it, polish, submit
| # | Block | Done |
|---|---|---|
| 1 | Chapter 7 | ☐ |
| 2 | Chapter 1 | ☐ |
| 3 | Abstract (5 sentences: problem, method, 3 results, conclusion) | ☐ |
| 4 | Figures: captions, numbering, lists of figures/tables | ☐ |
| 5 | Numbers check: every number in text matches RESULTS / JSON (I can help) | ☐ |
| 6 | Language pass ch. 1–4 | ☐ |
| 7 | Language pass ch. 5–7, references formatted | ☐ |
| 8 | Declaration, PDF export, print/upload per BHT rules | ☐ |

If you fall behind: drop 2.6 and 2.7 to half a page each, shorten Ch. 6.
Never drop 3.7, 5.5, 5.7 or the Limitations section.
