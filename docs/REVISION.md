# Revision sheet — the whole thesis, step by step

The story in the order it happened, with what was actually done at each step.
Read top to bottom once before the evaluation.

Depth lives elsewhere: [PRIMER.md](PRIMER.md) (concepts from zero),
[RESULTS.md](RESULTS.md) (every number), [FUSION.md](FUSION.md) (fusion model),
[DEFENSE_QA.md](DEFENSE_QA.md) (30 questions in full).

---

# Part 1 — The premise

## 1.1 The problem
A robot or vehicle perceives the world with a camera. In fog, smoke or darkness
the camera is useless. But a car still *sounds* like a car and a dog still
barks — **sound does not care about light**. So: give the machine hearing as a
backup sense for when vision fails.

## 1.2 Why that is hard
Teaching a machine to recognise sounds needs thousands of labelled examples
("this noise = car"). Labelling those by hand is infeasible.

## 1.3 The trick that solves it
Let the camera do the labelling. While the camera *can* see, an off-the-shelf
detector (YOLOv8) names what is there — and whatever the microphone recorded at
that same instant gets that name. The working sense teaches the unavailable
one. Later, in darkness, the audio model answers alone.

This is **cross-modal supervision**, and it means the system labels its own
data forever, with no human involved.

## 1.4 The actual thesis question
Because the system labels itself, it can keep learning forever. So:

> **Where should that continuous training run** — on a central server, partly
> on the device, or entirely on the device?

That is the thesis. Audio recognition is the *vehicle*; training **placement**
is the *subject*.

## 1.5 The one thing to never confuse
There are two models, doing two different jobs:

| | YOLOv8 | Our model |
|---|---|---|
| Role | the **teacher** (labeller) | the **student** |
| Trained by us? | no — fixed, pre-trained | **yes — this is the thesis subject** |
| Input | camera frame | audio (later: audio + frame) |
| Needs light? | yes | **no** |

---

# Part 2 — The setup: what is actually running where

Before any strategy makes sense, be clear about the physical picture. There are
only three kinds of machine in this project.

## 2.1 The two roles

**The edge / client — a Raspberry Pi with a camera and a microphone.**
It sits out in the world. It has sensors, it is weak, and it is on a slow or
expensive network. In the experiments there are **two** of them, called node A
and node B.

**The server — my laptop.**
It has no sensors. It is powerful, it has the test set, and it holds the
official model that everyone shares.

**The broker — mosquitto, also running on the laptop.**
The post office in between. Neither side talks to the other directly; both talk
to the broker.

```
   ┌─────────────────┐                        ┌──────────────────────┐
   │  EDGE (Pi #A)   │                        │  SERVER (laptop)     │
   │  camera + mic   │ ──┐                ┌── │  holds the model     │
   └─────────────────┘   │   ┌────────┐   │   │  holds the test set  │
                         ├──►│ BROKER │◄──┤   └──────────────────────┘
   ┌─────────────────┐   │   │mosquitto│  │
   │  EDGE (Pi #B)   │ ──┘   └────────┘   │
   │  camera + mic   │                     (broker also runs on the laptop)
   └─────────────────┘
```

## 2.2 What one client actually does, once
This loop is the heartbeat of the whole system. Every 500 milliseconds:

1. **Record** 500 ms of sound from the microphone.
2. **Grab** one frame from the camera at the same moment.
3. **Label it:** run YOLOv8 on the frame. If YOLO sees a car with enough
   confidence, the label is "car". *If YOLO is unsure, the sample is thrown
   away* — better no label than a wrong one.
4. **Throw the image away.** The picture has done its only job (producing the
   label). It is never stored and never transmitted.
5. **Keep** the pair `{audio, label}` in a local buffer.

That's one **tick**. Nothing is sent yet.

## 2.3 When and how data is sent
Sending happens **in batches, not every tick** — this matters, and examiners
ask about it. Sending 500 tiny messages costs far more overhead than one
message of 500 samples, and radios are most efficient in bursts.

So: the client buffers **500 samples** (~4 minutes of ticks), and only then:

1. **Pack** — all 500 audio clips are stacked into one array and written as a
   numpy binary blob, with the labels and timestamps as a small JSON block
   beside it. *Why not JSON for the audio too:* a number written as text costs
   ~10 bytes; the same number in int16 binary costs 2. Binary is 2–3× smaller
   before compression even starts.
2. **Compress** with zlib.
3. **Publish** the resulting bytes to the broker as a single MQTT message.

## 2.4 How the message finds its way — MQTT topics
MQTT is a publish/subscribe system. Nobody addresses anybody. A sender publishes
to a **topic** (a label like a folder path) and anyone subscribed to that topic
receives it.

| Topic | Direction | Carries |
|---|---|---|
| `thesis/edge/A/data` | client → server | Strategy A/B: the batch of audio or spectrograms |
| `thesis/edge/A/weights` | client → server | Strategy C: the trained model weights |
| `thesis/server/model/global` | server → clients | the updated shared model |

The server subscribes to `thesis/edge/+/data` — the `+` is a wildcard meaning
"any node", so one subscription catches both A and B.

Three transport details worth knowing if pushed:
- **QoS 1** — the broker guarantees delivery at least once, so a batch is not
  silently lost.
- **Retained** global model — the server's broadcast is marked "retained", so a
  client that connects *late* still immediately receives the current model
  instead of waiting for the next round.
- **Round numbers** on every message — the server ignores a straggler that
  arrives from a round that has already closed.

## 2.5 What the server does with it
The server subscribes, and when a batch arrives it puts it on a queue and
returns immediately — it does **not** train inside the network callback,
because training for 20 seconds inside a network handler would block the
connection and drop messages. A separate loop drains the queue, decompresses,
adds to a pool, and trains when the pool is large enough. Then it evaluates on
fold 10 and broadcasts the new model back.

## 2.6 The three strategies, in this picture
Now the thesis question is easy to state: **the machines are the same in all
three, only the division of labour moves.**

| | Client does | Wire carries | Server does |
|---|---|---|---|
| **A — Centralized** | record, label | **raw audio** | convert, train, test |
| **B — Hybrid** | record, label, **convert** | **spectrograms** | train, test |
| **C — Federated** | record, label, convert, **train** | **weights only** | **average**, test |

Notice the pattern: A → B → C moves one job at a time from the laptop to the
Pi. Fewer bytes on the wire, more work on the device, and the data itself stays
home.

**The method:** prove the comparison on a **simple audio-only model first**,
then repeat the whole thing with the harder audio+vision fusion model. One
variable at a time. That is why the work has two stages.

---

# Part 3 — Why the experiments are simulated (and what "simulated" means)

This is the question most likely to be asked sharply, so be precise about it.

## What is simulated — and what is not
**Simulated:** only the sensors. Instead of a live microphone, the client reads
a WAV file from UrbanSound8K and hands it to the code as if the mic had just
recorded it. Instead of YOLO labelling a live frame, the label comes from the
dataset's answer key.

**Real:** everything else. Two separate client processes, a real mosquitto
broker, real MQTT messages over the network stack, real serialization and
compression, real byte counts, real training, real evaluation on a held-out
fold.

```
     SIMULATION                              REAL PI
   read WAV from disk   ─── replaces ───►   record from USB mic
   label from CSV       ─── replaces ───►   YOLOv8 on a live frame
  ──────────────────────────────────────────────────────────────
   buffer → serialize → compress → MQTT → server   ← IDENTICAL CODE
```

## Reason 1 — a fair comparison needs identical input
The thesis compares three strategies. If A, B and C each captured their own
live audio, they would each be processing *different sounds*, and any
difference in accuracy or bytes could be caused by the data rather than by the
strategy. By replaying the same dataset in the same order, every strategy sees
**identical data with identical labels** — so any difference must be caused by
the strategy itself. That is a controlled experiment.

**Replay is the stronger method here, not a compromise.**

## Reason 2 — you cannot measure accuracy without ground truth
To say "the model is 71% accurate", you need to know the right answer. Live
capture in my room produces labels like "person" and "cell phone" with no
answer key to check against. UrbanSound8K has one.

## Reason 3 — what is being compared does not depend on the sensor
The thesis asks *where training should run*. That depends on what gets computed
and what crosses the wire — not on whether the audio arrived from a microphone
or a file. The bytes are the same bytes either way.

## And we validated it on real hardware anyway
Simulation proves the comparison; the Pi run proves it is not a simulation
artifact (Part 5.5).

---

# Part 4 — Stage one: build the student model

Before comparing *where* to train, we need something to train.

## Step 1 — Get labelled sound data
**UrbanSound8K**: 8,732 labelled city sounds (siren, dog bark, drilling, …) in
10 classes, pre-split into 10 official **folds**. We use folds 1–8 to train,
fold 9 to validate, and **fold 10 is never touched until the final score** —
so the model is graded on sounds it has never heard.

*Why a public dataset and not our own recordings:* to measure accuracy you need
ground truth to compare against. Our own live recordings have no answer key.

## Step 2 — Turn sound into a picture
Audio is not fed to the model raw. Each 500 ms clip is converted by **STFT**
into a **spectrogram** — a picture of sound: time across, pitch up, brightness
= energy. Our size is **51 × 128**.

*Why:* it turns "recognise a sound" into "recognise an image", and image
recognition is the thing neural networks are best at.

## Step 3 — Train the AudioCNN
A small **CNN** (convolutional neural network), **111,000 parameters** —
deliberately small enough to train on a Raspberry Pi later, which Strategy C
will require.

**Result: 70.97% accuracy on the unseen fold 10.** Random guessing across 10
classes is 10%, so the model genuinely learned. Weakest class: siren (~0.59 F1,
fewer samples and acoustically variable).

**This trained model is the starting point for every strategy run.** All three
strategies begin from the same weights ("warm-started"), so they are compared
on equal footing.

---

# Part 5 — Stage two: the three strategies, one at a time

## 5.0 How every run was set up (identical for A, B and C)
- **2 edge nodes**, called A and B, each a separate process pretending to be a
  Pi. Each gets a **disjoint half** of the training data (A takes even samples,
  B odd) — they never see the same sample.
- A **real MQTT broker** (mosquitto) between them and the server. The network
  traffic is real, not estimated.
- **10 rounds**, 5,000 samples streamed per node, all starting from the same
  70.97% model.
- Bytes are counted **twice** — once by the sender, once by the receiver — and
  cross-checked. This is what makes the bandwidth numbers trustworthy.

Only the **sensors** are simulated (audio is replayed from files instead of a
live mic). Everything measured — the training, the traffic, the bytes, the
accuracy — is real.

*Why replay rather than live capture:* the comparison is only fair if all three
strategies process **identical data with identical labels**. Then any
difference must have been caused by the strategy itself. That is a controlled
experiment; live capture cannot guarantee it.

---

## 5.1 Strategy A — Centralized (the baseline)

### What we did
The device is as dumb as possible: it only records and ships.

```
DEVICE                                    SERVER
record 500 ms audio                       receives raw audio batches
     │                                          │
buffer 500 samples                        converts to spectrogram (STFT)
     │                                          │
compress + send RAW AUDIO ──── MQTT ────► adds to a growing pool
                                                │
                                          trains on the WHOLE pool
                                                │
                                          tests on fold 10, broadcasts weights
```

Every 500 samples the device sends a compressed batch. The server does all the
work: STFT, training, evaluation. Each round it trains on **everything it has
ever received**.

### What we measured
| Round | Pool size | Accuracy | Train time |
|---|---|---|---|
| 1 | 1,000 | 0.712 | 3.7 s |
| 5 | 5,000 | 0.710 | 14.6 s |
| 10 | 10,000 | 0.712 | **26.3 s** |

**Upload: 65.9 MB per node.** Server training time **grows linearly** (3.7 →
26.3 s) because centralized retraining re-processes the entire history every
round.

### What it means
This is the baseline everything else is judged against. Two costs are visible
immediately: the most data on the wire, and a server whose workload grows
without bound as data accumulates. And the device's raw audio — the most
privacy-sensitive form — leaves it.

---

## 5.2 Strategy B — Hybrid (move the preprocessing to the device)

### What we changed
**Exactly one thing:** the STFT moves from the server to the device. The device
now converts audio to a spectrogram itself and ships that instead.

```
DEVICE                                    SERVER
record 500 ms audio                       receives SPECTROGRAMS
     │                                          │
STFT → spectrogram        ← moved here    (no STFT — already done)
     │                                          │
send SPECTROGRAM ──────────── MQTT ─────► pool → train → test → broadcast
```

Everything else — the buffering, the broker, the server's training loop, the
number of rounds — is untouched. That is what makes it a clean single-variable
comparison.

### The reasoning we were testing
A spectrogram throws away information the model doesn't use, so it should be
much smaller than raw audio. Our own analytical estimate predicted **~66% less
bandwidth**.

### What we measured
- Upload: **48.7 MB per node** → **26% less than A**, not 66%.
- Accuracy: **0.70–0.73** — no penalty, as expected.
- Device cost: **9.6 s** of STFT work per node over 5,000 samples.

### What it means — and this is a finding, not a disappointment
**Our own estimate was wrong, and measuring corrected it.** A raw clip is
~16 KB and a float16 spectrogram ~13 KB — they start close, and both compress
similarly on the wire. The theoretical model ignored compression.

The lesson is citable: *analytical bandwidth models overstate the benefit of
edge preprocessing.* Volunteer this before you are asked — correcting your own
published assumption is a strength.

---

## 5.3 Strategy C — Federated (move the training itself to the device)

### What we changed
Now the server stops training altogether. The devices train on their own data
and send only the **learned weights**. No audio, no spectrograms — **no data
ever leaves the device**.

The rhythm also changes: A and B are a continuous stream, C works in
**lock-step rounds**:

```
SERVER                                DEVICE A            DEVICE B
send global weights ───broadcast──►   overwrite local     overwrite local
                                      model with them     model with them
                                            │                   │
                                      train 1 epoch on    train 1 epoch on
                                      ITS OWN data        ITS OWN data
                                            │                   │
  ◄──── weights + sample count ─────────────┘───────────────────┘
FedAvg:  w_global = (n_A/n)·w_A + (n_B/n)·w_B
test on fold 10 → broadcast the new global model → next round
```

**FedAvg** is just a weighted average of the two devices' weights, weighted by
how much data each one had. The server never calls training at all.

### The problem we hit
The first runs used the same learning rate the server used in A and B (1e-3).
Accuracy **oscillated between 0.51 and 0.66** — the federated model was *worse*
than the model it started from. And this happened even though both devices had
evenly-split, statistically identical data, so it was not the usual
"clients see different things" explanation.

### How we fixed it — the learning-rate sweep
We swept the devices' local learning rate and the instability resolved
monotonically:

| Local learning rate | Accuracy | Behaviour |
|---|---|---|
| 1e-3 | 0.51–0.66 | unstable |
| 5e-4 | 0.55–0.72 | still oscillating |
| 1e-4 | 0.68–0.73 | stable |
| **5e-5** | **0.70–0.74** | **stable, equals centralized** |

**Why it happens:** the model was already converged — sitting at the bottom of
a valley. Each device takes a full epoch of large steps and walks some distance
up its own side of the valley. Averaging two points high on opposite slopes
lands you *off* the bottom, not at it. Small local steps keep both copies close
enough that their average is still near the bottom.

*Say it this way:* **"A converged model needs gentler local updates under
FedAvg than centralized training does."**

### What we measured (at the tuned rate)
- Upload: **3.99 MB per node** over 10 rounds — **16× less than A**.
- Accuracy: **0.70–0.74** — matches centralized.
- Server compute: **0.5 s/round** (averaging only) vs A's 3.7–26.3 s.
- Device compute: **158 s** of local training — the price paid.

### What it means
Federated gets centralized accuracy at 1/16 the bandwidth and almost no server
cost, with raw data never leaving the device — **but only after tuning.** The
naive version degrades the model. That instability, and its fix, is a genuine
result: it is *why* careful learning-rate schedules and algorithms like FedProx
exist in the literature.

---

## 5.4 Putting the three side by side

| | A Centralized | B Hybrid | C Federated |
|---|---|---|---|
| Upload per node | 65.9 MB | 48.7 MB | **3.99 MB** |
| Accuracy | 0.71–0.73 | 0.70–0.73 | 0.70–0.74 |
| Device compute | 0 | 9.6 s | 158 s |
| Server compute | 3.7 → 26.3 s/rd | trains | **0.5 s/rd** |
| Leaves the device | raw audio | spectrograms | **only weights** |

**The trade-off in one sentence:** moving training outward collapses bandwidth
(66 → 49 → 4 MB) and server load, and improves privacy — paid for in device
compute (0 → 10 → 158 s), with no accuracy cost.

**Why accuracy is flat everywhere — be ready for this.** It is the *correct*
result, not a broken experiment: the model was already trained on this
distribution, so more of the same data cannot teach it much. These runs measure
the **cost** of each placement. The experiments where accuracy genuinely moves
are the learning-rate sweep and the fusion blackout tests.

---

## 5.5 Running it for real: live Raspberry Pi hardware

Simulation proves the comparison is fair. This step proves the system is not
*only* a simulation.

### The physical setup
```
  Raspberry Pi 5                     WiFi / LAN                 Laptop
  ├── USB webcam (frames)   ──────────────────────────►  ├── mosquitto broker
  └── USB microphone (audio)                             └── the UNCHANGED server
```

Two configuration changes, and **no code changes**:
- The broker switches from listening on `localhost` to `0.0.0.0`, so the Pi can
  reach it across the LAN (`configs/mosquitto_lan.conf`).
- The Pi client is pointed at the laptop's LAN IP instead of localhost.

### What runs where
On the **laptop** — the broker and the exact same server used in the simulated
runs:
```bash
mosquitto -c configs/mosquitto_lan.conf
.venv/bin/python -m src.experiments.strategy_a_server --broker localhost
```

On the **Pi** — the live client:
```bash
.venv/bin/python -m src.experiments.strategy_a_pi_client \
    --node-id A --broker <LAPTOP_LAN_IP> --duration-min 2
```

### What changed in the code — almost nothing, and that's the point
Only the front of the client's loop differs:

| | Simulated client | Pi client |
|---|---|---|
| where audio comes from | a WAV file on disk | the USB microphone |
| where the label comes from | the dataset's CSV | **YOLOv8, on a live frame** |
| buffer → serialize → publish | **identical** | **identical** |
| broker + server | **identical** | **identical** |

### What the live run produced
Two minutes of live capture: **119 ticks**, of which **~26% were labelled** —
YOLO fired on person, cell phone, keyboard and traffic light; the rest produced
no confident detection and were correctly **skipped**. **452 KB uploaded**, and
the sender's byte count matched the receiver's **exactly**.

An evidence gallery was saved for the demo — for each labelled tick, the frame
YOLO saw, the 500 ms of audio, and the label it assigned. It contains a `vase`
mislabel: a live specimen of the ~20% label noise, which is honest to show.
*(Frames are saved only in this debug mode; the production path discards them.)*

### What it proves
The server **could not tell a real Pi from a simulated one** — it received the
same topics, the same payload format, the same byte counts. That is the
architecture working as designed: only the sensor front-end differs, everything
downstream is shared.

*Good story if asked what went wrong:* the webcam's built-in microphone output
a constant full-scale signal. There was no driver error — we caught it by
inspecting the data itself (RMS equalled the maximum ⇒ a constant, not sound).
Replaced with a separate USB mic. The lesson written into the thesis: **validate
sensors by looking at their data, not their status.**

*Honest scope:* only Strategy A has a Pi client so far. B and C would follow the
same pattern — swap the sensor source, reuse everything downstream — and the
Pi-side timing measurement for the fusion model is the one remaining task.

---

# Part 6 — Stage three: the fusion model

## 6.1 Why go further
Stage one proved the comparison method on an **audio-only** model. But the
thesis is audio-**visual**, and so far vision only *labelled*. Stage three
builds the model that actually uses both senses at once — and lets us test the
motivating claim directly: **does audio really save you when vision fails?**

## Step 1 — Build the two-sense model
```
   camera frame                    500 ms audio
        │                               │
        ▼                               ▼
  MobileNetV3-Small              audio CNN (3 conv blocks)
  pre-trained, FROZEN            TRAINED
        │                               │
   256 numbers ──── concatenate ──── 128 numbers
                        │
                  fusion head → prediction
```
- **The eye:** MobileNetV3-Small, pre-trained, **frozen** — never trained by
  us. A small trainable layer adapts its output to 256 numbers.
- **The ear:** the AudioCNN architecture → 128 numbers. **Trained**, because
  "what a dog sounds like" is the one skill nobody ships pre-trained.
- **The head:** glues 256 + 128 = 384 numbers together and decides. This is
  where judgment lives — which sense to trust for which class.

1.33M parameters, ~390K of them trainable. Merging encoded senses like this is
called **late fusion**.

## Step 2 — Get paired data (UrbanSound8K can't do this)
Training two senses needs examples with **both captured at the same moment**:
(image, audio, label). UrbanSound8K has no video, so it cannot train an eye.

**VGGSound** (Oxford): ~200,000 ten-second YouTube clips with human sound
labels. Oxford distributes only the list of video IDs, so everyone downloads
the clips themselves. We harvested **5 classes** (car, motorcycle, dog, cat,
bird), taking each clip's middle frame + its 16 kHz audio → **973 pairs**
(3,008 training samples, 98-clip test set, chance = 20%).

Two details worth stating before you are asked:
- Labels here are **VGGSound's human labels, not YOLO's**. YOLO is the labeller
  for live Pi capture — a different job.
- The split is **by clip**, not by audio window. All windows of one clip stay
  in one split, or the model would be tested on near-copies of its own training
  data (**leakage**).

## Step 3 — Train it, and watch it fail
The straightforward version ("naive fusion") scored **0.745 — worse than the
eye alone at 0.806.** Adding a second sense made it *worse*.

**Diagnosis: modality gradient starvation.** The pre-trained eye explains the
labels from the first minute. The loss drops. The learning signal reaching the
from-scratch ear shrinks to nothing, so the ear never develops — and the head
learns to ignore it entirely.

**How we proved that was the cause** (worth telling — it shows method):
1. Different training runs landed on *suspiciously identical* accuracies — an
   identical number is a bug smell.
2. With black frames the model predicted **one constant class** for all 98
   clips, and swapping the audio changed **0 of 20** predictions. Audio
   influence: exactly zero.
3. A probe inside the model showed the audio branch's output barely varied
   between completely different sounds — the branch itself was flat.

## Step 4 — The recipe that fixes it (order matters)
1. **Warm-start the ear** — copy the weights of the already-trained audio-only
   model into the fusion model's audio branch. The ear starts competent, so the
   head takes it seriously.
2. **Then modality dropout 0.5** — replace the image with black on half the
   training samples, so the model can never rely on the eye alone.

**Dropout alone does not work** — it cannot revive a branch that produces no
signal. The ear must be made competent *first*.

## Step 5 — Test the actual claim: the blackout ("fog") test
Every model is evaluated twice on the same 98 clips: **clean** (real frame +
audio), and **blackout** (every frame replaced with black, audio untouched —
simulating fog, darkness, or a dead camera).

| Model | Clean | Blackout | What it shows |
|---|---|---|---|
| vision-only | 0.806 | 0.204 | blind in the dark (0.20 = random) |
| naive fusion | 0.745 | 0.173 | ignores audio → dies with vision |
| **fusion + recipe** | **0.847** | **0.551** | **best clean AND survives the dark** |
| audio-only | 0.561 | 0.561 | the "camera dead" floor |

**Two claims, both measured:**
1. **Fusion beats either sense alone** — 0.847 > 0.806 (vision) > 0.561 (audio).
   The senses genuinely combine. But only with the recipe.
2. **Audio saves you when vision fails** — in blackout, vision-only collapses to
   random guessing (0.204) while the fused model holds **0.551**, essentially
   everything audio alone could give. **Graceful degradation instead of
   collapse.** This is the sentence the whole thesis is built on, now proven.

---

# Part 7 — Stage four: the strategies again, with fusion (FB / FC)

Same placement question, harder model. One rule decides every placement:

> **Frozen parts can sit on the device. Trainable parts must live wherever
> training happens.**

And one constant: **frames never leave the device.** The device ships the
frozen eye's 576-number **feature vector** (~1 KB, and *not* reversible into a
recognisable image) instead of the picture.

- **FB (hybrid fusion)** — server trains, so all trainable parts are on the
  server. The device runs only the frozen eye and the STFT, and ships
  spectrogram + feature vector.
- **FC (federated fusion)** — devices train, so each holds a full copy of every
  trainable part. Only weights move.

| | FB | FC |
|---|---|---|
| What ships | spectrogram + feature vector (10.3 KB/sample) | weights only (1.41 MB/round) |
| Upload/node | 14.78 MB | 13.76 MB |
| Clean accuracy | recovers to 0.837 | stable 0.84–0.88 |
| Blackout accuracy | **rises 0.52 → 0.67** | 0.59–0.68 |
| Server compute | trains (2.6→7.3 s/rd) | averages (0.2 s/rd) |

**Findings:** (1) the darkness-survival property **survives and even improves**
under continued strategy training (0.55 → ~0.67); (2) **federated cost scales
with model size, not data volume** — 1.41 MB/round is 3.5× the audio model's
400 KB, exactly matching the ratio of trainable parameters. At this data volume
FB and FC cost about the same; at 10× the data FB grows 10× while FC does not
move at all.

---

# Part 8 — Questions to be ready for

**How is the data actually transferred?**
Over **MQTT**, through a mosquitto broker running on the laptop. The client
buffers 500 samples, stacks the audio into one numpy array with the labels as a
small JSON block, compresses it with zlib, and publishes it as a single message
to the topic `thesis/edge/A/data`. The server subscribes to
`thesis/edge/+/data` — the wildcard catches every node with one subscription —
and broadcasts the updated model back on `thesis/server/model/global`. Delivery
is QoS 1, so a batch is not silently lost, and the global model is a *retained*
message so a client that connects late still gets the current model
immediately.

**Why batches instead of sending every sample as it is captured?**
Overhead and radio efficiency. Five hundred individual messages carry 500 sets
of MQTT and TCP headers and keep the radio awake continuously; one batched
message carries one. Compression also works far better across 500 clips than on
one. This is why the thesis says training is **batched, not real-time**.

**Does the camera image get transmitted?**
No — and this is a deliberate privacy property. The frame exists only long
enough for YOLO to read a label off it, then it is discarded on the device.
Only `{audio, label}` is ever buffered or sent. In the fusion strategies, where
the server does need something visual, the device sends a 576-number **feature
vector** rather than the picture — about 1 KB, and not reversible into a
recognisable image.

**How would this run on real edge hardware rather than on your laptop?**
It already did, for Strategy A. The only changes are configuration: the broker
binds to `0.0.0.0` instead of localhost so the Pi can reach it over the LAN,
and the Pi client is pointed at the laptop's IP. In the code, only the front of
the client's loop differs — audio comes from a USB microphone instead of a WAV
file, and the label comes from YOLOv8 running on a live frame instead of from a
CSV. Buffering, serialization, compression, MQTT and the entire server are the
same code. The live run uploaded 452 KB in two minutes with sender and receiver
byte counts matching exactly.

**Isn't labelling audio with vision circular?**
No — the two models answer different questions at different times. YOLO answers
"what is visible NOW" and needs light. The audio model learns "what does X
sound like" so it can answer LATER without light. And we measured the
assumption instead of assuming it: YOLO's frame label matches the human sound
label ~80% of the time.

**20% of your labels are wrong. How can that work?**
Standard **weak supervision**. Wrong labels are *inconsistent* — a dog clip
mislabelled "car" today, "bird" tomorrow — while correct labels always agree.
So correct signals accumulate and noise partly cancels. Deep networks are known
to tolerate substantial label noise. Also, when YOLO is unsure it produces no
label and the sample is *skipped* — that costs data volume, not correctness.

**Your experiments are simulated. Why trust them?**
Only the sensors are replayed. The training, the MQTT traffic, the byte counts
and the accuracies are all real. And replay is the *stronger* method here:
every strategy sees identical data, so differences are caused by the strategy
alone. Separately, the whole pipeline was validated live on a Pi 5, with
sender and receiver byte counts matching exactly.

**Why is accuracy flat across rounds? Is anything learning?**
Flat is the correct result — the model was already trained on that
distribution, so more of the same adds little. These runs measure the *cost* of
each placement. Accuracy moves in the LR sweep and the blackout tests.

**Federated matched centralized. Suspicious?**
Only *after* tuning the local learning rate down to 5e-5. Naive FedAvg at 1e-3
degraded the model. I show the full sweep, not only the good number.

**Walk me through one federated round.**
Server broadcasts the global weights → each device overwrites its local model
with them → each trains one epoch on its **own** data → each sends back weights
plus its sample count → the server averages them weighted by sample count →
broadcasts the result. Repeat. The data never moves; the model commutes.

**Is federated learning actually private? Weights still leave.**
It removes the raw-data exposure, which is the biggest and most legible risk.
It is **not** information-theoretically private — gradient-inversion and
membership-inference attacks are known. Secure aggregation and differential
privacy are the standard hardening, out of scope but acknowledged. My claim is
precisely *"raw audio never leaves the device"*, not *"zero information
leaves"*.

**Why only 2 devices?**
Two is the minimum that makes federation meaningful, and it matches the stated
hardware scope. FedAvg is a weighted mean over any number of clients, so the
architecture scales to N unchanged.

**Why did naive fusion do worse than vision alone?**
Modality gradient starvation — the pre-trained eye soaks up the learning
signal, the from-scratch ear never develops, the head wires it out. The fix
(warm-start, then modality dropout) is the finding, not a workaround.

**How do you know it really uses audio now?**
Three probes, all of which failed before the fix and pass after: blackout
accuracy 0.551 vs 0.204 chance; predictions **change** when only the audio is
swapped (before: 0 of 20 changed); the audio branch's feature variance is
healthy rather than near-constant.

**Real fog isn't pure black — doesn't the blackout test flatter you?**
Fair. Pure black is the extreme end, chosen because it is a clean, reproducible
worst case. The proper next step is intermediate degradations — blur, noise,
low contrast — to show the whole curve rather than two endpoints.

**Why ship a feature vector instead of the frame?**
~1 KB versus 30–50 KB, and 576 numbers cannot be inverted back into a
recognisable image — bandwidth *and* privacy improve together. It is only safe
because the eye is frozen: its output never changes, so device and server
always agree. We verified 100% prediction agreement between the
feature-vector model and the image model.

**Why is the vision branch frozen?**
Three reasons: a device cannot afford to train a 940K-parameter image backbone;
freezing keeps the feature-vector format valid (a fine-tuned server copy would
drift from the device's); and pre-trained features are already strong for these
classes. The trade-off — a few points of vision accuracy — is acknowledged.

**Your title says audio-visual but the strategy comparison used an audio-only
model.**
Both parts are true. The system was always audio-visual because **vision labels
audio**. And the full fusion model is now trained, evaluated (0.847 / 0.551)
and run under the strategies too (FB / FC). The simpler model came first
deliberately — establish the method on one variable at a time.

**What is your single main contribution?**
A measured, end-to-end comparison of *training placement* for a self-labelling
audio-visual edge system — plus four honest secondary findings: compression
erases most of hybrid's theoretical advantage; naive FedAvg destabilises a
converged model; naive multimodal fusion silently ignores the weaker sense and
a warm-start + modality-dropout recipe fixes it; and federated bandwidth scales
with model size, not data volume.

**What are the limitations?**
Single-fold evaluation rather than 10-fold cross-validation; a small 98-clip
fusion test set (treat those numbers as ±5%); evenly-split data so far, with
the uneven (non-IID) drift experiment still planned; and VGGSound frames are
unusually vision-friendly — they are videos *of* the labelled object — which
flatters the vision baseline. Real street scenes would be kinder to audio.

**What would you do with three more months?**
In priority order: the model-freshness experiment (how fast each strategy
adapts to a brand-new object — the sharpest difference between A and C);
uneven data distribution and FedProx; fusion training on the real Pi;
10-fold cross-validation and a larger fusion test set; and the
intermediate-degradation fog curve.

---

*If a number is challenged, the winning move is provenance: every figure in
[RESULTS.md](RESULTS.md) traces to a JSON file under `results/` with the exact
command in `invocation.txt`, and byte counts are recorded independently by both
sender and receiver and cross-checked.*
