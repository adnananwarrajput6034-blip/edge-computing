# 5 Results

This chapter presents what I measured. Each section shows a table or a
figure and describes what it shows; what the results mean is discussed in
Chapter 6. Every number comes from a result file in my repository, and
Appendix A lists the commands for the main experiments. The strategy,
fusion and labelling experiments were run again on 1 October 2026, so that
every number can be traced to a file; the baseline model is from July and
the live Pi results are from September 2026.

## 5.1 Building Blocks

### 5.1.1 YOLOv8 as a labeller

To check how good the automatic labels are, YOLOv8n was run on 150 VGGSound
video thumbnails (30 per class) and its label was compared with the human
label of the sound.

| | Count | Share |
|---|---|---|
| YOLO label matches the sound label | 40 | 26.7% of all, **85.1% of detections** |
| YOLO gives a different label (label noise) | 7 | 4.7% of all, 14.9% of detections |
| YOLO detects nothing (sample skipped) | 103 | 68.7% |

*Table 5.1: YOLOv8n as an automatic audio labeller. Source:
`data/vggsound/label_noise_results.json`.*

When YOLO made a detection, it was correct in 85% of the cases, so about 15%
of the automatic labels are wrong. The typical mistakes are between similar
classes, such as motorcycle and car or dog and cat. In about two thirds of
the frames YOLO found none of the five objects; these samples are skipped,
which costs data but does not add wrong labels. Thumbnails are often title
images rather than real scenes. During the live enrollment (Section 5.6) YOLO made a confident detection in 64.8% of the frames, although there any COCO class counted, not only these five. YOLOv8n took
about 36 ms per image on the laptop and about 318 ms per frame on the
Raspberry Pi 5.

### 5.1.2 Baseline AudioCNN

The AudioCNN was trained on UrbanSound8K folds 1–8 with fold 9 for
validation. Training stopped after 21 epochs (early stopping), which took
about 45 minutes on the laptop. On the unseen fold 10 (837 clips) the model
reached a **test accuracy of 64.9%** and a macro F1 score of 0.649. Random
guessing between ten classes would give 10%.

| Class | Precision | Recall | F1 |
|---|---|---|---|
| air conditioner | 0.52 | 0.85 | 0.65 |
| car horn | 0.69 | 0.73 | 0.71 |
| children playing | 0.53 | 0.36 | 0.43 |
| dog bark | 0.75 | 0.62 | 0.68 |
| drilling | 0.74 | 0.82 | 0.78 |
| engine idling | 0.53 | 0.73 | 0.62 |
| gun shot | 0.68 | 0.94 | 0.79 |
| jackhammer | 0.99 | 0.69 | 0.81 |
| siren | 0.76 | 0.27 | 0.39 |
| street music | 0.62 | 0.68 | 0.65 |
| **macro average** | **0.68** | **0.67** | **0.65** |

*Table 5.2: Per-class results of the baseline AudioCNN on fold 10. Source:
`data/urbansound8k/training_metrics.json`.*

The weakest classes are siren and children playing. The confusion matrix
shows that sirens are often mistaken for air conditioners, children playing
and dog barks. Many siren recordings contain long stretches of traffic noise,
and a single 500 ms window often does not contain the siren itself. The
strongest classes, gun shot and jackhammer, have short and very
characteristic patterns.

This model is the starting point for all strategy runs in stage one.

## 5.2 Strategy A: Centralized

| Round | Samples in pool | Test accuracy | Server training time |
|---|---|---|---|
| 1 | 1,000 | 0.638 | 3.6 s |
| 2 | 2,000 | 0.681 | 7.3 s |
| 3 | 3,000 | 0.644 | 10.0 s |
| 4 | 4,000 | 0.638 | 12.2 s |
| 5 | 5,000 | 0.612 | 14.5 s |
| 6 | 6,000 | 0.591 | 18.2 s |
| 7 | 7,000 | 0.639 | 22.0 s |
| 8 | 8,000 | 0.664 | 23.1 s |
| 9 | 9,000 | 0.671 | 26.3 s |
| 10 | 10,000 | 0.661 | 30.1 s |

*Table 5.3: Strategy A per round. Source: `results/strategy_a/run_20261001_160849/server.json`.*

Each node streamed 5,000 samples in about 42 minutes (one sample every
500 ms) and uploaded them in 10 batches. Node A uploaded 69.24 MB and node B
69.06 MB of compressed raw audio, about 13.8 KB per sample. The byte counts
recorded by the clients and by the server matched exactly, and no upload
failed.

The server training time grew linearly, from 3.6 s in round 1 to 30.1 s in
round 10, because the centralized server trains on everything it has ever
received. The test accuracy moved between 0.59 and 0.68 and ended at 0.661,
close to the 0.649 of the starting model. It dipped in rounds 5 and 6 and
recovered afterwards.

![Figure 5.1: Test accuracy (left) and server training time (right) per round for Strategy A. Source: own work.](docs/figures/14_result_strategy_a.png)

## 5.3 Strategy B: Hybrid

| Round | Test accuracy | Server training time |
|---|---|---|
| 1 | 0.621 | 3.7 s |
| 5 | 0.654 | 13.8 s |
| 10 | 0.688 | 30.1 s |

*Table 5.4: Strategy B, selected rounds. Source: `results/strategy_b/run_20261001_165113/server.json`.*

Moving the STFT to the device reduced the upload to 51.01 MB (node A) and
51.07 MB (node B) per node, which is **26.2% less** than Strategy A. The
accuracy moved between 0.62 and 0.70 and ended at 0.688, so there was no
accuracy penalty. Each device spent 11.3 s in total computing spectrograms
for its 5,000 samples, about 2.3 ms per clip.

The saving is modest because a raw 500 ms clip is 16 KB (8,000 samples ×
2 bytes) and a 16-bit spectrogram is about 13 KB (51 × 128 × 2 bytes), so
the two are already close in size. Section 5.10.2 derives the expected
saving from these sizes and explains the difference from the measured value.

## 5.4 Strategy C: Federated and the Learning-Rate Sweep

The first federated run used a local learning rate of 1e-3, the same value
the server uses in strategies A and B. The accuracy fell to 0.42 in the
first round, far below the 0.649 of the starting model, and then jumped up
and down. To find the cause, the local learning rate was varied while
everything else stayed the same.

| Local learning rate | Accuracy range over 10 rounds | Final accuracy | Behaviour |
|---|---|---|---|
| 1e-3 | 0.42–0.66 | 0.554 | unstable, mostly far below the start |
| 5e-4 | 0.51–0.70 | 0.607 | good start, then collapses and oscillates |
| 1e-4 | 0.62–0.70 | 0.699 | mostly above the start, still swings |
| 5e-5 | 0.65–0.69 | 0.676 | stable, only round 1 marginally below the start |

*Table 5.5: Effect of the local learning rate on federated training. Source:
the four `results/strategy_c/run_20261001_18*` folders.*

![Figure 5.2: Test accuracy per federated round for four local learning rates (dashed: starting model). Source: own work.](docs/figures/15_result_lr_sweep.png)

The behaviour improved step by step as the learning rate went down. At
5e-5 the accuracy stayed within a band of 0.05, and only round 1 dipped
marginally below the starting model (0.645 against 0.649). At 1e-4 the best
single round was slightly higher, but the swings were about 1.7 times as
large. Because a deployed system cannot choose which
round it stops at, 5e-5 was chosen as the setting for the comparison.

With this learning rate, each node uploaded 4.18 MB over ten rounds, about
418 KB per round, and the upload was the same for all four learning rates,
because it depends only on the size of the model. The server only averaged
the weights, which took 0.66–0.74 s per round. Each device spent 233 s in
total on local training, about 23 s per round for 3,540 clips.

## 5.5 Comparison of the Three Strategies

| | A Centralized | B Hybrid | C Federated (LR 5e-5) |
|---|---|---|---|
| Upload per node | 69.2 MB | 51.0 MB | 4.18 MB |
| Upload compared with A | – | −26.2% | **16.5× less** |
| Accuracy range (final) | 0.59–0.68 (0.661) | 0.62–0.70 (0.688) | 0.65–0.69 (0.676) |
| Device compute | 0 | 11.3 s (STFT) | 233 s (training) |
| Server compute per round | 3.6 → 30.1 s, growing | 3.7 → 30.1 s, growing | 0.7 s, constant |
| Data leaving the device | raw audio | spectrograms | weights only |

*Table 5.6: Comparison of strategies A, B and C.*

![Figure 5.3: Upload per node for strategies A, B and C. Source: own work.](docs/figures/16_result_upload.png)

From A to C, upload bandwidth falls from 69 to 51 to 4 MB per node and the
server's work falls from a growing training job to a constant averaging
step of under a second. The work on the device rises from 0 to 11 to 233
seconds. All three strategies end within 0.03 of each other in accuracy,
and all three stay close to the starting model. In every run the sender's
and the receiver's byte counts matched exactly and no upload failed.

## 5.6 Live Validation on the Raspberry Pi

### 5.6.1 Enrollment

During live enrollment the Raspberry Pi 5 recorded 247 ticks in about four
minutes. YOLO produced a confident label for 160 of them (64.8%). The
classes car (40 samples), cell phone (33) and person (55) appeared often
enough and became the vocabulary. Rare detections such as traffic light,
bird and tv were dropped. The 128 samples were split into 96 for training
and 32 for testing.

![Figure 5.4: Frames from the live enrollment with the label YOLO assigned (top right and bottom left are wrong labels). Source: own work.](docs/figures/13_evidence_gallery.jpg)

During enrollment the objects were shown as videos on a phone screen held in
front of the camera, and the phone's speaker played the matching sound. This
made it possible to test classes like car indoors. It also produced live
examples of label noise (Figure 5.4): YOLO sometimes labelled the phone
itself ("cell phone") or a small object inside the video ("traffic light")
instead of the object that made the sound.

### 5.6.2 Strategies A, B and C on the Pi

The same enrolled dataset was then replayed by the Pi clients of all three
strategies over the local network to the unchanged laptop server.

| | A Centralized | B Hybrid | C Federated |
|---|---|---|---|
| Upload | 1,337 KB | 979 KB (−26.7%) | 1,245 KB (3 rounds, ~415 KB each) |
| Work on the Pi | – | STFT 1.1 s | STFT 1.1 s + training 52.7 s |
| Server time per round | 2.3–3.0 s training | 2.2–2.8 s training | 0.09–0.18 s (one client, so no real averaging) |
| Final test accuracy | 0.531 | 0.531 | 0.563 |
| Raw audio kept on device | 0 | 1,536 KB (spectrograms sent instead) | 1,536 KB |

*Table 5.7: Live run on the Raspberry Pi 5 with 96 training samples and three
classes (chance level 0.333). Source: `data/pi_captures/results/live/`.*

![Figure 5.5: Upload and final accuracy of the three strategies on the real Raspberry Pi 5. Source: own work, from the result files in `data/pi_captures/results/live/` and `results/demo/`.](docs/figures/19_result_live_pi.png)

With only 96 training samples the accuracy is modest, but clearly above
chance. The pattern is the same as in the replayed experiments: B uploads
less than A, C uploads a fixed amount per round, and the work on the device
grows from A to C. With this small dataset, C uploads about as much as A,
because three rounds of model weights are about as large as 96 audio clips.
The server handled the data from the real Pi in the same way as the data
from the replayed clients.

The live demonstrator will also be shown during the defense: the objects
are chosen on the spot, the Pi enrolls them, the three strategies train on
the new data, and the trained model is tried on live sound. The numbers of
that run will differ from Table 5.7, because the objects, the room and the
amount of data will differ, but the pattern should be the same.

For the defense demonstration, the live setup will be extended to two Raspberry Pis, as originally planned. Each Pi records and labels its own data,
trains locally and sends only its weights to the laptop server, which
averages them with FedAvg and sends the new model back to both Pis. This will show in real time what the two replayed nodes showed in software in
Sections 5.4 and 5.5. The software side stays the same; the second device will be added with only small changes (Section 7.3).

### 5.6.3 Which training settings work on the live data?

The live run used one setting (learning rate 1e-3, 8–10 epochs per round).
To check whether this was a good choice, the recorded live dataset was
trained again offline on the laptop with different settings: two learning
rates, with and without the warm start from the UrbanSound8K model, 40
epochs, batch size 8 as on the Pi, and three repetitions with different
random seeds. The audio is exactly what the Pi microphone recorded and the
labels are exactly what YOLO assigned.

| Setting | after 1 epoch | after 5 | after 10 | after 20 | after 40 |
|---|---|---|---|---|---|
| Warm start, LR 1e-3 | 0.41 ± 0.03 | 0.41 ± 0.09 | 0.49 ± 0.06 | 0.50 ± 0.11 | 0.51 ± 0.03 |
| Warm start, LR 1e-4 | 0.36 ± 0.03 | 0.39 ± 0.03 | 0.45 ± 0.05 | 0.47 ± 0.08 | 0.50 ± 0.03 |
| From scratch, LR 1e-3 | 0.29 ± 0.03 | 0.31 ± 0.00 | 0.29 ± 0.03 | 0.29 ± 0.03 | 0.30 ± 0.01 |
| From scratch, LR 1e-4 | 0.31 ± 0.00 | 0.31 ± 0.00 | 0.31 ± 0.00 | 0.31 ± 0.00 | 0.31 ± 0.00 |

*Table 5.8: Test accuracy on the live dataset (32 test clips, chance 0.33),
mean ± standard deviation over three seeds. Source:
`data/pi_captures/results/live/training_sweep.json`.*

Three things stand out. First, the warm start is essential: without it the
model never got above chance, and at LR 1e-4 it predicted the same class for
every clip in every run (10 of 32 = 0.31). With only 96 training clips, the
convolution layers learned on UrbanSound8K are what makes learning possible
at all. Second, more epochs help up to about 10–20 epochs, after which the
accuracy levels off around 0.50. Third, the learning rate matters much less
here than in federated training: both rates end at about the same level.
The setting used in the live demo (LR 1e-3, about 10 epochs) was therefore a
reasonable choice, and the 0.53–0.56 reached in the demo lies within the
spread of this sweep. With 32 test clips, one clip is 3 percentage points,
so differences smaller than about 0.1 should not be over-interpreted.

### 5.6.4 Live inference

The trained model was copied back to the Pi and run on live microphone
audio. With a threshold of 0.95, 3 of 24 ticks were recognised as person
and the rest were reported as unknown, which is the expected behaviour for a
quiet room with sounds the model was not trained on.

## 5.7 The Fusion Model and the Blackout Test

The fusion experiments use the VGGSound subset with five classes (987 clips:
788 for training, 96 for validation, 103 for testing), so random guessing
gives 20%. Each model was evaluated on the same 103 test clips, once with the
real images (clean) and once with every image replaced by a black one
(blackout).

The vision-only and audio-only models were trained once. The fusion models
were trained twice: once with the default settings of my training script
(up to 40 epochs, stopping after 8 epochs without improvement), and once for
a fixed 25 epochs. In both cases the script keeps the weights of the epoch
with the lowest validation loss. The dropout-0.3 model was only trained in
the first run.

| Model | Run 1: clean | Run 1: blackout | Run 2: clean | Run 2: blackout |
|---|---|---|---|---|
| Vision only | 0.806 | 0.204 | (same) | (same) |
| Audio only | 0.505 | 0.505 | (same) | (same) |
| Naive fusion | 0.767 | 0.204 | 0.738 | 0.204 |
| Fusion + modality dropout 0.3 | 0.777 | 0.194 | – | – |
| Fusion + modality dropout 0.5 | 0.680 | 0.175 | 0.748 | 0.126 |
| **Fusion + audio warm start + dropout 0.5** | 0.757 | **0.359** | 0.718 | **0.398** |

*Table 5.9: Clean and blackout accuracy of the fusion experiments (103 test
clips, chance 0.20). Source: `data/vggsound/fusion_blackout_results_early_stopping.json`
and `data/vggsound/fusion_blackout_results.json`.*

![Figure 5.6: Clean and blackout accuracy for each model (run 2). Source: own work.](docs/figures/17_result_fusion_blackout.png)

**The best epoch came very early.** In both runs, the lowest validation
loss was reached after only one or two epochs, so every fusion model kept
those early weights, even when it trained for 25 epochs. The validation set
has only 96 clips, and the pre-trained vision branch fits it almost at once.
The two recipe results (0.359 and 0.398 in blackout) therefore come from
nearly the same point in training and differ only by normal run-to-run
variation. Training longer and keeping the last epoch instead of the best
one was not tested.

**Without the recipe, the camera failure is total.** In blackout, the naive
fusion model and both dropout-only models fell to chance level or below
(0.13–0.20), exactly like the vision-only model. To check whether these
models use the audio at all, two probes were run on every fusion model
(Table 5.10).

| Model | Different classes predicted in blackout | Predictions changed when audio is swapped |
|---|---|---|
| Naive fusion | 1 of 5 | 2 of 103 |
| Fusion + dropout 0.3 | 2 of 5 | 2 of 103 |
| Fusion + dropout 0.5 | 2 of 5 | 3 of 103 |
| **Fusion + warm start + dropout 0.5** | **5 of 5** | **23 of 103** |

*Table 5.10: Does the fusion model listen? Blackout: number of different
classes predicted when all images are black. Audio swap: real images, audio
shuffled between test clips. Source: `data/vggsound/fusion_audio_probe.json`.*

With black images the audio is the only input that still changes, yet the
naive model predicted one single class for all 103 clips. Its blackout score
of 0.204 is exactly the share of that class in the test set (21 of 103), not
a sign of hearing anything. Swapping the audio between clips changed only 2
of 103 predictions. The naive model has learned to ignore the audio
completely. Modality dropout alone did not change this: blacking out the
image during training does not help if the audio branch never learned
anything useful in the first place.

**The full recipe made the model listen.** With the audio branch
warm-started from the trained audio-only model and modality dropout of 0.5,
the model used all five classes in blackout, swapping the audio changed 23
of 103 predictions, and it kept 0.398 accuracy with a black camera, about
twice the chance level and 79% of what the audio-only model achieves.

**The price is clean accuracy.** On clean data, the vision-only model was
the best model (0.806). No fusion model beat it in these runs; the recipe
model reached 0.718. Fusion made the system robust to camera failure, but in
this setup it did not make it more accurate when the camera works.

**Comparison with the earlier run.** An earlier version of this experiment,
whose result files are not available, reported 0.847 clean and 0.551
blackout for the recipe model. That run used a different download of
VGGSound (98 test clips, and a different set of bird clips) and its
audio-only model was stronger (0.561 against 0.505 here). The qualitative
result is the same in both runs: only the recipe model keeps working when
the camera fails. The exact numbers are not, which shows how much a test set
of about 100 clips can move.

## 5.8 Fusion Model under the Hybrid and Federated Strategies

The recipe model (0.718 clean, 0.398 blackout) was then used as the starting
point for the hybrid (FB) and federated (FC) strategies, with two nodes that
each hold half of the 3,152 training samples (1,576 each). FB streamed
samples every 50 ms in batches of 250 and trained after every 500 new
samples, for 6 rounds (3,000 samples in the pool at the end). FC ran 10
rounds with a local learning rate of 1e-4 and one local epoch per round. Both runs measured
clean and blackout accuracy after every round.

| | FB hybrid fusion | FC federated fusion |
|---|---|---|
| What is sent | spectrogram + feature vector (10.3 KB per sample) | weights only (1.44 MB per round) |
| Upload per node | 16.2 MB (1,576 samples) | 14.4 MB (10 rounds) |
| Clean accuracy (final) | 0.67–0.85 (0.845) | 0.79–0.88 (0.854) |
| Blackout accuracy (final) | 0.49–0.60 (0.583) | 0.43–0.67 (0.573) |
| Server compute per round | trains, 2.8 → 9.2 s, growing | averages, 0.2–0.3 s |
| Device compute | feature vector + STFT only | 114 s local training per node |

*Table 5.11: Fusion model under the hybrid and federated strategies. Source:
`results/strategy_fb/run_20261001_193556/` and `results/strategy_fc/run_20261001_193733/`.*

![Figure 5.7: Clean (left) and blackout (right) accuracy per round for FB and FC. Source: own work.](docs/figures/18_result_fb_fc.png)

In Figure 5.7, round 0 is the starting model, and the dashed lines show the
vision-only model (left) and the audio-only model (right). Both strategies
improved the model in both conditions. On clean data, FB
ended at 0.845 and FC at 0.854 (best round 0.883), both above the
vision-only model (0.806). In blackout, both rose from 0.398 to a final 0.583
(FB) and 0.573 (FC), above the audio-only model (0.505); FC reached 0.670 in
its best round. The ability to work without the camera did not
get lost under continued training; it got stronger.

Part of this improvement is simply more training. FB and FC continue to
train on the same kind of data as the starting model, so they effectively
add more epochs. The vision-only and audio-only models did not get this
extra training, so the comparison with them in the last rounds is not fully
equal.

The federated upload per round was 1.44 MB, 3.45 times the 418 KB of the
audio model. The ratio of the numbers they send is 389,893 to 111,370,
which is 3.50. The upload in federated learning follows the size of the
model almost exactly.

## 5.9 Latency

While I was planning this work, latency came up as an important point next to
bandwidth, separated into inference (using the model) and training
(improving the model). This section collects both.

### 5.9.1 Inference latency

Table 5.12 shows how long each step takes for one clip. The laptop values
were measured with batch size 1 and 200 repetitions after a warm-up. The
Raspberry Pi values come from the live run (Section 5.6).

| Step | Laptop (median) | Raspberry Pi 5 | Needed in |
|---|---|---|---|
| YOLOv8n, one camera frame | 36 ms | 318 ms | all strategies (labelling) |
| STFT, one 500 ms clip | 0.65 ms | 11 ms | A on the server; B and C on the device |
| AudioCNN, one prediction | 5.4 ms | not measured | all strategies (inference) |
| MobileNetV3 feature vector, one frame | 44 ms | not measured | FB and FC |
| Fusion head, one prediction | 7.0 ms | not measured | FB and FC |

*Table 5.12: Latency per step for one clip. Sources:
`results/latency/inference_latency_MacBookAir.json` (laptop, except YOLO:
`logs/rerun_20261001/12_label_noise.log`),
`data/pi_captures/results/live/` (Pi: YOLO from the enrollment report, STFT
= 1.06 s / 96 clips).*

Two things follow from this table. First, **inference latency is the same in
all three strategies.** By design, inference always
runs on the device with the same model; the strategies only differ in where
the model is trained. Second, the camera label is by far the slowest step on
the Pi. YOLO uses 318 ms of the 500 ms tick, while the STFT that strategies B
and C add on the device costs only 11 ms. Moving the STFT to the device
therefore does not threaten the real-time loop.

### 5.9.2 Update latency

The strategies differ in how long it takes until a new, improved model
exists. Table 5.13 compares this.

| | A Centralized | B Hybrid | C Federated |
|---|---|---|---|
| Work for one model update | server trains on everything received | server trains on everything received | each device trains on its own data, server averages |
| Time of that work | 3.6 → 30.1 s, growing every round | 3.7 → 30.1 s, growing every round | 23 s per round on the laptop, 17.5 s on the Pi (96 clips), plus 0.7 s averaging |
| Grows with all data ever collected? | yes | yes | no |
| Time between new models in my runs | about 250 s | about 250 s | about 24 s |

*Table 5.13: Update latency of the three strategies. Sources: the run
folders of Sections 5.2–5.4 and the live run in Section 5.6.*

The interval of about 250 s in A and B follows directly from the settings:
the server trains after every 1,000 new samples, and two nodes each deliver
one sample every 500 ms, so

$$\frac{1000 \text{ samples}}{2 \text{ nodes} \times 2 \text{ samples/s}} = 250 \text{ s}$$

which matches the measured 2,503 s for 10 rounds. Most of this time is spent
waiting for data, not training. In my federated runs the data was already on
the devices, so a round took only about 24 s. In a real deployment, Strategy
C would also have to wait for new recordings, but its training time per round
stays constant, while the training time of A and B keeps growing with every
round.

The latency on the Pi for the AudioCNN and the fusion model, as well as
memory use, were not measured and are listed as future work.

## 5.10 Checking the Measurements against the Formulas

A measured number is easier to trust when it can be predicted on paper. In
this section I take the formulas from Chapter 2, put in the settings from my
code, and compare the prediction with what I measured. Where the two differ,
I explain why. The STFT and FedAvg formulas come from the sources cited in
Chapter 2 [1], [27]; the size, break-even and timing calculations are my own
derivations from those formulas.

### 5.10.1 Spectrogram shape

The code (`src/edge/processing/stft.py`) uses $N = 8000$ samples, window
$W = 400$, hop $H = 160$ and $N_{FFT} = 512$.

$$T = 1 + \left\lfloor \frac{N}{H} \right\rfloor = 1 + \left\lfloor \frac{8000}{160} \right\rfloor = 51 \text{ frames}$$

$$K = \frac{N_{FFT}}{2} + 1 = \frac{512}{2} + 1 = 257 \text{ frequency bins}$$

$$\text{overlap} = \frac{W - H}{W} = \frac{400 - 160}{400} = 60\%$$

The 257 bins are reduced to 128 mel bands. With the formula from Section
2.1, the highest frequency, 8 kHz, is
$\text{mel}(8000) = 2595 \cdot \log_{10}(1 + 8000/700) \approx 2840$ mel, so
the 128 bands would be about $2840 / 129 \approx 22$ mel apart (librosa's
Slaney version spaces them slightly differently, with the same idea). The
measured output shape is 51 × 128, as predicted.

### 5.10.2 Bytes per sample in strategies A and B

One raw clip in Strategy A is stored as 16-bit integers:

$$S_A = 8000 \times 2 \text{ B} = 16{,}000 \text{ B}$$

One spectrogram in Strategy B is stored as 16-bit floats:

$$S_B = 51 \times 128 \times 2 \text{ B} = 13{,}056 \text{ B}$$

Before compression, the expected saving of B over A is therefore

$$1 - \frac{S_B}{S_A} = 1 - \frac{13{,}056}{16{,}000} = 18.4\%$$

The measured upload per sample was 13,830 B for A and 10,208 B for B. The
compression factor $c = \text{measured} / \text{uncompressed}$ is

$$c_A = \frac{13{,}830}{16{,}000} = 0.86$$

$$c_B = \frac{10{,}208}{13{,}056} = 0.78$$

So the measured saving is

$$1 - \frac{c_B \cdot S_B}{c_A \cdot S_A} = 1 - \frac{10{,}208}{13{,}830} = 26.2\%$$

The simple size formula predicts 18%; the measurement gives 26%. The extra
8 percentage points come from compression: spectrograms are smoother than
raw audio and compress better (0.78 against 0.86). Both numbers are far from
the large saving one might expect from "sending features instead of raw
data", because a 500 ms spectrogram stored at 16 bit is almost as large as
the 500 ms of audio it was made from.

### 5.10.3 Upload per round in federated learning

From Section 2.5.6, the upload per client and round is the number of
parameters $P$ sent times 4 bytes. For the AudioCNN with
$P = 111{,}370$:

$$B_C \approx 111{,}370 \times 4 \text{ B} = 445{,}480 \text{ B} \approx 445 \text{ KB}$$

Measured: 4,183,866 B over 10 rounds, so 418 KB per round. The ratio
measured / predicted is 0.94: zlib compression saves about 6%, more than the
small JSON header adds.

For the fusion model, which sends $P = 389{,}893$ values (its 389,445
trainable parameters plus 448 batch-normalisation statistics):

$$B_{FC} \approx 389{,}893 \times 4 \text{ B} = 1{,}559{,}572 \text{ B} \approx 1.56 \text{ MB}$$

Measured: 1.44 MB per round (ratio 0.92). The ratio between the two models
is predicted by the parameter counts alone:

$$\frac{B_{FC}}{B_C} \approx \frac{389{,}893}{111{,}370} = 3.50$$

$$\text{measured: } \frac{1.441 \text{ MB}}{0.418 \text{ MB}} = 3.45$$

### 5.10.4 When does federated learning save bandwidth?

In Strategy A the upload grows with the number of samples $n$ collected
between two rounds; in Strategy C it does not. Setting the two equal gives
the break-even point:

$$n \cdot 13{,}830 \text{ B} = 418{,}387 \text{ B} \text{   } \Rightarrow \text{   } n^* \approx 30 \text{ samples per round}$$

If a device records more than about 30 samples between two federated rounds
(15 seconds of labelled audio at one sample per 500 ms), federated learning
uploads less than centralized training. In the replay experiments each node
contributed 500 samples per round of A, so C was 16.5 times cheaper:

$$\frac{69.15 \text{ MB}}{4.18 \text{ MB}} = 16.5$$

The live Pi run confirms the formula from the other side. There, the 96
samples were spread over three rounds, on average 32 new samples per round,
close to $n^*$, and A and C uploaded almost the
same amount (1,337 KB against 1,245 KB, Table 5.7).

### 5.10.5 Bytes per sample in the hybrid fusion strategy (FB)

FB sends a 16-bit spectrogram and a 576-number feature vector stored as
16-bit floats:

$$S_{FB} = 13{,}056 \text{ B} + 576 \times 2 \text{ B} = 14{,}208 \text{ B}$$

Measured: 10,302 B per sample (compression factor 0.73). The feature vector
adds only 1,152 B before compression, about 9% on top of the spectrogram,
while a JPEG frame of the same scene
would add 24–66 KB (Section 3.6.3).

### 5.10.6 Growth of server training time in Strategy A

The centralized server trains one epoch on its whole pool every round. If
one sample costs $k$ seconds to train, the time in round $r$ with 1,000 new
samples per round is

$$t_r = k \cdot 1000 \cdot r$$

and the total server time after $R$ rounds is

$$T_R = \sum_{r=1}^{R} k \cdot 1000 \cdot r = k \cdot 1000 \cdot \frac{R(R+1)}{2}$$

The measured times give $k \approx 3.0$ ms per sample (between 2.9 and 3.7 ms
in every round). With $k = 3.04$ ms the formula predicts $t_{10} = 30.4$ s
(measured 30.1 s) and $T_{10} = 3.04 \text{ ms} \times 1000 \times 55 = 167$ s
(measured 167.3 s). The time per round grows linearly and the total grows
with the square of the number of rounds. In Strategy C, the server time per
round stays constant (0.7 s), because averaging two models does not depend
on how much data the clients have.

### 5.10.7 FedAvg weights in the experiments

Node A held $n_A = 3{,}540$ clips and node B $n_B = 3{,}539$. The FedAvg
weights are

$$\frac{n_A}{n} = \frac{3{,}540}{7{,}079} = 0.50007$$

$$\frac{n_B}{n} = \frac{3{,}539}{7{,}079} = 0.49993$$

With an even split, FedAvg is almost exactly the plain mean of the two
models. This also shows that the instability at high learning rates
(Section 5.4) cannot come from one node dominating the average.

### 5.10.8 A blackout score that looks like chance

The test set has 21 bird, 21 car, 20 cat, 21 dog and 20 motorcycle clips. A
model that always predicts one of the 21-clip classes scores

$$\frac{21}{103} = 0.204$$

which is exactly the blackout score of the vision-only and the naive fusion
model. Together with the probe in Table 5.10 (one predicted class in
blackout), this shows that these models do not guess randomly in the dark;
they always give the same answer.

### 5.10.9 Summary

| Quantity | Formula predicts | Measured | Difference explained by |
|---|---|---|---|
| Spectrogram shape | 51 × 128 | 51 × 128 | – |
| B saving over A | 18.4% | 26.2% | spectrograms compress better |
| C upload per round | 445 KB | 418 KB | zlib compression |
| FC upload per round | 1.56 MB | 1.44 MB | zlib compression |
| FC / C upload ratio | 3.50 | 3.45 | – |
| Break-even samples per round | 30 | live run with 32: A ≈ C | – |
| A server time, round 10 | 30.4 s | 30.1 s | – |
| A server time, total | 167 s | 167.3 s | – |
| Blackout score of a constant model | 0.204 | 0.204 | – |

*Table 5.14: Predictions from the formulas compared with the measurements.*

Every measured quantity is within a few percent of the prediction, or the
difference has a clear cause. This makes it unlikely that the measurements
contain a hidden error.
