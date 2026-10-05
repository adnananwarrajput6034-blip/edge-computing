# 6 Discussion

In this chapter I explain what the results from Chapter 5 mean. I answer
each research question, look at privacy, compare my results with related
work and with what I predicted in my exposé, and then list the limitations.
The chapter ends with a personal reflection on what was easy, what was hard
and what I would do differently.

## 6.1 RQ1: How Do the Three Strategies Compare?

The results show a clear trade-off. Moving work from the server to the
device reduces upload and server load and improves privacy, and the price is
paid by the device. Accuracy was not part of the trade-off in my
experiments: all three strategies ended at about the same accuracy.

**Bandwidth.** Strategy C needed 16.5 times less upload than Strategy A. The
reason is simple: A and B send data, C sends a model. The data grows with
every new recording, but the model always has the same size. The formula
from Section 2.5.6 predicts about 445 KB per round for my audio model, and I
measured 418 KB (zlib compresses the weights a little).

**Server load.** In Strategy A the server trains again on everything it has
ever received, so every round takes longer than the last. In a system that
runs for months this would grow without limit. In Strategy C the server only
averages two models, which took less than a second per round and does not
depend on how much data the devices have.

**No cloud needed.** All server work ran on an ordinary laptop (Apple M1,
8 GB RAM), so a cloud server was not necessary for this scale. A cloud
server would add internet delay to every message, but it would not change
the byte counts, because the same messages are sent either way. With many
more devices, the growing training time of Strategy A would be the first
reason to move the server to the cloud; the averaging in Strategy C would
stay cheap.

**Device workload.** In Strategy C the device does about 20 times more work
than in Strategy B: 233 s of training against 11.3 s of STFT over the whole
run. In the replayed experiments the two "devices" were processes on my
laptop, so these are laptop times. On the real Raspberry Pi 5, one federated
round on the 96 live samples took about 17.5 s of training (Section 5.6).
That is fine for a system that trains every few minutes, but on weaker
devices or with bigger models the device could become the bottleneck.

**Why accuracy hardly changed.** The accuracy stayed close to the starting
model in all strategies. This can be explained and is not a failure: the starting model had already been trained on the same
kind of data, so more of the same could not teach it much. These experiments
measure the cost of each strategy. The experiments where accuracy really
moves are the learning-rate sweep and the blackout tests.

**The hybrid strategy.** Moving the STFT to the device saved 26% of the
upload. It sounds like "sending features instead of raw data" should save a
lot, but Section 5.10.2 shows why it does not here: a 51 × 128 spectrogram at
16 bit is 13 KB, almost as big as the 16 KB of audio it comes from. A simple
size calculation predicts 18%, and the smoother spectrograms compress a bit
better, which gives the rest. Preparing data on the device only saves real
bandwidth when the result is much smaller than the input, for example a
576-number feature vector instead of a camera image.

## 6.2 RQ2: How Does FedAvg Combine the Weights, and When Does It Match Centralized Training?

FedAvg combines the devices' models by averaging every weight array element
by element, weighted by each device's share of the training samples
(Sections 2.5.2 and 2.5.3). The weights do not cancel out, because every
device starts each round from the same shared model.

FedAvg matched centralized training only after I reduced the local learning rate by a factor of 10 to 20 (1e-4 or 5e-5; I chose 5e-5 because it was the most stable). This happened even though both nodes had evenly
split data, so it was not the usual explanation of different data on the
clients (non-IID data) [2], [3].

The explanation is the starting point. The model was already trained, so it
sat near the bottom of a valley in the loss landscape. With a large learning
rate, each client takes a full epoch of big steps on its own data and climbs
some distance up the side of the valley. The two clients climb in slightly
different directions, and the average of two points high up on different
slopes is not at the bottom. With small steps, both clients stay close to the
starting point and their average stays near the bottom.

In short: **a model that is already trained needs gentler local updates
under FedAvg than under centralized training.** FedProx tackles the same
problem with its extra penalty term [4]. This matters for any system that,
like mine, keeps improving a deployed model with federated learning.

## 6.3 RQ3: Does Fusion Survive When the Camera Fails?

**Only the right training makes a fusion model listen.** The fusion model
trained the obvious way was deaf: with a black camera it predicted the same
class for every clip, and swapping its audio changed only 2 of 103 answers.
This matches what Wang et al. describe: the input that learns fastest takes
over the training [21]. The pre-trained vision branch explained the labels
from the first epoch, the loss became small, and almost no learning signal
reached the untrained audio branch. The fusion head learned to ignore it.

The order of the two fixes matters. Modality dropout alone did not help: it
makes the model look at the audio when the image is missing, but if the
audio branch produces nothing useful, there is nothing to look at.
Warm-starting the audio branch first gives it useful features from the
start; modality dropout then teaches the head to use them. Only this
combination gave a model that used all five classes in blackout and reacted
to the audio.

**Graceful degradation.** In blackout, the vision-only model fell to chance
level, while the recipe model kept 0.398, about twice chance. After further
training under FB and FC it ended at 0.58 (FB) and 0.57 (FC) in blackout,
above the audio-only model (0.505); FC reached 0.67 in its best round. Instead of failing completely when the camera stops
working, the system falls back to hearing. This is what I set out to show.

**Robustness first, accuracy second.** Right after fusion training, the
recipe model was less accurate on clean data than vision alone (0.718
against 0.806). Only after further training under FB and FC did it pass the
vision-only model (final 0.845 for FB and 0.854 for FC). Part of this gain is extra training that the
single-sense models did not get, so the safe conclusion is: fusion with the
recipe makes the system robust to camera failure, and with enough training
it does not have to cost accuracy when the camera works.

**Federated cost follows model size.** The federated upload per round grew
from 418 KB for the audio model to 1.44 MB for the fusion model, a factor of
3.45, almost exactly the ratio of their parameter counts (3.50). With the
amount of data I used, FB and FC cost about the same (16.2 against 14.4 MB
per node). With ten times more data, FB would send ten times more, while FC
would send exactly the same.

## 6.4 Privacy

In Strategy A, raw audio leaves the device. Raw audio can contain speech and
is the most sensitive form of the data. In Strategy B only spectrograms are
sent. They still contain much of the sound: an approximate waveform can be
rebuilt from a spectrogram with the Griffin–Lim algorithm [33], although the
mel compression and the missing phase make the result rough. In Strategy C
only model weights are sent. Camera images never leave the device in any of
my strategies, and in the fusion strategies only a 576-number feature
vector is sent instead of the image.

Federated learning is not a complete privacy solution. Attacks that rebuild
training data from gradients [7] are known. My claim is therefore limited: in
Strategy C, raw audio never leaves the device. Secure aggregation [8] or
differential privacy would be needed for stronger guarantees.

## 6.5 Comparison with Related Work

| Work | Setting | Main result | Relation to my thesis |
|---|---|---|---|
| McMahan et al. [1] | FedAvg in simulation | FedAvg needs far fewer communication rounds than plain federated SGD | same algorithm; I measured it over a real MQTT network |
| Ridolfi et al. [6] | Flower on Raspberry Pi | federated learning works on Raspberry Pi | confirmed; I also compared it with centralized and hybrid training |
| Aytar et al. [14] | SoundNet, vision teaches sound | sound features can be learned from unlabelled video | same idea, used here for continuous training on edge devices |
| Wang et al. [21] | multimodal training | multimodal models often do worse than their best single input | I saw the same problem; warm start and modality dropout fixed it |

*Table 6.1: Comparison with related work.*

## 6.6 Comparison with My Exposé

### 6.6.1 Predictions versus results

In my exposé I made a table with what I expected from each strategy. Table
6.2 puts those expectations next to what I actually found.

| Aspect | What I expected (exposé) | What I found | Verdict |
|---|---|---|---|
| Bandwidth | A high, B medium, C low | 69.2 / 51.0 / 4.18 MB per node | confirmed, but B was closer to A than I expected |
| Privacy | A low, B medium, C high | A sends raw audio, B spectrograms, C only weights | confirmed (by design, not measured as a number) |
| Accuracy | A best (most data), B good, C variable | all three ended within 0.03 of each other | not confirmed: with even data, C matched A once the learning rate was right |
| Edge power use | A low, B low–medium, C high | device time 0 / 11 / 233 s (plug-in powered, watts out of scope) | confirmed as computing time |
| Implementation effort | A simple, B medium, C complex | same order (Section 6.8) | confirmed |
| Main challenge of C | client drift from non-IID data | instability from a too-large learning rate, even with even data | partly different: the problem appeared earlier than expected |

*Table 6.2: My expectations from the exposé compared with the results.*

The biggest surprise was accuracy. I expected centralized training to win
because the server sees all the data. In my experiments it did not, because
the model was already trained and both nodes had the same kind of data.
With very different data on the nodes, I would still expect centralized
training to have an advantage.

### 6.6.2 The five metrics

The exposé named five metrics, and latency and memory were added during
planning.
Table 6.3 shows how far I could measure each one.

| Metric (exposé) | Planned measurement | What I measured | Status |
|---|---|---|---|
| Accuracy | precision, recall, F1 on a shared test set | accuracy for all runs; per-class precision, recall and F1 for the baseline (Table 5.2) | done |
| Privacy | data-flow check; spectrogram reversibility | data flow checked for all strategies; reversibility discussed from literature (Section 6.4) | partly |
| Power consumption | USB power meter in watts | device computing time as a substitute | out of scope: plug-in powered devices |
| Bandwidth | MB per round | bytes counted at sender and receiver for every run | done |
| Model convergence | accuracy per round, FedAvg vs. baseline, non-IID | accuracy per round for every run; learning-rate sweep; non-IID not tested | partly |
| Latency and memory (added during planning) | inference vs. inference plus training | inference latency per step on the laptop, YOLO and STFT on the Pi, update latency per strategy (Section 5.9); memory not measured | partly |

*Table 6.3: The five metrics from the exposé and how far they were measured.*

### 6.6.3 Changes to the plan

Several things turned out differently from the exposé plan. Table 6.4 lists
them with the reason.

| Exposé plan | What I did | Why |
|---|---|---|
| Two Raspberry Pi 4 devices in different places | one Raspberry Pi 5 for the live tests; two replayed nodes on the laptop for the comparison | a fair comparison needs identical data for all strategies (Section 3.7.1); the live two-Pi run is planned for the defense demonstration |
| Camera Module 3 and I2S microphone | USB webcam and separate USB microphone | simpler to set up; the webcam's own microphone turned out to be broken (Section 4.7) |
| TensorFlow Lite on the Pi | full TensorFlow on the Pi | Strategy C needs training on the device, which TensorFlow Lite does not support well |
| Flower framework | own FedAvg over MQTT | one transport for all strategies keeps the byte counts comparable (Section 3.3) |
| Strategy A sends raw audio and video | raw audio only; images never leave the device | images are only needed for the label; sending them would add privacy risk for nothing |
| Non-IID data from different environments | evenly split data | the comparison came first; different data on two Pis in two labs is the next step (Section 7.3) |
| Power in watts | computing time | the Pis were plug-in powered, so battery operation and power measurement were outside the scope of this thesis |
| Local PC as server, AWS or Azure free tier as fallback | the laptop only; no cloud server | the laptop was enough: averaging took 0.7 s per round and server training at most 30 s |
| Select and implement one strategy | all three implemented, recommendation per situation | the results show that the best choice depends on the situation (Section 7.2) |

*Table 6.4: Differences between the exposé plan and the final work.*

## 6.7 Limitations and Threats to Validity

- **One test fold.** The UrbanSound8K results use fold 10 as the only test
  fold. Ten-fold cross-validation would give an average and a spread. The federated learning rate (Section 5.4) was also chosen by looking at the fold-10 accuracy, without a separate validation fold, so the result for 5e-5 is slightly optimistic.
- **Small fusion test set and no repeated runs.** The fusion test set has
  103 clips, so one clip is about one percentage point, and an earlier run of
  the same experiment gave quite different numbers (Section 5.7). Each
  fusion model was trained at most twice, with different settings but not with several seeds; several random seeds would give a mean and
  a spread.
- **Evenly split data.** The two nodes had almost the same mix of classes.
  In a real deployment, devices in different places hear different things
  (non-IID data), which makes federated learning harder.
- **VGGSound images favour vision.** VGGSound clips are videos of the
  labelled object, so the object is usually easy to see. In real street
  scenes the camera view would be worse, and audio would matter more.
- **Black is not fog.** The blackout test is the extreme case of losing the
  camera completely. Real fog or darkness makes the image worse step by
  step.
- **Small live dataset and one Pi.** The live tests used one Raspberry Pi 5
  and 96 training clips from about four minutes of enrollment. The results
  with two edge nodes come from the replayed software setup; the live run
  with two physical Pis is planned for the defense demonstration.
- **Phone videos in the live demo.** The live enrollment used videos played
  on a phone instead of real cars or animals. The pipeline is the same, but
  real outdoor recordings would contain more background noise.
- **Two clients.** FedAvg works the same way for any number of clients, but
  I did not measure effects of scale.
- **Plug-in powered devices.** The Pis were powered from the mains, so battery life and power in watts were outside the scope of this thesis; computing time is used as an indicator of the energy cost.
- **Not measured.** Memory use, the model inference latency on the Pi itself, and model freshness (how quickly each strategy learns a new object) were planned but not measured.
- **Baseline model.** The baseline audio model reaches 64.9%. Better audio
  models exist, but my thesis compares strategies, and all strategies start
  from the same model.

## 6.8 Personal Reflection

**Strategy A was the easiest.** It needs the least code on the edge device.
The Pi only records the sound, gets the label from the camera and sends the
raw data quickly to the central server. The server trains the model and
sends it back, and the Pi then uses it to decide what it hears, for example
"cat". Because the server does all the heavy work, there were few things
that could go wrong, and when something did go wrong it was easy to find.
That is also why I built it first, as the baseline for the other two.

**Strategy C was the hardest.** Here the Pi does not only collect data but
also has to process it and train its own copy of the model, so there is much
more code on the edge side. The server has to work in rounds, wait for both
Pis, ignore late messages from an old round and send the shared model back.
On top of that, I had to understand the formulas for averaging model weights
(Section 2.5), which was not easy for me. And when everything finally ran,
the model became worse instead of better. The reason turned out to be the
learning rate, not a bug in the code.

**The question about the weights.** My supervisor asked me whether the
weights of two different Pis would cancel each other out when they are
averaged, so that the result is useless. My first ideas were about timing:
maybe the weights must arrive at the server at the same moment, or the
clocks of the Pis must be synchronised, or the server must keep the data of
each Pi separately. When I read the FedAvg paper [1] and worked through the
example in Section 2.5.3, one matrix element at a time, I understood the
real reason why they do not cancel: in every round, both Pis start from
exactly the same shared model, so each position in their matrices means the
same thing. My timing idea was not completely wrong, though: the server does
wait until both Pis have sent their weights for the same round before it
averages them.

**Difficulties and what I learned from them.**

- The microphone built into the webcam was broken. The system showed it as
  working, but it only recorded a constant signal. I learned to check a
  sensor by looking at what it records, not only at its status.
- Understanding federated learning took me a long time, especially what
  happens to the weight matrices when they are averaged. Working through a
  small example by hand helped me most.
- The live tests used only one Raspberry Pi, and the live dataset was small:
  about four minutes of enrollment gave 96 training clips. This was enough to
  show that the pipeline works on real hardware, but not enough to train a
  strong model. I plan to show the live run with two Raspberry Pis in the
  defense demonstration.
- During the thesis period I had serious health problems, which slowed my
  work and created a lot of time pressure. Because of this, some planned parts, such as the experiment with different data on each device and the model-freshness experiment, could not be done within the thesis and are left for
  future work.

**Which strategy I would choose.** For now, I would use Strategy A, because
it is fast, efficient for the device and the easiest to implement. For the future, I would
move to Strategy C. A robot in the real world keeps meeting new sounds and
new objects, and with federated learning it can learn from this new data on
the device and keep its model up to date, while the raw data stays private.
