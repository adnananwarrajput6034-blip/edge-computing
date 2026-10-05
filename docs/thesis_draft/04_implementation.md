# 4 Implementation

This chapter explains how I turned the design from Chapter 3 into working
software and hardware. I focus on the decisions that mattered and on the
problems I ran into, not on listing code. The full source code is in my
project repository (Appendix D).

## 4.1 Hardware and Software

| Component | Used in this work |
|---|---|
| Edge device | Raspberry Pi 5 Model B, 8 GB RAM, Debian 13 |
| Camera | USB webcam, 1080p, MJPG |
| Microphone | separate USB microphone |
| Server | MacBook Air laptop (Apple M1, 8 cores, 8 GB RAM, 256 GB SSD) |
| Network | Pi and laptop on the same local network |

*Table 4.1: Hardware.*

| Software | Version | Used for |
|---|---|---|
| Python | 3.11 (laptop), 3.13 (Pi) | everything |
| TensorFlow / Keras | 2.21 | AudioCNN, fusion model, training |
| librosa | 0.11 | STFT and mel spectrogram |
| Ultralytics YOLOv8 | 8.4 | object detection (labelling) |
| paho-mqtt | 2.1 | MQTT client |
| Eclipse Mosquitto | 2.1.2 | MQTT broker |
| NumPy | 2.4 | arrays, serialisation |
| yt-dlp, ffmpeg | – | VGGSound download and frame/audio extraction |

*Table 4.2: Software.*

The code is organised in three parts: `src/edge` for code that runs on the
device (sensors, STFT, YOLO wrapper, MQTT client), `src/server` for the models
and the FedAvg aggregator, and `src/experiments` for the clients and servers
of each strategy. The script `scripts/run_strategy.py` starts one server and
two clients for a chosen strategy and writes all results to a timestamped
folder.

## 4.2 Audio Pipeline

The STFT is implemented in `src/edge/processing/stft.py` with librosa using
the parameters in Table 2.1. The steps for each 500 ms clip are:

1. compute the mel power spectrogram (n_fft 512, window 400, hop 160,
   128 mel bands, 0–8 kHz),
2. convert power to decibels relative to the maximum,
3. scale the result to the range [0, 1].

The output has the shape 51 × 128. The same code runs on the laptop (A) and
on the Raspberry Pi (B, C), so both sides produce exactly the same
spectrograms.

## 4.3 Labelling Pipeline

YOLOv8n is loaded once at start-up. For each tick it runs on the current
frame; the detection with the highest confidence above 0.5 gives the label.
Only a list of classes of interest is kept, and during live enrollment
further classes can be excluded with an ignore list. The frame is then dropped. On the Raspberry Pi 5 CPU, YOLO took
about 318 ms per frame, which fits within the 500 ms tick.

## 4.4 Client and Server Logic

The logic of the clients and servers is shown as pseudocode. Strategies A
and B share the same structure; only the payload differs.

**Edge client, strategies A and B**

```
loop every 500 ms:
    audio  = record_500ms()
    label  = yolo(frame)            # skip tick if no confident detection
    if strategy == B: audio = stft(audio)
    buffer.append(audio, label)
    if len(buffer) == 500:
        payload = zlib(savez(buffer))
        mqtt.publish("thesis/edge/<id>/data", payload, qos=1)
        buffer.clear()
```

**Server, strategies A and B**

```
on message: queue.put(message)        # return immediately
loop:
    batch = decode(queue.get())
    if strategy == A: batch = stft(batch)
    pool.extend(batch)
    if new samples since last round >= 1000:
        model.fit(pool, epochs=1)       # train on everything received
        evaluate(model, fold10)
        mqtt.publish("thesis/server/model/global", model.weights)
```

**Federated round, strategy C**

```
server: publish global weights (retained), round = r
client: model.set_weights(global)
        model.fit(local_data, epochs=1, lr=local_lr)   # fresh Adam each round
        publish("thesis/edge/<id>/weights", weights, n_samples, round=r)
server: wait until both clients reported for round r
        w = sum(n_k / n * w_k)          # FedAvg, layer by layer
        evaluate(w, fold10); r = r + 1
```

FedAvg is implemented in `src/server/aggregation/fedavg.py`. The core of the
aggregation is only a few lines and follows the formula in Section 2.5.2
directly (Listing 4.1).

```
total_samples = sum(sample_counts)
aggregated = []
for layer_idx in range(n_layers):
    layer_weights = [client_weights[i][layer_idx] for i in range(n_clients)]
    weighted_sum = np.zeros_like(layer_weights[0])
    for i, weights in enumerate(layer_weights):
        weighted_sum += (sample_counts[i] / total_samples) * weights
    aggregated.append(weighted_sum)
```

*Listing 4.1: Weighted averaging of client weights (FedAvg), shortened from
`src/server/aggregation/fedavg.py`.*

The aggregator averages
every weight array of the model, including the batch-normalisation
statistics, weighted by the number of samples each client used. The
optimiser state is not shared; each client starts a fresh Adam optimiser
every round, so that no optimiser state is shared; the original FedAvg algorithm uses plain SGD on the clients [1].

**Serialisation.** Strategy A sends 16-bit integer audio, Strategy B 16-bit
float spectrograms, and Strategy C 32-bit float weights. In all cases the
arrays are packed with NumPy's compressed format together with a small JSON
header. In strategies A and B the result is compressed once more with zlib
before publishing.

## 4.5 Fusion Model Training

The fusion model is trained with the script `scripts/train_fusion.py`. Several versions were trained on the same data so that they can be compared:

1. vision only (frozen MobileNetV3 and a classification head),
2. audio only (the AudioCNN structure),
3. naive fusion (both branches trained together from the start),
4. fusion with modality dropout only (0.3 and 0.5),
5. fusion with the training recipe described below.

**Training recipe.** The recipe was found after the naive fusion model
failed (Section 5.7). It has two steps, and the order matters:

1. **Warm start the audio branch.** The weights of the trained audio-only
   model are copied into the audio branch of the fusion model, so the audio
   branch is already useful when fusion training starts.
2. **Modality dropout with p = 0.5.** During training, the image of each
   sample is replaced by a black image with probability 0.5. The model cannot
   rely on the camera alone and must use the audio as well.

**Feature-vector model.** For the hybrid fusion strategy (FB), a second
version of the model takes the 576-number MobileNetV3 output directly instead
of the image. The trained layers are copied over, and the script
`scripts/prepare_fusion_fv.py` checks that both versions give the same
prediction on every test sample, with real and with black images. The two versions agreed on 100% of the 103 test predictions, both with real
and with black images (accuracy 0.718 and 0.398 for both). Computing the
feature vectors took about 3.5 ms per frame on the laptop when frames are
processed in batches (44 ms for a single frame, Section 5.9).

## 4.6 Live Deployment on the Raspberry Pi

For the first live run (Strategy A), moving from the replayed experiments to
the real Raspberry Pi needed only two configuration changes and no change to
the server code:

- the broker listens on all network interfaces (`0.0.0.0`) instead of only
  on the laptop itself, so that the Pi can reach it over the local network,
- the Pi client is given the laptop's IP address as broker address.

Only the start of the client loop is different. Table 4.3 shows the
comparison.

| | Replayed client | Raspberry Pi client |
|---|---|---|
| Audio source | WAV file from the dataset | USB microphone |
| Label source | dataset label | YOLOv8 on a live camera frame |
| Buffer, serialise, publish | identical | identical |
| Broker and server | identical | identical |

*Table 4.3: Differences between the replayed client and the Pi client.*

**Runtime vocabulary.** The live camera sees things like people, cars and
phones, which are COCO classes. The UrbanSound8K model knows classes like
siren and jackhammer. The two sets of names have nothing in common, so at
first none of the live samples could be used for training. The solution was
an enrollment step: the Pi records for a few minutes, YOLO labels each tick,
and the classes that appear at least 30 times become the vocabulary for this
deployment. The resulting dataset is saved once and replayed by all three
strategies, so that their numbers can be compared. For this, the servers got
one extra option that loads the enrolled class list and test set instead of
UrbanSound8K; the rest of the server code stayed the same. The audio model is
warm-started from the UrbanSound8K model: all convolution layers are copied
and only the output layer is replaced with a new one of the right size.

**Inference with a reject option.** After training, the model can be copied
back to the Pi and used on live audio. A plain softmax classifier always
picks one of its known classes, even for a sound it has never heard. To
avoid this, the Pi reports "unknown" when the highest probability is below a
threshold. If several unknown sounds come in a row while the camera sees an
object, the device reports that it has noticed something new, which in a real deployment would trigger a new
enrollment and a new round of training.

![Figure 4.1: Live pipeline: enrollment on the Pi, replay by the A, B and C clients, training on the unchanged server, and live inference on the Pi. Source: own work.](docs/figures/12_live_pi_pipeline.png)

## 4.7 Problems and How They Were Solved

**A broken microphone that reported no error.** The microphone built into
the webcam produced a constant signal at full scale. There was no driver
error, and the device looked fine in the system. The problem was found by
looking at the data itself: the RMS value of every clip was equal to the
maximum possible value, which means the signal was constant and not sound.
The webcam microphone was replaced with a separate USB microphone. Lesson:
sensors should be checked by looking at their data, not only at their
status.

**TensorFlow and PyTorch in one process.** On the Raspberry Pi, loading
TensorFlow (for the audio model) and PyTorch (for YOLO) in the same process
crashed with a segmentation fault, caused by a clash between their OpenMP
libraries. This could not be fixed by changing the import order or the
number of threads. Because enrollment only needs YOLO and inference only
needs TensorFlow, they were split into separate programs.

**Federated training made the model worse.** The first federated runs used
the same learning rate as the server (1e-3), and accuracy went down instead
of up. This was solved with a learning-rate sweep (Section 5.4).

**The fusion model ignored audio.** Different training runs gave
suspiciously identical accuracy values, and with black images the model gave
the same answer for every clip. Section 5.7 explains how this was traced to
an audio branch that had stopped learning.

**YouTube download changes.** YouTube stopped serving combined video and
audio streams in the format the download script asked for, so many VGGSound
clips failed to download. The script was changed to download the best
low-resolution video stream and the best audio stream separately and let
ffmpeg merge them.

**Wrong frame count in early tests.** An early test expected 49 frames per
spectrogram. The correct value is 51 with centred frames (librosa default)
and 48 without. The difference comes from padding at the edges of the clip
(Section 2.1).

## 4.8 Order of Implementation

The system was not built in one go. Table 4.4 shows the order in which I
built it, what worked and what did not work at first. The dates come from
the version history of the repository.

| When | Step | What happened |
|---|---|---|
| June 2026 | YOLOv8 labeller, STFT pipeline, first AudioCNN | worked; the labeller and the spectrogram shape were checked on test images and UrbanSound8K |
| July 2026 | first Strategy A simulation with mocked messaging; first Strategy B with 8-bit spectrograms | worked as a prototype, but without a real broker the bytes were not real network traffic |
| early August 2026 | first federated version with Flower; then Strategy A over a real MQTT broker and the first live run on the Raspberry Pi | Flower used a different transport than A and B, so the bytes were not comparable and I moved C to MQTT; the live run worked after replacing the broken webcam microphone (Section 4.7) |
| August 2026 | Strategy B, then Strategy C with my own FedAvg over MQTT | B worked directly; C ran, but made the model worse until I lowered the local learning rate (Section 5.4) |
| mid-August 2026 | fusion model, VGGSound data, FB and FC | the first fusion model ignored the audio; I found the cause with probes and fixed it with warm start and modality dropout (Section 5.7) |
| September 2026 | live enrollment, Pi clients for A, B and C, live inference | worked on the Pi; TensorFlow and YOLO had to run in separate programs (Section 4.7) |
| October 2026 | strategy, fusion and labelling experiments run again; latency and live-data training measured | needed because the result files of the first runs were not kept; the baseline model (July) and the live Pi results (September) were kept; some earlier fusion numbers could not be reproduced exactly (Section 5.7) |

*Table 4.4: Order of implementation. Source: version history of the project repository.*

The Raspberry Pis were powered from the mains, as planned for this thesis, so battery operation and power consumption in watts were outside its scope; the computing time on the device is used as an indicator of the energy cost instead. The live tests in this thesis used one Raspberry Pi 5;
the two-node comparison was done with two replayed nodes in software. Running
federated training live between two physical Raspberry Pis, as originally planned, is the next step and is prepared for the defense demonstration
(Section 7.3).
