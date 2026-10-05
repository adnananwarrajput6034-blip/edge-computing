# 3 System Design and Methodology

This chapter describes what I built and how I measured it, in enough
detail that someone else could repeat my experiments. Chapter 4 then
explains how I implemented the design.

## 3.1 Research Approach

My thesis is a comparative experimental study. I carry out the same task,
learning to recognise objects from sound, with three different placements
of the training, and I measure each placement with the same metrics on the
same data.

To keep the comparison clean, I worked in two stages and changed only one
thing at a time:

1. **Stage one, audio only.** The three strategies were first compared with a
   small audio-only model. This isolates the effect of the placement from
   the complexity of the model.
2. **Stage two, audio and vision.** A fusion model that uses both the camera
   image and the audio was then built, tested with the camera blacked out,
   and run again under the hybrid and federated strategies.

Finally, I ran the pipeline on a real Raspberry Pi to show that the results
are not just an effect of the simulation.

## 3.2 System Architecture

The system has three kinds of machines (Figure 3.1).

**Edge nodes.** Raspberry Pi computers with a camera and a microphone. They
sit where the data is produced, they are weak in compute and they are often
on a slow or expensive network. The comparison experiments use two edge nodes, called A and B, which run as two separate processes on the laptop (Section 3.7); the live tests use one Raspberry Pi 5.

**Server.** A laptop with no sensors. It is powerful, it holds the test set
and it keeps the official global model that all nodes share.

**Broker.** A Mosquitto MQTT broker, which runs on the same laptop. Edge
nodes and server never talk to each other directly; all messages go through
the broker.

![Figure 3.1: System architecture with two edge nodes, the MQTT broker and the server. Source: own work.](docs/figures/01_architecture_current_audiocnn.png)

### 3.2.1 The edge loop

Each edge node repeats the same loop every 500 ms. One pass of the loop is
called a tick:

1. Record 500 ms of audio from the microphone.
2. Take one camera frame at the same moment.
3. Run YOLOv8 on the frame. If an object is detected with a confidence of at
   least 0.5, its class becomes the label. If not, the tick is skipped.
4. Discard the image. It is never sent, and it is only stored in an optional
   evidence mode that I used to build the demo gallery (Section 5.6).
5. Store the pair {audio, label} in a local buffer.

Nothing is sent during a tick. Training in this system is batched, not
real-time.

![Figure 3.2: The edge loop that runs every 500 ms. Source: own work.](docs/figures/07_edge_tick_loop.png)

### 3.2.2 Batching

Data is sent in batches of 500 samples. Sending each sample on its own would
add the MQTT and TCP headers 500 times, keep the radio active all the time
and compress badly. One message with 500 samples carries the overhead once
and compresses much better.

When the buffer is full the node:

1. stacks all audio clips into one array and stores it in NumPy's binary
   format, with the labels and timestamps in a small JSON block,
2. compresses the result with zlib, and
3. publishes it as a single MQTT message.

Binary storage matters: a 16-bit sample takes 2 bytes in binary, but around
6–10 bytes when written as text in JSON.

## 3.3 Communication Design

All communication uses MQTT with the topics in Table 3.1.

| Topic | Direction | Content |
|---|---|---|
| `thesis/edge/{id}/data` | edge → server | Strategy A/B: a batch of audio or spectrograms |
| `thesis/edge/{id}/weights` | edge → server | Strategy C: locally trained weights |
| `thesis/server/model/global` | server → edges | the current global model |

*Table 3.1: MQTT topics.*

All messages use QoS 1. The global model in the federated strategy is
published as a retained message, so a node that connects late receives it
immediately. In the federated strategies every message carries a round
number, and the server ignores messages from rounds that are already
closed.

On the server, the MQTT callback only puts incoming messages into a queue
and returns. A separate loop takes messages from the queue, decodes them and
starts training. Training inside the network callback would block the
connection for many seconds and could lead to lost messages.

**Why not Flower.** The Flower framework [5] was planned at first for federated learning, and an early version of the project did. In the final
system FedAvg was implemented directly on top of MQTT. The reason is
measurement fairness: if Strategy C used Flower's gRPC transport while A and
B used MQTT, the bandwidth numbers would compare two different protocols, not
two placements. With one transport for all strategies, every byte is counted
in the same way.

## 3.4 The Three Strategies

The machines, the network and the data are the same in all three
strategies. Only the division of work changes (Table 3.2).

| | Edge does | Network carries | Server does |
|---|---|---|---|
| **A, centralized** | record, label | raw audio | STFT, train, evaluate |
| **B, hybrid** | record, label, STFT | spectrograms | train, evaluate |
| **C, federated** | record, label, STFT, train | model weights only | average (FedAvg), evaluate |

*Table 3.2: Division of work in the three strategies.*

In all three strategies, the trained model makes its decisions on the
device. In a fully centralized variant, the device could also send every
sound to the server and receive the decision back (cloud inference). In this
work, the server sends the trained model back and the device makes the
decision itself, so it keeps working without a network connection, as in fog
or darkness.

![Figure 3.3: What the edge does, what crosses the network and what the server does in each strategy. Source: own work.](docs/figures/08_three_strategies.png)

From A to C, one job at a time moves from the server to the device. The
expected effect is less data on the network and more privacy, paid for with
more work on the device.

**Strategy A, centralized.** The device only records, labels and sends raw
16-bit audio. The server converts the audio to spectrograms, adds them to a
growing pool and, every time 1,000 new samples have arrived, trains the
model for one epoch on everything it has received so far. It then evaluates
the model and broadcasts it.

**Strategy B, hybrid.** Exactly one thing changes: the STFT moves to the
device. The device sends spectrograms as 16-bit floats instead of raw audio.
Everything else is identical to A.

**Strategy C, federated.** The server no longer trains. In each round it
sends the global weights; every node trains one local epoch on its own data
and sends back its weights together with its sample count; the server
averages them with FedAvg (Section 2.5.2), evaluates the result and
broadcasts it. The server waits until both nodes have reported before it
averages.

![Figure 3.4: Message sequence for strategies A and B. Source: own work.](docs/figures/03_sequence_strategies_ab.png)

## 3.5 Datasets

### 3.5.1 UrbanSound8K

UrbanSound8K [9], a standard benchmark for environmental sound classification like ESC-50 [10], contains 8,732 labelled sound clips of up to 4 seconds in
ten urban classes: air conditioner, car horn, children playing, dog bark,
drilling, engine idling, gun shot, jackhammer, siren and street music. It is
split into ten official folds. In this work folds 1–8 are used for training,
fold 9 for validation and fold 10 only for the final test, so every reported
accuracy is measured on recordings the model has never seen.

A public dataset was needed because accuracy can only be measured if the
correct answer is known. Recordings made live with the Raspberry Pi have no
reliable ground truth.

### 3.5.2 VGGSound

UrbanSound8K has no video, so it cannot be used to train a model that also
looks at images. VGGSound [13] contains about 200,000 ten-second YouTube
clips with human labels for the sound. The authors only publish the video
IDs and timestamps, so the clips must be downloaded by each user.

Five classes were chosen that YOLO can see and that also make a typical
sound: car, motorcycle, dog, cat and bird. For each clip, the middle frame
and the 16 kHz audio were extracted. The resulting dataset has 987
clips, between 195 and 200 per class.

The split into training, validation and test sets (80/10/10) is done **by
clip**, not by audio window. Several 500 ms windows are cut from each
ten-second clip, and windows from the same clip are almost identical. If some
windows of a clip were in the training set and others in the test set, the
model would be tested on near copies of its training data. This is called
data leakage and it would make the results look better than they are.

The labels used for the fusion experiments are the human labels from
VGGSound, not YOLO labels, so that the accuracy can be measured correctly.
The quality of YOLO as a labeller was measured separately (Section 5.1).

## 3.6 Models

### 3.6.1 AudioCNN

The audio classifier takes a 51 × 128 × 1 spectrogram as input (Table 3.3).

| Layer | Output | Details |
|---|---|---|
| Input | 51 × 128 × 1 | log-mel spectrogram |
| Conv block 1 | 25 × 64 × 32 | Conv 3×3 (32), BatchNorm, ReLU, MaxPool 2×2, Dropout 0.15 |
| Conv block 2 | 12 × 32 × 64 | Conv 3×3 (64), BatchNorm, ReLU, MaxPool 2×2, Dropout 0.15 |
| Conv block 3 | 6 × 16 × 128 | Conv 3×3 (128), BatchNorm, ReLU, MaxPool 2×2, Dropout 0.15 |
| Global average pooling | 128 | |
| Dense | 128 | ReLU, Dropout 0.3 |
| Dense (output) | N classes | softmax |

*Table 3.3: AudioCNN architecture (111,370 parameters for 10 classes).*

The model was kept small on purpose, because in Strategy C it has to be
trained on a Raspberry Pi.

### 3.6.2 Fusion model

The fusion model has two branches and a fusion head (Figure 3.5).

- **Vision branch.** MobileNetV3-Small [20], pre-trained on ImageNet and
  **frozen**, turns a 224 × 224 image into a 576-number feature vector. A
  small trainable dense layer projects it to 256 numbers.
- **Audio branch.** The same three convolution blocks as the AudioCNN,
  followed by a dense layer with 128 outputs. This branch is **trained**,
  because no pre-trained network knows what a dog sounds like in this
  setting.
- **Fusion head.** The 256 + 128 = 384 numbers are joined and passed through
  dense layers of 256 and 128 units with dropout 0.3, and a softmax output.

The model has about 1.33 million parameters, of which about 389,000 are
trainable. This is called late fusion.

The vision branch is frozen for three reasons: a Raspberry Pi cannot afford
to train it, its pre-trained features are already good for these everyday
classes, and freezing it means its output never changes, which allows the
device to send the feature vector instead of the image (Section 3.6.3).

![Figure 3.5: Fusion model architecture. Source: own work.](docs/figures/10_fusion_model.png)

### 3.6.3 Placing the fusion model in the strategies

When the fusion model is used in the strategies, one rule decides where each
part lives:

> Frozen parts can stay on the device. Trainable parts must be wherever
> training happens.

A second rule is that camera images never leave the device. In the hybrid
fusion strategy (FB), the server trains, so all trainable parts are on the
server. The device runs the frozen MobileNetV3 and the STFT and sends the
spectrogram together with the 576-number feature vector. The feature vector
is about 1 KB, much smaller than a JPEG frame (on average 24 KB for the VGGSound frames
and 66 KB for the Pi camera frames in my data), and it cannot easily be turned back into a recognisable picture. In the federated fusion strategy
(FC), each device holds a full copy of all trainable parts and only weights
are sent.

![Figure 3.6: Placement of the fusion model in the hybrid (FB) and federated (FC) strategies. Source: own work.](docs/figures/11_fusion_placement_fb_fc.png)

## 3.7 Experimental Protocol

All strategy experiments in stage one used the same setup:

- **Two edge nodes**, each a separate process with its own MQTT connection.
- **Disjoint data.** The training clips are shuffled with a fixed seed;
  node A then takes every even position and node B every odd position, so
  they never see the same sample. The class
  distribution on both nodes is almost the same (IID).
- **10 rounds.** In A and B each node streams **5,000 samples**. Node A
  owns 3,540 and node B 3,539 unique training clips (half of folds 1–8 each), so after the
  first 3,540 the clips are sent a second time, as a device in the field
  would keep hearing similar sounds. In C each node trains on all its own
  clips in every round.
- **Same starting point.** All runs start from the same trained AudioCNN
  (warm start), so the strategies are compared on equal terms.
- **Same evaluation.** After every round the server evaluates the model on
  fold 10.
- **Bytes are counted twice**, once by the sender and once by the receiver,
  and the two counts are compared.

### 3.7.1 Why the sensors are replayed

In the stage-one experiments the sensors are replayed: instead of recording
from a live microphone, each node reads audio clips from UrbanSound8K and
passes them on as if they had just been recorded, and the label comes from
the dataset instead of from YOLO. Everything after that point is real: two
separate client processes, a real MQTT broker, real serialisation and
compression, real byte counts, real training and real evaluation.

I had three reasons for this choice.

1. **A fair comparison needs identical input.** If each strategy captured
   its own live audio, each would process different sounds, and any
   difference in accuracy or bandwidth could come from the data and not
   from the strategy. With replay, all strategies see exactly the same data
   with the same labels, so any difference is caused by the strategy. This
   is a controlled experiment.
2. **Accuracy needs ground truth.** To report that a model is 65% accurate
   you have to know the correct answers. UrbanSound8K provides them; a live
   recording in a room does not.
3. **The question does not depend on the sensor.** The thesis asks where
   training should run. That depends on what is computed and what is sent,
   not on whether the audio came from a microphone or from a file. The bytes
   are the same either way.

To make sure the replay does not hide a problem, I also ran the same
pipeline live on a Raspberry Pi (Section 5.6).

## 3.8 Metrics

| Metric | Definition |
|---|---|
| Test accuracy | correct predictions on the test set divided by test set size |
| Upload per node | total bytes sent from one edge node to the broker, counted at sender and receiver |
| Device compute | seconds spent on the device for STFT and/or training |
| Server compute | seconds per round spent on the server for training or averaging |
| Data leaving the device | what kind of data crosses the network (raw audio, spectrograms, weights) |
| Blackout accuracy | test accuracy when every image is replaced by a black image (fusion only) |

*Table 3.4: Metrics used in the experiments.*

The blackout test is the main test for the claim that audio helps when
vision fails. Every fusion model is evaluated twice on the same test clips:
once with the real images and once with all images replaced by black
images, while the audio stays the same. This simulates complete loss of
vision, as in darkness or with a broken camera.

## 3.9 Tools and Use of AI Assistance

**Software and tools.** All code was written in Python. Models were built and
trained with TensorFlow/Keras, audio was processed with librosa and NumPy,
objects were detected with Ultralytics YOLOv8, and messages were sent with
paho-mqtt through a Mosquitto broker. The result charts in Chapter 5 were
made with Matplotlib directly from the JSON result files that each
experiment writes, so every chart shows exactly the stored numbers. The
architecture and flow diagrams were drawn with Mermaid, a tool that turns a
short text description into a diagram; their sources are kept in the
repository next to the images. No figure was taken from another publication.

**AI assistance.** I designed the project and the experiments, built and
tested the hardware, and used an AI assistant as a tool for programming,
running experiments, making the charts and drafting the text. I checked every
result, and I rewrote key parts in my own words. A colleague helped me with some parts of the coding.
