# 2 Background and Related Work

This chapter explains the theory I needed for the rest of the thesis. I do
not try to cover each field completely. For every formula I first say in
plain words what it means, then give the formula itself and the book or
paper it comes from. Each section ends with a short note on how the idea is
used in my work.

## 2.1 Audio Signals and the Spectrogram

A microphone measures air pressure many times per second. After
digitisation, a sound is just a long list of numbers, called samples. I use
a sample rate of 16,000 samples per second, so half a second of sound is
8,000 numbers. This rate is common for environmental sound because it
captures frequencies up to 8 kHz, which covers most of what is useful in
engines, voices, animals and sirens.

A classifier cannot do much with the raw list of numbers, because the same
sound can look very different depending on its loudness and phase. Most
audio classifiers therefore first turn the sound into a spectrogram: a
picture that shows how much energy there is at each frequency over time
[11], [27].

**Short-Time Fourier Transform (STFT).** In words: cut the sound into short,
overlapping pieces, fade the edges of each piece, and measure which
frequencies each piece contains. Written as a formula [27], [34]:

$$X(m,k) = \sum_{n=0}^{N_{FFT}-1} x(n + mH)\, w(n)\, e^{-j 2\pi k n / N_{FFT}}$$

where
- $x(n)$ is the audio signal,
- $m$ is the number of the piece (frame) and $k$ the frequency bin,
- $H$ is the hop, the step from one frame to the next,
- $w(n)$ is the window that fades the edges of each frame,
- $N_{FFT}$ is the FFT size,
- $X(m,k)$ is the result: how strong frequency $k$ is in frame $m$.

Table 2.1 lists the settings I used and why.

| Parameter | Value | Reason |
|---|---|---|
| Sample rate | 16,000 Hz | covers 0–8 kHz, standard for environmental sound |
| Clip length | 500 ms (8,000 samples) | short enough to react quickly, long enough to hold a sound event |
| Window length | 25 ms (400 samples) | a sound hardly changes within this time |
| Hop length | 10 ms (160 samples) | 60% overlap between frames |
| FFT size | 512 | next power of two above 400, gives 257 frequency bins |
| Mel bands | 128 | groups the 257 bins into 128 bands that match human hearing |

*Table 2.1: STFT settings.*

**Number of frames.** In words: the number of frames is how many times the
window can jump by one hop along the clip, plus the first position. With
the edge padding that librosa adds by default [12]:

$$T = 1 + \left\lfloor \frac{N}{H} \right\rfloor = 1 + \left\lfloor \frac{8000}{160} \right\rfloor = 51$$

where $T$ is the number of frames, $N$ the number of samples in the clip,
$H$ the hop in samples, and $\lfloor \cdot \rfloor$ means rounding down.
Every 500 ms clip therefore becomes a picture of 51 time frames by 128 mel
bands. Without the edge padding it would be
$1 + \lfloor (8000-400)/160 \rfloor = 48$ frames.

**Mel scale.** In words: our ears hear small differences between low notes
much better than between high notes, so low frequencies should get more
detail. The standard mel formula [28] is

$$\text{mel}(f) = 2595 \cdot \log_{10}\left(1 + \frac{f}{700}\right)$$

where $f$ is a frequency in hertz and $\text{mel}(f)$ is the same frequency
on the hearing scale. The constants are chosen so that 1,000 Hz equals
1,000 mel. The 257 frequency bins are collected into 128 triangular filters
that are evenly spaced on the mel scale [27]. librosa, which I use, applies
the closely related Slaney version of the mel scale by default: linear below
1 kHz and logarithmic above it. Both versions have the same shape and the
same purpose [12].

**Log scaling.** In words: add up the energy inside each mel band and write
it in decibels, because sound energy covers a huge range. As a formula
[27]:

$$S(m,b) = 10 \cdot \log_{10}\left(\sum_k |X(m,k)|^2 \, M_b(k)\right)$$

where $S(m,b)$ is one pixel of the final picture (frame $m$, mel band $b$),
$|X(m,k)|^2$ is the energy in frequency bin $k$, and $M_b(k)$ is how much bin
$k$ belongs to mel band $b$. Finally, I scale every spectrogram to the range
0 to 1.

![Figure 2.1: Three 500 ms clips recorded by the Pi (car, cell phone, person) as waveforms (top) and as log-mel spectrograms (bottom). Source: own work.](docs/study/waveform-vs-spectrogram.png)

![Figure 2.2: Steps from waveform to the 51 × 128 spectrogram. Source: own work.](docs/figures/09_stft_pipeline.png)

**In my work:** every 500 ms clip becomes a 51 × 128 picture, which the
model treats like a small grey-scale image.

## 2.2 Convolutional Neural Networks for Sound

A convolutional neural network (CNN) learns small filters that slide over
an image and react to local patterns, such as edges. Deeper layers combine
these into bigger patterns [29]. Because a spectrogram is a picture of
sound, the same idea works for audio: the first filters react to short tones
or bursts of noise, later layers to the typical shape of a siren or a dog
bark. Hershey et al. showed that normal image CNNs work well for large-scale
audio classification [11].

Each block of my network has a convolution, batch normalisation [31], a
ReLU activation and max pooling, which halves the size of the picture. At
the end, a dense layer gives one score per class.

**Softmax.** In words: turn the scores into probabilities that are all
positive and add up to 100%. The formula [29] is

$$p_c = \frac{e^{z_c}}{\sum_j e^{z_j}}$$

where $z_c$ is the raw score of class $c$, $e^{z_c}$ makes it positive, and
dividing by the sum over all classes $j$ makes the probabilities $p_c$ add
up to 1.

**Cross-entropy loss.** In words: the loss is small when the model gives a
high probability to the right answer and large when it is confident and
wrong. The formula [29] is

$$\ell = -\sum_c y_c \log\, p_c$$

where $y_c$ is 1 for the correct class and 0 for all others. Only the term
for the correct class is left, so the loss is simply $-\log$ of the
probability the model gave to the right answer.

**Gradient descent.** In words: move every weight a small step in the
direction that makes the loss smaller. The basic rule [29] is

$$w \leftarrow w - \eta \, \nabla_w \ell$$

where $w$ stands for all the weights of the network, $\nabla_w \ell$ (the
gradient) tells for each weight in which direction the loss would grow, the
minus sign makes the step go the other way, and $\eta$ is the learning rate,
the size of the step. I use the Adam optimiser [30], which adapts the step
size for each weight but follows the same idea. The learning rate turned out
to be the most important setting for federated learning (Section 5.4).

**In my work:** the audio classifier is a small CNN with three convolution
blocks and 111,370 parameters, small enough to train on a Raspberry Pi.

## 2.3 Object Detection as a Teacher: YOLO

YOLO (You Only Look Once) is a family of object detectors that find objects
and name them in one pass over the image [18]. YOLOv8 [19] knows the 80
everyday classes of the COCO dataset, such as person, car, dog, bird and
cell phone, and gives every detection a confidence between 0 and 1.

I do not train or change YOLO. It is only my labelling tool. If YOLO finds
an object with a confidence of at least 0.5, the audio recorded at the same
moment gets that name. If YOLO is not sure, the clip gets no label and is
skipped.

**In my work:** I use the smallest version, YOLOv8n. It needs about 36 ms per
frame on the laptop and about 318 ms per frame on the Raspberry Pi 5, which
is fast enough for one label every 500 ms.

## 2.4 Cross-Modal and Weak Supervision

Cross-modal supervision means that one sensor provides the labels for
another. Aytar et al. used this idea in SoundNet: they trained a sound
network to copy what image networks saw in millions of unlabelled videos
[14]. Owens et al. used sound to teach a vision model [15], and Arandjelović
and Zisserman learned audio and image features together simply from the fact
that they happen at the same time [16]. Combining a camera and a microphone
on a robot was also shown by Chan et al. [26].

Labels made this way are noisy. The camera may see a person while the sound
comes from a car outside the picture. This is called weak supervision. Deep
networks can live with a surprising amount of label noise, because wrong
labels are often random and disagree with each other, while correct labels keep
pointing to the same pattern [17].

**In my work:** I measured the noise instead of assuming it. On VGGSound
images, YOLO's label matched the human sound label in 85% of the cases where
it detected something, so about 15% of the automatic labels are wrong
(Section 5.1.1).

## 2.5 Federated Learning

### 2.5.1 The idea

In normal (centralized) machine learning, all data is collected on one
server and the model is trained there. In federated learning (FL), the data
stays on the devices. Each device trains its own copy of the model on its
own data and sends only the updated model back. The server combines these
models into a new shared model and sends it out again [1], [2]. A short way
to say it: the data stays home and the model travels.

This has three advantages for edge systems:
- **privacy:** raw recordings never leave the device,
- **bandwidth:** a model can be much smaller than the data,
- **server load:** the server only averages, it does not train.

The price is that every device must be able to train a model.

### 2.5.2 The FedAvg algorithm

FedAvg was introduced by McMahan et al. [1] and is still the standard
algorithm. One round works like this:

1. The server sends the current shared model $w_t$ to all clients.
2. Each client copies it and trains for $E$ epochs on its own data, using
   the gradient descent rule from Section 2.2 with learning rate $\eta$. The
   result is that client's local model $w_{t+1}^{k}$.
3. Each client sends its model and the number of samples $n_k$ it trained
   on back to the server.
4. The server averages the models, giving more weight to clients with more
   data.
5. The new shared model goes out, and the next round starts.

**The aggregation step.** In words: the new shared model is a weighted
average of the clients' models, and each client's weight is its share of
all the data. The formula from [1] is

$$w_{t+1} = \sum_{k=1}^{K} \frac{n_k}{n}\, w_{t+1}^{k}$$

where $K$ is the number of clients, $n_k$ the number of samples on client
$k$, $n = \sum_k n_k$ the total number of samples, and $n_k / n$ the share of
client $k$. With my two edge nodes, A and B, this becomes

$$w_{t+1} = \frac{n_A}{n_A + n_B}\, w_A + \frac{n_B}{n_A + n_B}\, w_B$$

A small example for one single weight: if node A trained on 3,000 samples
and ended with the value $w_A = 0.40$, and node B trained on 1,000 samples
and ended with $w_B = 0.80$, then

$$w_{t+1} = \frac{3000}{4000} \cdot 0.40 + \frac{1000}{4000} \cdot 0.80 = 0.30 + 0.20 = 0.50$$

The node with more data pulls the result towards its own value.

**What the clients are trying to achieve together.** In words: the goal is
the same as in normal training, a small loss over all data of all clients,
but each client only computes its own part [1]:

$$F(w) = \sum_{k=1}^{K} \frac{n_k}{n} F_k(w)$$

where $F(w)$ is the loss over all data and $F_k(w)$ is the average loss on
client $k$'s own data. The server never needs the data to work towards this
goal, because each client reports only its own part.

![Figure 2.3: One FedAvg round between the server and two edge nodes. Source: own work.](docs/figures/04_sequence_strategy_c_fedavg.png)

### 2.5.3 What actually happens to the weight matrices

The step I found hardest to picture is what "averaging two models" means in
practice. In a discussion, my supervisor asked a good question:
if two Raspberry Pis learn different things, would their weights not cancel
each other out when the rows and columns of their matrices are averaged? To
answer it, I went back to the original FedAvg paper [1]. The answer is in
the next two sections. A model is
not one number but a list of matrices. My AudioCNN, for example, is a list
of 22 arrays with 111,370 numbers in total (Table 2.2).

| Layer | Array | Shape | Numbers |
|---|---|---|---|
| Conv block 1 | filter weights | 3 × 3 × 1 × 32 | 288 |
| Conv block 2 | filter weights | 3 × 3 × 32 × 64 | 18,432 |
| Conv block 3 | filter weights | 3 × 3 × 64 × 128 | 73,728 |
| Dense layer | weight matrix | 128 × 128 | 16,384 |
| Output layer | weight matrix | 128 × 10 | 1,280 |
| all layers | biases and batch-norm values | various | 1,258 |
| **total** | **22 arrays** | | **111,370** |

*Table 2.2: The weight arrays of the AudioCNN. Source: read from the trained
model with Keras.*

FedAvg treats each of these arrays separately and averages them element by
element: the number in row 1, column 1 of node A's matrix is averaged with
the number in row 1, column 1 of node B's matrix, and so on. For a tiny
2 × 2 matrix with equal data on both nodes:

$$\frac{1}{2}\left(\begin{matrix} 0.2 & -0.4 \\ 0.6 & 0.1 \end{matrix}\right) + \frac{1}{2}\left(\begin{matrix} 0.4 & -0.2 \\ 0.2 & 0.3 \end{matrix}\right) = \left(\begin{matrix} 0.3 & -0.3 \\ 0.4 & 0.2 \end{matrix}\right)$$

The server does exactly this for all 22 arrays, which in my code is one loop
over the layers (Listing 4.1). The shapes never change; only the values
inside do.

### 2.5.4 Why averaging weights works at all

Averaging two networks only makes sense if they started from the same
weights. A network can solve the same task in many equivalent ways:
swapping two neurons in a layer, together with their connections, gives the
same result but different weight values. If the two clients started from
different random weights, neuron 5 in one model might do the job of neuron
12 in the other, and the element-by-element average from Section 2.5.3
would mix unrelated neurons. Because every FedAvg round starts all clients
from the same shared model $w_t$, the neurons stay aligned and the average
stays meaningful [1]. It is a design requirement of FedAvg, not an assumption.

### 2.5.5 Local steps and client drift

The more steps a client takes on its own, the further its weights move away
from the shared model. Roughly, this distance grows with the learning rate
$\eta$, the number of local epochs $E$ and the number of batches. If the
clients' data are different (non-IID data), each client moves in its own
direction and the average can end up in a poor place. This is called client
drift [2], [3]. FedProx reduces it by punishing a client for moving too far
from the shared model [4]:

$$\underset{w}{\mathrm{min}} \; F_k(w) + \frac{\mu}{2}\|w - w_t\|^2$$

where $\|w - w_t\|^2$ measures how far the local model has moved from the
shared model $w_t$, and $\mu$ sets how strongly this is punished. I did not
use FedProx, but the same effect, local models moving too far, appeared in
my experiments even with evenly split data (Section 5.4).

### 2.5.6 Communication cost

In words: in federated learning each client uploads its model, so the
upload per round is the number of model parameters times the size of one
number. Since every parameter is a 32-bit float (4 bytes), the upload
follows directly from the model size (my own derivation, based on the
setting in [1]):

$$B_{\text{round}} \approx P \times 4 \text{ bytes}$$

where $B_{\text{round}}$ is the upload per client and round and $P$ is the
number of parameters sent. For my audio model with 111,370 parameters this
is about 445 KB per round, no matter how many samples the client trained
on. In centralized training the upload grows with every new sample instead.
This simple formula explains one of the main results in Chapter 5.

### 2.5.7 Privacy limits

Federated learning keeps the raw data on the device, but it is not
perfectly private. Zhu et al. showed that training data can sometimes be
rebuilt from shared gradients [7]. Secure aggregation [8] and differential
privacy are the usual ways to reduce this risk, and they are outside the
scope of my thesis. My privacy claim is therefore limited: raw audio never
leaves the device, not that no information at all leaves it.

### 2.5.8 Frameworks

Flower [5] is a popular framework for federated learning, and Ridolfi et
al. used it on Raspberry Pi devices [6]. Flower was my first choice,
and an early version of my project did. The final system implements FedAvg
directly over MQTT, for the reason given in Section 3.3.

## 2.6 Edge Computing and MQTT

Edge computing means processing data close to where it is created, on small
devices such as a Raspberry Pi, instead of sending everything to a data
centre. This saves network traffic and delay, but edge devices have little
memory, computing power and energy.

MQTT is a lightweight publish/subscribe protocol made for such devices
[24]. Devices do not talk to each other directly. A central broker (in my
case Eclipse Mosquitto [25]) receives every message. A sender publishes a
message to a topic, a text label that looks like a file path, and every
client that has subscribed to that topic receives it. I use three MQTT
features:

- **Quality of Service (QoS) 1:** the broker makes sure a message arrives at
  least once, so a batch cannot get lost without anyone noticing.
- **Retained messages:** the broker keeps the last message on a topic and
  gives it to every client that subscribes later. I use this for the shared
  model, so a device that connects late still gets the current model.
- **Wildcards:** the server subscribes to `thesis/edge/+/data`, where `+`
  matches any node name, so one subscription receives data from all devices.

## 2.7 Multimodal Fusion and Modality Imbalance

Multimodal models combine different kinds of input, for example an image
and a sound. In early fusion the raw inputs are joined before any
processing. In late fusion each input first goes through its own branch,
and the results are joined near the end. Late fusion is simpler and lets me
reuse pre-trained branches, so I use it.

A known problem is that multimodal networks are often no better than their
best single input. Wang et al. showed that the input that learns fastest
takes over the training, and the others stay under-trained [21]. Peng et
al. call this modality imbalance and balance the learning of the branches
[22]. A simpler countermeasure is modality dropout (ModDrop), introduced by
Neverova et al. [23]: during training, one input is randomly removed so the
model cannot depend on it alone. It builds on the general dropout idea of
Srivastava et al. [32].

For the vision branch I use MobileNetV3-Small [20], a small image network
made for mobile devices and pre-trained on ImageNet.

**In my work:** exactly the imbalance described in [21] appeared. Section
5.7 shows how I found it and fixed it.

## 2.8 Research Gap

Many federated learning studies compare final accuracy on benchmark
datasets in simulation. Fewer measure the actual bytes that cross a real
network when the same work is placed in different locations, and I found
none that do this for a system that labels its own data with a second
sensor. Work on audio-visual learning, on the other hand, usually assumes
that all data is on one machine. My thesis connects the two: it measures
centralized, hybrid and federated placement for a self-labelling
audio-visual system on real messaging infrastructure, and it checks whether
a fusion model's ability to work without the camera survives federated
training.
