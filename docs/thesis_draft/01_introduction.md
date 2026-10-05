# 1 Introduction

This chapter explains why I chose this topic, what problem I worked on,
which questions I wanted to answer and how the rest of the thesis is
organised.

## 1.1 Motivation

I chose this topic because robots, edge devices and machine learning are
growing fast and already play an important role in many fields, from
delivery and rescue robots to self-driving cars, and they will matter even
more in the future.

Robots and autonomous systems mostly rely on cameras to recognise objects.
Modern object detectors such as YOLO can name dozens of objects in a camera
image within milliseconds [18], [19]. In fog, storms or darkness, however,
the camera has difficulty detecting objects accurately, and this is often
exactly the moment when good perception matters most.

In such situations, sound can take over. A car still sounds like a car in
the dark, and a dog still barks in fog. A microphone is cheap, small and does
not need light. People do the same every day: when we cannot see something,
we often still know what it is because we hear it. The idea behind my thesis
is to give a machine this backup sense, so that it can still recognise
objects by their sound when the camera fails, and to combine sound and image
when both are available.

The idea itself is not new. Chan et al. combined vision and hearing on a
sound-localising robot [26], and other work has shown that sound and video can teach each other [14]–[16]. What is less studied is how such a system
should keep learning once it runs on small edge devices, and where that
learning should happen.

At the start of this work I saw this as a trade-off between three things that pull
in different directions: privacy (a microphone that is always on also
records conversations), power (training on a small battery-powered device is
expensive) and accuracy (a central server with all the data usually learns
best). My supervisor asked me not to assume from the start that federated
learning is the answer, but to compare the options fairly. That is what this
thesis does.

## 1.2 Problem Statement

To recognise sounds, a neural network needs thousands of labelled examples
such as "this sound is a car". Labelling audio by hand is slow and
expensive, and it is impossible for a device that keeps recording new data
in the field.

I solve the labelling problem with the camera. While there is enough light,
a pre-trained object detector (YOLOv8) looks at the camera image and names
what it sees. The sound recorded at the same moment gets that name as its
label. In this way the sense that works now (vision) teaches the sense that
will be needed later (hearing). This is called cross-modal supervision. Once
the audio model is trained, it can recognise objects from sound alone, also
when the camera sees nothing.

![Figure 1.1: The core idea: while the camera can see, YOLOv8 labels the sound; when the camera fails, the trained audio model recognises the object from sound alone. Source: own work.](docs/figures/06_concept_camera_teaches_mic.png)

Because the system labels its own data, it can keep learning for as long as
it runs. This leads to the actual engineering question of my thesis: where
should this training happen?

- on a central server, which receives the raw recordings from all devices
  (Strategy A, centralized),
- partly on the device, which prepares the data before sending it
  (Strategy B, hybrid), or
- completely on the device, which only shares what it has learned
  (Strategy C, federated).

Each choice changes how much data goes over the network, how much work the
small device and the server must do, how private the recordings stay and how
accurate the model becomes. These factors pull in different directions, and
the right choice is not obvious without measuring.

It is important to keep two models apart from the start, because they do
different jobs:

| | YOLOv8 | Audio / fusion model |
|---|---|---|
| Role | teacher (gives the labels) | student (learns from the labels) |
| Trained in this thesis? | no, fixed and pre-trained | yes, this is the model I study |
| Input | camera image | audio (later: audio and image) |
| Needs light? | yes | no |

*Table 1.1: The two models in the system and their roles.*

## 1.3 Research Questions and Objectives

My exposé asked one main question: how do centralized, hybrid and federated
training compare in accuracy, privacy, power consumption, bandwidth and
convergence, and which one suits an audio-visual system on small edge
devices? While working on it, I split this into three research questions:

**RQ1.** How do centralized, hybrid and federated training compare in
accuracy, upload bandwidth, device workload, server workload and privacy?

**RQ2.** How does federated averaging (FedAvg) combine the model weights of
the devices, and under which conditions does it reach the same accuracy as
centralized training on edge devices?

**RQ3.** Can an audio-visual fusion model keep recognising objects when the
camera fails, and does this ability survive hybrid and federated training?

To answer them, I set myself these objectives:

1. Build a pipeline that records sound at 16,000 samples per second, cuts it
   into 500 ms clips, turns each clip from the time domain into a mel
   spectrogram (frequency over time) with the short-time Fourier transform,
   and labels it automatically with YOLOv8.
2. Train a small audio classifier that is light enough for a Raspberry Pi.
3. Implement the three strategies so that they share the same devices,
   network protocol and data, and only the place of the work changes,
   including my own implementation of federated averaging, in which the
   server combines the devices' weight matrices with the FedAvg formula.
4. Measure accuracy, real network bytes and computing time for each strategy
   under the same conditions.
5. Build an audio-visual fusion model and test it with the camera blacked
   out.
6. Run the hybrid and federated strategies again with the fusion model.
7. Test the pipeline on a real Raspberry Pi with a live camera and
   microphone.

The original plan was to pick one strategy at the end. Instead, I implemented
and measured all three, and in Chapter 7 I recommend which strategy fits
which situation. Section 6.6 compares my results with the five metrics and
my original predictions, including the ones I could not
measure.

## 1.4 Contributions

What I am most proud of is that I did not only study these strategies on
paper: I implemented them, built and tested the system, ran the experiments
and observed what really happens, also on a real Raspberry Pi. The main
contribution is therefore a measured, end-to-end comparison of where to
train a self-labelling audio-visual edge system. Along the way, I found four
further results:

1. **Preparing the data on the device saves little bandwidth.** Sending
   spectrograms instead of raw audio saved only 26% of the upload, because a
   half-second spectrogram is almost as large as the half second of audio it
   comes from. Federated learning needed 16.5 times less upload. Simple formulas explain
   all these numbers (Section 5.10).
2. **Federated averaging can damage a model that is already trained.** With
   the same learning rate as on the server, federated accuracy jumped up and
   down and fell below the starting point. A 20 times smaller local learning
   rate fixed this.
3. **A fusion model trained the obvious way ignores the weaker sense.** It
   learned to use only the camera. Warm-starting the audio branch and then
   using modality dropout made it use both senses and keep working with a black camera.
4. **Federated cost depends on model size, not on data volume.** The upload
   per federated round depends only on how many parameters the model has.

## 1.5 Structure of the Thesis

- **Chapter 2** explains the background: spectrograms, neural networks,
  YOLO, cross-modal supervision, federated learning, MQTT and multimodal
  fusion, together with related work.
- **Chapter 3** describes the system design, the three strategies, the
  datasets, the models and how I ran the experiments.
- **Chapter 4** describes the implementation on the laptop and on the
  Raspberry Pi, including the problems I ran into and how I solved them.
- **Chapter 5** presents the measured results and checks them against the
  formulas.
- **Chapter 6** discusses what the results mean, compares them with my original plan, and lists the limitations and my own reflections.
- **Chapter 7** answers the research questions, gives a recommendation and
  outlines future work.
