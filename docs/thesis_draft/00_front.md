## Task Description (Aufgabenstellung)

Task as agreed with the supervisor, Prof. Dr.-Ing. André Jakob, in the
exposé of 23 February 2026.

**Topic:** Integrated Edge-to-Cloud Systems: Federated Learning for Sensor
Fusion and Object Recognition

**Problem.** Camera-only robotic systems fail in low-light or visually
obstructed environments. Adding audio sensors improves perception, but
raises the question of where and how the machine learning should happen,
balancing privacy, power consumption and model accuracy.

**Task.** Compare three learning strategies for a multimodal edge system
and, based on the comparison, recommend a suitable strategy:

1. Develop a multimodal edge system that integrates a camera and a
   microphone on Raspberry Pi edge devices connected to a server.
2. Implement three strategies: centralized learning (raw data sent to the
   server for training), hybrid learning (spectrograms computed on the
   device and sent to the server) and federated learning (local training on
   the device, only model weights shared).
3. Design an audio processing pipeline based on the short-time Fourier
   transform (STFT) on the device, use a pre-trained vision model (YOLOv8)
   to label the audio automatically, and develop a model that combines
   visual and auditory inputs for object recognition.
4. Evaluate and compare the strategies by accuracy, privacy, power
   consumption, bandwidth, latency and convergence behaviour.
5. Select and justify the most suitable strategy based on the results.

## Abstract

Robots and autonomous systems mostly rely on cameras to recognise objects. In
fog, storms or darkness, however, the camera has difficulty detecting objects
accurately. In such situations, sound can take over, because objects can also
be recognised by the sound they make. To learn these sounds, the system uses
the camera itself as a teacher: while the camera can see, an object detector
(YOLOv8) names the object, and the sound recorded at the same moment receives
that name. In addition, sound and camera images are combined in a fusion
model, so that the system uses both senses when the camera works and still
recognises objects by sound when it fails.

In this thesis, different strategies for training such a system were
implemented and compared, to observe which one gives the best results and
what the trade-offs are. The strategies differ in where the training takes
place: on a central server (centralized), partly on the device, which
converts the sound into spectrograms (hybrid), or completely on the device,
which sends only model weights (federated). In all three, the trained model
makes its decisions on the device. They were implemented with a Raspberry Pi edge device, two simulated edge devices and a laptop server communicating over MQTT, and compared by
accuracy, upload bandwidth, workload on device and server, latency and
privacy.

Centralized training was the easiest to implement and keeps the device
light, but raw audio leaves the device. The hybrid strategy reduced this
exposure but saved only about a quarter of the bandwidth. Federated learning
needed about 16 times less upload, kept the data on the device and reached
similar accuracy, but only after its learning rate was made much smaller.
The fusion model kept recognising objects when the camera was blacked out,
while a camera-only model failed.

Federated learning is therefore recommended for a system that keeps learning
in the field, as it also makes frequent model updates cheap; how fast each
strategy adapts to new objects is left for future work. The complete system
was also tested live on a Raspberry Pi 5.

**Keywords:** federated learning, edge computing, sensor fusion, audio
classification, cross-modal supervision, MQTT, Raspberry Pi

## List of Abbreviations

| Abbreviation | Meaning |
|---|---|
| CNN | Convolutional Neural Network |
| COCO | Common Objects in Context (dataset) |
| FB | Hybrid strategy with the fusion model |
| FC | Federated strategy with the fusion model |
| FedAvg | Federated Averaging |
| FL | Federated Learning |
| FV | Feature Vector |
| IID | Independent and Identically Distributed |
| LR | Learning Rate |
| MQTT | Message Queuing Telemetry Transport |
| QoS | Quality of Service |
| RMS | Root Mean Square |
| STFT | Short-Time Fourier Transform |
| YOLO | You Only Look Once (object detector) |
