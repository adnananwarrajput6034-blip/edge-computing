# 7 Conclusion and Future Work

## 7.1 Summary

In this thesis I built an edge system in which a robot-like device, a
Raspberry Pi with a camera and a microphone, collects sounds and labels them
with the objects the camera sees. With this data I trained a machine
learning model to recognise objects by their sound, and later by sound and
image together. I then tried three different strategies for where the
training happens, measured them with the same metrics and compared the
trade-offs between them. I built the system step by step, and I learned a
lot from the mistakes I made along the way.

The main findings are:

- **Accuracy:** all three strategies reached about the same accuracy.
- **Bandwidth:** moving work to the device reduced the upload per
  device from 69.2 MB (centralized) to 51.0 MB (hybrid) and 4.18 MB
  (federated), about 16 times less for federated learning than for
  centralized training.
- **Workload:** in the centralized strategy the server's training time grows
  with every round, while in federated learning the server only averages and
  the work moves to the device.
- **Learning rate:** federated learning matched centralized training only
  after its local learning rate was made much smaller (from 1e-3 to 5e-5).
- **Hybrid strategy:** sending spectrograms instead of raw audio saved only
  about a quarter of the bandwidth, because a spectrogram is almost as large
  as the sound it is made from.
- **Fusion:** a model that combines sound and camera images kept recognising
  objects when the camera was blacked out (0.40 accuracy, and about 0.58
  after further training), while a camera-only model fell to chance (0.20).
- **Real hardware:** the complete system also ran live on a Raspberry Pi 5.

## 7.2 Recommendation

My original plan was to select one strategy. The results show that the best
choice depends on the situation:

| Situation | Recommended strategy |
|---|---|
| Very weak device, fast and cheap network, privacy not critical | **A, centralized** |
| Device can compute spectrograms but cannot train | **B, hybrid**: reduces the exposure of raw audio, but saves little bandwidth |
| Limited or expensive network, privacy matters, device can train | **C, federated** |
| Large or growing amounts of data | **C**, because its upload does not grow with the data |

*Table 7.1: Recommended strategy by situation.*

For now, I would use Strategy A, because it is fast, efficient for the device and the
easiest to implement. For the future, I recommend Strategy C: a robot in the
real world keeps meeting new sounds and objects, and with federated learning
it can learn from this new data on the device and keep its model up to date,
while the raw data stays private. The local learning rate must be chosen
carefully. Strategy B reduces the exposure of raw audio, because only
spectrograms leave the device, but a spectrogram is not encryption and it
saves only about a quarter of the bandwidth.

## 7.3 Future Work

1. **Model freshness.** Measure how fast each strategy learns when a new
   object appears. This is where federated learning is expected to have its
   biggest advantage.
2. **Latency, memory and battery operation on the Pi.** Measure the inference time and the memory use of each strategy on the device itself. For battery-powered robots, also measure the energy per round, which was outside the scope here because the Pis were plug-in powered.
3. **Two Pis in two places.** Run the federated rounds live between two
   Raspberry Pis, as originally planned; this is prepared for the defense
   demonstration. Placed in two different labs, each Pi hears different
   sounds, which also gives real cases of different data on each device
   (non-IID data).
4. **Better evaluation.** Use more test data for the fusion model and
   ten-fold cross-validation for UrbanSound8K, so that the results get an
   average and a spread instead of a single value.
5. **Stronger privacy.** Add secure aggregation or differential privacy to
   Strategy C, so that even the shared weights reveal as little as possible
   about the data.
