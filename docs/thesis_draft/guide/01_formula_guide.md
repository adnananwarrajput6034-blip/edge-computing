# Formula Guide — every formula in my thesis, explained word by word

This guide is for learning, not for the thesis. For each formula:

- **The question:** what the formula answers, in one line.
- **The symbols:** every letter, how to say it out loud, what it means, and its value in my project.
- **With my numbers:** the formula calculated step by step with my real settings.
- **In one sentence:** what to say if an examiner asks.

Read one formula at a time. Cover the table with your hand and try to explain the formula before you read it.

---

# Part 1 — From sound to spectrogram

## F1. How many numbers is half a second of sound?

$$N = f_s \times d$$

**The question:** how many samples does one audio clip contain?

| Symbol | Say it as | What it means | In my project |
|---|---|---|---|
| $N$ | "N" | number of samples in the clip | 8,000 |
| $f_s$ | "f s", sample rate | how many times per second the microphone measures | 16,000 per second |
| $d$ | "d", duration | length of the clip | 0.5 seconds |

**With my numbers:**

$$N = 16000 \times 0.5 = 8000$$

**In one sentence:** half a second of audio at 16 kHz is a list of 8,000 numbers.

Code: `CHUNK = 8000` in `src/edge/processing/stft.py`.

---

## F2. How many time frames does the spectrogram have?

$$T = 1 + \left\lfloor \frac{N}{H} \right\rfloor$$

**The question:** how many columns (time steps) does the spectrogram picture have?

| Symbol | Say it as | What it means | In my project |
|---|---|---|---|
| $T$ | "T" | number of time frames (columns of the picture) | 51 |
| $N$ | "N" | samples in the clip (from F1) | 8,000 |
| $H$ | "H", hop | how far the window moves each step | 160 samples = 10 ms |
| $\lfloor \cdot \rfloor$ | "floor of" | round down to a whole number | – |
| $1 +$ | "one plus" | the first frame at position 0 also counts | – |

**With my numbers:**

$$T = 1 + \left\lfloor \frac{8000}{160} \right\rfloor = 1 + 50 = 51$$

**Think of it like this:** a window slides along the sound, jumping 160 samples each time. From position 0 to position 8,000 it makes 50 jumps, so it stops at 51 positions.

**In one sentence:** with a 10 ms hop, half a second gives 51 time frames.

**Examiner trap:** "Why not 49 or 48?" Without padding at the edges, the formula is $1 + \lfloor (8000 - 400)/160 \rfloor = 48$. librosa pads the edges by default (`center=True`), so I get 51.

---

## F3. How many frequency bins does the FFT give?

$$K = \frac{N_{FFT}}{2} + 1$$

| Symbol | Say it as | What it means | In my project |
|---|---|---|---|
| $K$ | "K" | number of frequency bins | 257 |
| $N_{FFT}$ | "N F F T" | FFT size: how many points the Fourier transform uses | 512 |

**With my numbers:**

$$K = \frac{512}{2} + 1 = 257$$

**Why half?** For a real signal, the second half of the FFT is a mirror copy of the first half, so only half is kept (plus the 0 Hz bin).

**In one sentence:** a 512-point FFT gives 257 useful frequency bins from 0 to 8 kHz.

---

## F4. How much do the frames overlap?

$$\text{overlap} = \frac{W - H}{W}$$

| Symbol | Say it as | What it means | In my project |
|---|---|---|---|
| $W$ | "W", window | how many samples one frame looks at | 400 (25 ms) |
| $H$ | "H", hop | how far the next frame starts | 160 (10 ms) |

**With my numbers:**

$$\text{overlap} = \frac{400 - 160}{400} = \frac{240}{400} = 0.6 = 60\%$$

**In one sentence:** each frame shares 60% of its samples with the next one, so no short sound falls between two frames.

---

## F5. The Short-Time Fourier Transform (STFT)

$$X(m,k) = \sum_{n=0}^{N_{FFT}-1} x(n + mH)\, w(n)\, e^{-j 2\pi k n / N_{FFT}}$$

**The question:** how much of each frequency is in each short piece of the sound?

| Symbol | Say it as | What it means | In my project |
|---|---|---|---|
| $X(m,k)$ | "X of m, k" | the result: strength of frequency $k$ in frame $m$ | a 51 × 257 table |
| $m$ | "m" | which frame (which moment in time) | 0 … 50 |
| $k$ | "k" | which frequency bin | 0 … 256 |
| $\sum_{n}$ | "sum over n" | add up over all samples inside one frame | – |
| $x(n + mH)$ | "x of n plus m H" | the audio samples of frame $m$ | – |
| $w(n)$ | "w of n", window | a bell-shaped curve that fades the frame edges to zero | Hann window, 400 samples |
| $e^{-j 2\pi k n / N_{FFT}}$ | "e to the minus j two pi k n over N F F T" | a pure tone at frequency $k$; multiplying and summing measures "how much of this tone is in the frame" | – |

**Think of it like this:** for every frame, hold up 257 tuning forks, one per frequency, and measure how strongly each one vibrates. That strength is $X(m,k)$.

**In one sentence:** the STFT cuts the sound into short overlapping pieces and measures which frequencies are in each piece.

Code: `librosa.feature.melspectrogram(...)` does F5, F6 and F7 in one call.

---

## F6. The mel scale

$$\text{mel}(f) = 2595 \cdot \log_{10}\left(1 + \frac{f}{700}\right)$$

**The question:** how does a human ear "space" frequencies?

| Symbol | Say it as | What it means | In my project |
|---|---|---|---|
| $f$ | "f" | a frequency in hertz | 0 … 8,000 Hz |
| $\text{mel}(f)$ | "mel of f" | the same frequency on the hearing scale | 0 … 2,840 mel |
| $\log_{10}$ | "log ten" | logarithm: big numbers get squeezed | – |
| 2595, 700 | constants | chosen so that 1,000 Hz = 1,000 mel | – |

**With my numbers:**

$$\text{mel}(1000) = 2595 \cdot \log_{10}(1 + 1.43) \approx 1000$$

$$\text{mel}(8000) = 2595 \cdot \log_{10}(1 + 11.43) \approx 2840$$

The 128 mel bands are spread evenly from 0 to 2,840 mel, about 22 mel apart. Low frequencies get many narrow bands, high frequencies few wide bands, just like the ear.

**In one sentence:** the mel scale gives more detail to low frequencies, where the ear hears more detail.

---

## F7. The log-mel spectrogram

$$S(m,b) = 10 \cdot \log_{10}\left(\sum_k |X(m,k)|^2 \, M_b(k)\right)$$

| Symbol | Say it as | What it means | In my project |
|---|---|---|---|
| $S(m,b)$ | "S of m, b" | one pixel of the final picture: frame $m$, mel band $b$ | 51 × 128 picture |
| $|X(m,k)|^2$ | "X squared" | energy of frequency bin $k$ (from F5) | – |
| $M_b(k)$ | "M b of k" | triangle filter $b$: how much bin $k$ belongs to mel band $b$ | 128 filters |
| $\sum_k$ | "sum over k" | collect the energy of all bins in this band | – |
| $10 \cdot \log_{10}$ | "ten log ten" | convert to decibels (dB) | – |

**Think of it like this:** 257 thin frequency stripes are poured into 128 buckets (the mel bands), then the bucket levels are written in decibels.

**In one sentence:** the final input to my model is a 51 × 128 picture of sound energy in decibels.

---

# Part 2 — How the neural network learns

## F8. Softmax: from scores to probabilities

$$p_c = \frac{e^{z_c}}{\sum_j e^{z_j}}$$

| Symbol | Say it as | What it means | In my project |
|---|---|---|---|
| $z_c$ | "z c" | raw score of class $c$ (any number, can be negative) | 10 scores for UrbanSound8K |
| $e^{z_c}$ | "e to the z c" | makes every score positive | – |
| $\sum_j e^{z_j}$ | "sum over all j" | total of all positive scores | – |
| $p_c$ | "p c" | probability of class $c$ (between 0 and 1, all add up to 1) | – |

**Small example with 3 classes:** scores $z = (2, 1, 0)$ for car, dog, siren.

$$e^2 = 7.39, \text{   } e^1 = 2.72, \text{   } e^0 = 1, \text{   } \text{sum} = 11.11$$

$$p_{car} = \frac{7.39}{11.11} = 0.67, \text{   } p_{dog} = 0.24, \text{   } p_{siren} = 0.09$$

**In one sentence:** softmax turns the network's raw scores into probabilities that add up to 100%.

---

## F9. Cross-entropy loss: how wrong was the model?

$$\ell = -\sum_c y_c \log p_c$$

| Symbol | Say it as | What it means | In my project |
|---|---|---|---|
| $\ell$ | "loss" | how wrong the prediction was (0 = perfect) | – |
| $y_c$ | "y c" | 1 for the correct class, 0 for all others | – |
| $p_c$ | "p c" | probability the model gave to class $c$ (from F8) | – |

**Because only the correct class has $y_c = 1$, the formula becomes simply:**

$$\ell = -\log(p_{\text{correct}})$$

**With the example above, if the true answer is car:**

$$\ell = -\log(0.67) = 0.40$$

If the true answer were siren: $\ell = -\log(0.09) = 2.41$, a much bigger punishment, because the model was confident and wrong.

**In one sentence:** the loss is small when the model gives a high probability to the right answer.

---

## F10. Gradient descent: how the weights change

$$w \leftarrow w - \eta \, \nabla_w \ell$$

| Symbol | Say it as | What it means | In my project |
|---|---|---|---|
| $w$ | "w", weights | all the numbers inside the network | 111,370 for the AudioCNN |
| $\leftarrow$ | "becomes" | replace the old value with the new one | – |
| $\nabla_w \ell$ | "gradient of the loss" | for each weight: which direction makes the loss bigger | – |
| $-$ | "minus" | go the opposite way, so the loss gets smaller | – |
| $\eta$ | "eta", learning rate | how big each step is | server: 0.001; federated: 0.00005 |

**Think of it like this:** you stand on a hill in fog (the loss) and want to reach the valley. The gradient tells you which way is uphill. You take a step of size $\eta$ downhill. Too big a step and you jump over the valley.

**In one sentence:** each training step moves every weight a little in the direction that reduces the error, and the learning rate decides how big that step is.

**Why this matters for my thesis:** the learning-rate sweep (Section 5.4) is exactly about $\eta$. With $\eta = 0.001$ the federated clients jumped too far; with $\eta = 0.00005$ they stayed stable.

---

## F11. Accuracy

$$\text{accuracy} = \frac{\text{number of correct predictions}}{\text{number of test samples}}$$

**With my numbers:** baseline AudioCNN on fold 10: 543 of 837 clips correct.

$$\text{accuracy} = \frac{543}{837} = 0.649 = 64.9\%$$

**Chance level:** with $C$ equally common classes, random guessing gives $1/C$: 10% for UrbanSound8K (10 classes), 20% for my VGGSound subset (5 classes).

---

# Part 3 — Federated learning

## F12. What federated learning tries to minimise

$$F(w) = \sum_{k=1}^{K} \frac{n_k}{n} F_k(w)$$

| Symbol | Say it as | What it means | In my project |
|---|---|---|---|
| $F(w)$ | "F of w" | the total loss over ALL data on all devices | – |
| $K$ | "K" | number of clients (devices) | 2 (node A, node B) |
| $k$ | "k" | which client | A or B |
| $n_k$ | "n k" | how many samples client $k$ has | 3,540 and 3,539 |
| $n$ | "n" | total samples of all clients | 7,079 |
| $n_k / n$ | "client k's share" | how much of all data is on client $k$ | 0.50007 and 0.49993 |
| $F_k(w)$ | "F k of w" | the average loss on client $k$'s own data | – |

**In one sentence:** the goal is the same as in normal training, a small loss on all data together, but each client only computes its own part.

---

## F13. FedAvg: how the server combines the models

$$w_{t+1} = \sum_{k=1}^{K} \frac{n_k}{n}\, w_{t+1}^{k}$$

| Symbol | Say it as | What it means | In my project |
|---|---|---|---|
| $w_{t+1}$ | "w t plus one" | the new global model after round $t$ | sent to both nodes |
| $t$ | "t" | round number | 1 … 10 |
| $w_{t+1}^{k}$ | "w t plus one of client k" | the model client $k$ sends back after local training | – |
| $n_k / n$ | "share" | weight of client $k$ in the average (from F12) | ≈ 0.5 each |

**With two clients it is just a weighted average:**

$$w_{t+1} = \frac{n_A}{n_A + n_B}\, w_A + \frac{n_B}{n_A + n_B}\, w_B$$

**Example for one weight:** A has 3,000 samples and value 0.40; B has 1,000 samples and value 0.80.

$$w_{t+1} = \frac{3000}{4000} \cdot 0.40 + \frac{1000}{4000} \cdot 0.80 = 0.30 + 0.20 = 0.50$$

**With my real numbers:**

$$w_{t+1} = 0.50007 \cdot w_A + 0.49993 \cdot w_B$$

so in my experiments FedAvg is almost the plain average of the two models.

**In one sentence:** the server averages the clients' models, giving more weight to the client with more data, and it does this for every one of the model's weights.

Code: `FedAvgAggregator.aggregate()` in `src/server/aggregation/fedavg.py` (Listing 4.1).

---

## F13b. What "averaging two models" means for the matrices

A model is not one number. My AudioCNN is a list of **22 arrays** (matrices) with **111,370 numbers** in total. FedAvg takes the same array from both nodes and averages them **element by element**: row 1, column 1 with row 1, column 1, and so on.

**Tiny example, equal data on both nodes (each share = 1/2):**

$$\frac{1}{2}\begin{pmatrix} 0.2 & -0.4 \\ 0.6 & 0.1 \end{pmatrix} + \frac{1}{2}\begin{pmatrix} 0.4 & -0.2 \\ 0.2 & 0.3 \end{pmatrix} = \begin{pmatrix} 0.3 & -0.3 \\ 0.4 & 0.2 \end{pmatrix}$$

Check one element yourself: top-left is (0.2 + 0.4) / 2 = 0.3.

| My model's biggest arrays | Shape | Numbers |
|---|---|---|
| Conv block 3 filters | 3 × 3 × 64 × 128 | 73,728 |
| Conv block 2 filters | 3 × 3 × 32 × 64 | 18,432 |
| Dense layer | 128 × 128 | 16,384 |

**In one sentence:** the server lays the two models on top of each other and takes the weighted average of every single number; the shapes never change, only the values.

**Why it works:** both nodes started the round from the *same* model, so position (row 1, column 1) means the same thing in both. If they started from different random models, the average would mix unrelated neurons (thesis Section 2.5.4).

---

## F14. FedProx (related work, not used)

$$\min_w \; F_k(w) + \frac{\mu}{2}\|w - w_t\|^2$$

| Symbol | Say it as | What it means |
|---|---|---|
| $F_k(w)$ | "F k" | client $k$'s normal loss |
| $\|w - w_t\|^2$ | "distance squared" | how far the local model has moved from the global model |
| $\mu$ | "mu" | how strongly moving away is punished |

**In one sentence:** FedProx adds a penalty that keeps each client close to the global model, which fixes the same problem I fixed by lowering the learning rate.

---

# Part 4 — Bandwidth and cost (my own derivations, Section 5.10)

## F15. Upload per federated round

$$B_{\text{round}} \approx P \times 4 \text{ bytes}$$

| Symbol | Say it as | What it means | In my project |
|---|---|---|---|
| $B_{\text{round}}$ | "B round" | bytes one client uploads in one round | – |
| $P$ | "P" | number of values sent (trainable weights plus batch-norm statistics) | 111,370 (audio), 389,893 (fusion) |
| 4 bytes | | size of one 32-bit decimal number | – |

**With my numbers:**

$$\text{audio: } 111{,}370 \times 4 = 445{,}480 \text{ B} \approx 445 \text{ KB} \text{   } (\text{measured } 418 \text{ KB})$$

$$\text{fusion: } 389{,}893 \times 4 = 1{,}559{,}572 \text{ B} \approx 1.56 \text{ MB} \text{   } (\text{measured } 1.44 \text{ MB})$$

The measured value is about 6% (audio) to 8% (fusion) smaller because zlib compresses the weights a little.

**In one sentence:** in federated learning you upload the model, so the cost depends on the model's size, not on how much data you have.

---

## F16. Bytes per sample in strategies A and B

$$S_A = N \times 2 \text{ B}, \text{     } S_B = T \times M \times 2 \text{ B}$$

| Symbol | Say it as | What it means | In my project |
|---|---|---|---|
| $S_A$ | "S A" | size of one raw audio clip | 16,000 B |
| $S_B$ | "S B" | size of one spectrogram | 13,056 B |
| $N$ | "N" | samples per clip (F1) | 8,000 |
| $T$, $M$ | "T", "M" | frames and mel bands (F2) | 51, 128 |
| 2 B | | one 16-bit number = 2 bytes | – |

**With my numbers:**

$$S_A = 8000 \times 2 = 16{,}000 \text{ B}, \text{     } S_B = 51 \times 128 \times 2 = 13{,}056 \text{ B}$$

$$\text{expected saving} = 1 - \frac{13{,}056}{16{,}000} = 18.4\% \text{     } \text{measured: } 26.2\%$$

**In one sentence:** a half-second spectrogram is almost as big as the half-second of audio it came from, so Strategy B saves little.

---

## F17. Break-even: when is federated cheaper than centralized?

$$n^* \cdot s_A = B_{\text{round}} \text{   } \Rightarrow \text{   } n^* = \frac{B_{\text{round}}}{s_A}$$

| Symbol | Say it as | What it means | In my project |
|---|---|---|---|
| $n^*$ | "n star" | samples per round where A and C cost the same | ≈ 30 |
| $s_A$ | "s A" | measured upload per sample in A (after compression) | 13,830 B |
| $B_{\text{round}}$ | "B round" | measured upload per round in C | 418,387 B |

**With my numbers:**

$$n^* = \frac{418{,}387}{13{,}830} \approx 30 \text{ samples}$$

**In one sentence:** if a device collects more than about 30 labelled clips between two rounds, federated learning uploads less than centralized training.

**Proof from my own demo:** the live Pi run had 32 samples per round, right at the break-even point, and A and C uploaded almost the same (1,337 KB vs 1,245 KB).

---

## F18. Why centralized training gets slower every round

$$t_r = k \cdot 1000 \cdot r, \text{     } T_R = k \cdot 1000 \cdot \frac{R(R+1)}{2}$$

| Symbol | Say it as | What it means | In my project |
|---|---|---|---|
| $t_r$ | "t r" | server training time in round $r$ | 3.6 s … 30.1 s |
| $k$ | "k" | training time per sample | ≈ 3.0 ms |
| 1000 | | new samples per round (the training trigger) | 1,000 |
| $r$ | "r" | round number | 1 … 10 |
| $T_R$ | "T R" | total server time after $R$ rounds | 167 s |
| $\frac{R(R+1)}{2}$ | "R times R plus one over two" | $1 + 2 + \dots + R$ | 55 for $R = 10$ |

**With my numbers:**

$$t_{10} = 0.00304 \times 1000 \times 10 = 30.4 \text{ s} \text{   } (\text{measured } 30.1 \text{ s})$$

$$T_{10} = 0.00304 \times 1000 \times 55 = 167 \text{ s} \text{   } (\text{measured } 167.3 \text{ s})$$

**In one sentence:** the centralized server retrains on everything it has ever received, so each round takes longer than the last, while the federated server only averages and stays constant.

---

## F19. A blackout score that only looks like chance

$$\text{accuracy}_{\text{constant}} = \frac{\text{clips of the predicted class}}{\text{all test clips}}$$

**With my numbers:** the test set has 21 bird, 21 car, 20 cat, 21 dog, 20 motorcycle clips (103 in total). A model that always says "bird":

$$\frac{21}{103} = 0.204$$

That is exactly the blackout score of the vision-only and the naive fusion model.

**In one sentence:** 0.204 is not random guessing; it is a model that gives the same answer for every clip, which the probe in Table 5.10 confirms.

---

# Quick check: can you explain these without looking?

1. Why 51 frames and not 50? → F2
2. Why is the learning rate the key to federated learning? → F10, Section 5.4
3. What does the server actually compute in FedAvg? → F13
4. Why does federated learning upload 418 KB per round no matter how much data? → F15
5. Why does Strategy B save only about a quarter? → F16
6. When does federated learning start saving bandwidth? → F17
7. Why does the centralized server get slower? → F18
