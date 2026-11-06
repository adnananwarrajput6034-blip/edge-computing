# Settings cheat sheet: change the experiment without changing code

> Written 6 Nov 2026 for the demo (post-submission). Every tool below has **options**
> (the `--something` words after the command). Changing an option changes the
> experiment; nobody needs to edit the code for that.

**Tip:** every tool lists all its options with `--help`, for example:

```bash
.venv/bin/python -m src.experiments.pi_teach --help
```

## The pipeline: what runs where, in order

| # | Step | Tool (file) | Runs on | Camera | Mic |
|---|---|---|---|---|---|
| 1 | **Record**: sound + photo + YOLO's names, every tick | `src/experiments/pi_record.py` | Pi | on | on |
| 2 | **Build the dataset**: choose labels, loudness filter, human check, train/test split | `scripts/build_dataset.py` (+ `scripts/merge_vocab.py` for two rooms) | Mac | – | – |
| 3 | **Train**: A, B or C (C = federated) | `strategy_c_server.py` (Mac) + `strategy_c_pi_client.py` (each Pi) | Mac + Pis | – | – |
| 4 | **Listen**: the ear alone (like fog or darkness) | `src/experiments/pi_infer.py` | Pi | off | on |
| 5 | **Teach**: it asks, you name or correct, it learns | `src/experiments/pi_teach.py` | Pi | off | on |
| – | **Live graphs** of step 3 | `.venv/bin/tensorboard --logdir results/experiments_20261106` | Mac | – | – |

Start order for step 3: **broker → server → Pis**.

## The knobs, and "if the professor asks…"

### Step 5: teaching (`pi_teach`)
| Option | Default | Meaning | If the professor asks… |
|---|---|---|---|
| `--want` | 12 | How many examples of a new sound it collects | "Can it learn from 5?" → `--want 5` (**few-shot learning**; less reliable, faster) |
| `--mode` | `quiz` | `quiz`: asks after every sound · `listen`: only asks when unsure, or when you press Enter | "Without a human?" → `--mode listen` and never press Enter: it only listens and says `unknown` |
| `--threshold` | 0.60 | Below this confidence it says `unknown` | "Make it ask more often" → `--threshold 0.8` |
| `--novelty-ticks` | 4 | listen mode: unknown sounds in a row before it asks | "Ask sooner" → `--novelty-ticks 2` |
| `--retrain-every` | 6 | quiz mode: answers before it learns again | "Learn after every answer" → `--retrain-every 1` |
| `--epochs` | 8 | Training passes each time it learns | "Faster" → `--epochs 4` |
| `--train` | `head` | `head`: only the last layers learn (less forgetting) · `all`: the whole network | "What if everything learns?" → `--train all` (watch the old test score) |

### Step 3: federated training (`strategy_c_server`, on the Mac)
| Option | Our value | Meaning | If the professor asks… |
|---|---|---|---|
| `--num-rounds` | 5 | How often the Pis meet at the server | "More rounds?" → `--num-rounds 10` |
| `--epochs-per-round` | 8 | Local epochs per round (how far each Pi walks alone) | "Less client drift?" → `--epochs-per-round 1` (and more rounds) |
| `--local-lr` | 1e-3 | Step size on the Pis | "Smaller steps?" → `--local-lr 1e-4` (steadier in our runs) |
| `--num-clients` | 2 | How many Pis must answer before averaging | "Only one Pi?" → `--num-clients 1` |
| `--init-model` | – | Start from an earlier model (relay training) | "Train one Pi after the other?" |
| `--no-class-check` | off | Accepts a Pi with a different class order (negative control only) | "What if the order differs?" |

### Step 3: each Pi (`strategy_c_pi_client`)
| Option | Meaning |
|---|---|
| `--node-id A` / `B` | Which Pi this is (its MQTT topic) |
| `--dataset` | Which recordings this Pi trains on (`results/live/room1_shared.npz` …) |
| `--shard 0/2` | Use only half of the data (even split) |
| `--class-order` | Number the classes differently (negative control only) |
| `--batch-size` | Clips per step (8) |

### Steps 1, 2 and 4
| Tool | Option | Meaning |
|---|---|---|
| `pi_record` | `--target "bottle"` | Counts the good examples of that object live |
| `pi_record` | `--min-rms auto` | Loudness gate = 1.5 × the quiet room |
| `pi_record` | `--duration-min 1` | Stop after 1 minute |
| `build_dataset` | `--classes "bottle,laptop"` | Which YOLO names become classes (order = priority) |
| `build_dataset` | `--min-conf 0.5` | How sure YOLO must be |
| `build_dataset` | `--background` | Add the quiet room as a class |
| `pi_infer` | `--threshold 0.6` | Below this confidence it says `unknown` |

## Where the settings live in the code
In every tool, the options are defined in one function at the top: **`parse_args()`**. Each line
`p.add_argument("--want", type=int, default=12, help="...")` is one knob: its name, its default
and its explanation. To change a default, change the number after `default=`.

## Honest answer for "can you change the code now?"
> "Most changes are options, and I can run them now. For example, `--want 5` learns from five
> examples. A new behaviour would be a code change. As I wrote in Section 3.9, I used AI help
> for programming, so I would describe the change, implement it with that help, and test it."
