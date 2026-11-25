# Build your own Looped Transformer

A standard transformer gets one pass through its layers to answer. A **looped transformer** sends its state through
the same block of layers again and again: more thinking per token, no more weights. Inkling 2 Mini runs its middle 12
layers three times (48 layer visits from 24 stored layers); NVIDIA's Dory runs a 20-layer block 1 to 4 times; Huginn
runs a 4-layer core anywhere from 4 to 64 times.

In this course you build one from scratch, in plain PyTorch, on a laptop CPU:

1. a tiny character-level GPT, the nanoGPT way (stages 01-05);
2. the same GPT turned into a looped model: a prelude, a shared core run K times, a coda (stage 06);
3. a puzzle where the amount of thinking is a dial: follow a chain of assignments like `q=k`, `k=d`, `d=7` to its
   digit (stage 07);
4. random-K training, input injection, and a fair fight against standard models with the same weights and with the
   same compute, on chains longer than any seen in training (stages 08-10);
5. six ways to look inside the loops: logit lens, tuned and Jacobian lenses, linear probes, activation patching, loop
   ablation with a fixed-point check, and one sparse autoencoder for every loop (stages 11-16);
6. two extensions: adaptive exit, and Inkling-style message passing between passes (stages 17-18).

The layout follows the CodeCrafters "build your own X" courses: a `course-definition.yml`, one description per stage
in `stage_descriptions/`, a starter in `starter_templates/python/`, the reference program after every stage in
`solutions/python/`, and tests per stage. The model code follows the structure of nanoGPT. Nothing is copied
from either; both are credited below. This is an independent project, not affiliated with or endorsed by CodeCrafters.

## What you'll find

Three seeds, every number from `./run_all.sh` on an M-series Mac CPU (about 32 minutes per seed with three seeds side
by side). Puzzles have 24 statements; training sees chains of 1-6 links; tests go to 12.

| model | parameters | longest chain solved, seeds 0 / 1 / 2 |
|---|---|---|
| standard, 3 blocks (same weights as the looped model) | 157,578 | 4 / 4 / 3 |
| standard, 10 blocks (same compute as the looped model at K = 8) | 507,466 | 4 / 6 / 5 |
| looped, 1 prelude + 1 shared core block + 1 coda, K = 1 | 157,578 | 2 / 3 / 3 |
| looped, K = 8 | 157,578 | 7 / 8 / 8 |
| looped, K = 16 | 157,578 | **9 / 9 / 9** |
| looped with Inkling-style messages (extension), K = 16 | 170,122 | 4 / **12 / 12** |

"Longest chain solved": the longest length with at least 90% right on it and on every shorter length (500 puzzles per
length). Inside the looped model:

- **One link per loop, backward.** Probes on the chain's statements show the answer digit moving back one statement
  per loop (a perfect staircase, stage 13); the question picks it up when it arrives, which is why K loops solve about
  K + 1 links.
- **One lens fits every loop when loops carry their state forward**; when every pass restarts from the input
  (Inkling-style), the logit and Jacobian lenses read little at intermediate loops (Jacobian 0.04-0.19) and a fitted
  tuned lens is needed (0.78-0.99) (stage 12).
- **Late edits stick, early ones wash out**: patching the clean state after loop 1 flips 2-5% of answers; after loop
  3, 96-99% (stage 14). Every loop matters equally (stage 15), and the state never reaches a fixed point.

## Limitations

- Toy scale: 157k-507k parameters, one synthetic task (chain following) plus a tiny Shakespeare run. The findings
  describe this task and these sizes; they are not claims about large looped language models.
- Three seeds per result. The "longest chain solved" numbers move by one or two links between seeds, so read them as
  ranges.
- The interpretability stages (11-16) use one trained looped model per seed and simple controls (shuffled labels,
  untrained models, random directions); they show methods, not a full mechanistic account.
- Stage 18's message passing is a simplified, single-gate version of the published design.
- CPU only and unoptimised: training is written for clarity, not speed.

## Stages

| # | Stage | Command | What you learn |
|---|---|---|---|
| 01 | [Characters to numbers](stage_descriptions/base-01-tokenize.md) | `tokenize` | a character vocabulary, encode and decode |
| 02 | [Embeddings and the forward pass](stage_descriptions/base-02-forward.md) | `forward` | token and position vectors, logits, the loss of a model that knows nothing (`ln V`) |
| 03 | [Causal self-attention](stage_descriptions/base-03-attention.md) | `attention` | queries, keys, values, the mask, and a test that nobody reads the future |
| 04 | [The transformer block](stage_descriptions/base-04-block.md) | `block` | residual connections, LayerNorm, the MLP, counting parameters |
| 05 | [Train a tiny GPT](stage_descriptions/base-05-train-gpt.md) | `train-gpt` | the training loop, validation loss, greedy generation (tinyshakespeare: 1.72 in 30 s) |
| 06 | [Loop it](stage_descriptions/base-06-loop.md) | `loop` | prelude / shared core × K / coda; parameters vs compute; a shared block's gradient is a sum |
| 07 | [A task that needs depth](stage_descriptions/base-07-chain.md) | `chain` | chain-following puzzles; a twin chain that blocks the "longest chain" shortcut |
| 08 | [Random loop counts](stage_descriptions/base-08-train-looped.md) | `train-looped` | random K per batch, a task curriculum, sweeping K = 1-16 at test time |
| 09 | [Input injection](stage_descriptions/base-09-injection.md) | `injection` | re-adding the input every loop vs a plain loop |
| 10 | [Looped vs standard](stage_descriptions/base-10-compare.md) | `compare` | same-parameter and same-compute baselines, length extrapolation |
| 11 | [Logit lens per loop](stage_descriptions/base-11-logit-lens.md) | `logit-lens` | decoding the state after each loop, with and without the coda |
| 12 | [Tuned and Jacobian lenses](stage_descriptions/base-12-tuned-lens.md) | `tuned-lens` | when one lens serves every loop, and when it can't |
| 13 | [Linear probes per loop](stage_descriptions/base-13-probe.md) | `probe` | where the chain is followed; shuffled-label and untrained-model controls |
| 14 | [Activation patching across loops](stage_descriptions/base-14-patch.md) | `patch` | clean vs corrupted runs; a random-vector control |
| 15 | [Loop ablation and convergence](stage_descriptions/base-15-ablate.md) | `ablate` | skipping one loop; is there a fixed point? |
| 16 | [A sparse autoencoder shared across loops](stage_descriptions/base-16-sae.md) | `sae` | one dictionary for every loop; following one feature; a random-direction check |
| 17 | [Adaptive exit](stage_descriptions/ext-17-adaptive-exit.md) (extension) | `exit` | stop when the answer stops changing (and why that fails on long chains) |
| 18 | [Message passing between passes](stage_descriptions/ext-18-message-passing.md) (extension) | `injection --kinds looped_message` | Inkling 2 Mini's gated cross-pass messages, simplified |

## How to run

Requirements: Python 3.10+, PyTorch 2.x (`pip install torch`), and `pip install -r requirements.txt` (numpy,
matplotlib, pytest, ruff). Everything runs on a CPU. The only download is the 1 MB tinyshakespeare file for stage 05
(`make data`).

```bash
cp -r starter_templates/python/code my-looped          # your working copy; start with app/main.py
make test STAGE=01 DIR=my-looped                        # one stage's tests against your code
make test-upto STAGE=08 DIR=my-looped                   # every stage so far
make test-solutions                                     # every reference solution against its stage (what CI runs)
make run-all                                            # every number quoted in the stage descriptions (3 seeds)
make solutions                                          # regenerate solutions/python/01-17 from 18
```

Stuck? `solutions/python/NN-<slug>/code/` is the program right after stage NN, and `run_all.sh` lists every command.
Tests for stages 08-18 train small models, so they take a minute or two each.

## Further reading and code

What each one is, and what this course borrows. We wrote our own code; nothing below is copied.

**The structure of this course and its code**

- nanoGPT, https://github.com/karpathy/nanoGPT, and the "Let's build GPT" lecture code,
  https://github.com/karpathy/ng-video-lecture: the shape of stages 01-05.
- CodeCrafters "build your own" courses (MIT), https://github.com/codecrafters-io/build-your-own-tokenizer and
  https://github.com/codecrafters-io/build-your-own-inference-engine: the course layout.

**Looped and recurrent-depth models**

- Universal Transformers (2018), https://arxiv.org/abs/1807.03819: one block applied over and over, with adaptive
  halting. Our `looped_none` is its plain loop.
- Looped Transformers as Programmable Computers (2023), https://arxiv.org/abs/2301.13196: a looped block can run a
  small instruction set; loops as steps of an algorithm.
- Looped Transformers are Better at Learning Learning Algorithms (ICLR 2024), code
  https://github.com/Leiay/looped_transformer: a looped GPT-2 block on nanoGPT for in-context regression;
  `z = f(z + x)` starting from `z = 0`, a curriculum on the number of loops, and a loss on only the last T loops with
  the earlier ones run without gradients. We borrow `z0 = 0` and `f(z + x)` (stage 06) and compare our random K with
  its loop curriculum (stage 08).
- Reasoning with Latent Thoughts: On the Power of Looped Transformers (2025), https://arxiv.org/abs/2502.17416: a
  k-layer block looped L times nearly matches a kL-layer model on reasoning; the reason for our same-compute baseline.
- Huginn, "Scaling up test-time compute with latent reasoning" (2025), https://arxiv.org/abs/2502.05171, code
  https://github.com/seal-rg/recurrent-pretraining: prelude / core / coda, input re-injected each recurrence through a
  learned map, random recurrence counts with truncated backpropagation. We borrow the three-part layout and random K.
- Ouro, "Scaling latent reasoning via looped language models" (2025), https://ouro-llm.github.io/, paper
  https://arxiv.org/abs/2510.25741, a reimplementation https://github.com/rkstgr/LoopLM: a whole stack looped 4 times,
  a loss at every loop weighted by a learned exit distribution. Its exit gate is the better version of stage 17.
- NVIDIA Nemotron-Labs Dory 1.0, https://huggingface.co/nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16: 8 / 20 / 8
  layers, 1-4 loops, the input re-added each loop with no extra parameters (stage 06's injection).
- Inkling 2 Mini, https://huggingface.co/blog/inkling2mini and
  https://www.mindstudio.ai/blog/inkling-2-mini-looped-transformer: layers 6-17 of 24 run three times; each pass
  restarts from the same input and receives gated messages from the previous pass. Stage 18 is a one-gate version.
- Mixture-of-Recursions (2025), https://arxiv.org/abs/2507.10524, code
  https://github.com/raymin0223/mixture_of_recursions: each token chooses its own depth, with matching KV caches.
- Relaxed Recursive Transformers (2024), https://arxiv.org/abs/2410.20672: turn a trained model into a looped one, with
  a small LoRA per loop so the loops need not be identical.
- CoTFormer (2024), https://arxiv.org/abs/2310.10845: later passes attend to earlier passes' tokens.
- How Much Is One Recurrence Worth? (2026), code https://github.com/kschwethelm/looped-lm-scaling: one recurrence is
  worth about r^0.46 of a unique layer; truncated backpropagation through loops hurts; several injection modes and a KV
  budget across recurrences.
- Towards Looped Models Done Right (2026), code https://github.com/ifm-ai/xllm-loop: diagonal, linear and orthogonal
  input injection, sampled depths, many released checkpoints.
- Two other small from-scratch looped GPTs: https://github.com/naatyu/Looped-Transformer (a nanochat fork with a loss
  per loop and a learned exit) and https://github.com/avikds/looped-transformer-from-scratch (Tiny Shakespeare with ACT
  halting, recursion routing and per-pass KV caches).

**Looking inside looped models**

- Looped Transformers under the Jacobian Lens (2026), https://arxiv.org/abs/2609.01924: the Jacobian lens on Ouro and
  Huginn. A workspace forms in the looped part, but Ouro rebuilds its content every loop, so reads and edits must be
  made per loop, and Huginn's reads and edits act only within about two recurrences. Stages 12 and 14 show small
  versions of both.
- Latent Chain-of-Thought? Decoding the Depth-Recurrent Transformer (2025), https://arxiv.org/abs/2507.02199, code
  https://github.com/wenquanlu/huginn-latent-cot: logit lens and "coda lens" on Huginn (stage 11 uses both).
- Loop, Think, & Generalize (2026), https://arxiv.org/abs/2604.07822: lenses on bridge entities in multi-hop reasoning
  in recurrent-depth models.
- A Mechanistic Analysis of Looped Reasoning Language Models (2026), https://arxiv.org/abs/2604.11791.
- Monitoring latent reasoning in looped language models (2026), https://arxiv.org/abs/2610.18127.
- Tuned lens (2023), https://arxiv.org/abs/2303.08112: one learned affine map per layer (stage 12 fits one per loop).
- A Dory-style looped model on a harder version of the chain task, with probes after every layer and loop:
  https://albertlieyingadrian.github.io/steerlab/experiments/looped-transformer/report/

## License

MIT. See [LICENSE](LICENSE).
