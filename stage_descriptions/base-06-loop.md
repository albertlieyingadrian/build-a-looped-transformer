In this stage, you'll turn the stack of blocks into a looped transformer.

### The idea: the same worker, several shifts

A standard transformer is an assembly line of distinct workers: block 1, block 2, block 3, each run once. A looped
transformer hires one worker for the middle of the line and sends the work past them again and again. Same wages
(parameters), more hours (compute).

```
embed ──► prelude (P blocks, once) ──► e ─────────────────────────────┐  re-added before every loop
                                       │                              │  ("input injection")
                                       ▼                              ▼
                                 s0 = 0 ──► core (C shared blocks) ──► s1 ──► core (same weights) ──► ... ──► sK
                                                                                                          │
                                                         coda (D blocks, once) ◄──────────────────────────┘
```

- **Prelude**: reads the input into the model's internal format, once.
- **Core**: the shared blocks, run K times. K is an argument of `forward`, so one trained model can run with any K.
- **Coda**: turns the final state into the answer, once.

This is the layout of Huginn (2 / 4 / 2 blocks, the core run 4 to 64 times), NVIDIA's Dory (8 / 20 / 8, 1 to 4 loops)
and Inkling 2 Mini (6 / 12 / 6, 3 loops: 48 layer visits from 24 stored layers). Ours is 1 / 1 / 1.

### The diff from your GPT

About twenty lines. The blocks list becomes three lists, and `forward` gets a loop:

```python
e = run_prelude(idx)                 # once
s = zeros_like(e)
for _ in range(k):                   # the only new idea
    s = core(s + e)                  # input injection: re-add e every loop (stage 09 asks why)
logits = head(ln_f(coda(s)))         # once
```

`s0 = 0` and `s = f(s + e)` follow the looped in-context-learning code (`z = f(z + x)`); Dory re-adds its input the
same way, with no extra parameters.

### Parameters versus compute

With 1 / 1 / 1 the looped model stores exactly the weights of a 3-block GPT (162,561 with a 65-character vocabulary),
whatever K is. Its compute is that of a `1 + K + 1`-block GPT. Stage 10 compares it with both: a standard model with
the same parameters (3 blocks) and one with the same compute as K = 8 (10 blocks).

### The gradient of a shared block

When the same weights are used K times, backpropagation adds up the gradient from each use. Your command must check
this: copy the core into K separate blocks, run them in a row, and compare.

```bash
$ ./your_program.sh loop
one block: 49984 parameters
looped (1 prelude + 1 core + 1 coda): 162561 parameters
standard (3 blocks): 162561 parameters
core: 49984 parameters (one block, shared by every loop)
k= 1  depth  3 blocks  parameters 162561  logits (2, 16, 65)
k= 2  depth  4 blocks  parameters 162561  logits (2, 16, 65)
k= 4  depth  6 blocks  parameters 162561  logits (2, 16, 65)
k= 8  depth 10 blocks  parameters 162561  logits (2, 16, 65)
k=16  depth 18 blocks  parameters 162561  logits (2, 16, 65)
shared gradient = sum of the 3 unrolled copies' gradients: True
```

### Tests

The tests check that the looped and 3-block models have the same parameter count, that the core is one block, that
the depth is `K + 2` at every K with the parameter count unchanged, and that the gradient check prints `True`.

### Check yourself

1. Inkling 2 Mini stores 24 layers and runs 48 layer visits per token. Which costs more memory at inference, its
   weights or the cache of keys and values for every visit? (Each visit keeps its own KV cache.)
2. Why can't you simply train a 3-block model and run its middle block twice at test time?
