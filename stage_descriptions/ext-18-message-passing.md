In this extension stage, you'll try Inkling 2 Mini's way of carrying work between passes.

### The idea

Inkling 2 Mini (24 layers, layers 6-17 run three times) does not add the input to the running state. **Every pass
starts again from the same input** (the first six layers' output), and what the previous pass worked out arrives as a
**message**: each looped layer reads the previous pass's outputs with its own learned query, and a small gate decides
how much of the current input to forget and how much of the message to add.

Ours is the one-gate version, at the core's entrance:

```python
m = Linear(LayerNorm(s_prev))                  # the message: what the last pass computed
g = sigmoid(Linear([e, m]))                    # per number: 0 = keep the input, 1 = take the message
s = core((1 - g) * e + g * m)                  # the gate starts mostly closed (bias -2)
```

Run it with `./your_program.sh injection --kinds looped_message`.

### What we found (3 seeds)

Longest chain solved, by loop count:

| model | seed | K = 1 | 2 | 4 | 8 | 16 |
|---|---|---|---|---|---|---|
| re-add the input (`looped`) | 0 / 1 / 2 | 2 / 3 / 3 | 3 / 5 / 4 | 5 / 6 / 6 | 7 / 8 / 8 | 9 / 9 / 9 |
| gated message (`looped_message`) | 0 / 1 / 2 | 4 / 3 / 4 | 4 / 4 / 5 | 4 / 6 / 7 | 4 / **10** / **11** | 4 / **12** / **12** |

When it trains, the gated message is the strongest model in the course: in seeds 1 and 2 it solves every chain the
test has, 12 hops (twice the longest training chain), at K = 16, and it gains more than one hop per loop between K = 4
and K = 8. In seed 0 it never learned to use its loops: stuck at 4 hops at every K (its curriculum stalled before 6
hops). One run in three failing is exactly why Inkling's real gates (one per layer and input, low-rank) and careful
training recipes matter; here it is a reason to run more seeds before believing either number.

### Check yourself

1. The gate's bias starts at -2, so `g` is about 0.12. Why start mostly closed?
2. The message model has 12,544 more parameters than `looped` (170,122 against 157,578: the message map and the gate). Is the comparison still
   fair? What would you add to make it fairer?
