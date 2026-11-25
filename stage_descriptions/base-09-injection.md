In this stage, you'll find out why looped models re-add their input before every loop.

### Two ways to loop

```
inject = "add":   s = core(s + e)      Dory, Huginn (with a learned map), the looped in-context-learning code
inject = "none":  s = core(s)          the plain loop of Universal Transformers
```

`e` is the prelude's output: the puzzle itself, read once. Without re-injection, the core only ever sees its own
previous output. An analogy: solving a long-division problem while glancing back at the original question every step,
against copying your scratch work forward and never looking at the question again. Errors in the copy accumulate, and
after enough steps you are working on a different problem.

### The `injection` command

Train one model per kind (`--kinds looped,looped_none`), same seed, same data, same steps, and print each one's
accuracy grid. It reuses `looped_s<seed>.pt` from stage 08 if it is there.

### What we found (3 seeds)

Longest chain solved (at least 90% right on that length and every shorter one), by loop count:

| model | seed | K = 1 | 2 | 4 | 8 | 16 |
|---|---|---|---|---|---|---|
| re-add the input (`looped`) | 0 / 1 / 2 | 2 / 3 / 3 | 3 / 5 / 4 | 5 / 6 / 6 | 7 / 8 / 8 | **9 / 9 / 9** |
| plain loop (`looped_none`) | 0 / 1 / 2 | 4 / 4 / 3 | 5 / 5 / 5 | 7 / 7 / 7 | 7 / 7 / 7 | 7 / 7 / 7 |

Two different strategies:

- **The plain loop is better with few loops** (4 hops at K = 1 in two seeds, against 2-3). Nothing distracts it.
- **But it stops at 7 hops, one past training, at every K from 4 up.** Looping more does nothing, and on 8-hop chains
  seed 0 drops from 0.84 at K = 8 to 0.77 at K = 16: extra loops slowly wash the puzzle out of the state.
- **Re-adding the input keeps gaining hops with every loop**, to 9 hops at K = 16 in all three seeds: three past the
  longest training chain.

That matches the steerlab Dory-style note on one point and differs on another: there, without re-injection the model
also used its loops more at small K; here we see that it pays for that past the training range.

### Check yourself

1. In `s = core(s + e)`, what does the model see on loop 1, given `s0 = 0`?
2. Huginn re-injects through a learned matrix on `[s; e]` instead of `s + e`. The steerlab note found that version
   hard to train from scratch at this size. Why might a freshly initialised matrix on every loop's input hurt?
