In this stage, you'll knock out single loops and ask whether the looping settles down.

### Two questions

1. **Does every loop do the same job?** Skip loop i (the state passes through unchanged that time) and measure the
   accuracy. If loops specialised (an early "find" loop, a late "answer" loop), skipping some would hurt more.
2. **Does the state converge?** A looped model that settled on a **fixed point**, a state the core no longer changes,
   could stop looping by itself. Measure, out to 16 loops, the relative change of the question's state
   (`|s_i - s_{i-1}| / |s_i|`), its length, and how often the decoded answer flips between loops.

### What we found (K = 8)

Skip one loop, accuracy on 6-hop and 8-hop chains (seeds 0 / 1 / 2):

| skipped loop | none | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|---|
| 6 hops | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| 8 hops, s0 | 0.91 | 0.75 | 0.75 | 0.75 | 0.75 | 0.75 | 0.75 | 0.75 | 0.75 |
| 8 hops, s1 | 0.97 | 0.95 | 0.95 | 0.95 | 0.95 | 0.95 | 0.95 | 0.95 | 0.95 |
| 8 hops, s2 | 0.98 | 0.93 | 0.93 | 0.93 | 0.93 | 0.93 | 0.93 | 0.93 | 0.93 |

6-hop chains need 5 loops, so with 8 there is slack and no single loop matters. On 8-hop chains every loop matters
**exactly equally**: skipping loop 1 costs the same as skipping loop 8. The loops are interchangeable steps, one link
each, which is what stage 13's staircase predicts.

Convergence (seed 0, chains of 1-6 hops):

| loop | 1 | 2 | 4 | 6 | 8 | 12 | 16 |
|---|---|---|---|---|---|---|---|
| relative change | 0.78 | 0.58 | 0.35 | 0.25 | 0.16 | 0.09 | 0.07 |
| length of the state | 22 | 42 | 85 | 139 | 196 | 307 | 417 |
| answer flips since last loop | - | 0.40 | 0.12 | 0.00 | 0.00 | 0.00 | 0.00 |

**No fixed point.** The relative change falls like 1/i only because the state keeps growing by about 28 per loop;
the core keeps adding the same-sized update forever. That is what re-adding the input does, and the steerlab looped
note found the same 1/i curve at 0.8M parameters. The *answer* does settle (no flips after loop 6), so "the answer
stopped changing" is a better stopping signal than "the state stopped changing". Stage 17 tests it.

### Check yourself

1. Why does a growing state make the relative change fall like 1/i even if the core's update has a constant size?
2. Huginn and Ouro use RMSNorm or LayerNorm around the recurrence. How would normalising the state each loop change
   both the convergence numbers and stage 14's patching result?
