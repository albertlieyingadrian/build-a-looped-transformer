In this stage, you'll fit lenses instead of borrowing the model's own head, and find out when one lens can serve
every loop.

### Two learned lenses

- **Tuned lens** (https://arxiv.org/abs/2303.08112): a small affine map per layer, trained so that
  `head(LayerNorm(map(h)))` matches the model's *final* answer distribution. Here: one map per loop, starting at the
  identity, fitted on 2,000 puzzles and scored on 1,000 others. Also one **shared** map for every loop.
- **Jacobian lens** (J-lens): no training at all. For loop i, measure `J_i`, the average over puzzles of how a small
  nudge to the state after loop i moves the final state (one backward pass per output number). Then read
  `head(LayerNorm(J_i @ h))`. It "fast-forwards" the state through the rest of the model with a straight line. Also
  one **averaged** `J`, the mean of `J_1 .. J_K`, for every loop.

### Why loops make lenses harder

A stack of distinct layers has one position per depth, so one lens per layer is natural. A looped model reuses the
same weights at many depths. Is "the state after loop 3" in the same format as "the state after loop 7"? If yes, one
lens fits every loop. If each loop rebuilds its content, a lens has to be fitted per loop, and a straight-line
transport from one loop to the end may fail. The 2026 preprint "Looped Transformers under the Jacobian Lens"
(https://arxiv.org/abs/2609.01924) found both kinds at scale: Huginn carries content forward through its recurrences,
while Ouro rebuilds it every loop, so linear maps can't carry content across loop boundaries and reads and edits must
be made per loop.

### The `tuned-lens` command

`--kind` picks which loop type to read: `looped` (re-add the input), `looped_none` (plain loop) or `looped_message`
(Inkling-style restart plus a gated message, stage 18). Score every lens after every loop on chains of 1-6 hops.

### What we found (K = 8, accuracy after loop 4 / loop 8, three seeds)

| model | logit lens | tuned, per loop | tuned, shared | Jacobian, per loop | Jacobian, averaged |
|---|---|---|---|---|---|
| `looped` (re-add input), s0 / s1 / s2 | .85 .99 .84 / 1.0 | .86 .98 .85 / 1.0 | .85 .98 .85 / 1.0 | .85 .98 .84 / 1.0 | .85 .98 .84 / 1.0 |
| `looped_none`, s0 / s1 / s2 | .98 .98 .89 / 1.0 | .98 .99 .96 / 1.0 | .96 .98 .93 / 1.0 | .97 .98 .91 / 1.0 | .97 .98 .91 / 1.0 |
| `looped_message`, s1 / s2 (s0 never used its loops) | .57 .41 / .69 .42 | .75 .68 / **.99 .78** | .75 .67 / .96 .75 | **.13 .12** / .76 .43 | .60 .38 / .77 .40 |

- **Loops that carry their state forward** (re-added input, or a plain loop) keep one format at every loop: every
  lens, including the free logit lens and one averaged Jacobian, reads about the same accuracy (within 0.07 from loop 1 on). One lens serves
  every loop.
- **Loops that restart from the input** (the message model) don't. The logit lens reads little (0.42-0.69 at loop 8),
  the Jacobian lens fails at every loop but the last (0.04-0.19 at loops 1-7: a straight-line transport across pass
  boundaries carries almost nothing), and a fitted tuned lens recovers the content (0.78-0.99). The same split as
  Huginn against Ouro in the preprint, in a model you can train in five minutes.

### Check yourself

1. Why is the per-loop Jacobian lens perfect after the last loop even for the message model?
2. A shared tuned lens did almost as well as per-loop ones here (0.96 vs 0.99). What does that tell you about the
   message model's format from loop to loop, as opposed to the Jacobian's transport?
