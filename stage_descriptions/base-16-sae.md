In this stage, you'll train one sparse autoencoder for every loop and follow a single feature through the loops.

### Why one SAE for all loops?

A sparse autoencoder (SAE) rewrites a state as a short list of parts from a learned dictionary: "these 8 of 256
directions, in these amounts". In a standard model you train one SAE per layer. A looped model reuses the same
weights at every loop, so one dictionary can be shared by every loop, and then **latent 42 at loop 3 is the same
latent 42 at loop 7**. That lets you follow a feature through the loops, which separate per-layer SAEs (or a
crosscoder, which learns one dictionary across layers) would have to line up first.

### The `sae` command

1. Collect the question's state after loops 1-8 on 4,000 puzzles of 1-6 hops. Scale each loop's states to the same
   average length (the state grows every loop, stage 15, and late loops would otherwise dominate).
2. Train a TopK SAE: 256 latents, 8 active per state, 1,500 steps.
3. For each answer digit, pick the latent that best separates "the answer is this digit" at the last loop
   (selectivity = P(fires | digit) - P(fires | other digit)).
4. Follow that latent back through the loops, split by hop count.
5. Control: do the same with 256 random directions, each "firing" on its top 10% at the last loop.

### What we found

The SAE rebuilds the states well (fraction of variance unexplained 0.04-0.06; 87-103 of 256 latents ever fire) and
has one latent per answer digit (selectivity 0.86-0.99). Followed back through the loops (seed 0, share of puzzles
where the true digit's latent fires):

| after loop | 1 hop | 2 | 3 | 4 | 5 | 6 hops |
|---|---|---|---|---|---|---|
| 1 | 1.00 | 1.00 | 0.03 | 0.04 | 0.04 | 0.04 |
| 2 | 1.00 | 1.00 | **1.00** | 0.03 | 0.03 | 0.04 |
| 3 | 1.00 | 1.00 | 1.00 | **1.00** | 0.05 | 0.04 |
| 4 | 1.00 | 1.00 | 1.00 | 1.00 | **0.93** | 0.06 |
| 5 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | **0.84** |
| 6 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.96 |

The same latent switches on at loop `hops - 1`, the loop the digit reaches the question (stage 13), and stays on.

**The boring explanation, checked.** Random directions separate the digits at the last loop almost as well
(selectivity 0.64-0.99): the answer dominates the final state so much that almost any direction picks out one digit.
Followed back through the loops, the random directions show the same staircase, but blurred: for 5-hop chains after
loop 4 they fire on 0.46 / 0.78 / 0.08 of puzzles (seeds 0 / 1 / 2) where the SAE latent fires on 0.93 / 1.00 / 0.98,
and they keep firing for some wrong digits. So on this toy the SAE does not discover the answer feature (a probe or
the logit lens already finds it); what it adds is a sharp on-off signal for *when* it arrives, in one dictionary
shared by every loop.

### Check yourself

1. Why scale each loop's states to the same length before training the shared SAE? What would the SAE spend its
   latents on otherwise?
2. A crosscoder trains one dictionary that reads from several layers at once. For a looped model, what would it read,
   and what could it show that our shared SAE cannot?
