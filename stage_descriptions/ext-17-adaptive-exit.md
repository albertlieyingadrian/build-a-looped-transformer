In this extension stage, you'll stop looping when the answer stops changing.

### The idea

Random-K training gave us one model that works at every K. But a 2-hop chain is done after 1 loop, and a 7-hop chain
needs 6. Spending 16 loops on every question wastes compute; spending 2 fails the hard ones. Ouro learns an exit gate
for this; Huginn stops when the output distribution stops moving. The simplest rule: **stop at the first loop whose
answer equals the previous loop's answer.**

### The `exit` command

Run every loop up to `--k-max`, read the coda's answer after each, and pick the first loop that agrees with the one
before it. Print, for each hop count, the mean number of loops used and the accuracy, next to always running
`--k-max` loops.

### What we found (seed 0, `--k-max 16`, 300 puzzles per hop count)

```bash
$ ./your_program.sh exit --ckpt runs/looped_s0.pt
adaptive exit: stop at the first loop whose answer equals the previous loop's (at most 16)
  1 hops: mean loops used  2.00  accuracy 1.00  (always 16 loops: 1.00)
  2 hops: mean loops used  2.00  accuracy 1.00  (always 16 loops: 0.99)
  3 hops: mean loops used  2.64  accuracy 1.00  (always 16 loops: 1.00)
  4 hops: mean loops used  2.95  accuracy 0.78  (always 16 loops: 1.00)
  5 hops: mean loops used  3.01  accuracy 0.60  (always 16 loops: 1.00)
  6 hops: mean loops used  2.96  accuracy 0.47  (always 16 loops: 0.99)
  7 hops: mean loops used  3.00  accuracy 0.41  (always 16 loops: 1.00)
  8 hops: mean loops used  2.93  accuracy 0.37  (always 16 loops: 0.99)
```

**The simple rule fails on long chains.** It saves loops on easy puzzles (2 instead of 16) but stops after about 3
loops on every chain, so 8-hop accuracy falls from 0.99 to 0.37. Seeds 1 and 2 agree (0.42 and 0.34 on 8 hops).
The reason is in stage 11: before the digit arrives, the coda's guess is often wrong but *stable* from one loop to
the next, so "unchanged" does not mean "finished". Stage 13 suggests a better
signal: stop when the question's state first holds the digit, which a probe can read. Ouro trains its exit gate for
exactly this reason.

### Check yourself

1. Why can "the answer didn't change" stop too early on a long chain? Look at what the coda says after loop 1 and 2 on
   a 6-hop chain (stage 11's coda lens).
2. How would you train a model so that stopping early is safe? (Hint: Ouro's loss at every loop.)
