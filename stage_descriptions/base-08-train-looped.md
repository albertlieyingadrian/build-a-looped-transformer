In this stage, you'll train the looped model on the chain puzzles with a random loop count, then run it at every loop
count from 1 to 16.

### Random K: one model, every depth

If you always train with K = 4, the coda learns to read the state after exactly four loops and nothing else. Draw K
at random for every batch (here uniformly from 1 to 8) and the coda has to answer from the state after *any* number
of loops. Then K becomes a dial you can turn at test time, as Dory (1-4 loops) and Huginn (4-64) allow.

Other recipes you'll meet:

- **A loop curriculum** (the looped in-context-learning code, https://github.com/Leiay/looped_transformer): start
  with few loops and raise the maximum during training, and compute the loss only on the last T loops, running the
  earlier ones without gradients. That saves memory when K is large.
- **Truncated backpropagation** (Huginn): gradients flow through only the last few loops.
- **A loss after every loop** (Ouro): every loop's output is scored, weighted by a learned exit distribution.

We keep the simplest: one random K per batch, the full gradient through every loop. Our models are tiny, so memory is
not the problem.

### A curriculum on the task

Starting straight on 6-hop chains, a model sees mostly noise. So training starts at 1-2 hops and adds one hop whenever
the last 50 batches were 90% right, up to 6. The chains of 7-12 hops are **never** seen in training: they test
whether looping more lets the model go further than it was taught.

### Settings

`n_embd = 64`, 4 heads, 1 prelude block + 1 core block + 1 coda block (157,578 parameters with the puzzle's
embeddings), 4,000 steps of 128 puzzles, AdamW at 2e-3 with warm-up and cosine decay. About 5 minutes on a laptop CPU
(6-7 when three seeds share the machine).

### What you should see (seed 0)

```bash
$ ./your_program.sh train-looped --seed 0
looped: 157578 parameters, trained in 394s, curriculum [(0, 2), (1030, 3), (1577, 4), (1724, 5), (2118, 6)]
looped: accuracy by hop count (columns) and loop count (rows); training saw 1-6 hops
           1    2    3    4    5    6    7    8    9   10   11   12
   k=1  1.00 1.00 0.33 0.22 0.25 0.26 0.29 0.27 0.31 0.35 0.43 0.53
   k=2  1.00 1.00 1.00 0.47 0.32 0.32 0.32 0.33 0.35 0.42 0.55 0.52
   k=3  1.00 1.00 1.00 1.00 0.63 0.40 0.38 0.41 0.42 0.45 0.54 0.51
   k=4  1.00 1.00 1.00 1.00 1.00 0.68 0.43 0.43 0.42 0.45 0.53 0.49
   k=6  1.00 1.00 1.00 1.00 1.00 1.00 0.96 0.42 0.47 0.44 0.51 0.51
   k=8  1.00 1.00 1.00 1.00 1.00 1.00 1.00 0.90 0.50 0.42 0.46 0.52
  k=12  1.00 0.99 1.00 1.00 1.00 1.00 1.00 0.98 0.85 0.49 0.35 0.39
  k=16  1.00 0.99 1.00 1.00 1.00 1.00 1.00 0.98 0.91 0.60 0.43 0.36
```

Read it diagonally. With K loops the model solves chains of K + 1 hops, and no longer: **one extra hop per extra
loop**. That holds past the training range: at K = 8 it solves 8 hops (0.90) and at K = 16, 9 hops (0.91), lengths it
never saw. It stops gaining around 9-10 hops, so looping is not unlimited.

The numbers at 11-12 hops that look like 0.5 are the twin-chain floor of stage 07: a model that knows "it's the end of
one of the two long chains" but can't tell which gets half right.

Seeds 1 and 2 give the same picture, shifted: seed 1 solves 3 hops at K = 1 and 9 hops at K = 16 (0.97 at 9, 0.79 at
10); seed 2 matches seed 0 (0.93 at 9 hops, K = 16).

### Tests

The tests train for 1,500 steps (about two minutes) and check that 1-hop chains are solved at K = 4, that 2-hop
chains are well above chance, that the checkpoint and the JSON grid were written, and that the curriculum starts at 2
hops.

### Check yourself

1. Why does the K = 1 row fall to about 0.25 at 4 hops rather than to 0.10 (guessing a digit)?
2. If you trained with K always 4, what would you expect the K = 1 and K = 8 rows to look like? (Try it:
   `--k-min 4 --k-max 4`.)
