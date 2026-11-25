In this stage, you'll give the looped model a fair fight against two standard transformers.

### Two fair baselines

A looped model can be compared with a standard one in two ways, and both matter:

- **Same parameters** (`standard3`): 3 distinct blocks, exactly the weights of `looped` (157,578). Does looping beat
  simply storing the same weights as a stack?
- **Same compute** (`standard10`): 10 distinct blocks, the compute of `looped` at K = 8 (1 + 8 + 1 blocks) with 3.2
  times the weights (507,466). Does a looped block do as much as distinct blocks given the same work per token?

All three train on the same puzzles (1-6 hops, curriculum), the same number of steps and the same optimiser, with the
same seed. Then we score them on 1-12 hops: chains of 7-12 links were never seen in training (**length
extrapolation**).

### The `compare` command

Train (or load) the three models, print their grids, save `compare_s<seed>.json` and a plot `compare_s<seed>.png`, and
print the **longest chain solved**: the longest chain length with at least 90% right, and every shorter length too.

### What we found (3 seeds, 500 puzzles per hop count)

| model | parameters | blocks run | longest chain solved, seeds 0 / 1 / 2 |
|---|---|---|---|
| standard, 3 blocks (same weights) | 157,578 | 3 | 4 / 4 / 3 |
| standard, 10 blocks (same compute as K = 8) | 507,466 | 10 | 4 / 6 / 5 |
| looped, K = 1 | 157,578 | 3 | 2 / 3 / 3 |
| looped, K = 4 | 157,578 | 6 | 5 / 6 / 6 |
| looped, K = 8 | 157,578 | 10 | **7 / 8 / 8** |
| looped, K = 16 | 157,578 | 18 | **9 / 9 / 9** |

- **Same weights:** the 3-block model stops at 3-4 hops, and its curriculum never got past 5 hops in two seeds. At
  K = 1 the looped model is a 3-block model too, and does no better (2-3 hops). Its advantage is entirely in the loops.
- **Same compute:** at K = 8 the looped model runs as many blocks as `standard10` and solves 2-3 more hops, with under
  a third of the weights. `standard10` learned the training range (6 hops in seed 1) but never went past it: on 7 hops
  it scores 0.25-0.58.
- **Extrapolation:** only the looped model solves chains longer than any it trained on: 7-8 hops at K = 8 and 9 at
  K = 16 in every seed, against a training maximum of 6.

Seed 0's 10-block model is the weakest (0.73 on 5 hops): deep standard models are slower to train at this learning
rate, and all models got the same 4,000 steps. A longer schedule might help it inside the training range; nothing in
its grids suggests it would reach past it.

### Check yourself

1. Why compare at the same compute *and* at the same parameters? Which comparison would a company serving the model
   care about, and which would someone fitting it on a phone care about?
2. The looped model solves K + 1 hops in seed 0. Write down a mechanism that would give exactly "one more hop per
   loop". (Stage 13 tests one.)
