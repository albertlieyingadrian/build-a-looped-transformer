In this stage, you'll train linear probes after every loop to find where the chain-following happens.

### Probes

A **linear probe** is a logistic regression trained on the model's state to predict a label. If a straight line
separates the classes, the information is there in an easy-to-read form. Two controls keep it honest: the same probe
trained on **shuffled labels** (it should score chance), and on an **untrained model** of the same shape (it should
score near chance too, or the "information" is just the input).

### Where to look

Stage 11 showed the question's answer arriving all at once after loop `hops - 1`. Two hypotheses:

- **Forward walk**: the question position follows the chain, one link per loop (q, then k, then d...).
- **Backward flood**: every statement learns its chain's digit as soon as its right-hand side knows it. The statement
  `d=7` knows 7 from the start; after one loop `k=d` knows it; after two, `q=k`. The digit flows back along every chain
  at once, one link per loop, and the question reads it when it arrives.

So probe two places on 6-hop puzzles (3,000, 70/30 split): at the question, for the name j links along the chain; and
at each statement of the question's chain, for the chain's digit, by its distance d from the end.

### What we found (K = 8, seed 0)

Probe accuracy at the statement d links from the end of the chain, for the chain's digit (chance 0.10):

| after loop | d = 1 | 2 | 3 | 4 | 5 | 6 (the question's first statement) |
|---|---|---|---|---|---|---|
| 0 | **1.00** | 0.15 | 0.11 | 0.11 | 0.12 | 0.12 |
| 1 | 1.00 | **1.00** | 0.16 | 0.12 | 0.12 | 0.12 |
| 2 | 1.00 | 1.00 | **1.00** | 0.15 | 0.15 | 0.15 |
| 3 | 1.00 | 1.00 | 1.00 | **1.00** | 0.18 | 0.18 |
| 4 | 1.00 | 1.00 | 1.00 | 1.00 | **1.00** | 0.21 |
| 5 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | **0.98** |

A perfect staircase: **the digit moves back one statement per loop**, and the question's answer probe jumps from 0.23
to 1.00 at loop 5, the loop after the digit reaches the chain's first statement. Seed 2 is identical; seed 1 is one
step ahead (its prelude already does the first link, which is why it solves one more hop at every K).

At the question, by contrast, the name 2 links along is only partly readable (0.42 at loop 1, fading to 0.04) and
names 3-5 links along never are (≤ 0.05; chance 0.02). The question doesn't walk the chain. It waits.

Controls: shuffled labels 0.07-0.11; the untrained model 0.12-0.15.

This also explains stage 08's "one more hop per loop": each loop moves every digit one link, and the coda adds one
link of its own.

### Check yourself

1. Why does the statement at d = 1 know its digit even after the prelude, before any loop?
2. The backward flood works on every chain in the puzzle at once, the twin and the distractors too. Is that
   wasteful? What would a forward walk cost in comparison?
