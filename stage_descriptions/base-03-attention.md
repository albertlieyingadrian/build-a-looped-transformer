In this stage, you'll implement causal self-attention, the only place where positions talk to each other.

### The idea: a room full of index cards

Every position writes three cards from its vector: a **query** ("what am I looking for?"), a **key** ("what do I
contain?") and a **value** ("what I'll hand over if you pick me"). Each query is compared with every key (a dot
product); a softmax turns the scores into weights that sum to 1; the output is the weighted mix of the values.

```
scores = q @ kᵀ / sqrt(head_size)          (T × T): how well position i's question matches position j's label
scores[i, j] = -inf for j > i              causal mask: nobody reads the future
weights = softmax(scores, along j)         each row sums to 1
out = weights @ v                          (T × head_size)
```

**Multi-head** attention runs several of these side by side on slices of the vector (4 heads of 16 numbers for
`n_embd = 64`), then a linear layer mixes the heads back together.

### Why the mask?

A language model is trained to predict the next character. If position 3 could read position 4, it would copy the
answer. The mask sets those scores to minus infinity before the softmax, so their weight is exactly zero.

(Later, in stage 07, the chain puzzle turns the mask off: there every statement may read every other.)

### Worked example

With `T = 5` the weights form a lower-triangular matrix. Row 0 can only read itself, so it is `1.00 0 0 0 0`. Row 1
splits its attention between positions 0 and 1, and so on.

```bash
$ ./your_program.sh attention --T 5
attention weights (row = reading position, column = position read):
  1.00 0.00 0.00 0.00 0.00
  0.46 0.54 0.00 0.00 0.00
  0.39 0.32 0.29 0.00 0.00
  0.26 0.30 0.21 0.23 0.00
  0.20 0.16 0.19 0.33 0.12
rows sum to 1: True
weight on future positions: 0.0
change at earlier positions after editing the last one: 0.0
```

The last line is the real test of the mask: change the input at the last position by a lot, and the outputs at all
earlier positions must not move at all.

### Tests

The tests run `attention --T 6`, check every row sums to 1 and has zeros to the right of the diagonal, and check the
last two lines print `0.0`.

### Check yourself

1. Why divide by `sqrt(head_size)`? (Hint: what happens to a softmax when its inputs are all very large?)
2. Attention has no idea of order by itself. Where does order come from in this model?
