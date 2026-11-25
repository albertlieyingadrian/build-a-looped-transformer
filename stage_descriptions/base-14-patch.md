In this stage, you'll edit a run in the middle of its loops and see whether the edit sticks.

### Activation patching

Make two runs that differ in one thing. **Clean**: the real puzzle. **Corrupted**: the same 24 statements, but the
question asks about the twin chain's first name, so the right answer becomes the twin's digit. Run the corrupted
puzzle, and right after loop i copy in the clean run's state. If the corrupted run now gives the clean answer, the
state after loop i carried what decides the answer.

A **control**: instead of the clean state, add a random vector of the same length as the clean-minus-corrupted
difference. If that also flips answers, the patch is just breaking the model, not transplanting information.

### The `patch` command

4-hop chains, K = 8, 500 puzzle pairs. Patch at the question position only, and at every position. Since the
statements are identical in both runs, the two patches differ only at the question.

### What we found (share of corrupted runs that give the clean answer)

| patched after loop | 0 (the prelude's output) | 1 | 2 | 3 | 4-8 |
|---|---|---|---|---|---|
| seed 0 | 1.00 | **0.02** | 0.52 | 0.96 | 1.00 |
| seed 1 | 1.00 | **0.03** | 0.68 | 0.98 | 1.00 |
| seed 2 | 1.00 | **0.05** | 0.78 | 0.99 | 1.00 |
| random-vector control | 0.05-0.08 | 0.00 | 0.01 | 0.01 | 0.00-0.02 |

The odd one out is loop 1. Patching the prelude's output works because that output `e` is re-added on every loop: you
changed the input itself. Patching after loop 1 does almost nothing: on loop 2 the corrupted `e` is added back and
wins. Patching after loop 3, when the answer has arrived at the question (stage 13), sticks completely.

Why do late edits stick and early ones wash out? The state grows by about the same amount every loop (stage 15: its
length goes 22, 42, 62, 85 ... 196), while the re-added input stays the same size, so the input's share of what the
core reads shrinks roughly as 1/i. An edit made early is outvoted by the input on the next loop; an edit made late is not. This is
the toy version of the preprint's finding that edits to Huginn act only within a window of recurrences.

### Check yourself

1. If you patched after loop 1 and ran with K = 1, what would you expect? Why is that not a fair test of loop 1?
2. Design a patch that should work after loop 1. (Hint: what else would you have to change on loop 2?)
