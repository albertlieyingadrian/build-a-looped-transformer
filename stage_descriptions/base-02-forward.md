In this stage, you'll give every token a vector and turn vectors back into next-token scores.

### From ids to vectors

An id like `6` carries no meaning by itself; 6 is not "more" than 5. So the model looks each id up in a table of
learned vectors, the **token embedding**: row 6 of a `vocab_size × n_embd` matrix. With `n_embd = 64`, every character
becomes 64 numbers that training will shape so similar characters end up close together.

A transformer also needs to know *where* each token sits, so it adds a second learned vector per position (the
**position embedding**). The sum, one vector per position, is the **residual stream**: think of it as a whiteboard per
position that every later layer reads and writes on.

```
ids (1, T) ──► token table ──┐
                             (+) ──► x (1, T, 64) ──► LayerNorm ──► head (64 → V) ──► logits (1, T, V)
positions 0..T-1 ──► table ──┘
```

The **head** is a linear layer that turns each 64-number vector into one score per vocabulary entry: the **logits**.
A softmax turns them into probabilities for "which character comes next".

### The loss of a model that knows nothing

The **loss** is cross-entropy: minus the log of the probability the model gave to the true next character. A model
that guesses uniformly gives every character `1/V`, so its loss is `ln V`. A freshly initialised model should be
close to that; if it is far above, something is wrong (often the initial weights are too large).

### The `forward` command

Build a model with no transformer blocks yet (embeddings, a final LayerNorm and the head), run the text through it,
and print the shapes and the loss of predicting each next character:

```bash
$ ./your_program.sh forward corpus.txt "hello loops"
ids shape: (1, 11)
embeddings shape: (1, 11, 64)
logits shape: (1, 11, 13)
parameters: 5901
loss at initialisation: 2.723
ln(vocab size): 2.565
```

(Your parameter count and loss depend on your layout and seed; the tests check the shapes and that the loss is within
1.0 of `ln V`.)

### Check yourself

1. The logits have shape `(1, T, V)`. Which of the `T` rows is the prediction for the character after the last one?
2. Why is the loss computed on `ids[:, :-1]` predicting `ids[:, 1:]`?
