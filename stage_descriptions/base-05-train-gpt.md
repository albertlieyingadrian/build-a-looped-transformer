In this stage, you'll write the training loop and train a tiny GPT on Shakespeare.

### The loop

```
for step in range(steps):
    x, y = random windows of the text, and the same windows shifted by one character
    logits, loss = model(x, y)
    loss.backward()             # how should each weight move to lower the loss?
    optimizer.step()            # move them a little (AdamW)
```

That's all of training. The model is the stage 02 embeddings, a stack of stage 04 blocks, a final LayerNorm and the
head. Generation is the same model run in a loop: feed the text so far, take the most likely next character
(greedy), append it, repeat.

### Worked example (tinyshakespeare, a laptop CPU)

`make data` downloads the 1.1 MB tinyshakespeare file. Four blocks, `n_embd = 64`, 64 characters of context, batches
of 32, 2,000 steps, learning rate 3e-3 with a cosine decay:

```bash
$ ./your_program.sh train-gpt data/tinyshakespeare.txt --steps 2000 --layers 4
parameters: 212545
step     0  train 4.311  val 4.308
step   400  train 2.014  val 2.066
step   800  train 1.743  val 1.887
step  1200  train 1.623  val 1.764
step  1600  train 1.539  val 1.739
step  2000  train 1.511  val 1.712
final val loss: 1.717  (29s)
sample (greedy):

The sent the stand the stand of the stands of the stand
To the stands and the common the stands and the common
```

The loss starts at `ln 65 = 4.17` (guessing) and falls to 1.71. Greedy sampling repeats itself; that is what greedy
decoding does to a small model, not a bug. The validation loss is measured on the last 10% of the text, which the
model never trains on.

### Tests

The tests train for 300 steps on a short, repetitive corpus and check that the final validation loss is below half
of `ln V`, that `gpt.pt` was saved, and that a sample was printed.

### Check yourself

1. Why does the training loss keep falling a little faster than the validation loss?
2. Train with `--layers 1` and then `--layers 4`. Which one wins, and is the difference worth four times the blocks?
   Keep this question in mind for stage 06.
