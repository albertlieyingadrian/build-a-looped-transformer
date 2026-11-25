In this stage, you'll read the model's state after every loop, as if it had to answer right then.

### Two ways to read a half-finished state

- **Logit lens**: skip the coda and apply the final LayerNorm and head straight to `states[i]`, the state after loop
  i. It asks: is the answer already written in the stream, in the form the head reads?
- **Coda lens**: run the coda on `states[i]`, exactly what the model would answer if you set K = i. (Huginn analyses
  call this the coda lens; it's the natural "early exit" reading of a looped model.)

An analogy: the logit lens reads a student's scratch paper with the teacher's answer key; the coda lens asks the
student to stop and write their final answer now.

### The `logit-lens` command

For chains of 2, 4 and 6 hops (500 puzzles each), print both lenses' accuracy after every loop (0 = after the
prelude) and how often the coda lens agrees with the model's real answer at K.

### What we found (seed 0, K = 8)

| hops | after loop 0 | 1 | 2 | 3 | 4 | 5 | 6-8 |
|---|---|---|---|---|---|---|---|
| 2: logit / coda lens | 0.14 / 0.25 | **1.00 / 1.00** | 1.00 / 1.00 | ... | | | 1.00 |
| 4: logit / coda lens | 0.16 / 0.19 | 0.17 / 0.22 | 0.15 / 0.47 | **1.00 / 1.00** | 1.00 | | 1.00 |
| 6: logit / coda lens | 0.13 / 0.21 | 0.19 / 0.26 | 0.19 / 0.30 | 0.19 / 0.37 | 0.19 / 0.63 | **1.00 / 1.00** | 1.00 |

The answer does not fade in; it **arrives**, all at once, after loop `hops - 1`, and from then on even the plain
logit lens reads it perfectly. Before that loop the logit lens is near the 0.10-0.20 floor. The coda lens rises a loop
early (0.47, 0.63): the coda can do one more hop on its own.

So for this model the logit lens answers "has the question got its answer yet?" exactly, but it says nothing about
the work in between. Where is that work? Stage 13 finds it.

### Check yourself

1. Why should the coda lens after loop K agree with the model's real answer 100% of the time? (The tests check it.)
2. On Huginn, the published analyses found the logit lens readable at some positions inside the core and not at
   others. Our core has one block. What would you expect with a 2-block core?
