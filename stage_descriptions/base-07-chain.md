In this stage, you'll generate the puzzle the rest of the course runs on: follow a chain of assignments to a digit.

### Why a new task?

Shakespeare doesn't tell you whether depth helps, because a loss of 1.71 against 1.74 is hard to read. We need a task
where the amount of thinking needed is a dial. Chain following is one:

```
statements (shuffled):  k=d   q=k   d=7   x=4   m=x   ...      question: "which digit does q hold?"
the question's chain:   q=k ──► k=d ──► d=7                      answer 7, 3 hops
```

Each extra link (a **hop**) is one more lookup the model has to chain. A model that does one lookup per layer needs
more layers for longer chains; a looped model can loop more.

### The format

- 48 variable names, 10 digits, **24 statements per puzzle**, shuffled, plus the question.
- Each statement `a=b` is **one position**, embedded as `lhs_vector[a] + rhs_vector[b]`. The question is one more
  position, `q=?`, with a special `?` id. No position vectors (the order is random) and **no causal mask** (every
  statement may read every other). The answer is read at the last position, and the head has 10 outputs.
- The question's chain has `h` hops. **A twin chain** of the same length ends in a different digit, and the rest are
  distractor chains of 1-4 statements. Every puzzle has 24 statements, so longer chains mean fewer distractors, never
  a longer input.

### Why the twin chain?

An earlier version of this task (the steerlab looped-transformer note) let models cheat on long chains: when the
question's chain is longer than every distractor, "pick the end of the longest chain" answers without following
anything. With a twin chain of the same length, that shortcut is right only half the time.

```bash
$ ./your_program.sh chain --hops 3 --n 3
w=C t=5 g=E I=5 q=J K=S s=c l=b k=w E=7 i=2 b=t u=l J=T C=3 c=7 v=I e=s T=3 B=8 S=3 d=K x=B R=v | e? -> 7 (3 hops)
...
statements per puzzle: 24
guess a digit: 0.10;  guess one chain end at random: 0.11;  pick the longest chain: 0.50 (the twin chain is as long)
```

Check the first puzzle by hand: `e=s`, `s=c`, `c=7`. Three hops, answer 7.

The chance levels matter later: at 12 hops the question's chain and its twin fill all 24 statements, so a model that
only knows "it's one of the two chain ends" gets 50%. Read every accuracy against these baselines.

### Tests

The tests run `chain` at 1, 3 and 6 hops, follow every printed puzzle themselves, and check the answer, the hop count,
the 24 statements and that a second chain of the same length exists.

### Check yourself

1. Why does the puzzle have no position embeddings, when the GPT of stage 05 needed them?
2. How many hops can a twin-chain puzzle with 24 statements have at most?
