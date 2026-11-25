In this stage, you'll implement a `tokenize` command that turns text into numbers and back.

### Why numbers?

A model only does arithmetic. Before it can read "Speak, speak." it needs each piece of text as a number, a
**token id**. Think of a cloakroom: every character you hand over gets a ticket number, and the same number always
gets the same character back.

The simplest tokenizer, and the one nanoGPT starts with, uses one token per character. The **vocabulary** is the
sorted list of every distinct character in the training text (the **corpus**); a character's id is its position in
that list.

### Worked example

Corpus: `hello world, hello loops.\n`. Its distinct characters, sorted, are

```
'\n'  ' '  ','  '.'  'd'  'e'  'h'  'l'  'o'  'p'  'r'  's'  'w'
  0    1    2    3    4    5    6    7    8    9   10   11   12
```

so the vocabulary size is 13 and `hello` becomes `[6, 5, 7, 7, 8]`. Decoding looks each id up again: `6 → h`,
`5 → e`, and so on, back to `hello`.

### The `tokenize` command

```bash
$ ./your_program.sh tokenize corpus.txt "hello"
corpus characters: 26
vocab size: 13
ids: [6, 5, 7, 7, 8]
decoded: hello
round trip ok: True
```

### Tests

`make test STAGE=01 DIR=<your code>` writes a small corpus to a file, runs `tokenize` on it, and checks the vocab size,
the ids (as a Python list), the decoded text and the round-trip line.

### Notes

- Sort the characters, so the ids do not depend on the order they appear in.
- Real models use sub-word tokenizers (byte-pair encoding) with tens of thousands of tokens. Characters keep the
  vocabulary tiny (65 for Shakespeare), which is all a laptop-sized model needs.

### Check yourself

1. What happens if the text you encode contains a character that never appears in the corpus? What would you do
   about it in a real system?
2. Why does a bigger vocabulary make the model's last layer bigger?
