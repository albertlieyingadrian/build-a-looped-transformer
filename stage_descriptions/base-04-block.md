In this stage, you'll wrap attention and a small MLP into a transformer block.

### The block

```
x = x + Attention(LayerNorm(x))     read other positions, add a note to the whiteboard
x = x + MLP(LayerNorm(x))           think about each position on its own, add a note
```

- **Residual connections** (`x = x + ...`): a block never overwrites the stream, it adds to it. A block that
  writes zeros is the identity, so stacking blocks can't make things worse at the start of training, and gradients
  have a straight road back to the start. This is what will make looping (running the same block many times) safe.
- **LayerNorm** rescales a copy of each position's vector to mean 0 and standard deviation 1 before a sub-layer reads
  it, so the sub-layer sees numbers of a steady size however large the stream grows.
- **MLP**: widen 64 → 256, a GELU bend, narrow 256 → 64. Most of a transformer's weights live here.

### Counting parameters

For `n_embd = 64`: two LayerNorms (2 × 128), attention's `qkv` (64 × 192 + 192) and `proj` (64 × 64 + 64), the MLP's
`up` (64 × 256 + 256) and `down` (256 × 64 + 64). Total **49,984**. Remember this number: in stage 06 a looped model
will use one block like this for every loop.

### The `block` command

```bash
$ ./your_program.sh block
block parameters: 49984
in shape: (2, 8, 64)  out shape: (2, 8, 64)
with the write-back layers at zero the block is the identity: True
LayerNorm output mean 0.00, std 1.00
```

The third line zeroes the two layers that write into the stream (attention's output projection and the MLP's last
layer) and checks the block then returns its input unchanged.

### Check yourself

1. Why does the block's output have exactly the same shape as its input? What would break if it didn't?
2. "Pre-norm" puts LayerNorm before each sub-layer; the original 2017 transformer put it after. Which one keeps the
   residual stream untouched?
