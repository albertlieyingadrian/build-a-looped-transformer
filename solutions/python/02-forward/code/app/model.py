"""A tiny GPT (stages 02-05).

The structure follows Karpathy's nanoGPT (https://github.com/karpathy/nanoGPT), written fresh for this course with
more comments. One change: the input layer (`embed`) is a separate, swappable piece, so the same blocks can read
characters (stages 01-05) or chain puzzles (stages 07 onward).

    token ids ──► embed ──► x  (the residual stream: one vector of n_embd numbers per position)
                             │
                             ├──► Block 1:  x = x + Attention(LayerNorm(x))     "look at other positions"
                             │              x = x + MLP(LayerNorm(x))           "think about what you saw"
                             ├──► Block 2:  (same shape, its own weights)
                             ▼
                          LayerNorm ──► head (one linear layer) ──► logits: a score for every possible next token

The residual stream is a shared whiteboard. Every block reads the whole board and adds a note to it; nobody erases.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class Config:
    n_embd: int = 64  # width of the residual stream (numbers per position)
    n_head: int = 4  # attention heads; each works on n_embd / n_head numbers
    block_size: int = 64  # longest sequence the position table covers
    causal: bool = True  # True: a position may only look left (language modelling). False: look everywhere.


class SelfAttention(nn.Module):
    """Every position asks a question (query) and offers a label (key) and a payload (value). A position's output is
    the payload of the positions whose labels best match its question, mixed by a softmax."""

    def __init__(self, cfg: Config):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head, self.causal = cfg.n_head, cfg.causal
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd)  # one matrix makes q, k and v at once
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd)  # mixes the heads' outputs back into the stream
        self.last_attn = None  # (B, heads, T, T) from the latest call, for inspection

    def forward(self, x):
        B, T, C = x.shape  # batch, time (positions), channels (n_embd)
        q, k, v = self.qkv(x).split(C, dim=2)
        # (B, T, C) -> (B, heads, T, C / heads): each head works on its own slice
        q, k, v = (t.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) for t in (q, k, v))
        att = (q @ k.transpose(-2, -1)) / math.sqrt(k.size(-1))  # (B, heads, T, T): how well each pair matches
        if self.causal:  # hide the future: position t may not read positions after t
            future = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1)
            att = att.masked_fill(future, float("-inf"))
        att = F.softmax(att, dim=-1)  # every row now sums to 1: "how much to read from each position"
        self.last_attn = att.detach()
        y = att @ v  # (B, heads, T, C / heads): the weighted mix of payloads
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # glue the heads back together
        return self.proj(y)


class MLP(nn.Module):
    """A two-layer network applied to each position on its own: widen 4x, bend (GELU), narrow back."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.up = nn.Linear(cfg.n_embd, 4 * cfg.n_embd)
        self.down = nn.Linear(4 * cfg.n_embd, cfg.n_embd)

    def forward(self, x):
        return self.down(F.gelu(self.up(x)))


class Block(nn.Module):
    """One transformer layer, pre-norm: normalise a copy, compute, add the result back to the stream."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.ln1, self.attn = nn.LayerNorm(cfg.n_embd), SelfAttention(cfg)
        self.ln2, self.mlp = nn.LayerNorm(cfg.n_embd), MLP(cfg)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))  # read other positions, add a note
        x = x + self.mlp(self.ln2(x))  # think per position, add a note
        return x


class TextEmbed(nn.Module):
    """Characters in: a learned vector per character plus a learned vector per position."""

    def __init__(self, vocab_size: int, cfg: Config):
        super().__init__()
        self.tok = nn.Embedding(vocab_size, cfg.n_embd)
        self.pos = nn.Embedding(cfg.block_size, cfg.n_embd)

    def forward(self, idx):  # idx: (B, T) integer token ids
        return self.tok(idx) + self.pos(torch.arange(idx.shape[1], device=idx.device))


class GPT(nn.Module):
    """A standard transformer: n_layer distinct blocks, each run once."""

    def __init__(self, embed: nn.Module, n_out: int, n_layer: int, cfg: Config):
        super().__init__()
        self.cfg, self.embed = cfg, embed
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(n_layer))
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, n_out)

    def unembed(self, x):
        """The model's last two steps: final LayerNorm, then the head. The logit lens (stage 11) reuses them."""
        return self.head(self.ln_f(x))

    def forward(self, idx, targets=None, k=None, return_states=False):
        """k is ignored (a standard model has no loops); it is accepted so every model has the same call signature.
        Returns (logits, loss) or (logits, loss, states); states[i] is the stream after block i (states[0]: input)."""
        x = self.embed(idx)
        states = [x]
        for blk in self.blocks:
            x = blk(x)
            states.append(x)
        logits = self.unembed(x)
        loss = None if targets is None else F.cross_entropy(logits.flatten(0, 1), targets.flatten())
        return (logits, loss, states) if return_states else (logits, loss)

    def depth(self, k=None):
        return len(self.blocks)

    @torch.no_grad()
    def generate(self, idx, n_new: int):
        """Greedy decoding: append the single most likely next token, n_new times."""
        for _ in range(n_new):
            logits, _ = self(idx[:, -self.cfg.block_size :])
            idx = torch.cat([idx, logits[:, -1].argmax(-1, keepdim=True)], dim=1)
        return idx


def text_model(vocab_size: int, n_layer: int, cfg: Config | None = None) -> GPT:
    """The character-level GPT of stages 02-05."""
    cfg = cfg or Config()
    return GPT(TextEmbed(vocab_size, cfg), vocab_size, n_layer, cfg)


def n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
