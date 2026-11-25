"""The chain-following task (stage 07) and how models are trained and scored on it (stages 08-10).

    statements (shuffled):  k=d   q=k   d=7   x=4   m=x   ...      question: "which digit does q hold?"
    the question's chain:   q=k ──► k=d ──► d=7                      answer 7, 3 hops

Each statement is one input position, embedded as lhs_vector[name] + rhs_vector[name or digit]; the question is one
more position at the end, whose rhs slot is a special "?" id. No position vectors (the order is random and carries
nothing) and no causal mask (every statement may look at every other). The answer is read at the last position.

Every puzzle has the same number of statements, so a longer chain means fewer distractors, never a longer input. One
distractor chain (the "twin") is always exactly as long as the question's and ends in a different digit, so the model
cannot find the answer by picking the longest chain: it has to follow the links.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, fields

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from app.looped import LoopedGPT
from app.model import GPT, Config

N_VARS = 48  # variable names
N_DIGITS = 10
N_STATEMENTS = 24
QUERY_ID = N_VARS + N_DIGITS  # the rhs id of the question position
RHS_VOCAB = N_VARS + N_DIGITS + 1
MAX_HOPS = N_STATEMENTS // 2  # the twin chain needs 2 x hops statements
TRAIN_HOPS = 6  # the curriculum stops here; 7-12 hops are never seen in training


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class PairEmbed(nn.Module):
    """A statement 'a=b' becomes lhs[a] + rhs[b]. Input: (B, T, 2) integer pairs."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.lhs = nn.Embedding(N_VARS, cfg.n_embd)
        self.rhs = nn.Embedding(RHS_VOCAB, cfg.n_embd)

    def forward(self, pairs):
        return self.lhs(pairs[..., 0]) + self.rhs(pairs[..., 1])


@dataclass
class ChainBatch:
    pairs: torch.Tensor  # (B, N_STATEMENTS + 1, 2): [lhs, rhs] per position; the last position is the question
    answer: torch.Tensor  # (B,) digit 0-9
    hops: torch.Tensor  # (B,)
    chain: torch.Tensor  # (B, MAX_HOPS + 1): chain[:, j] = the name j hops along from the question (-1 past the end)
    twin_answer: torch.Tensor  # (B,) the digit at the end of the twin chain

    def to(self, device):
        return ChainBatch(*(getattr(self, f.name).to(device) for f in fields(self)))


def make_chain_batch(rng: np.random.Generator, size: int, hops_range=(1, TRAIN_HOPS), hops: int | None = None):
    """`size` random puzzles. The question's chain has `hops` links (or a random count in hops_range, inclusive). The
    twin chain has the same length and a different end digit; the other statements form distractor chains of random
    length 1 to 4. All names within one puzzle are distinct."""
    N = N_STATEMENTS
    pairs = np.zeros((size, N + 1, 2), np.int64)
    answer, hop_arr, twin = np.zeros(size, np.int64), np.zeros(size, np.int64), np.zeros(size, np.int64)
    chain = np.full((size, MAX_HOPS + 1), -1, np.int64)
    for b in range(size):
        h = hops if hops is not None else int(rng.integers(hops_range[0], hops_range[1] + 1))
        names = rng.permutation(N_VARS)
        lengths, left = [h, h], N - 2 * h  # the question's chain, then its twin, then distractors
        while left > 0:
            n = int(min(left, rng.integers(1, 5)))
            lengths.append(n)
            left -= n
        digits = rng.integers(0, N_DIGITS, len(lengths))
        if digits[1] == digits[0]:
            digits[1] = (digits[0] + 1 + rng.integers(0, N_DIGITS - 1)) % N_DIGITS
        stmts, cursor = [], 0
        for c, n in enumerate(lengths):
            link = names[cursor : cursor + n]  # the chain v0=v1, v1=v2, ..., v_{n-1}=digit
            for j in range(n):
                stmts.append((int(link[j]), int(link[j + 1]) if j + 1 < n else N_VARS + int(digits[c])))
            if c == 0:
                chain[b, :h] = link
            cursor += n
        pairs[b, :N] = np.array(stmts)[rng.permutation(N)]
        pairs[b, N] = (names[0], QUERY_ID)
        answer[b], hop_arr[b], twin[b] = digits[0], h, digits[1]
    return ChainBatch(*(torch.from_numpy(x) for x in (pairs, answer, hop_arr, chain, twin)))


def render(batch: ChainBatch, i: int = 0) -> str:
    """One puzzle as text, e.g. 'k=d q=k d=7 ... | q? -> 7 (3 hops)'."""
    letters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUV"

    def name(v):
        return str(v - N_VARS) if v >= N_VARS else letters[v]

    p = batch.pairs[i].tolist()
    body = " ".join(f"{name(a)}={name(b)}" for a, b in p[:-1])
    return f"{body} | {name(p[-1][0])}? -> {batch.answer[i].item()} ({batch.hops[i].item()} hops)"


def build(kind: str, cfg: Config | None = None, seed: int = 0):
    """'looped' (1 prelude + 1 shared core block x k + 1 coda: 3 blocks of weights), 'looped_none' (same, no
    re-injection), 'looped_message' (same, Inkling-style message), 'standard3' (3 distinct blocks: the same weights as
    'looped'), 'standard10' (10 distinct blocks: the compute of 'looped' at k = 8)."""
    cfg = cfg or Config(causal=False)
    seed_everything(seed)
    if kind.startswith("looped"):
        inject = {"looped": "add", "looped_none": "none", "looped_message": "message"}[kind]
        return LoopedGPT(PairEmbed(cfg), N_DIGITS, cfg, prelude=1, core=1, coda=1, inject=inject)
    return GPT(PairEmbed(cfg), N_DIGITS, n_layer=int(kind.removeprefix("standard")), cfg=cfg)


def answer_logits(model, batch: ChainBatch, k=None, return_states=False):
    """The answer is read at the last position (the question)."""
    out = model(batch.pairs, k=k, return_states=return_states)
    return (out[0][:, -1], out[2]) if return_states else out[0][:, -1]


@torch.no_grad()
def accuracy(model, batch: ChainBatch, k=None) -> float:
    model.eval()
    return (answer_logits(model, batch, k=k).argmax(-1) == batch.answer).float().mean().item()


def test_sets(seed: int = 1234, n: int = 500, max_hops: int = MAX_HOPS):
    """A fixed test set per hop count, from its own seed, never used for training."""
    rng = np.random.default_rng(seed)
    return {h: make_chain_batch(rng, n, hops=h) for h in range(1, max_hops + 1)}


def train(model, seed=0, steps=4000, batch_size=128, lr=2e-3, max_hops=TRAIN_HOPS, k_range=(1, 8), log=print):
    """AdamW with warm-up and cosine decay; a curriculum on the hop count (start at 1-2 hops, add one whenever the last
    50 batches were 90% right, up to max_hops); for looped models a random loop count per batch, drawn from k_range
    (inclusive). Returns the curriculum as [(step, max hops)]."""
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05, betas=(0.9, 0.98))
    warm = min(200, steps // 10)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / steps)))
    )
    looped = isinstance(model, LoopedGPT)
    cur, recent, curriculum = min(2, max_hops), [], [(0, min(2, max_hops))]
    for step in range(steps):
        batch = make_chain_batch(rng, batch_size, hops_range=(1, cur))
        k = int(rng.integers(k_range[0], k_range[1] + 1)) if looped else None
        logits = answer_logits(model, batch, k=k)
        loss = F.cross_entropy(logits, batch.answer)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        recent = (recent + [(logits.argmax(-1) == batch.answer).float().mean().item()])[-50:]
        if cur < max_hops and len(recent) == 50 and np.mean(recent) > 0.9:
            cur, recent = cur + 1, []
            curriculum.append((step + 1, cur))
        if log and (step % 500 == 0 or step == steps - 1):
            log(f"step {step:5d}  loss {loss.item():.3f}  curriculum up to {cur} hops")
    model.eval()
    return curriculum
