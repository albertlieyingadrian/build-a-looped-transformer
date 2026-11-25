"""Training a tiny GPT on text (stage 05)."""

from __future__ import annotations

import math

import torch


def get_batch(data: torch.Tensor, block_size: int, batch_size: int, gen: torch.Generator):
    """batch_size random windows; the target is the same window shifted one character to the right."""
    ix = torch.randint(len(data) - block_size - 1, (batch_size,), generator=gen)
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return x, y


@torch.no_grad()
def estimate_loss(model, data, block_size, gen, batches=20, batch_size=32, k=None) -> float:
    model.eval()
    losses = [model(*get_batch(data, block_size, batch_size, gen), k=k)[1].item() for _ in range(batches)]
    model.train()
    return sum(losses) / len(losses)


def train_lm(model, train_data, val_data, steps=1000, batch_size=32, lr=3e-3, seed=0, log=print, k_sampler=None):
    """The standard loop: sample a batch, compute the loss, backpropagate, step. A cosine learning-rate decay.
    k_sampler (looped models only) picks the loop count for each batch."""
    gen = torch.Generator().manual_seed(seed)
    block = model.cfg.block_size
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * s / steps)))
    model.train()
    for step in range(steps + 1):
        if step % max(1, steps // 5) == 0 and log:
            log(
                f"step {step:5d}  train {estimate_loss(model, train_data, block, gen):.3f}  "
                f"val {estimate_loss(model, val_data, block, gen):.3f}"
            )
        if step == steps:
            break
        x, y = get_batch(train_data, block, batch_size, gen)
        k = k_sampler(gen) if k_sampler else None
        _, loss = model(x, y, k=k)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
    model.eval()
    return estimate_loss(model, val_data, block, gen, batches=50)
