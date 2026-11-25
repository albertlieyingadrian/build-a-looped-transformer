"""The looped GPT (stages 06-09). The whole change from model.GPT is in `forward`:

    embed ──► prelude (P blocks, once) ──► e ──────────────────────────────┐  re-added before every loop
                                           │                               │  ("input injection")
                                           ▼                               ▼
                                     s0 = 0 ──► core (C shared blocks) ──► s1 ──► core (same weights) ──► s2 ... sk
                                                                                                          │
                                                               coda (D blocks, once) ◄────────────────────┘
                                                                        │
                                                                   LayerNorm ──► head

Same parameters for any k. More loops = more compute per token, not more weights.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from app.model import Block, Config


class Message(nn.Module):
    """Inkling-2-Mini-style hand-off between passes (extension stage), much simplified. Every pass starts again from
    the prelude's output e; what the previous pass computed arrives as a "message", and a gate decides, number by
    number, how much of e to forget and how much of the message to take. In the real model each looped layer reads its
    own message with a learned query over the previous pass's outputs; here one gate sits at the core's entrance."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.ln = nn.LayerNorm(cfg.n_embd)
        self.msg = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.gate = nn.Linear(2 * cfg.n_embd, cfg.n_embd)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, -2.0)  # start mostly closed, so pass 2 begins close to pass 1

    def forward(self, e, prev):
        m = self.msg(self.ln(prev))
        g = torch.sigmoid(self.gate(torch.cat([e, m], -1)))  # 0 = keep e, 1 = take the message
        return (1 - g) * e + g * m


class LoopedGPT(nn.Module):
    """inject='add':     s_i = core(s_{i-1} + e)        Dory's re-injection; the simplest form of Huginn's
    inject='none':    s_i = core(s_{i-1})            the plain loop of Universal Transformers
    inject='message': s_i = core(gate(e, s_{i-1}))   Inkling-style: restart from e, plus a gated message"""

    def __init__(self, embed: nn.Module, n_out: int, cfg: Config, prelude=1, core=1, coda=1, inject="add"):
        super().__init__()
        self.cfg, self.embed, self.inject = cfg, embed, inject
        self.prelude = nn.ModuleList(Block(cfg) for _ in range(prelude))
        self.core = nn.ModuleList(Block(cfg) for _ in range(core))
        self.coda = nn.ModuleList(Block(cfg) for _ in range(coda))
        self.message = Message(cfg) if inject == "message" else None
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, n_out)
        self.sizes = (prelude, core, coda)

    def unembed(self, x):
        return self.head(self.ln_f(x))

    def loop_once(self, s, e):
        """One pass through the shared core: the only new idea in this file."""
        if self.inject == "add":
            x = s + e
        elif self.inject == "message":
            x = self.message(e, s)
        else:
            x = s
        for blk in self.core:
            x = blk(x)
        return x

    def readout(self, s):
        """The coda, then the final LayerNorm and head."""
        for blk in self.coda:
            s = blk(s)
        return self.unembed(s)

    def run_prelude(self, idx):
        x = self.embed(idx)
        for blk in self.prelude:
            x = blk(x)
        return x

    def forward(self, idx, targets=None, k=4, return_states=False):
        e = self.run_prelude(idx)
        # The state starts empty for 'add' (z = f(z + x) with z0 = 0, as in the looped in-context-learning code), so
        # loop 1 sees exactly e; the other two forms start from e itself.
        s = torch.zeros_like(e) if self.inject == "add" else e
        states = [e]  # states[i] = the stream after loop i (states[0] = after the prelude)
        for _ in range(k):
            s = self.loop_once(s, e)
            states.append(s)
        logits = self.readout(s)
        loss = None if targets is None else F.cross_entropy(logits.flatten(0, 1), targets.flatten())
        return (logits, loss, states) if return_states else (logits, loss)

    def depth(self, k=4):
        p, c, d = self.sizes
        return p + c * k + d
