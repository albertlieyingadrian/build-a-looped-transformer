"""Looking inside a looped model, one loop at a time (stages 11-16).

Every tool here reads or edits the residual stream at the question position (the last position) after each loop:

    states[0] = after the prelude,  states[i] = after loop i,  ...,  states[k] = what the coda reads

The question is the same for every tool: "what does the model already know after loop i?"
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from app.chain import N_DIGITS, N_VARS, ChainBatch, make_chain_batch

Q = -1  # the question position


@torch.no_grad()
def loop_states(model, batch: ChainBatch, k: int):
    """(logits at the question, list of k + 1 full states (B, T, d))."""
    logits, _, states = model(batch.pairs, k=k, return_states=True)
    return logits[:, Q], states


def start_state(model, e):
    return torch.zeros_like(e) if model.inject == "add" else e


# ------------------------------------------------------------------------------------------------ stage 11: logit lens
@torch.no_grad()
def logit_lens(model, batch: ChainBatch, k: int):
    """For each loop i, decode states[i] two ways and score against the true answer.
    logit lens: skip the coda, apply the final LayerNorm + head directly: unembed(states[i]).
    coda lens:  run the coda on states[i], as if the model had stopped looping there (the same as running k = i)."""
    final, states = loop_states(model, batch, k)
    rows = []
    for i, s in enumerate(states):
        logit = model.unembed(s[:, Q]).argmax(-1)
        coda = model.readout(s)[:, Q].argmax(-1)
        rows.append(
            {
                "loop": i,
                "logit_lens": (logit == batch.answer).float().mean().item(),
                "coda_lens": (coda == batch.answer).float().mean().item(),
                "coda_agrees_with_final": (coda == final.argmax(-1)).float().mean().item(),
            }
        )
    return rows


# ------------------------------------------------------------------------------------------------ stage 12: tuned lens
def fit_affine(xs: list[torch.Tensor], target_logits: torch.Tensor, unembed, steps=300, lr=1e-2, shared=False):
    """A tuned lens: one small learned map per loop (or one for all loops when shared=True), trained so that
    unembed(map(h_i)) matches the model's own final answer distribution (KL divergence). Starts at the identity.
    xs[i]: (N, d) question states after loop i. Returns one map (nn.Linear) per loop."""
    d = xs[0].shape[1]
    maps = [torch.nn.Linear(d, d) for _ in range(1 if shared else len(xs))]
    for m in maps:
        torch.nn.init.eye_(m.weight)
        torch.nn.init.zeros_(m.bias)
    opt = torch.optim.Adam([p for m in maps for p in m.parameters()], lr=lr)
    target = F.log_softmax(target_logits, -1)
    for _ in range(steps):
        loss = 0.0
        for i, x in enumerate(xs):
            m = maps[0 if shared else i]
            loss = loss + F.kl_div(F.log_softmax(unembed(m(x)), -1), target, log_target=True, reduction="batchmean")
        opt.zero_grad()
        loss.backward()
        opt.step()
    return [maps[0]] * len(xs) if shared else maps


def jacobian_maps(model, batch: ChainBatch, k: int):
    """A Jacobian lens: for each loop i, J_i = the average over puzzles of d(coda output at the question) /
    d(states[i] at the question): "if I nudge the state after loop i, how does the final state move?". One backward
    pass per output number. Returns one (d, d) matrix per state (index 0 = after the prelude)."""
    e = model.run_prelude(batch.pairs)
    s = start_state(model, e)
    states = [e]
    for _ in range(k):
        s = model.loop_once(s, e)
        states.append(s)
    out = s
    for blk in model.coda:
        out = blk(out)
    out = out[:, Q]  # (B, d): the stream the final LayerNorm + head read
    d = out.shape[1]
    J = [torch.zeros(d, d) for _ in states]
    for j in range(d):
        grads = torch.autograd.grad(out[:, j].sum(), states, retain_graph=True)
        for i, g in enumerate(grads):
            J[i][j] = g[:, Q].mean(0)
    return [m.detach() for m in J]


def tuned_lens_report(model, k: int, seed=7, n_fit=2000, n_test=1000, hops_range=(1, 6)):
    """Per-loop tuned lens vs one shared tuned lens vs the per-loop Jacobian lens vs one Jacobian lens averaged over
    the loops, all scored on fresh puzzles: does the decoded answer match the true answer?"""
    rng = np.random.default_rng(seed)
    fit, test = make_chain_batch(rng, n_fit, hops_range), make_chain_batch(rng, n_test, hops_range)
    f_logits, f_states = loop_states(model, fit, k)
    _, t_states = loop_states(model, test, k)
    xs = [s[:, Q] for s in f_states]
    per_loop = fit_affine(xs, f_logits, model.unembed)
    shared = fit_affine(xs, f_logits, model.unembed, shared=True)
    J = jacobian_maps(model, make_chain_batch(rng, 500, hops_range), k)
    J_avg = torch.stack(J[1:]).mean(0)  # one map for every loop (the prelude's map left out)
    rows = []
    with torch.no_grad():

        def score(z):
            return (model.unembed(z).argmax(-1) == test.answer).float().mean().item()

        for i, s in enumerate(t_states):
            h = s[:, Q]
            rows.append(
                {
                    "loop": i,
                    "logit_lens": score(h),
                    "tuned_per_loop": score(per_loop[i](h)),
                    "tuned_shared": score(shared[i](h)),
                    "jacobian_per_loop": score(h @ J[i].T),
                    "jacobian_averaged": score(h @ J_avg.T),
                }
            )
    return rows


# ------------------------------------------------------------------------------------------------ stage 13: probes
def fit_probe(x_tr, y_tr, x_te, y_te, n_classes, steps=300, lr=0.05):
    """Multinomial logistic regression with a little weight decay, on standardised inputs. Returns test accuracy."""
    mu, sd = x_tr.mean(0), x_tr.std(0) + 1e-6
    x_tr, x_te = (x_tr - mu) / sd, (x_te - mu) / sd
    lin = torch.nn.Linear(x_tr.shape[1], n_classes)
    opt = torch.optim.Adam(lin.parameters(), lr=lr, weight_decay=1e-3)
    for _ in range(steps):
        loss = F.cross_entropy(lin(x_tr), y_tr)
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        return (lin(x_te).argmax(-1) == y_te).float().mean().item()


def probe_report(model, k: int, hops=6, seed=11, n=3000, untrained=None):
    """Two sets of probes after every loop, on puzzles whose chain has `hops` links.
    At the question position: the name j links along the chain (j = 1 .. hops - 1), and the answer digit.
    At the statements of the question's chain: does the statement d links from the chain's end already hold the
    chain's digit? (d = 1 is the statement whose right-hand side is the digit.) If the model walks backward from the
    digit, the digit should reach distance d after about d - 1 loops.
    Controls: the answer probe with shuffled training labels, and (optionally) on an untrained model of the same
    shape."""
    rng = np.random.default_rng(seed)
    batch = make_chain_batch(rng, n, hops=hops)
    _, states = loop_states(model, batch, k)
    split = int(0.7 * n)
    gen = torch.Generator().manual_seed(seed)
    targets = {f"hop{j}": (batch.chain[:, j], N_VARS) for j in range(1, hops)}
    targets["answer"] = (batch.answer, N_DIGITS)
    rand_states = loop_states(untrained, batch, k)[1] if untrained is not None else None
    # pos[d] = input position of the chain statement d links from the end (its lhs is chain[hops - d])
    lhs = batch.pairs[:, :-1, 0]
    pos = {d: (lhs == batch.chain[:, hops - d : hops - d + 1]).float().argmax(1) for d in range(1, hops + 1)}
    every = torch.arange(n)
    rows = []
    for i, s in enumerate(states):
        torch.manual_seed(seed + i)
        x = s[:, Q]
        row = {"loop": i}
        for name, (y, c) in targets.items():
            row[name] = fit_probe(x[:split], y[:split], x[split:], y[split:], c)
        for d, p in pos.items():
            xs, y = s[every, p], batch.answer
            row[f"digit_at_dist{d}"] = fit_probe(xs[:split], y[:split], xs[split:], y[split:], N_DIGITS)
        y, c = targets["answer"]
        shuffled = y[:split][torch.randperm(split, generator=gen)]
        row["answer_shuffled_labels"] = fit_probe(x[:split], shuffled, x[split:], y[split:], c)
        if rand_states is not None:
            xr = rand_states[i][:, Q]
            row["answer_untrained_model"] = fit_probe(xr[:split], y[:split], xr[split:], y[split:], c)
        rows.append(row)
    return rows


# ------------------------------------------------------------------------------------------------ stage 14: patching
def corrupt(batch: ChainBatch) -> ChainBatch:
    """The same puzzles, but the question asks about the twin chain's first name, so the right answer becomes the twin
    chain's digit. Every statement is identical; only the question's name changes."""
    pairs = batch.pairs.clone()
    for b in range(len(pairs)):
        h, q = int(batch.hops[b]), int(pairs[b, -1, 0])
        lhs, rhs = pairs[b, :-1, 0].tolist(), pairs[b, :-1, 1].tolist()
        nxt = dict(zip(lhs, rhs, strict=False))
        pointed = {r for r in rhs if r < N_VARS}
        for start in lhs:  # chain starts are names no statement points to
            if start in pointed or start == q:
                continue
            v, n = start, 0
            while v < N_VARS:
                v, n = nxt[v], n + 1
            if n == h and v - N_VARS == int(batch.twin_answer[b]):
                pairs[b, -1, 0] = start
                break
    return ChainBatch(pairs, batch.twin_answer.clone(), batch.hops.clone(), batch.chain.clone(), batch.answer.clone())


@torch.no_grad()
def run_with_edit(model, pairs, k: int, loop: int, edit):
    """Run k loops; right after loop `loop` (0 = the prelude's output), replace the state with edit(state)."""
    e = model.run_prelude(pairs)
    s = start_state(model, e)
    if loop == 0:
        e = edit(e)
    for i in range(1, k + 1):
        s = model.loop_once(s, e)
        if i == loop:
            s = edit(s)
    return model.readout(s)[:, Q]


def patch_report(model, k: int, hops=4, seed=13, n=500):
    """Clean run: the real question. Corrupted run: the same statements, but the question asks about the twin chain.
    For each loop i, copy the clean state into the corrupted run right after loop i: (a) at the question position
    only, (b) at every position; (c) control: instead add a random vector as long as the clean - corrupted difference
    at the question. Score: share of puzzles where the corrupted run now gives the clean answer."""
    rng = np.random.default_rng(seed)
    clean = make_chain_batch(rng, n, hops=hops)
    bad = corrupt(clean)
    _, clean_states = loop_states(model, clean, k)
    _, bad_states = loop_states(model, bad, k)
    gen = torch.Generator().manual_seed(seed)

    def score(logits):
        return (logits.argmax(-1) == clean.answer).float().mean().item()

    rows = []
    for i in range(0, k + 1):
        cs, bs = clean_states[i], bad_states[i]
        diff_norm = (cs[:, Q] - bs[:, Q]).norm(dim=-1, keepdim=True)

        def at_question(s, cs=cs):
            s = s.clone()
            s[:, Q] = cs[:, Q]
            return s

        def random_dir(s, diff_norm=diff_norm):
            r = torch.randn(s[:, Q].shape, generator=gen)
            s = s.clone()
            s[:, Q] = s[:, Q] + r / r.norm(dim=-1, keepdim=True) * diff_norm
            return s

        rows.append(
            {
                "loop": i,
                "question_only": score(run_with_edit(model, bad.pairs, k, i, at_question)),
                "all_positions": score(run_with_edit(model, bad.pairs, k, i, lambda s, cs=cs: cs.clone())),
                "random_control": score(run_with_edit(model, bad.pairs, k, i, random_dir)),
            }
        )
    base = {
        "clean_run_gives_clean_answer": score(loop_states(model, clean, k)[0]),
        "corrupted_run_gives_clean_answer": score(loop_states(model, bad, k)[0]),
    }
    return base, rows


# ------------------------------------------------------------------------------------------------ stage 15: ablation
@torch.no_grad()
def ablation_report(model, k: int, seed=17, n=500, k_max=16):
    """(a) skip loop i: the core is not run that time (the state passes through unchanged); accuracy by hop count.
    (b) convergence: the relative change of the question's state from loop to loop, out to k_max loops, and how
    often the decoded answer flips between consecutive loops."""
    rng = np.random.default_rng(seed)
    sets = {h: make_chain_batch(rng, n, hops=h) for h in (2, 4, 6, 8)}
    skip = []
    for i in range(0, k + 1):
        row = {"skipped_loop": i}
        for h, b in sets.items():
            e = model.run_prelude(b.pairs)
            s = start_state(model, e)
            for j in range(1, k + 1):
                if j != i:
                    s = model.loop_once(s, e)
            row[f"hops{h}"] = (model.readout(s)[:, Q].argmax(-1) == b.answer).float().mean().item()
        skip.append(row)
    conv = []
    b = make_chain_batch(rng, n, hops_range=(1, 6))
    _, states = loop_states(model, b, k_max)
    prev = None
    for i in range(1, k_max + 1):
        a, c = states[i - 1][:, Q], states[i][:, Q]
        pred = model.readout(states[i])[:, Q].argmax(-1)
        conv.append(
            {
                "loop": i,
                "relative_change": ((c - a).norm(dim=-1) / c.norm(dim=-1)).mean().item(),
                "norm": c.norm(dim=-1).mean().item(),
                "answer_flips": None if prev is None else (pred != prev).float().mean().item(),
                "accuracy": (pred == b.answer).float().mean().item(),
            }
        )
        prev = pred
    return skip, conv


# ------------------------------------------------------------------------------------------------ stage 16: SAE
class TopKSAE(torch.nn.Module):
    """A sparse autoencoder: score n_latents directions, keep the top k, rebuild from those k (unit-length) rows."""

    def __init__(self, d: int, n_latents: int, k: int):
        super().__init__()
        self.k = k
        self.enc = torch.nn.Linear(d, n_latents)
        self.dec = torch.nn.Parameter(torch.randn(n_latents, d) / d**0.5)
        self.b_dec = torch.nn.Parameter(torch.zeros(d))

    def encode(self, x):
        pre = F.relu(self.enc(x - self.b_dec))
        top = pre.topk(self.k, dim=-1)
        return torch.zeros_like(pre).scatter(-1, top.indices, top.values)

    def forward(self, x):
        z = self.encode(x)
        return z @ F.normalize(self.dec, dim=-1) + self.b_dec, z


def sae_report(model, k: int, seed=19, n=4000, n_latents=256, topk=8, steps=1500):
    """One SAE shared by every loop: trained on the question's state pooled over loops 1..k (each loop scaled to the
    same average length, so late loops do not dominate). For each answer digit, pick the latent that best separates
    "the answer is this digit" at the last loop; then follow that same latent back through the loops, by hop count.
    Control: the same search over random unit directions instead of SAE latents."""
    rng = np.random.default_rng(seed)
    batch = make_chain_batch(rng, n, hops_range=(1, 6))
    _, states = loop_states(model, batch, k)
    xs = [states[i][:, Q] for i in range(1, k + 1)]
    xs = [x / x.norm(dim=-1).mean() for x in xs]
    data = torch.cat(xs)
    torch.manual_seed(seed)
    sae = TopKSAE(data.shape[1], n_latents, topk)
    opt = torch.optim.Adam(sae.parameters(), lr=3e-3)
    for _ in range(steps):
        idx = torch.randint(len(data), (512,))
        rec, _ = sae(data[idx])
        loss = (rec - data[idx]).pow(2).sum(-1).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        rec, _ = sae(data)
        fvu = ((rec - data).pow(2).sum() / (data - data.mean(0)).pow(2).sum()).item()
        fired = (sae.encode(xs[-1]) > 0).float()
        best = []  # selectivity: P(fires | answer = d) - P(fires | answer != d)
        for dgt in range(N_DIGITS):
            m = batch.answer == dgt
            sel = fired[m].mean(0) - fired[~m].mean(0)
            best.append((dgt, int(sel.argmax()), round(sel.max().item(), 3)))
        track = []  # P(the true digit's latent fires), loop by loop, by hop count
        for i, x in enumerate(xs, start=1):
            z = sae.encode(x)
            row = {"loop": i}
            for h in range(1, 7):
                m = batch.hops == h
                lat = torch.tensor([best[int(a)][1] for a in batch.answer[m]])
                row[f"hops{h}"] = (z[m].gather(1, lat[:, None]) > 0).float().mean().item()
            track.append(row)
        g = torch.Generator().manual_seed(seed)
        dirs = F.normalize(torch.randn(n_latents, data.shape[1], generator=g), dim=-1)
        proj = xs[-1] @ dirs.T
        rf = (proj > proj.quantile(0.9, dim=0)).float()  # each random direction "fires" on its top 10%
        rand_sel, rand_best = [], []
        for d in range(N_DIGITS):
            sel = rf[batch.answer == d].mean(0) - rf[batch.answer != d].mean(0)
            rand_sel.append(sel.max().item())
            rand_best.append(int(sel.argmax()))
        thresh = proj.quantile(0.9, dim=0)
        rand_track = []  # the same tracking for the best random direction per digit (thresholds from the last loop)
        for i, x in enumerate(xs, start=1):
            pr = x @ dirs.T
            row = {"loop": i}
            for h in range(1, 7):
                m = batch.hops == h
                lat = torch.tensor([rand_best[int(a)] for a in batch.answer[m]])
                row[f"hops{h}"] = (pr[m].gather(1, lat[:, None]) > thresh[lat][:, None]).float().mean().item()
            rand_track.append(row)
        alive = int((sae.encode(data) > 0).any(0).sum())
    return {
        "fvu": fvu,
        "alive_latents": alive,
        "best_latents": best,
        "track": track,
        "random_direction_selectivity": rand_sel,
        "random_direction_track": rand_track,
    }
