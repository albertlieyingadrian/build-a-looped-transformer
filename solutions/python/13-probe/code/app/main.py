"""Build a looped transformer: one command per stage.

    ./your_program.sh <command> [options]

Stage 01 tokenize   02 forward   03 attention   04 block   05 train-gpt   06 loop   07 chain   08 train-looped
      09 injection  10 compare   11 logit-lens  12 tuned-lens  13 probe   14 patch  15 ablate  16 sae
Extensions: 17 exit (adaptive exit), 18 injection --kinds looped_message (message passing)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

MAX_STAGE = 13  # commands from later stages are hidden; the staged solutions lower this number

torch.set_num_threads(4)


def out_path(args, name):
    path = Path(args.out_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path / name


def save_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=1))
    print(f"wrote {path}")


# ------------------------------------------------------------------------------------------------ stages 01-05: a GPT
def cmd_tokenize(args):
    from app.tokenizer import CharTokenizer

    corpus = Path(args.corpus).read_text(encoding="utf-8")
    tok = CharTokenizer(corpus)
    ids = tok.encode(args.text)
    print(f"corpus characters: {len(corpus)}")
    print(f"vocab size: {tok.vocab_size}")
    print(f"ids: {ids}")
    print(f"decoded: {tok.decode(ids)}")
    print(f"round trip ok: {tok.decode(ids) == args.text}")


def cmd_forward(args):
    from app.model import Config, n_params, text_model
    from app.tokenizer import CharTokenizer

    tok = CharTokenizer(Path(args.corpus).read_text(encoding="utf-8"))
    torch.manual_seed(args.seed)
    model = text_model(tok.vocab_size, n_layer=0, cfg=Config())  # no blocks yet: embed, LayerNorm, head
    ids = torch.tensor([tok.encode(args.text)])
    _, loss = model(ids[:, :-1], ids[:, 1:])
    print(f"ids shape: {tuple(ids.shape)}")
    print(f"embeddings shape: {tuple(model.embed(ids).shape)}")
    print(f"logits shape: {tuple(model(ids)[0].shape)}")
    print(f"parameters: {n_params(model)}")
    print(f"loss at initialisation: {loss.item():.3f}")
    print(f"ln(vocab size): {torch.log(torch.tensor(float(tok.vocab_size))).item():.3f}")


def cmd_attention(args):
    from app.model import Config, SelfAttention

    torch.manual_seed(args.seed)
    cfg = Config(n_embd=16, n_head=1, causal=True)
    attn = SelfAttention(cfg)
    x = torch.randn(1, args.T, cfg.n_embd)
    y = attn(x)
    w = attn.last_attn[0, 0]
    print("attention weights (row = reading position, column = position read):")
    for row in w.tolist():
        print("  " + " ".join(f"{v:.2f}" for v in row))
    print(f"rows sum to 1: {torch.allclose(w.sum(-1), torch.ones(args.T))}")
    print(f"weight on future positions: {w.triu(1).abs().sum().item():.1f}")
    x2 = x.clone()
    x2[:, -1] += 10.0  # change only the last position
    y2 = attn(x2)
    print(f"change at earlier positions after editing the last one: {(y2[:, :-1] - y[:, :-1]).abs().max().item():.1f}")


def cmd_block(args):
    from app.model import Block, Config, n_params

    torch.manual_seed(args.seed)
    cfg = Config()
    block = Block(cfg)
    x = torch.randn(2, 8, cfg.n_embd)
    print(f"block parameters: {n_params(block)}")
    print(f"in shape: {tuple(x.shape)}  out shape: {tuple(block(x).shape)}")
    for lin in (block.attn.proj, block.mlp.down):  # zero the two layers that write into the stream
        torch.nn.init.zeros_(lin.weight)
        torch.nn.init.zeros_(lin.bias)
    print(f"with the write-back layers at zero the block is the identity: {torch.allclose(block(x), x)}")
    big = x * 100
    mean = round(block.ln1(big).mean().item(), 2) + 0.0  # + 0.0 turns -0.0 into 0.0
    print(f"LayerNorm output mean {mean:.2f}, std {block.ln1(big).std().item():.2f}")


def cmd_train_gpt(args):
    from app.model import Config, n_params, text_model
    from app.tokenizer import CharTokenizer
    from app.train import train_lm

    text = Path(args.corpus).read_text(encoding="utf-8")
    tok = CharTokenizer(text)
    data = torch.tensor(tok.encode(text))
    n = int(0.9 * len(data))
    torch.manual_seed(args.seed)
    cfg = Config(n_embd=args.n_embd, block_size=args.block_size)
    model = text_model(tok.vocab_size, args.layers, cfg)
    print(f"parameters: {n_params(model)}")
    t = time.time()
    val = train_lm(model, data[:n], data[n:], steps=args.steps, seed=args.seed, batch_size=args.batch_size)
    print(f"final val loss: {val:.3f}  ({time.time() - t:.0f}s)")
    start = torch.tensor([tok.encode(args.prompt)])
    print("sample (greedy):")
    print(tok.decode(model.generate(start, args.sample_len)[0].tolist()))
    torch.save(model.state_dict(), out_path(args, "gpt.pt"))


# ------------------------------------------------------------------------------------------------ stages 06-10: loops
def cmd_loop(args):
    from app.looped import LoopedGPT
    from app.model import GPT, Block, Config, TextEmbed, n_params

    torch.manual_seed(args.seed)
    cfg, V = Config(), 65
    looped = LoopedGPT(TextEmbed(V, cfg), V, cfg, prelude=1, core=1, coda=1)
    standard = GPT(TextEmbed(V, cfg), V, n_layer=3, cfg=cfg)
    print(f"one block: {n_params(Block(cfg))} parameters")
    print(f"looped (1 prelude + 1 core + 1 coda): {n_params(looped)} parameters")
    print(f"standard (3 blocks): {n_params(standard)} parameters")
    print(f"core: {n_params(looped.core)} parameters (one block, shared by every loop)")
    idx = torch.randint(V, (2, 16))
    for k in (1, 2, 4, 8, 16):
        logits, _ = looped(idx, k=k)
        print(
            f"k={k:2d}  depth {looped.depth(k):2d} blocks  parameters {n_params(looped)}  logits {tuple(logits.shape)}"
        )
    # The shared core's gradient is the sum of the gradients that k separate copies of it would get.
    k = 3
    looped.zero_grad()
    looped(idx, idx, k=k)[1].backward()
    shared_grad = looped.core[0].mlp.up.weight.grad.clone()
    copies = [Block(cfg) for _ in range(k)]
    for c in copies:
        c.load_state_dict(looped.core[0].state_dict())
    e = looped.run_prelude(idx)
    s = torch.zeros_like(e)
    for c in copies:
        s = c(s + e)
    F.cross_entropy(looped.readout(s).flatten(0, 1), idx.flatten()).backward()
    summed = sum(c.mlp.up.weight.grad for c in copies)
    print(
        f"shared gradient = sum of the {k} unrolled copies' gradients: {torch.allclose(shared_grad, summed, atol=1e-6)}"
    )


def cmd_chain(args):
    import numpy as np

    from app.chain import N_STATEMENTS, N_VARS, make_chain_batch, render

    batch = make_chain_batch(np.random.default_rng(args.seed), args.n, hops=args.hops)
    for i in range(args.n):
        print(render(batch, i))
    many = make_chain_batch(np.random.default_rng(args.seed + 1), 1000, hops=args.hops)
    ends = (many.pairs[:, :-1, 1] >= N_VARS).sum(1).float()  # chains = statements that end in a digit
    print(f"statements per puzzle: {N_STATEMENTS}")
    print(
        f"guess a digit: 0.10;  guess one chain end at random: {(1 / ends).mean().item():.2f};  "
        f"pick the longest chain: 0.50 (the twin chain is as long)"
    )


def chain_sets(args):
    from app.chain import test_sets

    return test_sets(n=args.n_test)


def accuracy_grid(model, sets, ks):
    from app.chain import accuracy

    return {str(k): {str(h): round(accuracy(model, b, k=k), 3) for h, b in sets.items()} for k in ks}


def print_grid(name, grid):
    hops = list(next(iter(grid.values())).keys())
    print(f"{name}: accuracy by hop count (columns) and loop count (rows); training saw 1-6 hops")
    print("        " + " ".join(f"{h:>4}" for h in hops))
    for k, row in grid.items():
        label = f"k={k}" if k != "None" else "-"
        print(f"{label:>6}  " + " ".join(f"{row[h]:4.2f}" for h in hops))


def train_chain_model(kind, args):
    from app.chain import build, train
    from app.model import n_params

    model = build(kind, seed=args.seed)
    t = time.time()
    curriculum = train(
        model, seed=args.seed, steps=args.steps, k_range=(args.k_min, args.k_max), log=print if args.verbose else None
    )
    print(f"{kind}: {n_params(model)} parameters, trained in {time.time() - t:.0f}s, curriculum {curriculum}")
    return model, curriculum


def load_or_train(kind, args):
    from app.chain import build

    ckpt = out_path(args, f"{kind}_s{args.seed}.pt")
    if ckpt.exists() and not getattr(args, "retrain", False):
        model = build(kind, seed=args.seed)
        model.load_state_dict(torch.load(ckpt))
        model.eval()
        print(f"{kind}: loaded {ckpt}")
        return model
    model, _ = train_chain_model(kind, args)
    torch.save(model.state_dict(), ckpt)
    return model


def cmd_train_looped(args):
    model, curriculum = train_chain_model("looped", args)
    torch.save(model.state_dict(), out_path(args, f"looped_s{args.seed}.pt"))
    grid = accuracy_grid(model, chain_sets(args), [1, 2, 3, 4, 6, 8, 12, 16])
    print_grid("looped", grid)
    save_json(out_path(args, f"looped_s{args.seed}.json"), {"grid": grid, "curriculum": curriculum})


def cmd_injection(args):
    sets = chain_sets(args)
    results = {}
    for kind in args.kinds.split(","):
        results[kind] = accuracy_grid(load_or_train(kind, args), sets, [1, 2, 4, 8, 16])
        print_grid(kind, results[kind])
    save_json(out_path(args, f"injection_s{args.seed}.json"), results)


def cmd_compare(args):
    sets = chain_sets(args)
    results = {}
    for kind in ("looped", "standard3", "standard10"):
        ks = [1, 2, 4, 8, 12, 16] if kind == "looped" else [None]
        results[kind] = accuracy_grid(load_or_train(kind, args), sets, ks)
        print_grid(kind, results[kind])
    reach = {}
    for kind, grid in results.items():
        for k, row in grid.items():
            longest = 0
            for h in sorted(row, key=int):
                if row[h] < 0.9:
                    break
                longest = int(h)
            reach[f"{kind} k={k}" if k != "None" else kind] = longest
    print("longest chain solved (>= 90% right, and every shorter chain too):")
    for name, h in reach.items():
        print(f"  {name:<18} {h} hops")
    save_json(out_path(args, f"compare_s{args.seed}.json"), {"grid": results, "reach": reach})
    plot_compare(results, out_path(args, f"compare_s{args.seed}.png"))


def plot_compare(results, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    for kind, grid in results.items():
        for k, row in grid.items():
            if kind == "looped" and k not in ("1", "4", "8", "16"):
                continue
            hops = sorted(row, key=int)
            label = f"{kind} k={k}" if k != "None" else kind
            ax.plot(
                [int(h) for h in hops],
                [row[h] for h in hops],
                "-" if kind == "looped" else "--",
                marker="o",
                ms=3,
                label=label,
            )
    ax.axvspan(6.5, 12.5, color="0.9", zorder=0)
    ax.text(9.5, 0.05, "longer than any training chain", ha="center", fontsize=8)
    ax.set_xlabel("hops in the question's chain")
    ax.set_ylabel("accuracy")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    print(f"wrote {path}")


# ------------------------------------------------------------------------------------------------ stages 11-16: inside
def tagged(args, name):
    """Results of a looped_none or looped_message checkpoint get the model's name in front."""
    return name if args.kind == "looped" else f"{args.kind}_{name}"


def load_looped(args):
    from app.chain import build

    model = build(args.kind, seed=args.seed)
    model.load_state_dict(torch.load(args.ckpt))
    model.eval()
    return model


def print_rows(rows, title):
    print(title)
    keys = list(rows[0].keys())
    print("  " + " ".join(f"{k:>12.12}" for k in keys))
    for r in rows:
        print("  " + " ".join(f"{r[k]:>12.2f}" if isinstance(r[k], float) else f"{str(r[k]):>12}" for k in keys))


def cmd_logit_lens(args):
    import numpy as np

    from app.chain import make_chain_batch
    from app.interp import logit_lens

    model = load_looped(args)
    out = {}
    for hops in (2, 4, 6):
        batch = make_chain_batch(np.random.default_rng(args.seed + hops), 500, hops=hops)
        out[hops] = logit_lens(model, batch, args.k)
        print_rows(out[hops], f"{hops}-hop chains, k={args.k}: accuracy of each lens after loop i (0 = after prelude)")
    save_json(out_path(args, tagged(args, "logit_lens.json")), out)


def cmd_tuned_lens(args):
    from app.interp import tuned_lens_report

    rows = tuned_lens_report(load_looped(args), args.k)
    print_rows(rows, f"k={args.k}, chains of 1-6 hops: accuracy of each lens after loop i")
    save_json(out_path(args, tagged(args, "tuned_lens.json")), rows)


def cmd_probe(args):
    from app.chain import build
    from app.interp import probe_report

    untrained = build(args.kind, seed=args.seed + 100).eval()
    rows = probe_report(load_looped(args), args.k, untrained=untrained)
    print_rows(
        rows,
        f"6-hop chains, k={args.k}: linear-probe test accuracy at the question after loop i "
        "(hopj = the name j links along, chance 0.02; answer chance 0.10)",
    )
    save_json(out_path(args, tagged(args, "probe.json")), rows)


COMMANDS = {
    "tokenize": (1, cmd_tokenize),
    "forward": (2, cmd_forward),
    "attention": (3, cmd_attention),
    "block": (4, cmd_block),
    "train-gpt": (5, cmd_train_gpt),
    "loop": (6, cmd_loop),
    "chain": (7, cmd_chain),
    "train-looped": (8, cmd_train_looped),
    "injection": (9, cmd_injection),
    "compare": (10, cmd_compare),
    "logit-lens": (11, cmd_logit_lens),
    "tuned-lens": (12, cmd_tuned_lens),
    "probe": (13, cmd_probe),
}


def parser():
    p = argparse.ArgumentParser(
        prog="your_program.sh", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = p.add_subparsers(dest="command", required=True)
    for name, (stage, _) in COMMANDS.items():
        if stage > MAX_STAGE:
            continue
        s = sub.add_parser(name)
        s.add_argument("--seed", type=int, default=0)
        s.add_argument("--out-dir", default="runs")
        if name in ("tokenize", "forward"):
            s.add_argument("corpus")
            s.add_argument("text")
        if name == "attention":
            s.add_argument("--T", type=int, default=5)
        if name == "train-gpt":
            s.add_argument("corpus")
            s.add_argument("--steps", type=int, default=2000)
            s.add_argument("--layers", type=int, default=4)
            s.add_argument("--n-embd", type=int, default=64)
            s.add_argument("--block-size", type=int, default=64)
            s.add_argument("--batch-size", type=int, default=32)
            s.add_argument("--prompt", default="\n")
            s.add_argument("--sample-len", type=int, default=200)
        if name == "chain":
            s.add_argument("--hops", type=int, default=3)
            s.add_argument("--n", type=int, default=3)
        if name in ("train-looped", "injection", "compare"):
            s.add_argument("--steps", type=int, default=4000)
            s.add_argument("--k-min", type=int, default=1)
            s.add_argument("--k-max", type=int, default=8)
            s.add_argument("--n-test", type=int, default=500)
            s.add_argument("--retrain", action="store_true")
            s.add_argument("--verbose", action="store_true")
        if name == "injection":
            s.add_argument("--kinds", default="looped,looped_none")
        if name in ("logit-lens", "tuned-lens", "probe", "patch", "ablate", "sae", "exit"):
            s.add_argument("--ckpt", default="runs/looped_s0.pt")
            s.add_argument("--kind", default="looped", help="looped, looped_none or looped_message (match --ckpt)")
            s.add_argument("--k", type=int, default=8)
        if name == "exit":
            s.add_argument("--k-max", type=int, default=16)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    COMMANDS[args.command][1](args)


if __name__ == "__main__":
    sys.exit(main())
