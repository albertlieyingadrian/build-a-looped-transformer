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
from pathlib import Path

import torch

MAX_STAGE = 4  # commands from later stages are hidden; the staged solutions lower this number

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


COMMANDS = {
    "tokenize": (1, cmd_tokenize),
    "forward": (2, cmd_forward),
    "attention": (3, cmd_attention),
    "block": (4, cmd_block),
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
