"""Build your own looped transformer: one command per stage.

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

MAX_STAGE = 1  # commands from later stages are hidden; the staged solutions lower this number

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


COMMANDS = {
    "tokenize": (1, cmd_tokenize),
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
