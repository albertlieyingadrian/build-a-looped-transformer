"""Your looped transformer. Run it with ./your_program.sh <command> [options].

Each stage adds one command. Start with stage 01 (`tokenize`): read stage_descriptions/base-01-tokenize.md, uncomment
the code below and make it pass `make test STAGE=01 DIR=<your copy of this folder>`.

Suggested layout as you go (the reference solutions use it, but any layout passes the tests):
    app/tokenizer.py  stage 01        app/model.py   stages 02-04    app/train.py   stage 05
    app/looped.py     stage 06        app/chain.py   stages 07-10    app/interp.py  stages 11-16
"""

import sys


def main():
    # Debug output goes to stderr, so it never confuses the tests, which read stdout.
    print("Logs from your program will appear here!", file=sys.stderr)

    # TODO: Uncomment the code below to pass the first stage.
    #
    # command = sys.argv[1]
    # if command == "tokenize":
    #     corpus = open(sys.argv[2], encoding="utf-8").read()
    #     text = sys.argv[3]
    #     chars = sorted(set(corpus))                 # the vocabulary: every distinct character
    #     ids = [chars.index(c) for c in text]
    #     print(f"corpus characters: {len(corpus)}")
    #     print(f"vocab size: {len(chars)}")
    #     print(f"ids: {ids}")
    #     print(f"decoded: {''.join(chars[i] for i in ids)}")
    #     print(f"round trip ok: {''.join(chars[i] for i in ids) == text}")
    # else:
    #     raise RuntimeError(f"Unknown command {command}")


if __name__ == "__main__":
    main()
