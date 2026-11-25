"""Stage 07: chain. Puzzles with the right number of hops, a twin chain, and the right answer."""

import re

from conftest import run


def follow(statements, start):
    table = dict(s.split("=") for s in statements)
    v, hops = start, 0
    while not v.isdigit():
        v, hops = table[v], hops + 1
    return int(v), hops


def test_chain(tmp_path):
    for hops in (1, 3, 6):
        out = run(["chain", "--hops", hops, "--n", 5, "--seed", hops], cwd=tmp_path)
        puzzles = [line for line in out.splitlines() if "|" in line]
        assert len(puzzles) == 5
        for p in puzzles:
            body, tail = p.split("|")
            statements = body.split()
            assert len(statements) == 24
            q, ans, h = re.match(r"\s*(\w)\? -> (\d) \((\d+) hops\)", tail).groups()
            assert follow(statements, q) == (int(ans), hops) and int(h) == hops
            lengths = [follow(statements, s.split("=")[0])[1] for s in statements]
            assert lengths.count(hops) >= 2, "a second chain (the twin) must be as long as the question's"
