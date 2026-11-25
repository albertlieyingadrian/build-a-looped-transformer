"""Stage 06: loop. Same parameters at every loop count; the shared gradient is a sum over passes."""

import re

from conftest import line_value, run


def test_loop(tmp_path):
    out = run(["loop"], cwd=tmp_path)
    block = int(line_value(out, "one block:").split()[0])
    looped = int(line_value(out, "looped (1 prelude + 1 core + 1 coda):").split()[0])
    standard = int(line_value(out, "standard (3 blocks):").split()[0])
    assert looped == standard
    assert int(line_value(out, "core:").split()[0]) == block
    depths = {
        int(k): (int(d), int(p)) for k, d, p in re.findall(r"k=\s*(\d+)\s+depth\s+(\d+) blocks\s+parameters (\d+)", out)
    }
    assert set(depths) == {1, 2, 4, 8, 16}
    for k, (d, p) in depths.items():
        assert d == k + 2 and p == looped
    assert out.strip().endswith("True"), "the shared gradient must equal the sum of the unrolled copies' gradients"
