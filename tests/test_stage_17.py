"""Stage 17 (extension): exit. Stop when the answer stops changing."""

import re

from conftest import run


def test_adaptive_exit(trained):
    out = run(["exit", "--ckpt", trained / "looped_s0.pt", "--k-max", 8], cwd=trained)
    used = [float(x) for x in re.findall(r"mean loops used\s+([\d.]+)", out)]
    assert len(used) == 8
    assert all(1 <= u <= 8 for u in used)
