"""Stage 05: train-gpt. The loss falls well below guessing."""

import math

from conftest import CORPUS, line_value, run


def test_train_gpt(corpus, tmp_path):
    out = run(
        [
            "train-gpt",
            corpus,
            "--steps",
            "300",
            "--layers",
            "2",
            "--block-size",
            "32",
            "--sample-len",
            "40",
            "--prompt",
            "All:",
            "--out-dir",
            tmp_path,
        ],
        cwd=tmp_path,
    )
    val = float(line_value(out, "final val loss:").split()[0])
    assert val < 0.5 * math.log(len(set(CORPUS))), f"val loss {val} is not far enough below guessing"
    assert (tmp_path / "gpt.pt").exists()
    assert "sample (greedy):" in out
