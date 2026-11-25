"""Stage 04: block. Residual connections and LayerNorm."""

from conftest import line_value, run


def test_block(tmp_path):
    out = run(["block"], cwd=tmp_path)
    assert int(line_value(out, "block parameters:")) == 49984
    assert "out shape: (2, 8, 64)" in out
    assert line_value(out, "with the write-back layers at zero the block is the identity:") == "True"
    assert "LayerNorm output mean 0.00, std 1.00" in out
