"""Stage 03: attention. Rows sum to one and no position reads the future."""

from conftest import line_value, run


def test_causal_attention(tmp_path):
    out = run(["attention", "--T", "6"], cwd=tmp_path)
    rows = [line.split() for line in out.splitlines() if line.startswith("  ")]
    assert len(rows) == 6
    for i, row in enumerate(rows):
        vals = [float(v) for v in row]
        assert abs(sum(vals) - 1) < 0.02
        assert all(v == 0 for v in vals[i + 1 :]), f"row {i} reads the future"
    assert line_value(out, "rows sum to 1:") == "True"
    assert float(line_value(out, "weight on future positions:")) == 0.0
    assert float(line_value(out, "change at earlier positions after editing the last one:")) == 0.0
