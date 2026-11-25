"""Stage 09: injection. Re-adding the input vs a plain loop."""

from conftest import load_json, run


def test_injection(tmp_path):
    out = run(
        ["injection", "--kinds", "looped,looped_none", "--steps", "400", "--n-test", "100", "--out-dir", tmp_path],
        cwd=tmp_path,
    )
    res = load_json(tmp_path / "injection_s0.json")
    assert set(res) == {"looped", "looped_none"}
    for grid in res.values():
        assert set(grid) == {"1", "2", "4", "8", "16"}
        assert grid["4"]["1"] > 0.5, "one-hop chains should be learnt by both models"
    assert "looped_none:" in out
