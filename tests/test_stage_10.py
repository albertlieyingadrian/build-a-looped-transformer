"""Stage 10: compare. Standard models with the same weights and the same compute."""

from conftest import load_json, run


def test_compare(trained):
    out = run(["compare", "--steps", "400", "--n-test", "100", "--out-dir", trained], cwd=trained)
    res = load_json(trained / "compare_s0.json")
    assert set(res["grid"]) == {"looped", "standard3", "standard10"}
    assert set(res["grid"]["standard3"]) == {"None"}
    assert "looped k=16" in res["reach"] and "standard10" in res["reach"]
    assert (trained / "compare_s0.png").exists()
    assert "longest chain solved" in out
