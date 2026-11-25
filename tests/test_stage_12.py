"""Stage 12: tuned-lens. Per-loop and shared tuned lenses, per-loop and averaged Jacobian lenses."""

from conftest import load_json, run


def test_tuned_lens(trained):
    run(["tuned-lens", "--ckpt", trained / "looped_s0.pt", "--k", 4, "--out-dir", trained], cwd=trained)
    rows = load_json(trained / "tuned_lens.json")
    assert [r["loop"] for r in rows] == list(range(5))
    keys = {"logit_lens", "tuned_per_loop", "tuned_shared", "jacobian_per_loop", "jacobian_averaged"}
    assert keys <= set(rows[0])
    assert rows[-1]["tuned_per_loop"] >= rows[-1]["logit_lens"] - 0.05, "a tuned lens should not read less"
