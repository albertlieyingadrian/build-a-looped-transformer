"""Stage 11: logit-lens. Decoding the state after every loop."""

from conftest import load_json, run


def test_logit_lens(trained):
    run(["logit-lens", "--ckpt", trained / "looped_s0.pt", "--k", 6, "--out-dir", trained], cwd=trained)
    res = load_json(trained / "logit_lens.json")
    for rows in res.values():
        assert [r["loop"] for r in rows] == list(range(7))
        assert rows[-1]["coda_agrees_with_final"] == 1.0, "the coda lens at the last loop is the model's own output"
        assert all(0 <= r["logit_lens"] <= 1 for r in rows)
