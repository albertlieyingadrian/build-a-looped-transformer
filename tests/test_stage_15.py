"""Stage 15: ablate. Skipping loops, and convergence out to 16 loops."""

from conftest import load_json, run


def test_ablate(trained):
    run(["ablate", "--ckpt", trained / "looped_s0.pt", "--k", 4, "--out-dir", trained], cwd=trained)
    res = load_json(trained / "ablate.json")
    assert [r["skipped_loop"] for r in res["skip"]] == list(range(5))
    assert len(res["convergence"]) == 16
    assert res["convergence"][0]["answer_flips"] is None
    assert all(r["relative_change"] >= 0 for r in res["convergence"])
