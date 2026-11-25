"""Stage 14: patch. Copying clean states into a corrupted run, loop by loop."""

from conftest import load_json, run


def test_patch(trained):
    run(["patch", "--ckpt", trained / "looped_s0.pt", "--k", 4, "--out-dir", trained], cwd=trained)
    res = load_json(trained / "patch.json")
    rows, base = res["rows"], res["base"]
    assert [r["loop"] for r in rows] == list(range(5))
    assert abs(rows[-1]["all_positions"] - base["clean_run_gives_clean_answer"]) < 1e-6, (
        "patching every position after the last loop must reproduce the clean run exactly"
    )
    assert base["corrupted_run_gives_clean_answer"] < 0.2
