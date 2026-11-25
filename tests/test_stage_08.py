"""Stage 08: train-looped. Random loop counts in training, a sweep of loop counts at test time."""

from conftest import load_json


def test_train_looped(trained):
    res = load_json(trained / "looped_s0.json")
    grid = res["grid"]
    assert {"1", "2", "4", "8", "16"} <= set(grid)
    assert grid["4"]["1"] >= 0.9, "one-hop chains must be solved after 1,500 steps"
    assert grid["4"]["2"] > 0.5, "two-hop chains must be well above chance (about 0.2) after 1,500 steps"
    assert (trained / "looped_s0.pt").exists()
    assert res["curriculum"][0] == [0, 2]
