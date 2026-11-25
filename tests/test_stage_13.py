"""Stage 13: probe. Linear probes per loop, with controls."""

from conftest import load_json, run


def test_probe(trained):
    run(["probe", "--ckpt", trained / "looped_s0.pt", "--k", 4, "--out-dir", trained], cwd=trained)
    rows = load_json(trained / "probe.json")
    assert [r["loop"] for r in rows] == list(range(5))
    for r in rows:
        assert r["answer_shuffled_labels"] < 0.25, "a probe trained on shuffled labels must stay near chance"
    assert max(r["hop1"] for r in rows) > 0.5, "the next link of the chain should be readable after some loop"
