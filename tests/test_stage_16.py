"""Stage 16: sae. One sparse autoencoder for every loop."""

from conftest import load_json, run


def test_sae(trained):
    run(["sae", "--ckpt", trained / "looped_s0.pt", "--k", 4, "--out-dir", trained], cwd=trained)
    res = load_json(trained / "sae.json")
    assert res["fvu"] < 0.5, "the SAE should rebuild most of the variance"
    assert res["alive_latents"] >= 16
    assert len(res["best_latents"]) == 10 and len(res["track"]) == 4
