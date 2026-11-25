"""Stage 18 (extension): message passing between passes, Inkling-style."""

from conftest import load_json, run


def test_message_passing(tmp_path):
    run(
        ["injection", "--kinds", "looped_message", "--steps", "400", "--n-test", "100", "--out-dir", tmp_path],
        cwd=tmp_path,
    )
    grid = load_json(tmp_path / "injection_s0.json")["looped_message"]
    assert grid["4"]["1"] > 0.5
