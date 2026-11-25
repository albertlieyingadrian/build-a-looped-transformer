"""Stage 02: forward. Embeddings, logits and the loss of an untrained model."""

import math

from conftest import CORPUS, line_value, run


def test_forward_shapes_and_initial_loss(corpus, tmp_path):
    out = run(["forward", corpus, "Speak, speak."], cwd=tmp_path)
    V = len(set(CORPUS))
    assert line_value(out, "ids shape:") == "(1, 13)"
    assert line_value(out, "embeddings shape:") == "(1, 13, 64)"
    assert line_value(out, "logits shape:") == f"(1, 13, {V})"
    loss = float(line_value(out, "loss at initialisation:"))
    assert abs(loss - math.log(V)) < 1.0, "an untrained model should be close to guessing uniformly"
