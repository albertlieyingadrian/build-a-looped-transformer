"""Stage 01: tokenize. Characters to ids and back."""

from conftest import CORPUS, line_value, run


def test_tokenize(corpus, tmp_path):
    out = run(["tokenize", corpus, "Speak, speak."], cwd=tmp_path)
    vocab = sorted(set(CORPUS))
    assert int(line_value(out, "vocab size:")) == len(vocab)
    assert line_value(out, "ids:") == str([vocab.index(c) for c in "Speak, speak."])
    assert line_value(out, "decoded:") == "Speak, speak."
    assert line_value(out, "round trip ok:") == "True"
