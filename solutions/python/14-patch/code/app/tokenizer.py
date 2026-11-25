"""Characters to numbers and back (stage 01)."""


class CharTokenizer:
    """The vocabulary is every distinct character in the corpus, sorted; a character's id is its place in that list."""

    def __init__(self, corpus: str):
        self.chars = sorted(set(corpus))
        self.stoi = {c: i for i, c in enumerate(self.chars)}

    @property
    def vocab_size(self) -> int:
        return len(self.chars)

    def encode(self, s: str) -> list[int]:
        return [self.stoi[c] for c in s]

    def decode(self, ids) -> str:
        return "".join(self.chars[i] for i in ids)
