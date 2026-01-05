from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import cast

type Token = int
type Tokens = tuple[Token, ...]


@dataclass(slots=True)
class PrefixNode:
    vocab_size: int
    value: Token | None = None
    _children: list["PrefixNode | None"] = field(init=False, repr=False)

    def __post_init__(self):
        self._children = [None] * self.vocab_size

    def __contains__(self, toks: Token | Tokens) -> bool:
        if isinstance(toks, int):
            return self._children[toks] is not None

        node = self
        for t in toks:
            if t not in node._children:
                return False
            node = node._children[t]
        return self.value is not None

    def __getitem__(self, toks: Tokens) -> Token:
        node = self
        for t in toks:
            if node._children[t] is None:
                raise KeyError(toks)
            node = node._children[t]
            assert node is not None
        assert node.value is not None
        return node.value

    def __repr__(self) -> str:
        if self.value is not None:
            return f"<{self.value}>"

        children_repr = []
        for i, child in self.children():
            children_repr.append(f"{i}→{child}")

        if not children_repr:
            return "∅"

        return "{" + ", ".join(children_repr) + "}"

    def get_child(self, tok: Token) -> "PrefixNode":
        child = self._children[tok]
        if child is None:
            raise KeyError(tok)
        return child

    def children(self) -> Iterable[tuple[Token, "PrefixNode"]]:
        return (
            (cast(Token, i), node)
            for (i, node) in enumerate(self._children)
            if node is not None
        )

    def insert(self, key: Tokens, value: Token):
        node = self
        for t in key:
            if node._children[t] is None:
                node._children[t] = PrefixNode(self.vocab_size)
            node = node._children[t]
            assert node is not None
        node.value = value
