#!/usr/bin/env python

from dataclasses import dataclass, field
import heapq
import sys
import itertools
from typing import Generator, Iterable, LiteralString, cast
import time

Input = list[str] | list[LiteralString]

CONTROL_1: Input = (
    """\
H => HO
H => OH
O => HH
""".splitlines()
)

type Token = str


@dataclass(slots=True)
class Tokenizer:
    # TODO: maybe do byte pair encoding
    tokens: set[Token]
    max_token_len: int

    def __init__(self, known_tokens: Iterable[str]):
        # self.tokens = sorted(known_tokens, key=lambda x: len(x), reverse=True)
        self.tokens = {x for x in known_tokens if len(x) > 1}
        if self.tokens:
            self.max_token_len = max(len(x) for x in self.tokens)
        else:
            self.max_token_len = 1

    def tokenize(self, s: str) -> Generator[Token]:
        i = 0
        while i < len(s):
            found = False
            for size in range(self.max_token_len, 1, -1):
                if (chunk := s[i : i + size]) in self.tokens:
                    yield chunk
                    found = True
                    i += size
                    break
            if not found:
                yield s[i]
                i += 1


def flatten(x) -> list[Token]:
    def inner(x) -> Generator[Token]:
        for t in x:
            if isinstance(t, str):
                yield t
            else:
                yield from flatten(t)

    return list(inner(x))


@dataclass(slots=True)
class RopeNode[Str: str | bytes | bytearray]:
    weight: int = field(repr=False)  # total length of left
    s: Str | None = None
    left: "RopeNode[Str] | None" = None
    right: "RopeNode[Str] | None" = None

    def __len__(self) -> int:
        if self.s:
            return len(self.s)  # ERROR: incompatible with type Sized
        out = 0
        if self.left:
            out += len(self.left)
        if self.right:
            out += len(self.right)
        return out

    def __str__(self) -> str:
        if self.s is not None:
            if isinstance(self.s, str):
                return self.s
            if isinstance(self.s, bytes):
                return str(self.s, "ascii")
            if isinstance(self.s, bytearray):
                return str(self.s, "ascii")
        assert self.left is not None and self.right is not None
        return str(self.left) + str(self.right)

    @staticmethod
    def concat(r1: "RopeNode", r2: "RopeNode") -> "RopeNode":
        return RopeNode(left=r1, right=r2, weight=len(r1))

    def split(self, idx: int) -> tuple["RopeNode", "RopeNode"]:
        """
        Given s = "abcdefg", idx = 3: ("abc", "defg")
        """
        if self.s is not None:
            left = self.s[:idx]
            right = self.s[idx:]
            return RopeNode(len(left), s=left), RopeNode(len(right), s=right)

        assert self.left is not None and self.right is not None
        if idx < self.weight:
            (l1, l2) = RopeNode.split(self.left, idx)
            return l1, RopeNode.concat(l2, self.right)

        (r1, r2) = RopeNode.split(self.right, idx - self.weight)
        return RopeNode.concat(self.left, r1), r2

    def insert(self, idx: int, s: str) -> "RopeNode":
        insert = RopeNode(0, s=s)
        left, right = self.split(idx)
        return RopeNode.concat(RopeNode.concat(left, insert), right)

    def replace(self, start: int, end: int, s: str) -> "RopeNode":
        insert = RopeNode(0, s=s)
        left, _ = self.split(start)
        _, right = self.split(end)
        return RopeNode.concat(RopeNode.concat(left, insert), right)


def rope(s: str | bytes | bytearray) -> RopeNode:
    return RopeNode(0, s=s)


# TEST ROPE
for s in (
    cast(str, "hello there"),
    cast(bytes, b"hello there"),
    bytearray(b"hello there"),
):
    r = rope(s)
    assert str(r) == "hello there"
    r2 = r.insert(3, "ahoy")
    assert str(r2) == "helahoylo there"
    r3 = r.insert(0, "ahoy ")
    assert str(r3) == "ahoy hello there"
    r4 = r.replace(4, 5, " no")
    assert str(r4) == "hell no there"


@dataclass(slots=True)
class Map:
    tokenizer: Tokenizer = field(repr=False)
    map: dict[Token, list[str]] = field(default_factory=dict)

    def add(self, key: str, val: str):
        self.map.setdefault(key, []).append(val)

    def extend(self, pairs: Iterable[tuple[str, str]]):
        for key, val in pairs:
            self.add(key, val)

    def substitutions(
        self, src: list[Token], mapping: tuple[Token, str]
    ) -> Generator[str]:
        from_, to_ = mapping
        for i, tok in enumerate(src):
            if tok == from_:
                yield "".join(itertools.chain(src[:i], (to_,), src[i + 1 :]))

    def replacements(self, s: str) -> Generator[str]:
        src_tokens = list(self.tokenizer.tokenize(s))
        for token in src_tokens:
            if token not in self.map:
                continue
            for dst in self.map[token]:
                yield from self.substitutions(src_tokens, (token, dst))


def parse(input: Input) -> tuple[Map, str]:
    "Returns map, molecule."
    mappings: list[tuple[str, str]] = []
    for line in input:
        if line == "":
            break
        left, right = line.split(" => ", maxsplit=1)
        mappings.append((left, right))
    tokenizer = Tokenizer({x[0] for x in mappings})
    map = Map(tokenizer)
    map.extend(mappings)
    return map, input[-1]


def part_1(input: Input):
    map, molecule = parse(input)
    return len(set(map.replacements(molecule)))


# TODO: possible optimizations:
# - the b string is always the same, so we can reuse/reset distances list
# - use bytearray instead of strs (more likely to trigger memcpy, no unicode)
def string_distance(a: str, b: str) -> int:
    """Wagner-Fischer algorithm for Levenshtein distance using 1-d table."""
    # shortcuts
    if a == b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    a_len, b_len = len(a), len(b)
    if b_len == 0:
        return a_len

    distances = list(range(b_len + 1))

    for i, a_char in enumerate(a, 1):
        prev_diag = distances[0]
        distances[0] = i
        for j, b_char in enumerate(b, 1):
            previous_distance = distances[j]
            distances[0] = i
            cost = 0 if a_char == b_char else 1
            distances[j] = min(
                # deletion
                distances[j] + 1,
                # insertion
                distances[j - 1] + 1,
                # substitution
                prev_diag + cost,
            )
            prev_diag = previous_distance
    return distances[b_len]


@dataclass(slots=True)
class QNode:
    count: int
    s: str
    distance: int

    def __lt__(self, other: "QNode") -> bool:
        # return self.count < other.count
        return (self.count, self.distance) < (other.count, other.distance)
        # return self.distance < other.distance


def search(map: Map, destination: str) -> int:
    q = [QNode(0, "e", 0)]
    visited = set()
    i = 0
    while q:
        i += 1
        node = heapq.heappop(q)
        if i % 100 == 0:
            print(node)
        if node.s in visited:
            continue
        visited.add(node.s)
        if node.s == destination:
            return node.count
        for tok in map.tokenizer.tokenize(node.s):
            if tok not in map.map:
                continue
            for sub in set(map.replacements(node.s)):
                heapq.heappush(
                    # q, QNode(node.count + 1, sub, string_distance(sub, destination))
                    q,
                    QNode(node.count + 1, sub, 0),
                )
    raise Exception("unreachable")


def part_2(input: Input):
    map, molecule = parse(input)
    print("STARTING PART 2")
    search(map, molecule)


def _test():
    def assert_eq(a, b):
        assert a == b, f"{a} != {b}"

    map, _ = parse(CONTROL_1)
    assert_eq(len(set(map.replacements("HOH"))), 4)
    assert_eq(len(set(map.replacements("HOHOHO"))), 7)

    map.add("e", "H")
    map.add("e", "O")
    assert_eq(search(map, "HOH"), 3)
    assert_eq(search(map, "HOHOHO"), 6)
    print("tests PASSED", file=sys.stderr)
    # assert_eq(part_2(CONTROL_1), 0)


def run(fn, year=2015, day=19, part=0):
    start = time.perf_counter_ns()
    res = fn()
    elapsed_ns = time.perf_counter_ns() - start
    print(f"{year}\t{day}\t{part}\t{res}\t{elapsed_ns}")
    return res


if __name__ == "__main__":
    with open("2015-19.input") as f:
        input_file = [line.rstrip("\n") for line in f.readlines()]
    _test()
    run(lambda: part_1(input_file), part=1)
    # run(lambda: part_2(input_file), part=2)
