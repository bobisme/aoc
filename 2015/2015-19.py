#!/usr/bin/env python

from collections import deque
from dataclasses import dataclass, field
import sys
from typing import Generator, Iterable, LiteralString
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
        out = list(s)
        i = 0
        while i < len(s):
            found = False
            for size in range(self.max_token_len, 1, -1):
                if (chunk := "".join(out[i : i + size])) in self.tokens:
                    yield chunk
                    found = True
                    i += size
                    break
            if not found:
                yield out[i]
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
class Map:
    tokenizer: Tokenizer = field(repr=False)
    map: dict[Token, list[tuple[Token, ...]]] = field(default_factory=dict)

    def add(self, key: str, val: str):
        self.map.setdefault(key, []).append(tuple(self.tokenizer.tokenize(val)))

    def extend(self, pairs: Iterable[tuple[str, str]]):
        for key, val in pairs:
            self.add(key, val)

    def substitute(
        self, src: list[Token], mapping: tuple[Token, tuple[Token, ...]]
    ) -> list[Token]:
        out: list[Token | tuple[Token, ...]] = [x for x in src]
        from_, to_ = mapping
        for i, tok in enumerate(out):
            if tok == from_:
                out[i] = to_
        return flatten(out)

    def substitutions(
        self, src: list[Token], mapping: tuple[Token, tuple[Token, ...]]
    ) -> Generator[list[Token]]:
        from_, to_ = mapping
        for i, tok in enumerate(src):
            if tok == from_:
                out: list[Token | tuple[Token, ...]] = [x for x in src]
                out[i] = to_
                yield flatten(out)

    def replacements(self, s: str) -> Generator[str]:
        src_tokens = list(self.tokenizer.tokenize(s))
        for token in src_tokens:
            dst_tokens = self.map.get(token, [])
            for dst in dst_tokens:
                for subst in self.substitutions(src_tokens, (token, dst)):
                    yield "".join(subst)


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


def search(map: Map, destination: str) -> int:
    q = deque([("e", 0)])
    while q:
        curr, sub_count = q.popleft()
        if curr == destination:
            return sub_count
        for tok in map.tokenizer.tokenize(curr):
            if tok in map.map:
                for sub in map.replacements(curr):
                    q.append((sub, sub_count + 1))
    raise Exception("unreachable")


def part_2(input: Input):
    map, molecule = parse(input)
    print("STARTING PART 2")
    search(map, molecule)


def _test(input_file):
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
    _test(input_file)
    run(lambda: part_1(input_file), part=1)
    run(lambda: part_2(input_file), part=2)
