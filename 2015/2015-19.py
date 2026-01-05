#!/usr/bin/env python

from dataclasses import dataclass, field
import itertools
import heapq
import re
from typing import Generator, Iterable, LiteralString
import time

Input = list[str] | list[LiteralString]

PAT = re.compile(r"[A-Z][a-z]?")


type Token = int
type Tokens = tuple[Token, ...]


@dataclass(slots=True)
class Tokenizer:
    """Elments to int tokens."""

    _to_toks: dict[str, int]
    _to_strs: dict[int, str]
    i: int = 1

    def __init__(self, vocab: Iterable[str]) -> None:
        self.i = 1
        self._to_toks = {"e": 0}
        self._to_strs = {0: "e"}
        for element in vocab:
            self.add(element)

    def add(self, element: str):
        if element in self._to_toks:
            return
        self._to_toks[element] = self.i
        self._to_strs[self.i] = element
        self.i += 1

    def to_str(self, toks: Token | Iterable[Token]) -> str:
        if isinstance(toks, int):
            return self._to_strs[toks]
        return "".join(self._to_strs[t] for t in toks)

    def to_token(self, s: str) -> Token:
        return self._to_toks[s]

    def to_tokens(self, s: str) -> tuple[Token, ...]:
        return tuple(self._to_toks[x.group()] for x in PAT.finditer(s))


@dataclass(slots=True)
class Map:
    tokenizer: Tokenizer = field(repr=False)
    map: dict[Token, list[tuple[Token, ...]]] = field(default_factory=dict)

    def add(self, key: str, val: str):
        self.map.setdefault(self.tokenizer.to_token(key), []).append(
            self.tokenizer.to_tokens(val)
        )

    def extend(self, pairs: Iterable[tuple[str, str]]):
        for key, val in pairs:
            self.add(key, val)

    def __contains__(self, val: Token) -> bool:
        return val in self.map

    def __getitem__(self, key: Token) -> list[tuple[Token, ...]]:
        return self.map[key]


def substitutions(src: Tokens, mapping: tuple[Token, Tokens]) -> Generator[Tokens]:
    from_, to_ = mapping
    for i in range(len(src)):
        tok = src[i]
        if tok == from_:
            yield tuple(itertools.chain(src[:i], to_, src[i + 1 :]))


def replacements(map: Map, molecule: Tokens) -> Iterable[Tokens]:
    for token in molecule:
        if token not in map:
            continue
        for dst in map[token]:
            yield from substitutions(molecule, (token, dst))


def parse(input: Input) -> tuple[Map, Tokens]:
    "Returns map, molecule."
    mappings: list[tuple[str, str]] = []
    for line in input:
        if not line:
            break
        left, right = line.split(" => ", maxsplit=1)
        mappings.append((left, right))
    all_strs = itertools.chain((x[0] for x in mappings), (x[1] for x in mappings))
    all_tokens = set(itertools.chain(*(PAT.findall(s) for s in all_strs)))
    tokenizer = Tokenizer(sorted(all_tokens))
    map = Map(tokenizer)
    map.extend(mappings)
    return map, tokenizer.to_tokens(input[-1])


def part_1(input: Input):
    map, molecule = parse(input)
    return len(set(replacements(map, molecule)))


def parse_molecule(map: Map, toks: Tokens) -> int:
    tr_list: list[tuple[Tokens, Token]] = []
    for k, val in map.map.items():
        for v in val:
            tr_list.append((v, k))
    tr_list.sort(key=lambda x: (len(x[0]), x[0]), reverse=True)

    q = [(len(toks), toks, 0)]

    while q:
        _, molecule, replacements = heapq.heappop(q)
        next_count = replacements + 1
        for from_, to_ in tr_list:
            for i in range(len(toks) - len(from_)):
                if molecule[i : i + len(from_)] == from_:
                    reduced = molecule[:i] + (to_,) + molecule[i + len(from_) :]
                    if reduced == (0,):
                        return next_count
                    heapq.heappush(q, (len(reduced), reduced, next_count))
    assert not "unreachable"
    return -1


def part_2(input: Input):
    map, molecule = parse(input)
    return parse_molecule(map, molecule)


def run(fn, year=2015, day=19, part=0):
    start = time.perf_counter_ns()
    res = fn()
    elapsed_ns = time.perf_counter_ns() - start
    print(f"{year}\t{day}\t{part}\t{res}\t{elapsed_ns}")
    return res


if __name__ == "__main__":
    with open("2015-19.input", "r") as f:
        input_file = f.read().splitlines()
    run(lambda: part_1(input_file), part=1)
    run(lambda: part_2(input_file), part=2)
