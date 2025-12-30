#!/usr/bin/env python

from dataclasses import dataclass, field
import heapq
from typing import Generator, Iterable, Iterator, LiteralString, cast
import time

Input = list[str] | list[LiteralString] | list[bytes]

CONTROL_1: Input = (
    b"""\
H => HO
H => OH
O => HH
""".splitlines()
)


def debug(*args, **kwargs):
    print("DEBUG:", *args, **kwargs)


@dataclass(slots=True)
class RopeNode[Str: str | bytes | bytearray]:
    left_len: int = field(repr=False)  # total length of left
    s: Str | None = None
    left: "RopeNode[Str] | None" = None
    right: "RopeNode[Str] | None" = None

    def __len__(self) -> int:
        if self.s is not None:
            return len(self.s)
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

    def __iter__(self) -> Iterator[Str]:
        if self.s is not None:
            if len(self.s) == 0:
                return
            if isinstance(self.s, str):
                yield from cast(Iterator[Str], self.s)
            else:
                for i in range(len(self.s)):
                    yield cast(Str, self.s[i : i + 1])
        else:
            assert self.left is not None and self.right is not None
            yield from self.left
            yield from self.right

    def __getitem__(self, key: int | slice) -> Str:
        if isinstance(key, slice):
            if self.s is not None:
                return cast(Str, self.s[key])
            assert self.left is not None and self.right is not None
            if key.start < self.left_len:
                if key.stop <= self.left_len:
                    return self.left[key]
                else:
                    left_slice = self.left[key.start :]
                    right_slice = self.right[: key.stop - self.left_len]
                    if isinstance(left_slice, str):
                        return cast(Str, left_slice + cast(str, right_slice))
                    if isinstance(left_slice, bytes):
                        return cast(Str, left_slice + cast(bytes, right_slice))
                    else:
                        return cast(Str, left_slice + cast(bytearray, right_slice))
            return self.right[key.start - self.left_len : key.stop - self.left_len]
        else:
            if self.s is not None:
                return self[key : key + 1]
            assert self.left is not None and self.right is not None
            if key < self.left_len:
                return self.left[key]
            return self.right[key - self.left_len]

    def __eq__(self, other: object, /) -> bool:
        assert isinstance(other, RopeNode)
        if len(self) != len(other):
            return False
        for a, b in zip(self, other):
            if a != b:
                return False
        return True

    def __add__(self, other: "RopeNode") -> "RopeNode":
        return RopeNode.concat(self, other)

    @staticmethod
    def concat(r1: "RopeNode", r2: "RopeNode") -> "RopeNode":
        return RopeNode(left=r1, right=r2, left_len=len(r1))

    def split(self, idx: int) -> tuple["RopeNode", "RopeNode"]:
        """
        Given s = "abcdefg", idx = 3: ("abc", "defg")
        """
        if self.s is not None:
            left = self.s[:idx]
            right = self.s[idx:]
            return RopeNode(len(left), s=left), RopeNode(len(right), s=right)

        assert self.left is not None and self.right is not None
        if idx < self.left_len:
            (l1, l2) = self.left.split(idx)
            return l1, RopeNode.concat(l2, self.right)

        (r1, r2) = self.right.split(idx - self.left_len)
        return RopeNode.concat(self.left, r1), r2

    def insert(self, idx: int, s: str) -> "RopeNode":
        insert = RopeNode(0, s=s)
        left, right = self.split(idx)
        return RopeNode.concat(RopeNode.concat(left, insert), right)

    def replace(self, start: int, end: int, s: Str) -> "RopeNode":
        insert = RopeNode(0, s=s)
        left, _ = self.split(start)
        _, right = self.split(end)
        return RopeNode.concat(RopeNode.concat(left, insert), right)


def rope(s: str | bytes | bytearray) -> RopeNode:
    return RopeNode(0, s=s)


def __test_rope():
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
        r5 = r.replace(6, 11, "dude")
        assert str(r5) == "hello dude", str(r5)
        if isinstance(s[3:5], str):
            assert r[3:5] == "lo"
        else:
            assert str(r[3:5], "ascii") == "lo", r[3:5]
    r = rope("abcdefgh")
    left, right = RopeNode.split(r, 3)
    assert str(left) == "abc"
    assert str(right) == "defgh"
    r2 = RopeNode.concat(*r.split(3))
    assert str(r2[1:5]) == "bcde"
    new_root = RopeNode(left_len=len(left), left=left, right=right)
    assert str(new_root) == "abcdefgh"
    r3 = new_root.replace(2, 5, "!!!")
    assert str(r3) == "ab!!!fgh"


__test_rope()


type Token = str | bytes


@dataclass(slots=True)
class Tokenizer[Tok: Token, Str: str | bytes | bytearray]:
    # TODO: maybe do byte pair encoding
    tokens: set[Tok]
    max_token_len: int

    def __init__(self, known_tokens: Iterable[Tok]):
        # self.tokens = sorted(known_tokens, key=lambda x: len(x), reverse=True)
        self.tokens = {x for x in known_tokens if len(x) > 1}
        if self.tokens:
            self.max_token_len = max(len(x) for x in self.tokens)
        else:
            self.max_token_len = 1

    def token(self, x: Str) -> Tok:
        if isinstance(x, bytearray):
            return cast(Tok, bytes(x))
        return cast(Tok, x)

    def scan_tokens(self, s: Str | RopeNode[Str]) -> Generator[Tok]:
        # debug(f"scanning {s=}")
        i = 0
        while i < len(s):
            found = False
            for size in range(self.max_token_len, 1, -1):
                chunk = self.token(cast(Str, s[i : i + size]))
                # debug(f"{chunk=}")
                if chunk in self.tokens:
                    yield chunk
                    found = True
                    i += size
                    break
            if not found:
                c = self.token(cast(Str, s[i : i + 1]))
                # debug(f"{c=}")
                yield c
                i += 1


# Tokenizer Tests
def _test_tokenizer():
    t = Tokenizer(("ab", "def"))
    r = rope("abcdefgh")
    l_ = list(t.scan_tokens(r))
    assert l_ == ["ab", "c", "def", "g", "h"]


_test_tokenizer()


@dataclass(slots=True)
class Map[Tok: Token, Str: str | bytes | bytearray]:
    tokenizer: Tokenizer[Tok, Str] = field(repr=False)
    map: dict[Tok, list[Str]] = field(default_factory=dict)

    def add(self, key: Str, val: Str):
        self.map.setdefault(self.tokenizer.token(key), []).append(val)

    def extend(self, pairs: Iterable[tuple[Str, Str]]):
        for key, val in pairs:
            self.add(key, val)

    def substitutions(
        self, src: RopeNode[Str], mapping: tuple[Tok, Str]
    ) -> Generator[RopeNode[Str]]:
        # debug(f"getting subs in {src}\n  for {mapping[0]} -> {mapping[1]}")
        from_, to_ = mapping
        # for i, tok in enumerate(src):
        for i in range(len(src) - (len(from_)) + 1):
            tok = src[i : i + len(from_)]
            # for i, tok in enumerate(src):
            if tok == from_:
                # debug(f"subtitution at {i}")
                yield src.replace(i, i + len(from_), to_)

    def replacements(
        self, s: RopeNode | str | bytes | bytearray
    ) -> Generator[RopeNode]:
        if not isinstance(s, RopeNode):
            s = rope(s)
        # debug(f"finding replacements in {s}")
        for token in self.tokenizer.scan_tokens(s):
            # debug(f"{token=}")
            if token not in self.map:
                continue
            for dst in self.map[token]:
                # debug(f"{token} => {dst}")
                yield from self.substitutions(s, (token, dst))


def parse(input: Input) -> tuple[Map, bytes]:
    "Returns map, molecule."
    mappings: list[tuple[bytes, bytes]] = []
    for line in input:
        assert isinstance(line, bytes)
        if not line:
            break
        left, right = line.split(b" => ", maxsplit=1)
        mappings.append((left, right))
    tokenizer = Tokenizer({x[0] for x in mappings})
    map = Map(tokenizer)
    map.extend(mappings)
    return map, cast(bytes, input[-1])


def part_1(input: Input):
    map, molecule = parse(input)
    return len({str(r) for r in map.replacements(molecule)})


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

    def print_thing(map: Map, s):
        print(f"src = {s}:")
        repls = list(map.replacements(s))
        for x in repls:
            print(x)
        set_ = set(repls)
        print("\nset")
        for x in set_:
            print(f"'{x}'", len(x))

    map, _ = parse(CONTROL_1)
    # print(map)
    # print_thing(map, b"HOH")
    assert_eq(len({str(s) for s in map.replacements(rope(b"HOH"))}), 4)
    assert_eq(len({str(s) for s in map.replacements(rope(b"HOHOHO"))}), 7)

    map.add("e", "H")
    map.add("e", "O")
    # assert_eq(search(map, "HOH"), 3)
    # assert_eq(search(map, "HOHOHO"), 6)
    # print("tests PASSED", file=sys.stderr)
    # assert_eq(part_2(CONTROL_1), 0)


def run(fn, year=2015, day=19, part=0):
    start = time.perf_counter_ns()
    res = fn()
    elapsed_ns = time.perf_counter_ns() - start
    print(f"{year}\t{day}\t{part}\t{res}\t{elapsed_ns}")
    return res


if __name__ == "__main__":
    with open("2015-19.input", "rb") as f:
        # input_file = [line.rstrip(b"\n") for line in f.readlines()]
        input_file = f.read().splitlines()
    _test()
    run(lambda: part_1(input_file), part=1)
    # run(lambda: part_2(input_file), part=2)
