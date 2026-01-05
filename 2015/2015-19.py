#!/usr/bin/env python

from dataclasses import dataclass, field
import itertools
import heapq
import re
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

PAT_BYTES = re.compile(rb"[A-Z][a-z]?")
PAT_STR = re.compile(r"[A-Z][a-z]?")


def debug(*args, **kwargs):
    print("DEBUG:", *args, **kwargs)


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

    def to_token(self, s: str | bytes) -> Token:
        if isinstance(s, bytes):
            s = str(s, "ascii")
        return self._to_toks[s]

    def to_tokens(self, s: str | bytes) -> tuple[Token, ...]:
        if isinstance(s, bytes):
            s = str(s, "ascii")
        return tuple(self._to_toks[x.group()] for x in PAT_STR.finditer(s))


@dataclass(slots=True)
class Map:
    tokenizer: Tokenizer = field(repr=False)
    map: dict[Token, list[tuple[Token, ...]]] = field(default_factory=dict)
    _terminals: set[Token] = field(init=False)

    def __post_init__(self):
        print("MAP POST INIT")
        for vals in self.map.values():
            print("MAP POST INIT", vals)
            for val in vals:
                for v in val:
                    print(v in self.map)

    def add(self, key: str, val: str):
        self.map.setdefault(self.tokenizer.to_token(key), []).append(
            self.tokenizer.to_tokens(val)
        )

    def extend(self, pairs: Iterable[tuple[str, str]]):
        for key, val in pairs:
            self.add(key, val)

    def grammar(self) -> str:
        out = "Map Grammar = {\n"
        out += "\n".join(
            f"  {key} → {' | '.join(' '.join(str(x) for x in v) for v in vals)}"
            for key, vals in sorted(self.map.items())
        )
        out += "\n}"
        return out

    @property
    def terminals(self):
        if hasattr(self, "_terminals") and self._terminals:
            return self._terminals
        self._terminals = {
            v
            for vals in self.map.values()
            for val in vals
            for v in val
            if v not in self.map
        }
        return self._terminals

    def substitutions(
        self, src: Tokens, mapping: tuple[Token, Tokens]
    ) -> Generator[Tokens]:
        # debug(f"getting subs in {src}\n  for {mapping[0]} -> {mapping[1]}")
        from_, to_ = mapping
        # for i, tok in enumerate(src):
        for i in range(len(src)):
            tok = src[i]
            # for i, tok in enumerate(src):
            if tok == from_:
                # debug(f"subtitution at {i}")
                # yield src.replace(i, i + len(from_), to_)
                yield tuple(itertools.chain(src[:i], to_, src[i + 1 :]))

    def replacements(self, s: Tokens) -> Generator[Tokens]:
        # debug(f"finding replacements in {s}")
        for token in s:
            # debug(f"{token=}")
            if token not in self.map:
                continue
            for dst in self.map[token]:
                # debug(f"{token} => {dst}")
                yield from self.substitutions(s, (token, dst))

    def vocab_size(self) -> int:
        return len(self.tokenizer._to_toks)


def parse(input: Input) -> tuple[Map, Tokens]:
    "Returns map, molecule."
    mappings: list[tuple[str, str]] = []
    for line in input:
        if not line:
            break
        left, right = line.split(" => ", maxsplit=1)
        mappings.append((left, right))
    all_strs = itertools.chain((x[0] for x in mappings), (x[1] for x in mappings))
    all_tokens = set(itertools.chain(*(PAT_STR.findall(s) for s in all_strs)))
    tokenizer = Tokenizer(sorted(all_tokens))
    map = Map(tokenizer)
    map.extend(mappings)
    return map, tokenizer.to_tokens(input[-1])


def part_1(input: Input):
    map, molecule = parse(input)
    return len(set(map.replacements(molecule)))


# TODO: possible optimizations:
# - the b string is always the same, so we can reuse/reset distances list
# - use bytearray instead of strs (more likely to trigger memcpy, no unicode)
def string_distance(a: Tokens, b: Tokens) -> int:
    """Wagner-Fischer algorithm for Levenshtein distance using 1-d table."""

    return abs(len(b) - len(a))

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
    s: Tokens
    distance: int

    def __lt__(self, other: "QNode") -> bool:
        # return self.count < other.count
        # return (self.count, self.distance) < (other.count, other.distance)
        return (self.distance, self.count) < (other.distance, other.count)
        # return self.distance < other.distance


def search(map: Map, destination: Tokens) -> int:
    q = [QNode(0, (0,), 0)]
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
        for tok in node.s:
            if tok not in map.map:
                continue
            # for sub in {bytes(x) for x in map.replacements(node.s)}:
            for sub in set(map.replacements(node.s)):
                heapq.heappush(
                    q,
                    QNode(node.count + 1, sub, string_distance(sub, destination)),
                    # q, QNode(node.count + 1, sub, 0),
                )
    raise Exception("unreachable")


_cache: dict[bytes, list[tuple[bytes, int]]] = {}


def tr(tr_list: list[tuple[bytes, bytes]], s: bytes) -> Generator[tuple[bytes, int]]:
    if len(s) == 0:
        yield b"", 0
        return
    if len(s) == 1:
        yield s, 0
        return

    for from_, to_ in tr_list:
        if s[: len(from_)] == from_:
            for rest, cnt in tr(tr_list, s[len(from_) :]):
                yield to_ + rest, 1 + cnt
    for rest, cnt in tr(tr_list, s[1:]):
        yield s[:1] + rest, cnt


def tr_best(tr_list: list[tuple[bytes, bytes]], s: bytes) -> tuple[bytes, int]:
    if len(s) == 0:
        return b"", 0

    best = None

    for from_, to_ in tr_list:
        if s[: len(from_)] == from_:
            if best is None or len(from_) > len(best[0]):
                best = (from_, to_)
    if best is not None:
        from_, to_ = best
        rest, cnt = tr_best(tr_list, s[len(from_) :])
        return to_ + rest, cnt + 1
    else:
        rest, cnt = tr_best(tr_list, s[1:])
        return s[1:], cnt


PAT = re.compile(rb"[A-Z][a-z]?")


def rev_search(map: Map, destination: Tokens) -> int:
    tknzr = map.tokenizer
    tr_list: list[tuple[Tokens, Token]] = []
    for k, val in map.map.items():
        for v in val:
            tr_list.append((v, k))
    tr_list.sort(key=lambda x: (len(x[0]), x[0]), reverse=True)

    for to_, from_ in tr_list:
        if to_ in destination:
            print(f"{tknzr.to_str(from_)} => {tknzr.to_str(to_)}")

    terminal = set()
    for to_, from_ in tr_list:
        for e in to_:
            if e not in map.map:
                terminal.add(e)
    debug("terminal", terminal)

    for to_, from_ in tr_list:
        if any(t in to_ for t in terminal):
            print(f"{tknzr.to_str(from_)} => {tknzr.to_str(to_)} is terminal")

    for to_, from_ in tr_list:
        if any(t in to_ for t in terminal):
            if to_ in destination:
                print(
                    f"{tknzr.to_str(from_)} => {tknzr.to_str(to_)} leads to destination"
                )
    return 0

    assert isinstance(destination, bytes)
    q = [QNode(0, destination, 0)]

    i = 0
    for dest, total in itertools.islice(tr(tr_list, destination), 0, 10_000_000):
        i += 1
        if i % 1 == 0:
            print(i)
        n = dest
        while True:
            n, cnt = tr_best(tr_list, n)
            total += cnt
            if n == b"e":
                return total
            if cnt == 0 or n == b"":
                break
        print(total)

    # n = destination
    # total = 0
    # while True:
    #     n, cnt = tr_best(tr_list, n)
    #     print(n, cnt, total)
    #     if cnt == 0 or n == b"":
    #         break
    #     total += cnt
    # print(n, total)

    return 0


@dataclass(slots=True)
class PrefixNode:
    vocab_size: int
    value: Token | None = None
    # children: dict[Token, "PrefixNode"] = field(default_factory=dict)
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
        # node = self
        # for t in toks:
        #     if t not in node.children:
        #         node.children[t] = PrefixNode(t)
        #     node = node.children[t]
        node = self
        for t in key:
            if node._children[t] is None:
                node._children[t] = PrefixNode(self.vocab_size)
            node = node._children[t]
            assert node is not None
        node.value = value


def rev_tree(map: Map) -> PrefixNode:
    tree = PrefixNode(map.vocab_size())
    for key, vals in map.map.items():
        for val in vals:
            tree.insert(val, key)
    return tree


@dataclass(slots=True)
class TokenNode:
    into: Token
    from_: "Tokens | TokenNode"

    def __repr__(self) -> str:
        if isinstance(self.from_, tuple):
            return f"{self.into}: {self.from_}"
        return f"{self.into}: ({self.from_})"


@dataclass(slots=True)
class Parser:
    map: Map = field(repr=False)
    tokens: Tokens
    pos: int = 0
    rev_map: PrefixNode = field(init=False, repr=False)
    input: list[Token] = field(init=False)
    stack: list[Token] = field(default_factory=list)
    reductions: int = 0

    def __post_init__(self):
        self.rev_map = rev_tree(self.map)
        self.input = list(self.tokens)

    def shift_to_stack(self):
        self.stack.append(self.input.pop(0))

    def stack_reductions(self, index: int) -> Generator[tuple[Token, int]]:
        "Yields (token, match len)."
        t = self.stack[index]
        node = self.rev_map
        offset = 1
        while t is not None and t in node:
            node = node.get_child(t)
            assert node is not None
            if node.value is not None:
                yield (node.value, offset)

            if index + offset >= len(self.stack):
                break
            t = self.stack[index + offset]
            offset += 1

    def reduce_stack(self) -> int:
        "Return number of reductions."
        i = 0
        reductions_made = 0  # TODO: count globally
        while True:
            while i < len(self.stack):
                reductions = list(self.stack_reductions(i))
                # TODO: don't just grab first
                if not reductions:
                    i += 1
                    continue
                match, match_len = reductions[0]
                print("reduce", match, match_len)
                self.stack = self.stack[:i] + [match] + self.stack[i + match_len :]
                reductions_made += 1
                i += match_len
                break
            if reductions_made == 0:
                break
            else:
                self.reductions += reductions_made
                reductions_made = 0
        return reductions_made

    def parse(self):
        while self.input or self.stack:
            print("stack", self.stack)
            print("input", self.input)
            reduction_count = self.reduce_stack()
            if not self.input and reduction_count <= 0:
                raise SyntaxError("failed")
            if self.input:
                self.shift_to_stack()
            elif len(self.stack) == 1:
                return self.stack[0]
        raise SyntaxError("failed")


@dataclass(slots=True)
class BranchingParser:
    map: Map = field(repr=False)
    tokens: Tokens
    target: Token = 0
    pos: int = 0
    rev_map: PrefixNode = field(init=False, repr=False)
    input: list[Token] = field(init=False)
    stack: list[Token] = field(default_factory=list)
    reductions: int = 0

    def __post_init__(self):
        self.rev_map = rev_tree(self.map)
        self.input = list(self.tokens)

    def shift_to_stack(self):
        self.stack.append(self.input.pop(0))

    def stack_reductions(self, index: int) -> Generator[tuple[Token, int]]:
        "Yields (token, match len)."
        t = self.stack[index]
        node = self.rev_map
        offset = 1
        while t is not None and t in node:
            node = node.get_child(t)
            assert node is not None
            if node.value is not None:
                yield (node.value, offset)

            if index + offset >= len(self.stack):
                break
            t = self.stack[index + offset]
            offset += 1

    def reduce_stack(self) -> int:
        "Return number of reductions."
        i = 0
        start_reductions = self.reductions
        reductions_made = 0
        while True:
            while i < len(self.stack):
                reductions = list(self.stack_reductions(i))
                # TODO: don't just grab first
                if not reductions:
                    i += 1
                    continue
                match, match_len = reductions[0]
                print("reduce", match, match_len)
                self.stack = self.stack[:i] + [match] + self.stack[i + match_len :]
                reductions_made += 1
                i += match_len
                break
            if reductions_made == 0:
                break
            else:
                self.reductions += reductions_made
                reductions_made = 0
        return self.reductions - start_reductions

    def parse(self):
        while self.input or self.stack:
            if not self.input and len(self.stack) == 1:
                assert self.stack[0] == self.target
                return self.stack[0]
            reduction_count = self.reduce_stack()
            if not self.input and reduction_count <= 0:
                print(self)
                raise SyntaxError("failed")
            if self.input:
                self.shift_to_stack()
        if len(self.stack) == 1:
            return self.stack[0]
        print(self)
        raise SyntaxError("failed")


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
    assert_eq(
        len({str(s) for s in map.replacements(map.tokenizer.to_tokens("HOH"))}), 4
    )
    assert_eq(
        len({str(s) for s in map.replacements(map.tokenizer.to_tokens("HOHOHO"))}), 7
    )

    map, _ = parse(CONTROL_1)
    map.add("e", "H")
    map.add("e", "O")
    assert_eq(search(map, map.tokenizer.to_tokens("HOH")), 3)
    assert_eq(search(map, map.tokenizer.to_tokens("HOHOHO")), 6)
    # print("tests PASSED", file=sys.stderr)
    # assert_eq(part_2(CONTROL_1), 0)


def run(fn, year=2015, day=19, part=0):
    start = time.perf_counter_ns()
    res = fn()
    elapsed_ns = time.perf_counter_ns() - start
    print(f"{year}\t{day}\t{part}\t{res}\t{elapsed_ns}")
    return res


if __name__ == "__main__":
    with open("2015-19.input", "r") as f:
        # input_file = [line.rstrip(b"\n") for line in f.readlines()]
        input_file = f.read().splitlines()
    _test()
    run(lambda: part_1(input_file), part=1)
    run(lambda: part_2(input_file), part=2)

    ###### PLAYGROUND ######
    # map, molecule = parse(input_file)
    # print(map.map)
    # # print(molecule)
    # # reduced = molecule
    # # for _ in range(10):
    # #     reduced = tuple(parse_molecule(map, reduced))
    # #     print(len(reduced), reduced)
    # parser = Parser(map, molecule)
    # print(parser)
    # tree = rev_tree(map)
    # print(tree)
    # print("-" * 40)
    # print(map.tokenizer.to_str(molecule))
    # print("-" * 40)
    # print(molecule)
    # print("-" * 40)
    # print(tree)
    # print("-" * 40)
    # print(map.tokenizer._to_strs)
    # print(tree.get_child(4))
    # print(tree.get_child(4).get_child(12))
    # print([i for i, _ in tree.get_child(4).get_child(12).children()])
    #
    # parser = Parser(map, molecule)
    # # print(parser.parse())
    #
    # print(map.grammar())
    # print(map.terminals)
    #
    # ca = map.tokenizer.to_token("Ca")
    # parser = BranchingParser(map, (ca, ca), target=ca)
    # parser.parse()
