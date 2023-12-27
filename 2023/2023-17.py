#!/usr/bin/env python
from collections import defaultdict
from collections.abc import Generator
from dataclasses import dataclass, field
from typing import (
    Any,
    DefaultDict,
    Deque,
    Dict,
    Iterator,
    Literal,
    NamedTuple,
    Optional,
    Self,
    Tuple,
)
import heapq
from enum import CONFORM, CONTINUOUS, UNIQUE, Enum, IntEnum, ReprEnum, auto, verify
from colored import Fore, Style
from itertools import islice, pairwise
import pyrsistent
from pyrsistent import PSet
from pprint import pp

BIG_NUM = 1_000_000_000

# Each city block is marked by a single digit that represents the amount of
# heat loss if the crucible enters that block.
# The crucible can move at most three blocks in a single direction before it
# must turn 90 degrees left or right. The crucible also can't reverse
# direction; after entering each city block, it may only turn left, continue
# straight, or turn right.
CONTROL_1 = """\
2413432311323
3215453535623
3255245654254
3446585845452
4546657867536
1438598798454
4457876987766
3637877979653
4654967986887
4564679986453
1224686865563
2546548887735
4322674655533
""".splitlines()

STRAIGHTS = 3

with open("2023-17.input") as f:
    input_file = [line.strip() for line in f.readlines()]

Pos = NamedTuple("Pos", [("i", int), ("j", int)])
Pos.__repr__ = lambda x: f"({x.i}, {x.j})"


def dist(a: Pos, b: Pos) -> int:
    return abs(b.i - a.i) + abs(b.j - a.j)


Input = tuple[tuple[int, ...], ...]


@verify(UNIQUE)
class D(str, ReprEnum):
    UP = "^"
    DOWN = "v"
    LEFT = "<"
    RIGHT = ">"
    INVALID = "X"

    @classmethod
    def detect(cls, prev: Pos, curr: Pos) -> str:
        if curr.i - prev.i > 0:
            return cls.RIGHT
        if curr.i - prev.i < 0:
            return cls.LEFT
        if curr.j - prev.j > 0:
            return cls.DOWN
        if curr.j - prev.j < 0:
            return cls.UP
        return cls.INVALID


def print_input(input: Input):
    for line in input:
        print("".join(str(x) for x in line))


def up(coord) -> Pos:
    return Pos(coord.i, coord.j - 1)


def right(coord) -> Pos:
    return Pos(coord.i + 1, coord.j)


def down(coord) -> Pos:
    return Pos(coord.i, coord.j + 1)


def left(coord) -> Pos:
    return Pos(coord.i - 1, coord.j)


def in_bounds(rows: int, cols: int, coord: Pos) -> bool:
    return 0 <= coord.i < cols and 0 <= coord.j < rows


def expand_coord(input: Input, coord: Pos) -> list[Pos]:
    rows = len(input)
    cols = len(input[0])
    next_dirs = [right(coord), down(coord), left(coord), up(coord)]
    next_dirs = [d for d in next_dirs if in_bounds(rows, cols, d)]
    return next_dirs


def retrace_path(prev, start: Pos):
    yield start
    n = prev[start.j][start.i]
    while n != Pos(-1, -1):
        yield n
        n = prev[n.j][n.i]


def back_track(prev: list[list[Pos]], curr: Pos, count=3) -> list[Pos]:
    rev = [curr]
    p = prev[curr.j][curr.i]
    i = count
    while p != Pos(-1, -1) and i > 0:
        rev.append(p)
        curr = p
        p = prev[curr.j][curr.i]
        i -= 1
    rev.reverse()
    return rev


def all_straight(coords: list[Pos]) -> bool:
    first = coords[0]
    return all(x.i == first.i for x in coords[1:]) or all(
        x.j == first.j for x in coords[1:]
    )


assert all_straight([Pos(1, 4), Pos(1, 3), Pos(1, 2), Pos(1, 1)])


def lerp(a: Pos, b: Pos) -> Iterator[Pos]:
    if a.i == b.i:
        step = 1 if a.j < b.j else -1
        for y in range(a.j, b.j, step):
            yield Pos(a.i, y)
    if a.j == b.j:
        step = 1 if a.i < b.i else -1
        for x in range(a.i, b.i, step):
            yield Pos(x, a.j)


def visualize_path(input, path: list[Pos]):
    plots = 0
    print(f"{Fore.blue}{'-'*len(input[0])}{Style.reset}")
    pathviz: list[list[str]] = [
        [f"{Fore.black}{c}{Style.reset}" for c in row] for row in input
    ]
    p = Pos(0, 0)
    for n in path[1:]:
        dir = D.detect(p, n)
        pathviz[n.j][n.i] = dir
        interpolated = list(lerp(p, n))[1:]
        plots += 1
        for inter in interpolated:
            plots += 1
            pathviz[inter.j][inter.i] = dir
        p = n
    for row in pathviz:
        print("".join(row))
    print(f"{Fore.blue}{'-'*len(input[0])}{Style.reset}")
    print("printed", plots, "plots")


@dataclass
class HeapItem:
    total_heat: int
    coord: Pos
    input_dir: Optional[D] = None
    v: int = 0

    def __lt__(self, other):
        return self.total_heat < other.total_heat


def neighbors(coord: Pos) -> list[Pos]:
    return [right(coord), down(coord), left(coord), up(coord)]


def expand_to_depth(
    input: Input,
    unvisited: set[Pos],
    prev: list[Pos],
    curr: Pos,
    rem: int,
    depth: int,
) -> Generator[list[Pos], Any, Any]:
    rows = len(input)
    cols = len(input[0])
    dir = D.detect(prev[-1], curr)

    def is_ok(dir):
        return in_bounds(rows, cols, dir) and dir in unvisited

    neighbors = right(curr), down(curr), left(curr), up(curr)
    neighbors = [d for d in neighbors if is_ok(d)]
    if depth <= 0:
        yield [curr]
        return
    for nbr in neighbors:
        if nbr in prev:
            continue
        next_dir = D.detect(curr, nbr)
        next_rem = rem - 1 if next_dir == dir else STRAIGHTS
        if next_rem <= 0:
            continue
        for sub in expand_to_depth(
            input, unvisited, prev + [curr], nbr, next_rem, depth - 1
        ):
            yield [curr] + sub


def eval_path(input: Input, path: list[Pos]) -> int:
    return sum(input[c.j][c.i] for c in path)


QItemKey = Tuple[Pos, D]


@dataclass
class QItem:
    pos: Pos
    dir: D
    total_heat: int
    v: int = 0

    def key(self):
        return (self.pos, self.dir)

    def __lt__(self, other):
        return self.total_heat < other.total_heat


class Q:
    q_set: set[QItemKey]
    q: list[QItem]
    versions: dict[QItemKey, int]

    def __init__(self):
        self.versions = dict()
        self.q_set = set()
        self.q = []

    def __contains__(self, key: QItemKey):
        return key in self.q_set

    def __bool__(self) -> bool:
        return len(self.q) > 0

    def __len__(self) -> int:
        return len(self.q)

    def expected_version(self, item: QItem) -> int:
        return self.versions.get(item.key(), 0)

    def has_latest_version(self, item: QItem) -> bool:
        return self.expected_version(item) == item.v

    def increment_version(self, item: QItem):
        next_v = self.expected_version(item) + 1
        item.v = next_v
        self.versions[item.key()] = next_v

    def pop(self) -> Optional[QItem]:
        while self.q:
            item = heapq.heappop(self.q)
            if not item:
                # Queue is empty
                return None
            if item.v != self.expected_version(item):
                continue
            try:
                self.q_set.remove(item.key())
            except Exception:
                print("WARNING: could not remove item from set")
            return item

    def push(self, node: QItem):
        self.q_set.add(node.key())
        heapq.heappush(self.q, node)

    def update(self, item: QItem):
        self.increment_version(item)
        self.push(item)

    def reheap(self):
        heapq.heapify(self.q)


def blue_line(length: int):
    print(f"{Fore.blue}{'━'*length}{Style.reset}")


class Field:
    def __init__(self, input: tuple[tuple[int, ...], ...]):
        self.input = input
        self.rows = len(input)
        self.cols = len(input[0])

    def in_bounds(self, pos: Pos) -> bool:
        return 0 <= pos.i < self.cols and 0 <= pos.j < self.rows

    def get(self, pos: Pos) -> int:
        return self.input[pos.j][pos.i]


def part_1(input):
    input = tuple(tuple(int(x) for x in line) for line in input)
    field = Field(input)
    blue_line(field.cols)
    print_input(input)
    blue_line(field.cols)

    def next_nodes(in_dir: D, pos: Pos) -> Iterator[tuple[Pos, D, int]]:
        if in_dir == D.RIGHT or in_dir == D.LEFT:
            up_heat = 0
            for i in range(1, 4):
                p = Pos(pos.i, pos.j - i)
                if not field.in_bounds(p):
                    break
                up_heat += field.get(p)
                yield p, D.UP, up_heat
            down_heat = 0
            for i in range(1, 4):
                p = Pos(pos.i, pos.j + i)
                if not field.in_bounds(p):
                    break
                down_heat += field.get(p)
                yield p, D.DOWN, down_heat
        if in_dir == D.UP or in_dir == D.DOWN:
            left_heat = 0
            for i in range(1, 4):
                p = Pos(pos.i - i, pos.j)
                if not field.in_bounds(p):
                    break
                left_heat += field.get(p)
                yield p, D.LEFT, left_heat
            right_heat = 0
            for i in range(1, 4):
                p = Pos(pos.i + i, pos.j)
                if not field.in_bounds(p):
                    break
                right_heat += field.get(p)
                yield p, D.RIGHT, right_heat

    def expand(field) -> dict[tuple[Pos, D], set[tuple[Pos, D, int]]]:
        adj_list = DefaultDict(set)
        for j in range(field.rows):
            for i in range(field.cols):
                pos = Pos(i, j)
                for dir in (D.DOWN, D.RIGHT, D.LEFT, D.UP):
                    for n in next_nodes(dir, pos):
                        adj_list[pos, dir].add(n)
        return adj_list

    def dijkstra():
        start = Pos(0, 0)
        adj_list = expand(field)
        # q = list(adj_list.keys())
        q = Q()
        for pos, dir in adj_list.keys():
            q.push(QItem(pos, dir, BIG_NUM))
        q.update(QItem(start, D.DOWN, 0))
        q.update(QItem(start, D.RIGHT, 0))
        print("POPULATED QUEUE, EXPLORING")
        heat: DefaultDict[tuple[Pos, D], int] = defaultdict(lambda: BIG_NUM)
        heat[start, D.RIGHT] = 0
        heat[start, D.DOWN] = 0
        prev: dict[tuple[Pos, D], tuple[Pos, D]] = {}
        while q:
            item = q.pop()
            if item is None:
                break
            # print(item)
            pos = item.pos
            dir = item.dir
            for npos, ndir, nheat in adj_list[pos, dir]:
                if (npos, ndir) not in q:
                    continue
                alt = heat[pos, dir] + nheat
                if alt < heat[npos, ndir]:
                    q.update(QItem(npos, ndir, alt))
                    heat[npos, ndir] = alt
                    prev[npos, ndir] = pos, dir
        return heat, prev

    heat, prev = dijkstra()

    def retrace(prev: dict[tuple[Pos, D], tuple[Pos, D]], end: Pos, end_dir: D):
        yield end
        n = prev.get((end, end_dir))
        while n is not None:
            yield n[0]
            n = prev.get(n)

    def interpolated_path(path) -> Iterator[Pos]:
        for p, n in pairwise(path):
            yield from lerp(p, n)
        yield path[-1]

    # pp(heat)
    # pp(prev)
    start = Pos(0, 0)
    end = Pos(field.cols - 1, field.rows - 1)
    for d in (D.RIGHT, D.DOWN):
        path = list(retrace(prev, end, d))
        path.reverse()
        path = list(interpolated_path(path))
        print(path)
        visualize_path(input, path)
        total_heat = sum(input[p.j][p.i] for p in path[1:])
        print("path len", len(path[1:]))
        print(f"{total_heat=}")


if __name__ == "__main__":
    part_1(input_file)
