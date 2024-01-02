#!/usr/bin/env python
from collections import defaultdict
from dataclasses import dataclass
from typing import (
    DefaultDict,
    Iterator,
    NamedTuple,
    Optional,
    Self,
    Tuple,
)
import heapq
from enum import UNIQUE, ReprEnum, verify
from colored import Fore, Style
from itertools import pairwise

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
CONTROL_2 = """\
111111111111
999999999991
999999999991
999999999991
999999999991
""".splitlines()

with open("2023-17.input") as f:
    input_file = [line.strip() for line in f.readlines()]


class Pos(NamedTuple("Pos", [("i", int), ("j", int)])):
    def __repr__(self: Self) -> str:
        return f"({self.i}, {self.j})"


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


@verify(UNIQUE)
class BinD(str, ReprEnum):
    HORIZONTAL = "-"
    VERTICAL = "|"


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


def lerp(a: Pos, b: Pos) -> Iterator[Pos]:
    if a.i == b.i:
        step = 1 if a.j < b.j else -1
        for y in range(a.j, b.j, step):
            yield Pos(a.i, y)
    if a.j == b.j:
        step = 1 if a.i < b.i else -1
        for x in range(a.i, b.i, step):
            yield Pos(x, a.j)


def draw_line(width=10, color=Fore.blue, char="━"):
    print(f"{color}{char*width}{Style.reset}")


def visualize_path(input, path: list[Pos]):
    draw_line(width=len(input[0]))
    pathviz: list[list[str]] = [
        [f"{Fore.black}{c}{Style.reset}" for c in row] for row in input
    ]
    p = Pos(0, 0)
    for n in path[1:]:
        dir = D.detect(p, n)
        pathviz[n.j][n.i] = dir
        interpolated = list(lerp(p, n))[1:]
        for inter in interpolated:
            pathviz[inter.j][inter.i] = dir
        p = n
    for row in pathviz:
        print("".join(row))
    draw_line(width=len(input[0]))


QItemKey = Tuple[Pos, BinD]


@dataclass
class QItem:
    pos: Pos
    dir: BinD
    total_heat: int
    v: int = 0

    def key(self) -> tuple[Pos, BinD]:
        return (self.pos, self.dir)

    def __lt__(self, other):
        return self.total_heat < other.total_heat


class Q:
    q_set: DefaultDict[QItemKey, int]
    q: list[QItem]

    def __init__(self):
        self.q_set = DefaultDict(int)
        self.q = []

    def __contains__(self, key: QItemKey):
        return self.q_set.get(key, 0) > 0

    def pop(self) -> Optional[QItem]:
        if not self.q:
            return None
        item = heapq.heappop(self.q)
        self.q_set[item.key()] -= 1
        return item

    def push(self, item: QItem):
        self.q_set[item.key()] += 1
        heapq.heappush(self.q, item)

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

    def next_nodes(
        in_dir: BinD, pos: Pos, min_dist, max_dist
    ) -> Iterator[tuple[Pos, BinD, int]]:
        def yield_dir(dir: D):
            heat = 0
            for i in range(min_dist, max_dist + 1):
                if dir == D.UP:
                    p = Pos(pos.i, pos.j - i)
                elif dir == D.DOWN:
                    p = Pos(pos.i, pos.j + i)
                elif dir == D.LEFT:
                    p = Pos(pos.i - i, pos.j)
                elif dir == D.RIGHT:
                    p = Pos(pos.i + i, pos.j)
                if dir in (D.UP, D.DOWN):
                    bin_dir = BinD.VERTICAL
                else:
                    bin_dir = BinD.HORIZONTAL
                if not field.in_bounds(p):
                    break
                heat += field.get(p)
                yield p, bin_dir, heat

        if in_dir == BinD.HORIZONTAL:
            yield from yield_dir(D.UP)
            yield from yield_dir(D.DOWN)
        else:
            yield from yield_dir(D.RIGHT)
            yield from yield_dir(D.LEFT)

    def expand(
        field: Field, min_dist, max_dist
    ) -> dict[tuple[Pos, BinD], set[tuple[Pos, BinD, int]]]:
        adj_list = DefaultDict(set)
        for j in range(field.rows):
            for i in range(field.cols):
                pos = Pos(i, j)
                for dir in (BinD.VERTICAL, BinD.HORIZONTAL):
                    for n in next_nodes(dir, pos, min_dist, max_dist):
                        adj_list[pos, dir].add(n)
        return adj_list

    def dijkstra(min_dist, max_dist):
        start = Pos(0, 0)
        adj_list = expand(field, min_dist, max_dist)
        q = Q()
        q.push(QItem(start, BinD.VERTICAL, 0))
        q.push(QItem(start, BinD.HORIZONTAL, 0))
        for pos, dir in adj_list.keys():
            if pos == start:
                continue
            q.push(QItem(pos, dir, BIG_NUM))
        print("POPULATED QUEUE, EXPLORING")
        heat: DefaultDict[tuple[Pos, BinD], int] = defaultdict(lambda: BIG_NUM)
        heat[start, BinD.HORIZONTAL] = 0
        heat[start, BinD.VERTICAL] = 0
        prev: dict[tuple[Pos, BinD], tuple[Pos, BinD]] = {}
        while item := q.pop():
            pos = item.pos
            dir = item.dir
            qheat = item.total_heat
            curr_heat = heat[pos, dir]
            if qheat > curr_heat:
                continue
            for npos, ndir, nheat in adj_list[pos, dir]:
                if (npos, ndir) not in q:
                    continue
                alt = curr_heat + nheat
                if alt < heat[npos, ndir]:
                    q.push(QItem(npos, ndir, alt))
                    heat[npos, ndir] = alt
                    prev[npos, ndir] = pos, dir
        return heat, prev

    def retrace(
        prev: dict[tuple[Pos, BinD], tuple[Pos, BinD]], end: Pos, end_dir: BinD
    ):
        yield end
        n = prev.get((end, end_dir))
        while n is not None:
            yield n[0]
            n = prev.get(n)

    def interpolated_path(path) -> Iterator[Pos]:
        for p, n in pairwise(path):
            yield from lerp(p, n)
        yield path[-1]

    heat, prev = dijkstra(min_dist=4, max_dist=10)
    end = Pos(field.cols - 1, field.rows - 1)

    min_heat = BIG_NUM
    min_path = []
    for d in (BinD.HORIZONTAL, BinD.VERTICAL):
        path = list(retrace(prev, end, d))
        path.reverse()
        path = list(interpolated_path(path))
        total_heat = sum(input[p.j][p.i] for p in path[1:])
        if total_heat < min_heat:
            min_heat = total_heat
            min_path = path
    visualize_path(input, min_path)
    print(f"{heat[end, BinD.HORIZONTAL]=}")
    print(f"{heat[end, BinD.VERTICAL]=}")
    print(f"{min_heat=}")


if __name__ == "__main__":
    part_1(input_file)
