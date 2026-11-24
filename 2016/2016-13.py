#!/usr/bin/env python

import time
from collections import deque
from collections.abc import Generator
from dataclasses import dataclass
from typing import LiteralString

Input = list[str] | list[LiteralString]

CONTROL_1: Input = """\
10
""".splitlines()


@dataclass
class Grid:
    offset: int

    def is_wall(self, x: int, y: int) -> bool:
        val = x * x + 3 * x + 2 * x * y + y + y * y + self.offset
        return val.bit_count() % 2 != 0

    def neighbors(self, p: "Pos") -> Generator["Pos"]:
        for off in ((0, -1), (-1, 0), (1, 0), (0, 1)):
            x = p.x + off[0]
            if x < 0:
                continue
            y = p.y + off[1]
            if y < 0:
                continue
            if not self.is_wall(x, y):
                yield Pos(x, y)

    def print(self):
        for y in range(7):
            for x in range(10):
                print("#" if self.is_wall(x, y) else ".", end="")
            print()


@dataclass(slots=True)
class Pos:
    x: int
    y: int

    def __hash__(self) -> int:
        return hash((self.x, self.y))

    def __eq__(self, other: object, /) -> bool:
        assert isinstance(other, Pos)
        return self.x == other.x and self.y == other.y


@dataclass(slots=True)
class Item:
    pos: Pos
    steps: int


def part_1(input: Input, dst=None):
    grid = Grid(int(input[0]))
    if dst is None:
        dst = Pos(31, 39)
    pos = Pos(1, 1)
    q: deque[Item] = deque([Item(pos, 0)])
    visited = {pos}

    while len(q) > 0:
        n = q.popleft()
        if n.pos == dst:
            return n.steps
        for ne in grid.neighbors(n.pos):
            if ne in visited:
                continue
            visited.add(ne)
            q.append(Item(ne, n.steps + 1))
    return -1


def part_2(input: Input):
    grid = Grid(int(input[0]))
    dst = Pos(31, 39)
    pos = Pos(1, 1)
    q: deque[Item] = deque([Item(pos, 0)])
    visited = {pos}

    while len(q) > 0:
        n = q.popleft()
        if n.pos == dst:
            return n.steps
        if n.steps >= 50:
            continue
        for ne in grid.neighbors(n.pos):
            if ne in visited:
                continue
            visited.add(ne)
            q.append(Item(ne, n.steps + 1))
    return len(visited)


def _test():
    def assert_eq(a, b):
        assert a == b, f"{a} != {b}"

    assert_eq(part_1(CONTROL_1, Pos(7, 4)), 11)
    # assert_eq(part_2(CONTROL_1), 0)


def run(fn, year=2016, day=13, part=0):
    start = time.perf_counter_ns()
    res = fn()
    elapsed_ns = time.perf_counter_ns() - start
    print(f"{year}\t{day}\t{part}\t{res}\t{elapsed_ns}")
    return res


if __name__ == "__main__":
    with open("2016-13.input") as f:
        input_file = [line.rstrip("\n") for line in f]
    _test()
    run(lambda: part_1(input_file), part=1)
    run(lambda: part_2(input_file), part=2)
