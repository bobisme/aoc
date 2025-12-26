#!/usr/bin/env python

import time
from typing import LiteralString, NamedTuple

Input = list[str] | list[LiteralString]

CONTROL_1: Input = (
    """\
.......S.......
...............
.......^.......
...............
......^.^......
...............
.....^.^.^.....
...............
....^.^...^....
...............
...^.^...^.^...
...............
..^...^.....^..
...............
.^.^.^.^.^...^.
...............
""".splitlines()
)

Pos = NamedTuple("Pos", [("i", int), ("j", int)])


def part_1(input: Input):
    grid = [[c for c in row] for row in input]
    h = len(grid)
    s_pos = Pos(0, next(i for (i, x) in enumerate(grid[0]) if x == "S"))

    def shoot(pos: Pos) -> int:
        for i in range(pos.i, h):
            if grid[i][pos.j] == "|":
                return 0
            if grid[i][pos.j] == "^":
                return 1 + shoot(Pos(i, pos.j - 1)) + shoot(Pos(i, pos.j + 1))
            else:
                grid[i][pos.j] = "|"
        return 0

    return shoot(s_pos)


def part_2(input: Input):
    h = len(input)
    s_pos = Pos(0, next(i for (i, x) in enumerate(input[0]) if x == "S"))

    cache_map = {}

    def shoot(pos: Pos) -> int:
        if (val := cache_map.get(pos)) is not None:
            return val
        for i in range(pos.i, h):
            if input[i][pos.j] == "^":
                val = shoot(Pos(i, pos.j - 1)) + shoot(Pos(i, pos.j + 1))
                for ii in range(pos.i, i):
                    cache_map[Pos(ii, pos.j)] = val
                return val
        for ii in range(pos.i, h):
            cache_map[Pos(ii, pos.j)] = 1
        return 1

    shoot(s_pos)
    return cache_map[s_pos]


def _test():
    def assert_eq(a, b):
        assert a == b, f"{a} != {b}"

    assert_eq(part_1(CONTROL_1), 21)
    assert_eq(part_2(CONTROL_1), 40)


def run(fn, year=2025, day=7, part=0):
    start = time.perf_counter_ns()
    res = fn()
    elapsed_ns = time.perf_counter_ns() - start
    print(f"{year}\t{day}\t{part}\t{res}\t{elapsed_ns}")


if __name__ == "__main__":
    with open("2025-07.input") as f:
        input_file = [line.rstrip("\n") for line in f.readlines()]
    _test()
    run(lambda: part_1(input_file), part=1)
    run(lambda: part_2(input_file), part=2)
