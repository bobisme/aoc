#!/usr/bin/env python

from typing import LiteralString, NamedTuple
import timeit

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

with open("2025-07.input") as f:
    input_file = [line.rstrip("\n") for line in f.readlines()]

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


def _bench(fn, count=100):
    return timeit.timeit(fn, number=count) / count * 1_000


if __name__ == "__main__":
    _test()
    print("tests: PASS")
    print("-" * 40)
    print("part_1:", part_1(input_file))
    print("part_2:", part_2(input_file))
    print("-" * 40)
    print("part_1 bench: {:.1f}ms".format(_bench(lambda: part_1(input_file), count=10)))
    print("part_2 bench: {:.1f}ms".format(_bench(lambda: part_2(input_file), count=10)))
