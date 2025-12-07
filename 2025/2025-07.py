#!/usr/bin/env python

from typing import Generator, LiteralString, NamedTuple
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
    h, w = len(grid), len(grid[0])
    s_pos = Pos(0, next(i for (i, x) in enumerate(grid[0]) if x == "S"))

    def shoot(pos: Pos) -> Generator[Pos]:
        if pos.i < 0 or pos.i >= h or pos.j < 0 or pos.j >= w:
            return
        for i in range(pos.i, h):
            if grid[i][pos.j] == "|":
                return
            if grid[i][pos.j] == "^":
                yield Pos(i, pos.j)
                yield from shoot(Pos(i, pos.j - 1))
                yield from shoot(Pos(i, pos.j + 1))
                return
            else:
                grid[i][pos.j] = "|"

    positions = set(shoot(Pos(s_pos.i + 1, s_pos.j)))
    return len(positions)


def part_2(input: Input):
    grid = [[c for c in row] for row in input]
    h, w = len(grid), len(grid[0])
    s_pos = Pos(0, next(i for (i, x) in enumerate(grid[0]) if x == "S"))

    cache_grid = [[0 for _ in row] for row in grid]

    def shoot(pos: Pos) -> int:
        if pos.i < 0 or pos.i >= h or pos.j < 0 or pos.j >= w:
            return 0
        if (val := cache_grid[pos.i][pos.j]) > 0:
            return val
        sub_path = []
        for i in range(pos.i, h):
            if grid[i][pos.j] == "^":
                val = +shoot(Pos(i, pos.j - 1)) + shoot(Pos(i, pos.j + 1))
                for p in sub_path:
                    cache_grid[p.i][p.j] = val
                return val
            sub_path.append(Pos(i, pos.j))
        for p in sub_path:
            cache_grid[p.i][p.j] = 1
        return 1

    shoot(s_pos)

    return cache_grid[s_pos.i][s_pos.j]


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
