#!/usr/bin/env python

import copy
from dataclasses import dataclass, field
from typing import Generator, LiteralString
import time

Input = list[str] | list[LiteralString]

CONTROL_1: Input = (
    """\
.#.#.#
...##.
#....#
..#...
#.#..#
####..
""".splitlines()
)


@dataclass(slots=True)
class Grid:
    grid: list[list[bool]]
    w: int = field(init=False)
    h: int = field(init=False)

    def __post_init__(self):
        self.w = len(self.grid[0])
        self.h = len(self.grid)

    def copy(self):
        return Grid(copy.deepcopy(self.grid))

    def __repr__(self) -> str:
        return "\n".join("".join("#" if v else "." for v in row) for row in self.grid)

    def __getitem__(self, ij: tuple[int, int]) -> bool:
        return self.grid[ij[0]][ij[1]]

    def __setitem__(self, ij: tuple[int, int], val: bool):
        self.grid[ij[0]][ij[1]] = val

    def neighbors(self, ij: tuple[int, int]) -> Generator[tuple[int, int]]:
        i, j = ij
        for ni in (i - 1, i, i + 1):
            if ni < 0 or ni >= self.h:
                continue
            for nj in (j - 1, j, j + 1):
                if (ni, nj) == ij or nj < 0 or nj >= self.w:
                    continue
                yield (ni, nj)

    def count_neighbors(self) -> list[list[int]]:
        counts = [[0] * self.w for _ in range(self.h)]
        for i, row in enumerate(self.grid):
            for j, val in enumerate(row):
                for ni, nj in self.neighbors((i, j)):
                    if val:
                        counts[ni][nj] += 1
        return counts

    def turn_stuck_lights_on(self):
        self[0, 0] = True
        self[self.h - 1, 0] = True
        self[0, self.w - 1] = True
        self[self.h - 1, self.w - 1] = True

    def step(self, stuck_lights=False):
        counts = self.count_neighbors()
        for i, row in enumerate(counts):
            for j, count in enumerate(row):
                if self[i, j]:
                    if not (count == 2 or count == 3):
                        self[i, j] = False
                elif count == 3:
                    self[i, j] = True
        if stuck_lights:
            self.turn_stuck_lights_on()

    def count_lights(self) -> int:
        return sum(sum(row) for row in self.grid)


def part_1(input: Input, steps=100):
    grid = Grid([[c == "#" for c in line] for line in input])
    for _ in range(steps):
        grid.step()
    return grid.count_lights()


def part_2(input: Input, steps=100):
    grid = Grid([[c == "#" for c in line] for line in input])
    grid.turn_stuck_lights_on()
    for _ in range(steps):
        grid.step(stuck_lights=True)
    return grid.count_lights()


def _test():
    def assert_eq(a, b):
        assert a == b, f"{a} != {b}"

    assert_eq(part_1(CONTROL_1, steps=4), 4)
    assert_eq(part_2(CONTROL_1, steps=5), 17)


def run(fn, year=2015, day=18, part=0):
    start = time.perf_counter_ns()
    res = fn()
    elapsed_ns = time.perf_counter_ns() - start
    print(f"{year}\t{day}\t{part}\t{res}\t{elapsed_ns}")
    return res


if __name__ == "__main__":
    with open("2015-18.input") as f:
        input_file = [line.rstrip("\n") for line in f.readlines()]
    _test()
    run(lambda: part_1(input_file), part=1)
    run(lambda: part_2(input_file), part=2)
