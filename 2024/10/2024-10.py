#!/usr/bin/env python

from collections.abc import Generator, Iterable
from dataclasses import dataclass
from pprint import pformat
from typing import Optional, Set

CONTROL_1 = """\
89010123
78121874
87430965
96549874
45678903
32019012
01329801
10456732
""".splitlines()

with open("2024-10.input") as f:
    input_file = [line.strip() for line in f.readlines()]


@dataclass
class Pos:
    i: int
    j: int

    def __hash__(self) -> int:
        return hash((self.i, self.j))

    def __add__(self, other: "Pos") -> "Pos":
        return Pos(self.i + other.i, self.j + other.j)

    def __sub__(self, other: "Pos") -> "Pos":
        return Pos(self.i - other.i, self.j - other.j)

    def __neg__(self) -> "Pos":
        return Pos(-self.i, -self.j)


class Grid:
    def __init__(self, input: Iterable[str]) -> None:
        self.grid = [[int(x) for x in line] for line in input]

    def __repr__(self) -> str:
        return pformat(self.grid)

    @property
    def height(self):
        return len(self.grid)

    @property
    def width(self):
        return len(self.grid[0])

    def is_in_bounds(self, pos: Pos) -> bool:
        return 0 <= pos.i < self.height and 0 <= pos.j < self.width

    def neighbors(self, pos: Pos) -> Generator[Pos, None, None]:
        if pos.i > 0:
            yield Pos(pos.i - 1, pos.j)
        if pos.j > 0:
            yield Pos(pos.i, pos.j - 1)
        if pos.j < self.width - 1:
            yield Pos(pos.i, pos.j + 1)
        if pos.i < self.height - 1:
            yield Pos(pos.i + 1, pos.j)


def get_trailheads(grid: Grid) -> Generator[Pos, None, None]:
    for i in range(grid.height):
        for j in range(grid.width):
            if grid.grid[i][j] == 0:
                yield Pos(i, j)


def find_peaks(grid: Grid, trailhead: Pos) -> Generator[Pos, None, None]:
    def climb(start: Pos, height: int) -> Generator[Pos, None, None]:
        for neighbor in grid.neighbors(start):
            if grid.grid[neighbor.i][neighbor.j] == height + 1:
                if height == 8:
                    yield neighbor
                yield from climb(neighbor, height + 1)

    yield from climb(trailhead, 0)


def part_1(input):
    grid = Grid(input)
    out = 0
    for trailhead in get_trailheads(grid):
        peaks = set(find_peaks(grid, trailhead))
        out += len(peaks)
    print(out)


def part_2(input):
    grid = Grid(input)
    out = 0
    for trailhead in get_trailheads(grid):
        for _ in find_peaks(grid, trailhead):
            out += 1
    print(out)


if __name__ == "__main__":
    part_1(input_file)
    part_2(input_file)
