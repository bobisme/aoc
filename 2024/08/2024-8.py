#!/usr/bin/env python

from collections import defaultdict
from collections.abc import Generator, Iterable
from dataclasses import dataclass
from pprint import pformat, pprint
from itertools import combinations
from typing import DefaultDict


CONTROL_1 = """\
............
........0...
.....0......
.......0....
....0.......
......A.....
............
............
........A...
.........A..
............
............
""".splitlines()

with open("2024-8.input") as f:
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
        self.grid = [list(line) for line in input]

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


def map_antennas(grid: Grid):
    map: DefaultDict[str, list[Pos]] = defaultdict(list)
    for i in range(grid.height):
        for j in range(grid.width):
            val = grid.grid[i][j]
            if val != ".":
                map[val].append(Pos(i, j))
    return map


def get_antinodes(antennas: dict[str, list[Pos]]) -> Generator[Pos, None, None]:
    for _freq, ants in antennas.items():
        for a, b in combinations(ants, 2):
            yield b + (b - a)
            yield a + (a - b)


def part_1(input):
    grid = Grid(input)
    antennas = map_antennas(grid)
    antinodes = {x for x in get_antinodes(antennas) if grid.is_in_bounds(x)}
    print(len(antinodes))


def get_repeat_antinodes(
    antennas: dict[str, list[Pos]], grid: Grid
) -> Generator[Pos, None, None]:
    for _freq, ants in antennas.items():
        for a, b in combinations(ants, 2):
            base = b
            delta = b - a
            d = delta
            while grid.is_in_bounds(base + d):
                yield base + d
                d += delta
            base = a
            delta = a - b
            d = delta
            while grid.is_in_bounds(base + d):
                yield base + d
                d += delta
            # yield b + (b - a)
            # yield a + (a - b)


def part_2(input):
    grid = Grid(input)
    antennas = map_antennas(grid)
    antinodes = {x for x in get_repeat_antinodes(antennas, grid)}
    for _, ants in antennas.items():
        if len(ants) == 1:
            continue
        for ant in ants:
            antinodes.add(ant)
    g = [line.copy() for line in grid.grid]
    for ant in antinodes:
        if g[ant.i][ant.j] == ".":
            g[ant.i][ant.j] = "#"
    pprint(["".join(line) for line in g])
    print(len(antinodes))


if __name__ == "__main__":
    part_1(input_file)
    part_2(input_file)
