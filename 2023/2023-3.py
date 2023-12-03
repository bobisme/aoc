#!/usr/bin/env python

from collections import deque
from typing import NamedTuple, List, Generator
import re
from string import digits


class Vec2(NamedTuple("Vec2", [("x", int), ("y", int)])):
    @property
    def i(self):
        return self.x

    @property
    def j(self):
        return self.y

    @property
    def ij(self):
        return self.x, self.y


class ColorGrid(NamedTuple("ColorGrid", [("data", List[List[int]])])):
    def __repr__(self) -> str:
        return "\n".join(" ".join(f"{col:02}" for col in row) for row in self.data)

    def get(self, ij: Vec2) -> int:
        return self.data[ij.j][ij.i]

    def put(self, ij: Vec2, val: int):
        self.data[ij.j][ij.i] = val

    def was_visited(self, ij) -> int:
        return self.data[ij.j][ij.i] >= 0


class Grid(NamedTuple("Grid", [("data", List[str])])):
    def __repr__(self) -> str:
        return "\n".join(self.data)

    @property
    def cols(self):
        return len(self.data[0])

    @property
    def rows(self):
        return len(self.data)

    def get(self, ij) -> str:
        return self.data[ij.j][ij.i]

    def positions(self) -> Generator:
        for j in range(len(self.data)):
            for i in range(len(self.data[0])):
                yield Vec2(i, j)

    def symbols(self) -> Generator:
        for position in self.positions():
            val = self.get(position)
            if is_symbol(val):
                yield (position, val)

    def part_numbers(self, colors: ColorGrid) -> Generator:
        for position, _ in self.symbols():
            ns = neighbors(self.data, position)
            for n in ns:
                if colors.was_visited(n):
                    continue
                n_val = self.get_number(n, colors)
                if n_val:
                    yield n_val

    def gears(self) -> Generator:
        for position, val in self.symbols():
            if val == "*":
                yield position

    def gear_ratios(self, colors: ColorGrid) -> Generator:
        for position, _ in self.symbols():
            adj_count = 0
            adj_prod = 1
            ns = neighbors(self.data, position)
            for n in ns:
                if colors.was_visited(n):
                    continue
                n_val = self.get_number(n, colors)
                if not n_val:
                    continue
                adj_count += 1
                adj_prod *= n_val
            if adj_count == 2:
                yield adj_prod

    def get_number(self, pos: Vec2, colors: ColorGrid) -> int | None:
        n = deque()
        if not is_digit(self.data[pos.j][pos.i]):
            return None
        for i in range(pos.i, -1, -1):
            val = self.data[pos.j][i]
            if not is_digit(val):
                break
            n.appendleft(int(val))
            colors.put(Vec2(i, pos.j), 1)
        for i in range(pos.i + 1, self.cols):
            val = self.data[pos.j][i]
            if not is_digit(val):
                break
            n.append(int(val))
            colors.put(Vec2(i, pos.j), 1)
        return sum(10**i * x for (i, x) in enumerate(reversed(n)))


with open("2023-3.input") as f:
    input = [line.strip() for line in f.readlines()]


def neighbors(
    input: List[str] | List[List[int]], ij: Vec2
) -> Generator[Vec2, None, None]:
    max_i = len(input[0]) - 1
    max_j = len(input) - 1

    for i in range(ij.i - 1, ij.i + 2):
        for j in range(ij.j - 1, ij.j + 2):
            if i < 0 or i > max_i:
                continue
            if j < 0 or j > max_j:
                continue
            yield Vec2(i, j)


def is_symbol(val: str) -> bool:
    return re.match("[^.0-9]", val) is not None


def is_digit(val: str) -> bool:
    return val in digits


def main(input):
    grid = Grid(input)
    color_grid = ColorGrid([[-1 for _ in range(grid.cols)] for _ in range(grid.rows)])
    print("grid")
    print(grid)
    part_numbers = list(grid.part_numbers(color_grid))
    print("colors")
    print(color_grid)
    print("part_numbers =", part_numbers)
    print("sum", sum(part_numbers))
    color_grid = ColorGrid([[-1 for _ in range(grid.cols)] for _ in range(grid.rows)])
    gear_ratios = list(grid.gear_ratios(color_grid))
    print("gear_ratios", gear_ratios)
    print("sum of gear_ratios", sum(gear_ratios))
    # q = []


CONTROL_1 = """\
467..114..
...*......
..35..633.
......#...
617*......
.....+.58.
..592.....
......755.
...$.*....
.664.598..
""".splitlines()

if __name__ == "__main__":
    main(input)
