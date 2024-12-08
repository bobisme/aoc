#!/usr/bin/env python

from collections.abc import Generator
from pprint import pp

CONTROL_1 = """\
MMMSXXMASM
MSAMXMSMSA
AMXSXMAAMM
MSAMASMSMX
XMASAMXAMM
XXAMMXXAMA
SMSMSASXSS
SAXAMASAAA
MAMMMXMMMM
MXMXAXMASX
""".splitlines()

with open("2024-4.input") as f:
    input_file = [line.strip() for line in f.readlines()]


def get_words(grid: list[list[str]], i: int, j: int) -> Generator[str, None, None]:
    height = len(grid)
    width = len(grid[0])
    if j < width - 3:
        yield "".join(grid[i][j : j + 4])
    if j >= 3:
        yield "".join(grid[i][j::-1][:4])
    if i < height - 3:
        yield "".join(line[j] for line in grid[i : i + 4])
    if i >= 3:
        yield "".join(line[j] for line in grid[i::-1][:4])
    if j < width - 3 and i < height - 3:
        yield "".join(grid[ii][jj] for ii, jj in zip(range(i, i + 4), range(j, j + 4)))
    if j < width - 3 and i >= 3:
        yield "".join(
            grid[ii][jj] for ii, jj in zip(range(i, i - 4, -1), range(j, j + 4))
        )
    if j >= 3 and i >= 3:
        yield "".join(
            grid[ii][jj] for ii, jj in zip(range(i, i - 4, -1), range(j, j - 4, -1))
        )
    if j >= 3 and i < height - 3:
        yield "".join(
            grid[ii][jj] for ii, jj in zip(range(i, i + 4), range(j, j - 4, -1))
        )


def part_1(input):
    grid = []
    for line in input:
        grid.append(list(line))
    height = len(grid)
    width = len(grid[0])
    count = 0
    for i in range(height):
        for j in range(width):
            # print(list(get_words(grid, i, j)))
            for word in get_words(grid, i, j):
                if word == "XMAS":
                    count += 1
    print(count)


def get_x(grid: list[list[str]], i: int, j: int) -> str:
    return "".join(
        (
            grid[i][j],
            grid[i][j + 2],
            grid[i + 1][j + 1],
            grid[i + 2][j],
            grid[i + 2][j + 2],
        )
    )


def part_2(input):
    grid = []
    for line in input:
        grid.append(list(line))
    height = len(grid)
    width = len(grid[0])
    count = 0
    for i in range(height - 2):
        for j in range(width - 2):
            x = get_x(grid, i, j)
            if x in ("MMASS", "SSAMM", "MSAMS", "SMASM"):
                count += 1
    print(count)


if __name__ == "__main__":
    part_1(input_file)
    part_2(input_file)
