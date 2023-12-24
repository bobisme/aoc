#!/usr/bin/env python

from dataclasses import dataclass
from typing import Self
from itertools import combinations

CONTROL_1 = """\
...#......
.......#..
#.........
..........
......#...
.#........
.........#
..........
.......#..
#...#.....
""".splitlines()

with open("2023-11.input") as f:
    input = [line.strip() for line in f.readlines()]


@dataclass
class Coords:
    i: int
    j: int

    def __hash__(self):
        return hash((self.i, self.j))

    def dist(self, other: Self) -> int:
        return abs(other.i - self.i) + abs(other.j - self.j)


def main(input, exp_rate=1):
    input = [list(line) for line in input]
    n_in_rows = len(input)
    n_in_cols = len(input[0])

    row_is_empty = [all(x == "." for x in line) for line in input]
    col_is_empty = [
        all(input[j][i] == "." for j in range(n_in_rows)) for i in range(n_in_cols)
    ]
    empty_rows = set(i for (i, x) in enumerate(row_is_empty) if x is True)
    empty_cols = set(i for (i, x) in enumerate(col_is_empty) if x is True)

    n_exp_rows = n_in_rows + len(empty_rows) * (exp_rate - 1)
    n_exp_cols = n_in_cols + len(empty_cols) * (exp_rate - 1)

    # expanded = []

    # for j, line in enumerate(input):
    #     exp_row = []
    #     for i, x in enumerate(line):
    #         if i in empty_cols:
    #             for _ in range(exp_rate):
    #                 exp_row.append(".")
    #         else:
    #             exp_row.append(x)
    #     if j in empty_rows:
    #         for _ in range(exp_rate - 1):
    #             expanded.append(exp_row)
    #     expanded.append(exp_row)

    galaxies: list[Coords] = []
    for j in range(n_in_rows):
        for i in range(n_in_cols):
            if input[j][i] == "#":
                expanding_cols = sum(1 for x in empty_cols if i > x)
                expanding_rows = sum(1 for x in empty_rows if j > x)
                galaxies.append(
                    Coords(
                        i + expanding_cols * (exp_rate - 1),
                        j + expanding_rows * (exp_rate - 1),
                    )
                )

    # for n, c in enumerate(galaxies):
    #     expanded[c.j][c.i] = hex(n + 1).replace("0x", "")

    # print([(i + 1, g) for (i, g) in enumerate(galaxies)])

    # for line in expanded:
    #     print("".join(line))

    total = 0
    for (ia, a), (ib, b) in combinations(enumerate(galaxies), 2):
        d = a.dist(b)
        print(ia + 1, "->", ib + 1, d)
        total += d
    print(total)


if __name__ == "__main__":
    main(input, exp_rate=1_000_000)
