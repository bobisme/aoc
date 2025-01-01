#!/usr/bin/env python

from itertools import product
from typing import Generator, NamedTuple, Sequence


CONTROL_1 = """\
#####
.####
.####
.####
.#.#.
.#...
.....

#####
##.##
.#.##
...##
...#.
...#.
.....

.....
#....
#....
#...#
#.#.#
#.###
#####

.....
.....
#.#..
###..
###.#
###.#
#####

.....
.....
.....
#....
#.#..
#.#.#
#####
""".splitlines()

with open("2024-25.input") as f:
    input_file = [line.strip() for line in f.readlines()]

Thing = NamedTuple("Thing", [("is_lock", bool), ("heights", tuple[int, ...])])


def parse(input: Sequence[str]) -> Generator[Thing, None, None]:
    grids = []
    grid = []
    for line in input:
        if line == "":
            grids.append(grid)
            grid = []
            continue
        grid.append(line)
    grids.append(grid)
    for grid in grids:
        is_lock = grid[0][0] == "#"
        heights = [0] * 5
        for col in range(5):
            for i in range(1, 6):
                if grid[i][col] == "#":
                    heights[col] += 1
        yield Thing(is_lock, tuple(heights))


def overlap(key: Thing, lock: Thing) -> bool:
    cmp = [0] * 5
    for i in range(5):
        cmp[i] = key.heights[i] + lock.heights[i]
    return any(c > 5 for c in cmp)


def part_1(input):
    things = list(parse(input))
    locks = [x for x in things if x.is_lock]
    keys = [x for x in things if not x.is_lock]

    out = 0
    for key, lock in product(keys, locks):
        if not overlap(key, lock):
            out += 1
        # print(lock.heights, key.heights, overlap(key, lock))
    print(out)


if __name__ == "__main__":
    part_1(input_file)
