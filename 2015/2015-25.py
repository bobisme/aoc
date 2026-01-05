#!/usr/bin/env python

import re
from typing import Iterator, LiteralString
import time

Input = list[str] | list[LiteralString]

INIT = 20151125
MUL = 252533
DIV = 33554393


def op(x: int) -> int:
    return x * MUL % DIV


def ijs() -> Iterator[tuple[int, int]]:
    row = 0
    col = 0
    max_row = 0
    while True:
        yield (row, col)
        if row > 0:
            row -= 1
            col += 1
        else:
            max_row += 1
            row = max_row
            col = 0


def part_1(input: Input):
    target_row, target_col = map(
        int, re.findall(r"row (\d+), column (\d+).", input[0])[0]
    )
    coords = ijs()
    next(coords)
    val = INIT
    for i, j in coords:
        val = op(val)
        if (i, j) == (target_row - 1, target_col - 1):
            return val


def run(fn, year=2015, day=25, part=0):
    start = time.perf_counter_ns()
    res = fn()
    elapsed_ns = time.perf_counter_ns() - start
    print(f"{year}\t{day}\t{part}\t{res}\t{elapsed_ns}")
    return res


if __name__ == "__main__":
    with open("2015-25.input") as f:
        input_file = [line.rstrip("\n") for line in f.readlines()]
    run(lambda: part_1(input_file), part=1)
