#!/usr/bin/env python

from functools import cache
from typing import Any, Generator


CONTROL_1 = """\
O....#....
O.OO#....#
.....##...
OO.#O....O
.O.....O#.
O.#..O.#.#
..O..#O..O
.......O..
#....###..
#OO..#....
""".splitlines()

with open("2023-14.input") as f:
    input = [line.strip() for line in f.readlines()]


def move(row_i, col_i):
    pass


def tilt_up(input: list[list[str]]):
    rows = len(input)
    cols = len(input[0])
    for row_i in range(1, rows):
        for col_i in range(cols):
            for dst_row_i in range(row_i, 0, -1):
                src = input[dst_row_i][col_i]
                if src != "O":
                    continue
                dst = input[dst_row_i - 1][col_i]
                if dst != ".":
                    continue
                input[dst_row_i - 1][col_i] = "O"
                input[dst_row_i][col_i] = "."


def col_up(col: list[str]) -> list[str]:
    out = col.copy()
    rows = len(col)
    dst_i = 0
    for row_i in range(rows):
        # print(out_i, row_i)
        if dst_i >= rows:
            break
        src = out[row_i]
        if src == "#":
            dst_i = row_i + 1
            continue
        elif src == "O":
            if dst_i != row_i:
                assert out[dst_i] == "."
                out[dst_i] = "O"
                out[row_i] = "."
            dst_i = dst_i + 1
            continue
    return out


def tilt_up2(input: list[list[str]]):
    rows = len(input)
    cols = len(input[0])
    for col_i in range(cols):
        col = [input[i][col_i] for i in range(rows)]
        tilted = col_up(col)
        for i, c in enumerate(tilted):
            input[i][col_i] = c

        # dst_row_i = 0
        # for row_i in range(rows):
        #     # print(col_i, row_i)
        #     if dst_row_i >= rows:
        #         break
        #     src = input[row_i][col_i]
        #     if src == "#":
        #         dst_row_i = row_i + 1
        #         continue
        #     elif src == "O":
        #         if dst_row_i != row_i:
        #             assert input[dst_row_i][col_i] == "."
        #             input[dst_row_i][col_i] = "O"
        #             input[row_i][col_i] = "."
        #         dst_row_i = dst_row_i + 1
        #         continue


def rotate_clockwise(input):
    return list(list(x) for x in zip(*input[::-1]))


def eval(input: list[list[str]]) -> Generator[int, Any, Any]:
    rows = len(input)
    cols = len(input[0])
    for j in range(rows):
        for i in range(cols):
            obj = input[j][i]
            if obj != "O":
                continue
            yield rows - j


def print_matrix(input):
    for line in input:
        print("".join(line))
    print("-" * len(input[0]))


def main(input):
    input = [list(line) for line in input]
    print_matrix(input)
    # tilt_up(input)
    # print_matrix(input)
    loads = []
    for i in range(100):
        for _ in ("north", "west", "south", "east"):
            tilt_up2(input)
            input = rotate_clockwise(input)
            # print_matrix(input)
        load = sum(eval(input))
        loads.append(load)
    # print_matrix(input)
    print(loads)

    load = sum(eval(input))
    print(f"{load=}")


if __name__ == "__main__":
    main(input)
