#!/usr/bin/env python

from collections.abc import Generator
from typing import Optional


CONTROL_1 = """\
2333133121414131402
""".splitlines()

with open("2024-9.input") as f:
    input_file = [line.strip() for line in f.readlines()]


def build_dense(map: list[int]) -> list[Optional[int]]:
    dense: list[Optional[int]] = [None for _ in range(sum(map))]
    dense_i = 0
    id = 0
    for i in range(len(map)):
        if i % 2 == 0:
            for _ in range(map[i]):
                dense[dense_i] = id
                dense_i += 1
            id += 1
        else:
            for _ in range(map[i]):
                dense_i += 1
    return dense


def free_spaces(dense: list[Optional[int]]) -> Generator[int, None, None]:
    for i in range(len(dense)):
        if dense[i] is None:
            yield i


def end_files(dense: list[Optional[int]]) -> Generator[int, None, None]:
    for i in range(len(dense) - 1, 0, -1):
        if dense[i] is not None:
            yield i


def checksum(dense: list[Optional[int]]) -> int:
    out = 0
    for i, x in enumerate(dense):
        if x is None:
            break
        out += i * x

    return out


def part_1(input):
    map = [int(x) for x in input[0]]
    dense = build_dense(map)
    for free, filled in zip(free_spaces(dense), end_files(dense)):
        if free >= filled:
            break
        dense[free] = dense[filled]
        dense[filled] = None
    print(checksum(dense))


def part_2(input):
    for line in input:
        print(line)


if __name__ == "__main__":
    part_1(input_file)
    # part_2(input_file)
