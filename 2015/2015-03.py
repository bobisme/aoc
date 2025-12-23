#!/usr/bin/env python

from typing import LiteralString
import time

Input = list[str] | list[LiteralString]


def move(start: tuple[int, int], dir: str) -> tuple[int, int]:
    match dir:
        case "^":
            return (start[0], start[1] + 1)
        case "v":
            return (start[0], start[1] - 1)
        case "<":
            return (start[0] - 1, start[1])
        case ">":
            return (start[0] + 1, start[1])
    raise Exception("nah")


def part_1(input: Input):
    houses = {(0, 0)}
    pos = (0, 0)
    for c in input[0]:
        pos = move(pos, c)
        houses.add(pos)
    return len(houses)


def part_2(input: Input):
    houses = {(0, 0)}
    posa = (0, 0)
    posb = (0, 0)
    for c in input[0][::2]:
        posa = move(posa, c)
        houses.add(posa)
    for c in input[0][1::2]:
        posb = move(posb, c)
        houses.add(posb)
    return len(houses)


def _test():
    def assert_eq(a, b):
        assert a == b, f"{a} != {b}"

    assert_eq(part_1([">"]), 2)
    assert_eq(part_1(["^>v<"]), 4)
    assert_eq(part_1(["^v^v^v^v^v"]), 2)
    assert_eq(part_2(["^v"]), 3)
    assert_eq(part_2(["^>v<"]), 3)
    assert_eq(part_2(["^v^v^v^v^v"]), 11)


def run(fn, year=2015, day=3, part=0):
    start = time.perf_counter_ns()
    res = fn()
    elapsed_ns = time.perf_counter_ns() - start
    print(f"{year}\t{day}\t{part}\t{res}\t{elapsed_ns}")
    return res


if __name__ == "__main__":
    with open("2015-03.input") as f:
        input_file = [line.rstrip("\n") for line in f.readlines()]
    _test()
    run(lambda: part_1(input_file), part=1)
    run(lambda: part_2(input_file), part=2)
