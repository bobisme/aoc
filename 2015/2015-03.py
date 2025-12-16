#!/usr/bin/env python

from typing import LiteralString
import timeit

Input = list[str] | list[LiteralString]

with open("2015-03.input") as f:
    input_file = [line.rstrip("\n") for line in f.readlines()]


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


def _bench(fn, count=100):
    return timeit.timeit(fn, number=count) / count * 1_000


if __name__ == "__main__":
    _test()
    print("tests: PASS")
    print("-" * 40)
    print("part_1:", part_1(input_file))
    print("part_2:", part_2(input_file))
    print("-" * 40)
    print("part_1 bench: {:.1f}ms".format(_bench(lambda: part_1(input_file), count=1)))
    print("part_2 bench: {:.1f}ms".format(_bench(lambda: part_2(input_file), count=1)))
