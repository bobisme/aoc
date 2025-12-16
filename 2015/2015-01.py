#!/usr/bin/env python

from typing import LiteralString
import timeit

Input = list[str] | list[LiteralString]

with open("2015-01.input") as f:
    input_file = [line.rstrip("\n") for line in f.readlines()]


def part_1(input: Input) -> int:
    return sum(1 if c == "(" else -1 for c in input[0])


def part_2(input: Input):
    s = 0
    for i, c in enumerate(input[0]):
        s += 1 if c == "(" else -1
        if s == -1:
            return i + 1


def _test():
    def assert_eq(a, b):
        assert a == b, f"{a} != {b}"

    assert_eq(part_1([")())())"]), -3)
    assert_eq(part_2(["()())"]), 5)


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
