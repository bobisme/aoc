#!/usr/bin/env python

from typing import LiteralString
import time

Input = list[str] | list[LiteralString]


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


def run(fn, year=2015, day=1, part=0):
    start = time.perf_counter_ns()
    res = fn()
    elapsed_ns = time.perf_counter_ns() - start
    print(f"{year}\t{day}\t{part}\t{res}\t{elapsed_ns}")
    return res


if __name__ == "__main__":
    with open("2015-01.input") as f:
        input_file = [line.rstrip("\n") for line in f.readlines()]
    _test()
    run(lambda: part_1(input_file), part=1)
    run(lambda: part_2(input_file), part=2)
