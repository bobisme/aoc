#!/usr/bin/env python

from hashlib import md5
from typing import LiteralString
import timeit

Input = list[str] | list[LiteralString]


with open("2015-04.input") as f:
    input_file = [line.rstrip("\n") for line in f.readlines()]


def part_1(input: Input):
    key = bytes(input[0], "ascii")
    for i in range(100_000_000):
        h = md5(key + bytes(str(i), "ascii")).hexdigest()
        if h[:5] == "00000":
            return i
    return 0


def part_2(input: Input):
    key = bytes(input[0], "ascii")
    for i in range(100_000_000):
        h = md5(key + bytes(str(i), "ascii")).hexdigest()
        if h[:6] == "000000":
            return i
    return 0


def _test():
    def assert_eq(a, b):
        assert a == b, f"{a} != {b}"

    assert_eq(part_1(["pqrstuv"]), 1048970)
    # assert_eq(part_2(CONTROL_1), 0)


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
