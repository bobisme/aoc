#!/usr/bin/env python

from hashlib import md5
from typing import LiteralString
import time

Input = list[str] | list[LiteralString]


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


def run(fn, year=2015, day=4, part=0):
    start = time.perf_counter_ns()
    res = fn()
    elapsed_ns = time.perf_counter_ns() - start
    print(f"{year}\t{day}\t{part}\t{res}\t{elapsed_ns}")
    return res


if __name__ == "__main__":
    with open("2015-04.input") as f:
        input_file = [line.rstrip("\n") for line in f.readlines()]
    _test()
    run(lambda: part_1(input_file), part=1)
    run(lambda: part_2(input_file), part=2)
