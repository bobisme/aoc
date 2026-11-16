#!/usr/bin/env python

import time
from typing import LiteralString

Input = list[str] | list[LiteralString]

CONTROL_1: Input = """\
""".splitlines()


def count_decompress(line: str) -> int:
    out = 0
    i = 0
    while i < len(line):
        if line[i] != "(":
            out += 1
            i += 1
        else:
            marker = ""
            i += 1
            while line[i] != ")":
                marker += line[i]
                i += 1
            i += 1
            a, b = map(int, marker.split("x"))
            out += a * b
            i += a
    return out


def count_full_decompress(line: str, repeat=1) -> int:
    out = 0
    i = 0
    while i < len(line):
        if line[i] != "(":
            out += 1
            i += 1
        else:
            marker = ""
            i += 1
            while line[i] != ")":
                marker += line[i]
                i += 1
            i += 1
            a, b = map(int, marker.split("x"))
            out += count_full_decompress(line[i : i + a], b)
            i += a
    return out * repeat


def part_1(input: Input):
    return count_decompress(input[0])


def part_2(input: Input):
    return count_full_decompress(input[0])


def _test():
    def assert_eq(a, b):
        assert a == b, f"{a} != {b}"

    assert_eq(part_1(["A(1x5)BC"]), 7)
    assert_eq(part_1(["X(8x2)(3x3)ABCY"]), 18)
    assert_eq(part_2(["(25x3)(3x3)ABC(2x3)XY(5x2)PQRSTX(18x9)(3x2)TWO(5x7)SEVEN"]), 445)


def run(fn, year=2016, day=9, part=0):
    start = time.perf_counter_ns()
    res = fn()
    elapsed_ns = time.perf_counter_ns() - start
    print(f"{year}\t{day}\t{part}\t{res}\t{elapsed_ns}")
    return res


if __name__ == "__main__":
    with open("2016-09.input") as f:
        input_file = [line.rstrip("\n") for line in f]
    _test()
    run(lambda: part_1(input_file), part=1)
    run(lambda: part_2(input_file), part=2)
