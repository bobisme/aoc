#!/usr/bin/env python

from typing import LiteralString
import time

Input = list[str] | list[LiteralString]


def part_1(input: Input):
    def surf_area(ln: int, w: int, h: int) -> int:
        return sum((2 * ln * w, 2 * w * h, 2 * h * ln))

    def side_area(ln: int, w: int, h: int) -> tuple[int, int, int]:
        return (ln * w, w * h, h * ln)

    out = 0
    for line in input:
        dims = tuple(map(int, line.split("x")))
        area = surf_area(*dims)
        sides = side_area(*dims)
        out += area + min(sides)
    return out


def part_2(input: Input):
    def perims(ln: int, w: int, h: int) -> tuple[int, int, int]:
        return (2 * ln + 2 * w, 2 * w + 2 * h, 2 * h + 2 * ln)

    def vol(ln: int, w: int, h: int) -> int:
        return ln * w * h

    out = 0
    for line in input:
        dims = tuple(map(int, line.split("x")))
        out += min(perims(*dims)) + vol(*dims)
    return out


def _test():
    def assert_eq(a, b):
        assert a == b, f"{a} != {b}"

    assert_eq(part_1(["2x3x4"]), 58)
    assert_eq(part_2(["2x3x4"]), 34)


def run(fn, year=2015, day=2, part=0):
    start = time.perf_counter_ns()
    res = fn()
    elapsed_ns = time.perf_counter_ns() - start
    print(f"{year}\t{day}\t{part}\t{res}\t{elapsed_ns}")
    return res


if __name__ == "__main__":
    with open("2015-02.input") as f:
        input_file = [line.rstrip("\n") for line in f.readlines()]
    _test()
    run(lambda: part_1(input_file), part=1)
    run(lambda: part_2(input_file), part=2)
