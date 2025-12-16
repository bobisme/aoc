#!/usr/bin/env python

from typing import LiteralString
import timeit

Input = list[str] | list[LiteralString]


with open("2015-02.input") as f:
    input_file = [line.rstrip("\n") for line in f.readlines()]


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
