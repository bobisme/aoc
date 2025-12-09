#!/usr/bin/env python

from typing import LiteralString, NamedTuple
import timeit

Input = list[str] | list[LiteralString]

CONTROL_1: Input = (
    """\
7,1
11,1
11,7
9,7
9,5
2,5
2,3
7,3
""".splitlines()
)

with open("2025-09.input") as f:
    input_file = [line.rstrip("\n") for line in f.readlines()]

Pos = NamedTuple("Pos", [("x", int), ("y", int)])


def areas(positions: list[Pos]) -> list[int]:
    areas = []
    for i in range(len(positions) - 1):
        a = positions[i]
        for j in range(i + 1, len(positions)):
            b = positions[j]
            areas.append((abs(b.x - a.x) + 1) * (abs(b.y - a.y) + 1))
    return areas


def part_1(input: Input):
    positions = [Pos(*map(int, line.split(","))) for line in input]
    areas_ = areas(positions)
    return max(areas_)


def part_2(input: Input):
    positions = [Pos(*map(int, line.split(","))) for line in input]
    # for line in input:
    #     print(line)
    return 0


def _test():
    def assert_eq(a, b):
        assert a == b, f"{a} != {b}"

    assert_eq(part_1(CONTROL_1), 50)
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
