#!/usr/bin/env python

from collections.abc import Iterator
import itertools
import math
from typing import LiteralString
import time

Input = list[str] | list[LiteralString]

CONTROL_1: Input = (
    """\
1
2
3
4
5
7
8
9
10
11
""".splitlines()
)


def get_weights(input: Input) -> tuple[int, ...]:
    return tuple(map(int, input))


def get_groups(weights: list[int], n_groups=3) -> Iterator[tuple[int, ...]]:
    expected = sum(weights) // n_groups
    smallest_group_size = 10000
    for count in range(2, len(weights) // 2):
        if count > smallest_group_size:
            break
        for comb in itertools.combinations(weights, count):
            if sum(comb) == expected:
                if count < smallest_group_size:
                    smallest_group_size = count
                yield comb


def part_1(input: Input):
    weights = list(get_weights(input))
    weights.sort(reverse=True)
    groups = get_groups(weights)
    return math.prod(min(groups, key=lambda x: math.prod(x)))


def part_2(input: Input):
    weights = list(get_weights(input))
    weights.sort(reverse=True)
    groups = get_groups(weights, n_groups=4)
    return math.prod(min(groups, key=lambda x: math.prod(x)))


def _test():
    def assert_eq(a, b):
        assert a == b, f"{a} != {b}"

    assert_eq(part_1(CONTROL_1), 99)
    assert_eq(part_2(CONTROL_1), 44)


def run(fn, year=2015, day=24, part=0):
    start = time.perf_counter_ns()
    res = fn()
    elapsed_ns = time.perf_counter_ns() - start
    print(f"{year}\t{day}\t{part}\t{res}\t{elapsed_ns}")
    return res


if __name__ == "__main__":
    with open("2015-24.input") as f:
        input_file = [line.rstrip("\n") for line in f.readlines()]
    _test()
    run(lambda: part_1(input_file), part=1)
    run(lambda: part_2(input_file), part=2)
