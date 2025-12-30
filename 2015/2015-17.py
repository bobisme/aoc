#!/usr/bin/env python

import itertools
from typing import Generator, LiteralString
import time


Input = list[str] | list[LiteralString]

CONTROL_1: Input = (
    """\
20
15
10
5
5
""".splitlines()
)


def part_1(input: Input, amount: int = 150):
    containers = list(map(int, input))
    out = 0
    for count in range(1, len(containers)):
        for comb in itertools.combinations(containers, count):
            if sum(comb) == amount:
                out += 1
    return out


def part_2(input: Input, amount: int = 150):
    containers = sorted(list(map(int, input)), reverse=True)

    def get_comb(containers: list[int]) -> Generator[tuple[int, ...]]:
        max_count = len(containers) - 1
        count = 1
        while count <= max_count:
            for comb in itertools.combinations(containers, count):
                if sum(comb) == amount:
                    max_count = count
                    yield comb
            count += 1

    return sum(1 for _ in get_comb(containers))


def _test():
    def assert_eq(a, b):
        assert a == b, f"{a} != {b}"

    assert_eq(part_1(CONTROL_1, amount=25), 4)
    assert_eq(part_2(CONTROL_1, amount=25), 3)


def run(fn, year=2015, day=1, part=0):
    start = time.perf_counter_ns()
    res = fn()
    elapsed_ns = time.perf_counter_ns() - start
    print(f"{year}\t{day}\t{part}\t{res}\t{elapsed_ns}")
    return res


if __name__ == "__main__":
    with open("2015-17.input") as f:
        input_file = [line.rstrip("\n") for line in f.readlines()]
    _test()
    run(lambda: part_1(input_file), part=1)
    run(lambda: part_2(input_file), part=2)
    # part_2(input_file)
