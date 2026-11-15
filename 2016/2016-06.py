#!/usr/bin/env python

import time
from collections import defaultdict
from typing import LiteralString

Input = list[str] | list[LiteralString]

CONTROL_1: Input = """\
eedadn
drvtee
eandsr
raavrd
atevrs
tsrnev
sdttsa
rasrtv
nssdts
ntnada
svetve
tesnvt
vntsnd
vrdear
dvrsen
enarar
""".splitlines()


def part_1(input: Input):
    length = len(input[0])
    counters = [defaultdict(int) for _ in range(length)]
    for line in input:
        for i in range(length):
            counters[i][line[i]] += 1
    out = ""
    for i in range(length):
        out += max(counters[i].items(), key=lambda item: item[1])[0]
    return out


def part_2(input: Input):
    length = len(input[0])
    counters = [defaultdict(int) for _ in range(length)]
    for line in input:
        for i in range(length):
            counters[i][line[i]] += 1
    out = ""
    for i in range(length):
        out += min(counters[i].items(), key=lambda item: item[1])[0]
    return out


def _test():
    def assert_eq(a, b):
        assert a == b, f"{a} != {b}"

    assert_eq(part_1(CONTROL_1), "easter")
    assert_eq(part_2(CONTROL_1), "advent")


def run(fn, year=2016, day=6, part=0):
    start = time.perf_counter_ns()
    res = fn()
    elapsed_ns = time.perf_counter_ns() - start
    print(f"{year}\t{day}\t{part}\t{res}\t{elapsed_ns}")
    return res


if __name__ == "__main__":
    with open("2016-06.input") as f:
        input_file = [line.rstrip("\n") for line in f]
    _test()
    run(lambda: part_1(input_file), part=1)
    run(lambda: part_2(input_file), part=2)
