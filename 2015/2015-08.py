#!/usr/bin/env python

from typing import LiteralString
import time


Input = list[str] | list[LiteralString]

CONTROL_1: Input = (
    """\
""
"abc"
"aaa\\"aaa"
"\\x27"
""".splitlines()
)


def part_1(input: Input):
    out = 0
    for line in input:
        unescaped = bytes(line[1:-1], "ascii").decode("unicode_escape")
        out += len(line) - len(unescaped)
    return out


def part_2(input: Input):
    out = 0
    for line in input:
        escaped = line.encode("unicode_escape").replace(b'"', b'\\"')
        out += len(escaped) - len(line) + 2
    return out


def _test():
    def assert_eq(a, b):
        assert a == b, f"{a} != {b}"

    assert_eq(part_1(CONTROL_1), 12)
    assert_eq(part_2(CONTROL_1), 19)


def run(fn, year=2015, day=8, part=0):
    start = time.perf_counter_ns()
    res = fn()
    elapsed_ns = time.perf_counter_ns() - start
    print(f"{year}\t{day}\t{part}\t{res}\t{elapsed_ns}")
    return res


if __name__ == "__main__":
    with open("2015-08.input") as f:
        input_file = [line.rstrip("\n") for line in f.readlines()]
    _test()
    run(lambda: part_1(input_file), part=1)
    run(lambda: part_2(input_file), part=2)
