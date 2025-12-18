#!/usr/bin/env python

from typing import LiteralString
import time


def bench(fn):
    def inner(*args, **kwargs):
        start = time.perf_counter()
        res = fn(*args, **kwargs)
        t_ms = (time.perf_counter() - start) * 1000
        print(f"{fn.__name__} = {res} in {t_ms:.2f}ms")
        return res

    return inner


Input = list[str] | list[LiteralString]

CONTROL_1: Input = (
    """\
""
"abc"
"aaa\\"aaa"
"\\x27"
""".splitlines()
)

with open("2015-08.input") as f:
    input_file = [line.rstrip("\n") for line in f.readlines()]


@bench
def part_1(input: Input):
    out = 0
    for line in input:
        unescaped = bytes(line[1:-1], "ascii").decode("unicode_escape")
        out += len(line) - len(unescaped)
    return out


@bench
def part_2(input: Input):
    out = 0
    for line in input:
        escaped = line.encode("unicode_escape").replace(b'"', b'\\"')
        print(line, "->", escaped)
        out += len(escaped) - len(line) + 2
    return out


def _test():
    def assert_eq(a, b):
        assert a == b, f"{a} != {b}"

    assert_eq(part_1(CONTROL_1), 12)
    assert_eq(part_2(CONTROL_1), 19)


if __name__ == "__main__":
    _test()
    print("tests: PASS")
    print("-" * 40)
    part_1(input_file)
    part_2(input_file)
