#!/usr/bin/env python

import re
from typing import LiteralString
import time

Input = list[str] | list[LiteralString]

CONTROL_1: Input = (
    """\
ugknbfddgicrmopn
aaa
jchzalrnumimnmhp
haegwjzuvuyypxyu
dvszwmarrgswjxmb
""".splitlines()
)

CONTROL_2: Input = (
    """\
qjhvhtzxzqqjkmpb
xxyxx
uurcxstgmygtbstg
ieodomkazucvgmuy
""".splitlines()
)


def part_1(input: Input):
    def count_repeats(s: str) -> int:
        return sum(
            map(lambda x: 1 if x[0] > 0 and s[x[0] - 1] == x[1] else 0, enumerate(s))
        )

    def has_bad(s: str) -> bool:
        return any(
            x
            for x in map(
                lambda x: s[x[0] - 1] + x[1],
                ((i, x) for (i, x) in enumerate(s) if i > 0),
            )
            if x in ("ab", "cd", "pq", "xy")
        )

    def is_nice(s: str) -> bool:
        if sum(1 for c in s if c in ("aeiou")) < 3:
            return False
        if count_repeats(s) < 1:
            return False
        return not has_bad(s)

    return sum(1 for s in input if is_nice(s))


def part_2(input: Input):
    def is_nice(s: str) -> int:
        return (any(re.finditer(r"(..).*(\1)", s))) and (
            any(re.finditer(r"(.).(\1)", s))
        )

    return sum(1 for s in input if is_nice(s))


def _test():
    def assert_eq(a, b):
        assert a == b, f"{a} != {b}"

    assert_eq(part_1(CONTROL_1), 2)
    assert_eq(part_2(CONTROL_2), 2)


def run(fn, year=2015, day=5, part=0):
    start = time.perf_counter_ns()
    res = fn()
    elapsed_ns = time.perf_counter_ns() - start
    print(f"{year}\t{day}\t{part}\t{res}\t{elapsed_ns}")
    return res


if __name__ == "__main__":
    with open("2015-05.input") as f:
        input_file = [line.rstrip("\n") for line in f.readlines()]
    _test()
    run(lambda: part_1(input_file), part=1)
    run(lambda: part_2(input_file), part=2)
