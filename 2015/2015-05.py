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

with open("2015-05.input") as f:
    input_file = [line.rstrip("\n") for line in f.readlines()]


def _bench(fn):
    def inner(*args, **kwargs):
        start = time.perf_counter()
        res = fn(*args, **kwargs)
        t_ms = (time.perf_counter() - start) * 1000
        print(f"{fn.__name__} = {res} in {t_ms:.2f}ms")
        return res

    return inner


@_bench
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


@_bench
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


if __name__ == "__main__":
    _test()
    print("tests: PASS")
    print("-" * 40)
    part_1(input_file)
    part_2(input_file)
