#!/usr/bin/env python

import time
from collections.abc import Generator
from typing import LiteralString

Input = list[str] | list[LiteralString]

CONTROL_1: Input = """\
abba[mnop]qrst
abcd[bddb]xyyx
aaaa[qwer]tyui
ioxxoj[asdfgh]zxcvbn
""".splitlines()


def parse_line(line: str):
    is_hypernet = False
    start = 0
    for i, c in enumerate(line):
        if c == "[":
            if i <= start:
                continue
            yield ((start, i - 1), is_hypernet)
            is_hypernet = True
            start = i + 1
            continue
        if c == "]":
            if i <= start:
                continue
            yield ((start, i - 1), is_hypernet)
            is_hypernet = False
            start = i + 1
            continue
    yield ((start, len(line) - 1), is_hypernet)


def has_abba(s: str) -> bool:
    for i in range(len(s) - 3):
        if s[i] != s[i + 1] and (s[i], s[i + 1]) == (s[i + 3], s[i + 2]):
            return True
    return False


def find_abba(s: str) -> int | None:
    for i in range(len(s) - 3):
        if s[i] != s[i + 1] and (s[i], s[i + 1]) == (s[i + 3], s[i + 2]):
            return i


def supports_tls(s: str) -> bool:
    does_support = False
    for (start, end), is_hypernet in parse_line(s):
        abba = find_abba(s[start : end + 1])
        if abba is not None:
            if is_hypernet:
                return False
            does_support = True
    return does_support


def part_1(input: Input):
    out = 0
    for line in input:
        if supports_tls(line):
            out += 1
    return out


def find_aba(s: str) -> Generator[str]:
    for i in range(len(s) - 2):
        if s[i] != s[i + 1] and s[i] == s[i + 2]:
            yield s[i : i + 2]


def has_bab(s, a, b) -> bool:
    for i in range(len(s) - 2):
        if s[i] == b and s[i + 1] == a and s[i + 2] == b:
            return True
    return False


def supernets(s: str) -> Generator[str]:
    for (start, end), is_hypernet in parse_line(s):
        if not is_hypernet:
            yield s[start : end + 1]


def hypernets(s: str) -> Generator[str]:
    for (start, end), is_hypernet in parse_line(s):
        if is_hypernet:
            yield s[start : end + 1]


def supports_ssl(s: str) -> bool:
    for supernet in supernets(s):
        for a, b in find_aba(supernet):
            for hypernet in hypernets(s):
                if has_bab(hypernet, a, b):
                    return True
    return False


def part_2(input: Input):
    out = 0
    for line in input:
        if supports_ssl(line):
            out += 1
    return out


def _test():
    def assert_eq(a, b):
        assert a == b, f"{a} != {b}"

    assert_eq(part_1(CONTROL_1), 2)
    # assert_eq(part_2(CONTROL_1), 0)


def run(fn, year=2016, day=7, part=0):
    start = time.perf_counter_ns()
    res = fn()
    elapsed_ns = time.perf_counter_ns() - start
    print(f"{year}\t{day}\t{part}\t{res}\t{elapsed_ns}")
    return res


if __name__ == "__main__":
    with open("2016-07.input") as f:
        input_file = [line.rstrip("\n") for line in f]
    _test()
    run(lambda: part_1(input_file), part=1)
    run(lambda: part_2(input_file), part=2)
