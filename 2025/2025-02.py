#!/usr/bin/env python

import re
from typing import NamedTuple


CONTROL_1 = """\
11-22,95-115,998-1012,1188511880-1188511890,222220-222224,1698522-1698528,446443-446449,38593856-38593862,565653-565659,824824821-824824827,2121212118-2121212124
""".splitlines()

with open("2025-2.input") as f:
    input_file = [line.strip() for line in f.readlines()]

Range = NamedTuple("Range", [("start", int), ("end", int)])


def parse(input: list[str]) -> list[Range]:
    out = []
    for r in input[0].split(","):
        start, end = map(int, r.split("-"))
        out.append(Range(start=start, end=end))
    return out


def part_1(input):
    ranges = parse(input)
    invalid_count = 0
    for r in ranges:
        for i in range(r.start, r.end + 1):
            s = str(i)
            n = len(s)
            if n % 2 != 0:
                continue
            if s[: n // 2] == s[n // 2 :]:
                invalid_count += i
    return invalid_count


def part_2(input):
    ranges = parse(input)
    invalid_count = 0
    for r in ranges:
        for i in range(r.start, r.end + 1):
            if re.match(r"^(\d+)(\1)+$", str(i)):
                invalid_count += i
    return invalid_count


# slower than re: 2.29s vs 1.66s
def part_2_without_re(input):
    def pat_fills(s: str, pat: str) -> bool:
        for start in range(len(pat), len(s), len(pat)):
            if s[start : start + len(pat)] != pat:
                return False
        return True

    def has_repeats(s: str) -> bool:
        n = len(s)
        for pat_len in range(1, n // 2 + 1):
            if n % pat_len != 0:
                continue
            pat = s[:pat_len]
            if pat_fills(s, pat):
                return True
        return False

    ranges = parse(input)
    invalid_count = 0
    for r in ranges:
        for i in range(r.start, r.end + 1):
            if has_repeats(str(i)):
                invalid_count += i
    return invalid_count


def _test():
    def assert_eq(a, b):
        assert a == b, f"{a} != {b}"

    assert_eq(part_1(CONTROL_1), 1227775554)
    assert_eq(part_2(CONTROL_1), 4174379265)
    assert_eq(part_2_without_re(CONTROL_1), 4174379265)


if __name__ == "__main__":
    _test()
    print(part_1(input_file))
    print(part_2(input_file))
