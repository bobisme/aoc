#!/usr/bin/env python

import math
import re
from typing import Generator, NamedTuple


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


def decimals(i: int):
    return math.floor(math.log10(i)) + 1


# original solution
def part_1_stringy(input):
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


def decompose_symmetric_ranges(r: Range) -> Generator[Range]:
    min_d, max_d = decimals(r.start), decimals(r.end)
    if min_d == max_d:
        if min_d % 2 == 0:
            yield r
        return
    for d in range(min_d, max_d + 1):
        if d % 2 != 0:
            continue
        if d == min_d:
            yield Range(r.start, 1 * 10**d - 1)
        elif d == max_d:
            yield Range(1 * 10 ** (d - 1), r.end)
        else:
            yield Range(1 * 10 ** (d - 1), 1 * 10**d - 1)


def part_1(input):
    ranges = parse(input)
    invalid_count = 0
    for full_range in ranges:
        for r in decompose_symmetric_ranges(full_range):
            # start_d = decimals(r.start)
            # end_d = decimals(r.end)
            for i in range(r.start, r.end + 1):
                n = decimals(i)
                if n % 2 != 0:
                    continue
                zeroes = 10 ** (n // 2)
                left = i // zeroes
                right = i - (left * zeroes)
                if left == right:
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

    assert_eq(
        list(decompose_symmetric_ranges(Range(22, 4444))),
        [Range(22, 99), Range(1000, 4444)],
    )


def _bench(fn, count=100):
    import timeit

    return timeit.timeit(fn, number=count) / count * 1_000


if __name__ == "__main__":
    _test()
    print("tests: PASS")
    print("-" * 40)
    print("part_1:", part_1(input_file))
    print("part_2:", part_2(input_file))
    print("-" * 40)
    print("part_1 bench: {:.1f}ms".format(_bench(lambda: part_1(input_file), count=10)))
    print("part_2 bench: {:.1f}ms".format(_bench(lambda: part_2(input_file), count=1)))
