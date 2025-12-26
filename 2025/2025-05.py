#!/usr/bin/env python

import time
from typing import Generator, LiteralString, NamedTuple

Input = list[str] | list[LiteralString]

CONTROL_1: Input = (
    """\
3-5
10-14
16-20
12-18

1
5
8
11
17
32
""".splitlines()
)

Range = NamedTuple("Range", [("start", int), ("end", int)])


def parse(input: Input) -> tuple[list[Range], list[int]]:
    ranges = []
    item_ids = []

    gen = (line for line in input)

    for line in gen:
        if line == "":
            break
        ranges.append(Range(*map(int, line.split("-"))))
    for line in gen:
        item_ids.append(int(line))
    return ranges, item_ids


def part_1(input: Input):
    ranges, item_ids = parse(input)

    def in_range(r: Range, n: int):
        return r.start <= n <= r.end

    fresh_count = 0
    for n in item_ids:
        for r in ranges:
            if in_range(r, n):
                fresh_count += 1
                break
    return fresh_count


def merge_2_ranges(a: Range, b: Range) -> tuple[Range, Range | None]:
    if b.start < a.start:
        a, b = b, a
    if b.start >= a.start and b.end <= a.end:
        return a, None
    if (a.end + 1) >= b.start:
        return Range(a.start, b.end), None
    return a, b


def merge_ranges(ranges: list[Range]) -> Generator[Range]:
    if len(ranges) <= 1:
        yield from ranges
        return

    ranges.sort(key=lambda r: r.start)
    i = 0
    j = 1
    left = ranges[i]
    while i < len(ranges):
        if j >= len(ranges):
            yield left
            break
        left, right = merge_2_ranges(left, ranges[j])
        if right is None:
            j += 1
            continue
        else:
            yield left
            left = right
            i = j
            j += 1


def part_2(input: Input):
    ranges, _ = parse(input)
    ranges = list(merge_ranges(ranges))
    return sum(r.end - (r.start - 1) for r in ranges)


def _test():
    def assert_eq(a, b):
        assert a == b, f"{a} != {b}"

    assert_eq(part_1(CONTROL_1), 3)
    # merge_2_ranges
    assert_eq(merge_2_ranges(Range(1, 3), Range(5, 6)), (Range(1, 3), Range(5, 6)))
    assert_eq(merge_2_ranges(Range(1, 3), Range(4, 6)), (Range(1, 6), None))
    assert_eq(merge_2_ranges(Range(1, 5), Range(4, 6)), (Range(1, 6), None))
    assert_eq(merge_2_ranges(Range(1, 10), Range(4, 6)), (Range(1, 10), None))
    assert_eq(merge_2_ranges(Range(4, 6), Range(1, 10)), (Range(1, 10), None))
    # part_2
    assert_eq(part_2(CONTROL_1), 14)


def run(fn, year=2025, day=5, part=0):
    start = time.perf_counter_ns()
    res = fn()
    elapsed_ns = time.perf_counter_ns() - start
    print(f"{year}\t{day}\t{part}\t{res}\t{elapsed_ns}")


if __name__ == "__main__":
    with open("2025-05.input") as f:
        input_file = [line.strip() for line in f.readlines()]
    _test()
    run(lambda: part_1(input_file), part=1)
    run(lambda: part_2(input_file), part=2)
