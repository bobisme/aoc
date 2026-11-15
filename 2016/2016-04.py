#!/usr/bin/env python

from typing import LiteralString
import time
from collections import Counter

Input = list[str] | list[LiteralString]

CONTROL_1: Input = """\
aaaaa-bbb-z-y-x-123[abxyz]
a-b-c-d-e-f-g-h-987[abcde]
not-a-real-room-404[oarel]
totally-real-room-200[decoy]
""".splitlines()


def parse(line: str):
    name, rest = line.rsplit("-", 1)
    sector, rest = rest.split("[")
    checksum = rest[:-1]
    return name, int(sector), checksum


def part_1(input: Input):
    out = 0
    for line in input:
        name, sector, checksum = parse(line)
        c = Counter(name.replace("-", ""))
        keys = [(-count, key) for key, count in c.items()]
        keys.sort()
        calculated = "".join(k[1] for k in keys)[:5]
        if calculated == checksum:
            out += sector
    return out


def part_2(input: Input):
    A = ord("a")
    for line in input:
        name, sector, _ = parse(line)
        rotated = "".join(
            chr(((ord(c) - A + sector) % 26) + A) if c != "-" else "-" for c in name
        )
        if rotated == "northpole-object-storage":
            return sector
    return 0


def _test():
    def assert_eq(a, b):
        assert a == b, f"{a} != {b}"

    assert_eq(part_1(CONTROL_1), 1514)


def run(fn, year=2016, day=4, part=0):
    start = time.perf_counter_ns()
    res = fn()
    elapsed_ns = time.perf_counter_ns() - start
    print(f"{year}\t{day}\t{part}\t{res}\t{elapsed_ns}")
    return res


if __name__ == "__main__":
    with open("2016-04.input") as f:
        input_file = [line.rstrip("\n") for line in f.readlines()]
    _test()
    run(lambda: part_1(input_file), part=1)
    run(lambda: part_2(input_file), part=2)
