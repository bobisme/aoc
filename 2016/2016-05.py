#!/usr/bin/env python

import time
from hashlib import md5
from typing import LiteralString

Input = list[str] | list[LiteralString]

CONTROL_1: Input = """\
abc
""".splitlines()


def part_1(input: Input):
    out = ""
    door_id = input[0].strip()
    index = 0
    for i in range(10_000_000_000):
        h = md5((door_id + str(i)).encode()).hexdigest()
        if h[:5] == "00000":
            out += h[5]
            index += 1
            if index > 7:
                return out


def part_2(input: Input):
    out = [""] * 8
    door_id = input[0].strip()
    count = 0
    for i in range(10_000_000_000):
        h = md5((door_id + str(i)).encode()).hexdigest()
        if h[:5] == "00000":
            try:
                index = int(h[5])
            except ValueError:
                continue
            if index > 7:
                continue
            if out[index] != "":
                continue
            out[index] = h[6]
            count += 1
            out_str = "".join(out)
            if len(out_str) == 8:
                return out_str


def _test():
    def assert_eq(a, b):
        assert a == b, f"{a} != {b}"

    assert_eq(part_1(CONTROL_1), "18f47a30")
    assert_eq(part_2(CONTROL_1), "05ace8e3")


def run(fn, year=2016, day=5, part=0):
    start = time.perf_counter_ns()
    res = fn()
    elapsed_ns = time.perf_counter_ns() - start
    print(f"{year}\t{day}\t{part}\t{res}\t{elapsed_ns}")
    return res


if __name__ == "__main__":
    with open("2016-05.input") as f:
        input_file = [line.rstrip("\n") for line in f]
    _test()
    run(lambda: part_1(input_file), part=1)
    run(lambda: part_2(input_file), part=2)
