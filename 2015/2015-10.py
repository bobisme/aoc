#!/usr/bin/env python

from typing import LiteralString
import time


Input = list[str] | list[LiteralString]


def x(seq: str):
    i = 0
    while i < len(seq):
        j = i + 1
        while j < len(seq):
            if seq[j] != seq[i]:
                break
            j += 1
        yield f"{j-i}{seq[i]}"
        i = j


def part_1(input: Input):
    seq = input[0]
    for _ in range(40):
        seq = "".join(x(seq))
    return len(seq)


def part_2(input: Input):
    seq = input[0]
    for _ in range(50):
        seq = "".join(x(seq))
    return len(seq)


def run(fn, year=2015, day=10, part=0):
    start = time.perf_counter_ns()
    res = fn()
    elapsed_ns = time.perf_counter_ns() - start
    print(f"{year}\t{day}\t{part}\t{res}\t{elapsed_ns}")
    return res


if __name__ == "__main__":
    with open("2015-10.input") as f:
        input_file = [line.rstrip("\n") for line in f.readlines()]
    run(lambda: part_1(input_file), part=1)
    run(lambda: part_2(input_file), part=2)
