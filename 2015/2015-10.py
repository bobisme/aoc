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

with open("2015-10.input") as f:
    input_file = [line.rstrip("\n") for line in f.readlines()]


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


@bench
def part_1(input: Input):
    seq = input[0]
    for _ in range(40):
        seq = "".join(x(seq))
    return len(seq)


@bench
def part_2(input: Input):
    seq = input[0]
    for _ in range(50):
        seq = "".join(x(seq))
    return len(seq)


if __name__ == "__main__":
    part_1(input_file)
    part_2(input_file)
