#!/usr/bin/env python

import math
from typing import DefaultDict, Iterator, LiteralString
import time

Input = list[str] | list[LiteralString]


def divisors(n: int) -> Iterator[int]:
    yield 1
    if n != 1:
        yield n
    for i in range(2, int(math.sqrt(n)) + 1):
        if n % i == 0:
            yield i
            if (b := n // i) != i:
                yield b


def part_1(input: Input):
    target = int(input[0])
    for i in range(800_000, 1_000_000):
        x = sum(divisors(i)) * 10
        if x >= target:
            return i
    assert not "unreachable"


def limited_divisors(n: int, limit=50) -> Iterator[int]:
    for i in range(1, int(math.sqrt(n)) + 1):
        if n % i == 0:
            b = n // i
            if b <= limit:
                yield i
            if b != i:
                yield b


def part_2(input: Input):
    target = int(input[0])
    counts = DefaultDict(int)
    for i in range(0, 2_000_000):
        sum_ = 0
        for div in divisors(i):
            if counts[div] >= 50:
                continue
            counts[div] += 1
            sum_ += div

        if sum_ * 11 >= target:
            return i
    assert not "unreachable"


def run(fn, year=2015, day=20, part=0):
    start = time.perf_counter_ns()
    res = fn()
    elapsed_ns = time.perf_counter_ns() - start
    print(f"{year}\t{day}\t{part}\t{res}\t{elapsed_ns}")
    return res


if __name__ == "__main__":
    with open("2015-20.input") as f:
        input_file = [line.rstrip("\n") for line in f.readlines()]
    run(lambda: part_1(input_file), part=1)
    run(lambda: part_2(input_file), part=2)
