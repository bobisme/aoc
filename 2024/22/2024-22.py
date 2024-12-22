#!/usr/bin/env python

from itertools import islice
from typing import DefaultDict, Generator


CONTROL_1 = """\
1
10
100
2024
""".splitlines()

CONTROL_2 = """\
1
2
3
2024
""".splitlines()

with open("2024-22.input") as f:
    input_file = [line.strip() for line in f.readlines()]


def ticks(n: int) -> Generator[int, None, None]:
    MOD = 16777216
    while True:
        n = (n ^ (n << 6)) % MOD
        n = (n ^ (n >> 5)) % MOD
        n = (n ^ (n << 11)) % MOD
        yield n


def nth_tick(n: int, count=1) -> int:
    return next(islice(ticks(n), count - 1, count))


def part_1(input):
    nums = [int(line) for line in input]
    s = sum(nth_tick(n, count=2000) for n in nums)
    print(s)


def gen_deltas(n: int) -> Generator[int, None, None]:
    prices = (t % 10 for t in ticks(n))
    last = n % 10
    while True:
        nxt = next(prices)
        yield nxt - last
        last = nxt


def search(deltas: list[int], pattern: tuple[int, ...]) -> int | None:
    for i in range(0, len(deltas) - len(pattern)):
        delta_chunk = deltas[i : i + len(pattern)]
        if pattern == delta_chunk:
            return i
    return None


def part_2(input):
    nums = [int(line) for line in input]
    patterns: DefaultDict[tuple[int, ...], dict[int, int]] = DefaultDict(dict)
    price_lists = []
    delta_lists = []
    for n_idx, n in enumerate(nums):
        prices = list(islice((x % 10 for x in ticks(n)), 0, 2000))
        price_lists.append(prices)
        deltas = list(islice(gen_deltas(n), 0, 2000))
        delta_lists.append(deltas)
        for i in range(len(deltas) - 4):
            pattern = deltas[i : i + 4]
            pdict = patterns[tuple(pattern)]
            if n_idx not in pdict:
                price = prices[i + 3]
                pdict[n_idx] = price

    print(max(sum(values.values()) for values in patterns.values()))


if __name__ == "__main__":
    part_1(input_file)
    part_2(input_file)
