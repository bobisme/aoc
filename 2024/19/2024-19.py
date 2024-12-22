#!/usr/bin/env python

from collections import deque
from typing import Iterable


CONTROL_1 = """\
r, wr, b, g, bwu, rb, gb, br

brwrr
bggr
gbbr
rrbgbr
ubwu
bwurrg
brgr
bbrgwb
""".splitlines()

with open("2024-19.input") as f:
    input_file = [line.strip() for line in f.readlines()]


def parse(input: list[str]) -> tuple[list[str], list[str]]:
    towel_line = input[0]
    patterns = input[2:]
    return towel_line.split(", "), patterns


def dfs(towels: list[str], pattern: str):
    memo = {}

    def inner(pattern: str):
        if pattern in memo:
            return memo[pattern]

        if len(pattern) == 0:
            return True

        for towel in towels:
            if pattern.startswith(towel):
                remaining = pattern[len(towel) :]
                if inner(remaining):
                    memo[remaining] = True
                    return True

        memo[pattern] = False
        return False

    return inner(pattern)


def part_1(input):
    towels, patterns = parse(input)
    count = 0
    for p in patterns:
        if dfs(towels, p):
            count += 1
    print(count)


def dfs2(towels: list[str], pattern: str) -> int:
    memo = {}

    def inner(pattern: str) -> int:
        if pattern in memo:
            return memo[pattern]

        if len(pattern) == 0:
            return 1

        subtotal = 0
        for towel in towels:
            if pattern.startswith(towel):
                remaining = pattern[len(towel) :]
                subtotal += inner(remaining)

        memo[pattern] = subtotal
        return subtotal

    return inner(pattern)


def part_2(input):
    towels, patterns = parse(input)
    count = 0
    for p in patterns:
        count += dfs2(towels, p)
    print(count)


if __name__ == "__main__":
    part_1(input_file)
    part_2(input_file)
