#!/usr/bin/env python

from itertools import pairwise

CONTROL_1 = """\
0 3 6 9 12 15
1 3 6 10 15 21
10 13 16 21 30 45
""".splitlines()

with open("2023-9.input") as f:
    input = [line.strip() for line in f.readlines()]


def print_trapezoid(trap: list[list[int]]) -> None:
    for i, line in enumerate(trap):
        print("   " * i, " ".join((str(x).rjust(6) for x in line)))


def parse_trapezoid(line):
    last_line = [int(x) for x in line.split(" ")]
    yield last_line
    while last_line[-1] != 0:
        next_line = [b - a for (a, b) in pairwise(last_line)]
        yield next_line
        last_line = next_line


def extrapolate_trapezoid(trap) -> int:
    next = 0
    for line in reversed(trap[:-1]):
        next += line[-1]
        # print(next)
    return next


def extrapolate_trapezoid_back(trap) -> int:
    next = 0
    for line in reversed(trap[:-1]):
        next = line[0] - next
        # print(next)
    return next


def main(input):
    traps = [list(parse_trapezoid(line)) for line in input]
    for trap in traps:
        # print_trapezoid(trap)
        next = extrapolate_trapezoid_back(trap)
        print(next)
    print("sum forward =", sum(extrapolate_trapezoid(trap) for trap in traps))
    print("sum back =", sum(extrapolate_trapezoid_back(trap) for trap in traps))
    # print(trap)


if __name__ == "__main__":
    main(input)
