#!/usr/bin/env python

import itertools
import re
from typing import DefaultDict, LiteralString
import time


Input = list[str] | list[LiteralString]

CONTROL_1: Input = (
    """\
Alice would gain 54 happiness units by sitting next to Bob.
Alice would lose 79 happiness units by sitting next to Carol.
Alice would lose 2 happiness units by sitting next to David.
Bob would gain 83 happiness units by sitting next to Alice.
Bob would lose 7 happiness units by sitting next to Carol.
Bob would lose 63 happiness units by sitting next to David.
Carol would lose 62 happiness units by sitting next to Alice.
Carol would gain 60 happiness units by sitting next to Bob.
Carol would gain 55 happiness units by sitting next to David.
David would gain 46 happiness units by sitting next to Alice.
David would lose 7 happiness units by sitting next to Bob.
David would gain 41 happiness units by sitting next to Carol.
""".splitlines()
)

Constraints = dict[str, dict[str, int]]


def happiness(constraints: Constraints, order: tuple[str, ...]) -> int:
    val = 0
    for i in range(len(order) - 1):
        a = order[i]
        b = order[i + 1]
        val += constraints[a][b] + constraints[b][a]
    return val + constraints[order[-1]][order[0]] + constraints[order[0]][order[-1]]


def exhaustive(constraints: Constraints) -> tuple[int, tuple[str, ...]]:
    """
    n! solution
    """
    maximal = 0
    best_order = tuple()
    for order in itertools.permutations(constraints.keys()):
        val = happiness(constraints, order)
        if val > maximal:
            maximal = val
            best_order = order
    return maximal, best_order


def parse(input: Input) -> Constraints:
    pattern = re.compile(
        r"(\w+) would (\w+) (\d+) happiness units by sitting next to (\w+)."
    )
    data = DefaultDict(dict)
    for line in input:
        matches = next(pattern.finditer(line))
        assert matches is not None
        person, action, units, other_person = matches.groups()
        amount = (1 if action == "gain" else -1) * int(units)
        data[person][other_person] = amount
    return data


def part_1(input: Input):
    constraints = parse(input)
    return exhaustive(constraints)[0]


def part_2(input: Input):
    constraints = parse(input)
    others = constraints.keys()
    for person in others:
        constraints[person]["Me"] = 0
    constraints["Me"] = {p: 0 for p in others}
    return exhaustive(constraints)[0]


def _test():
    def assert_eq(a, b):
        assert a == b, f"{a} != {b}"

    assert_eq(part_1(CONTROL_1), 330)


def run(fn, year=2015, day=13, part=0):
    start = time.perf_counter_ns()
    res = fn()
    elapsed_ns = time.perf_counter_ns() - start
    print(f"{year}\t{day}\t{part}\t{res}\t{elapsed_ns}")
    return res


if __name__ == "__main__":
    with open("2015-13.input") as f:
        input_file = [line.rstrip("\n") for line in f.readlines()]
    _test()
    run(lambda: part_1(input_file), part=1)
    run(lambda: part_2(input_file), part=2)
