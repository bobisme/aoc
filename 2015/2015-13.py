#!/usr/bin/env python

import itertools
import re
from typing import DefaultDict, LiteralString
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

with open("2015-13.input") as f:
    input_file = [line.rstrip("\n") for line in f.readlines()]

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


@bench
def part_1(input: Input):
    constraints = parse(input)
    return exhaustive(constraints)[0]


@bench
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


if __name__ == "__main__":
    _test()
    print("tests: PASS")
    print("-" * 40)
    part_1(input_file)
    part_2(input_file)
