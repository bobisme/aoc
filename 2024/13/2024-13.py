#!/usr/bin/env python

from collections.abc import Generator
import re
from typing import NamedTuple, Optional
import math

ADD = 10000000000000

CONTROL_1 = """\
Button A: X+94, Y+34
Button B: X+22, Y+67
Prize: X=8400, Y=5400

Button A: X+26, Y+66
Button B: X+67, Y+21
Prize: X=12748, Y=12176

Button A: X+17, Y+86
Button B: X+84, Y+37
Prize: X=7870, Y=6450

Button A: X+69, Y+23
Button B: X+27, Y+71
Prize: X=18641, Y=10279
""".splitlines()

with open("2024-13.input") as f:
    input_file = [line.strip() for line in f.readlines()]

Button = NamedTuple("Button", [("x", int), ("y", int)])
Prize = NamedTuple("Prize", [("x", int), ("y", int)])
Machine = NamedTuple("Machine", [("a", Button), ("b", Button), ("prize", Prize)])


def parse(input: list[str]) -> Generator[Machine, None, None]:
    button_re = re.compile(r"Button \w: X\+(\d+), Y\+(\d+)")
    prize_re = re.compile(r"Prize: X=(\d+), Y=(\d+)")

    for i in range(0, len(input), 4):
        a = Button(*(int(x) for x in button_re.findall(input[i])[0]))
        b = Button(*(int(x) for x in button_re.findall(input[i + 1])[0]))
        prize = Prize(*(int(x) for x in prize_re.findall(input[i + 2])[0]))
        yield Machine(a, b, prize)


def get_nums(machine: Machine, add=0) -> Generator[tuple[int, int], None, None]:
    a = machine.a
    b = machine.b
    p = machine.prize
    p = Prize(p.x + add, p.y + add)
    for n in range(0, 100):
        mx = (p.x - (n * a.x)) / b.x
        my = (p.y - (n * a.y)) / b.y
        if mx != my:
            continue
        if int(mx) != mx:
            continue
        if mx < 0:
            continue
        yield (n, int(mx))


def do_algebra(machine: Machine, add=0) -> Optional[tuple[int, int]]:
    a = machine.a
    b = machine.b
    p = Prize(machine.prize.x + add, machine.prize.y + add)
    d = a.x * b.y - a.y * b.x
    m = (p.x * b.y - b.x * p.y) / d
    n = (a.x * p.y - p.x * a.y) / d
    if math.trunc(m) != m:
        return None
    if math.trunc(n) != n:
        return None
    return (int(m), int(n))


def part_1(input):
    machines = list(parse(input))
    out = 0
    for machine in machines:
        nums = list(get_nums(machine, add=0))
        if nums:
            out += min(3 * n + m for (n, m) in nums)
    print(out)


def part_2(input):
    machines = list(parse(input))
    out = 0
    for machine in machines:
        res = do_algebra(machine, add=ADD)
        if res:
            out += 3 * res[0] + res[1]
    print(out)


if __name__ == "__main__":
    part_1(input_file)
    part_2(input_file)
