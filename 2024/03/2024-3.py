#!/usr/bin/env python

from collections.abc import Iterable
import re

CONTROL_1 = """\
xmul(2,4)%&mul[3,7]!@^do_not_mul(5,5)+mul(32,64]then(mul(11,8)mul(8,5))
""".splitlines()

CONTROL_2 = """\
xmul(2,4)&mul[3,7]!^don't()_mul(5,5)+mul(32,64](mul(11,8)undo()?mul(8,5))
""".splitlines()

MUL = re.compile(r"mul\((\d{1,3}),(\d{1,3})\)")
INST = re.compile(r"do\(\)|don't\(\)|mul\(\d{1,3},\d{1,3}\)")

with open("2024-3.input") as f:
    input_file = [line.strip() for line in f.readlines()]


def part_1(input: Iterable[str]):
    out = 0
    for line in input:
        for a, b in re.findall(MUL, line):
            out += int(a) * int(b)
    print(out)


def part_2(input: Iterable[str]):
    out = 0
    do = True
    for line in input:
        for inst in re.findall(INST, line):
            if inst == "do()":
                do = True
            elif inst == "don't()":
                do = False
            elif do:
                a, b = re.findall(MUL, inst)[0]
                out += int(a) * int(b)
    print(out)


if __name__ == "__main__":
    part_1(input_file)
    part_2(input_file)
