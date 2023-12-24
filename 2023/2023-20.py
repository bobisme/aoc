#!/usr/bin/env python
from dataclasses import dataclass

CONTROL_1 = """\
broadcaster -> a, b, c
%a -> b
%b -> c
%c -> inv
&inv -> a
""".splitlines()
CONTROL_2 = """\
broadcaster -> a
%a -> inv, con
&inv -> b
%b -> con
&con -> output
""".splitlines()

with open("2023-20.input") as f:
    input = [line.strip() for line in f.readlines()]


@dataclass
class FlipFlop:
    id: str
    outs: list[str]
    is_on: bool = False


@dataclass
class Conj:
    id: str
    outs: list[str]
    mem: int = 0

def parse_line(line):
    left, right = line.split(' -> ', 1)

def parse(input):
    for line in input:


def main(input):
    for line in input:
        print(line)


if __name__ == "__main__":
    main(CONTROL_1)
