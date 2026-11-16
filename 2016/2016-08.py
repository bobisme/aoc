#!/usr/bin/env python

import time
from dataclasses import dataclass, field
from typing import LiteralString

Input = list[str] | list[LiteralString]

CONTROL_1: Input = """\
rect 3x2
rotate column x=1 by 1
rotate row y=0 by 4
rotate column x=1 by 1
""".splitlines()


@dataclass
class Op:
    op: str
    a: int
    b: int

    @staticmethod
    def parse(line: str) -> "Op":
        op, rest = line.split(" ", 1)
        if op == "rect":
            a, b = rest.split("x")
            return Op(op, int(a), int(b))
        if op == "rotate":
            parts = rest.split(" ")
            _, a = parts[1].split("=")
            b = parts[3]
            return Op(f"{op} {parts[0]}", int(a), int(b))
        raise ValueError()


@dataclass
class Screen:
    rows: int = 6
    cols: int = 50
    pixels: list[list[bool]] = field(init=False)

    def __post_init__(self) -> None:
        self.pixels = [[False for _ in range(self.cols)] for _ in range(self.rows)]

    def apply(self, op: Op) -> None:
        match op.op:
            case "rect":
                for i in range(op.b):
                    for j in range(op.a):
                        self.pixels[i][j] = True
            case "rotate row":
                new_row = self.pixels[op.a].copy()
                for j in range(self.cols):
                    dst_j = (j + op.b) % self.cols
                    new_row[dst_j] = self.pixels[op.a][j]
                self.pixels[op.a] = new_row
            case "rotate column":
                new_col = [self.pixels[i][op.a] for i in range(self.rows)]
                for i in range(self.rows):
                    dst_i = (i + op.b) % self.rows
                    new_col[dst_i] = self.pixels[i][op.a]
                for i in range(self.rows):
                    self.pixels[i][op.a] = new_col[i]

    def print(self):
        for row in self.pixels:
            print("".join("#" if x else "." for x in row))

    def lit(self):
        return sum(sum(row) for row in self.pixels)


def part_1(input: Input, rows=6, cols=50):
    screen = Screen(rows=rows, cols=cols)
    for line in input:
        op = Op.parse(line)
        screen.apply(op)

    return screen.lit()


def part_2(input: Input):
    _ = input
    return 0


def _test():
    def assert_eq(a, b):
        assert a == b, f"{a} != {b}"

    assert_eq(part_1(CONTROL_1, rows=3, cols=7), 6)


def run(fn, year=2016, day=8, part=0):
    start = time.perf_counter_ns()
    res = fn()
    elapsed_ns = time.perf_counter_ns() - start
    print(f"{year}\t{day}\t{part}\t{res}\t{elapsed_ns}")
    return res


if __name__ == "__main__":
    with open("2016-08.input") as f:
        input_file = [line.rstrip("\n") for line in f]
    _test()
    run(lambda: part_1(input_file), part=1)
    run(lambda: part_2(input_file), part=2)
