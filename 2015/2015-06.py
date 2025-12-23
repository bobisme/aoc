#!/usr/bin/env python

from enum import Enum
from typing import Iterable, LiteralString, NamedTuple
import time

Input = list[str] | list[LiteralString]


class Pos(NamedTuple):
    i: int
    j: int


class Op(Enum):
    On = "on"
    Off = "off"
    Toggle = "toggle"

    @staticmethod
    def from_str(s: str) -> "Op":
        if s == "turn on":
            return Op.On
        if s == "turn off":
            return Op.Off
        if s == "toggle":
            return Op.Toggle
        assert False

    def __repr__(self) -> str:
        return self.value


class Direction(NamedTuple):
    op: Op
    start: Pos
    end: Pos

    def rows(self) -> Iterable[int]:
        return iter(range(self.start.i, self.end.i + 1))

    def cols(self) -> Iterable[int]:
        return iter(range(self.start.j, self.end.j + 1))


def op_from_line(line: str) -> Direction:
    op, start, _, end = line.rsplit(" ", 3)
    return Direction(
        Op.from_str(op),
        Pos(*map(int, start.split(",", 1))),
        Pos(*map(int, end.split(",", 1))),
    )


def parse(input: Input) -> list[Direction]:
    return [op_from_line(line) for line in input]


def part_1(input: Input):
    grid = [[0 for _ in range(1_000)] for _ in range(1_000)]
    dirs = parse(input)
    for dir in dirs:
        for row in dir.rows():
            for col in dir.cols():
                match dir.op:
                    case Op.On:
                        grid[row][col] = 1
                    case Op.Off:
                        grid[row][col] = 0
                    case Op.Toggle:
                        grid[row][col] ^= 1
    return sum(sum(row) for row in grid)


def part_2(input: Input):
    grid = [[0 for _ in range(1_000)] for _ in range(1_000)]
    dirs = parse(input)
    for dir in dirs:
        for row in dir.rows():
            for col in dir.cols():
                match dir.op:
                    case Op.On:
                        grid[row][col] += 1
                    case Op.Off:
                        grid[row][col] = max(0, grid[row][col] - 1)
                    case Op.Toggle:
                        grid[row][col] += 2
    return sum(sum(row) for row in grid)


def run(fn, year=2015, day=6, part=0):
    start = time.perf_counter_ns()
    res = fn()
    elapsed_ns = time.perf_counter_ns() - start
    print(f"{year}\t{day}\t{part}\t{res}\t{elapsed_ns}")
    return res


if __name__ == "__main__":
    with open("2015-06.input") as f:
        input_file = [line.rstrip("\n") for line in f.readlines()]
    run(lambda: part_1(input_file), part=1)
    run(lambda: part_2(input_file), part=2)
