#!/usr/bin/env python

from dataclasses import dataclass
from typing import LiteralString
import time

Input = list[str] | list[LiteralString]


def part_1(input: Input):
    turns = input[0].split(", ")
    posx, posy = 0, 0
    current_dir = 0
    for turn in turns:
        dir = turn[0]
        steps = int(turn[1:])
        if dir == "L":
            current_dir -= 1
        else:
            current_dir += 1
        current_dir %= 4
        match current_dir:
            case 0:
                posy += steps
            case 1:
                posx += steps
            case 2:
                posy -= steps
            case 3:
                posx -= steps

    return abs(posx) + abs(posy)


@dataclass(slots=True)
class Pos:
    x: int
    y: int

    def __hash__(self) -> int:
        return hash((self.x, self.y))

    def dist(self) -> int:
        return abs(self.x) + abs(self.y)


def part_2(input: Input):
    turns = input[0].split(", ")
    current_dir = 0
    current_pos = Pos(0, 0)
    visited: set[Pos] = {current_pos}
    for turn in turns:
        dir, steps = turn[0], int(turn[1:])

        if dir == "L":
            current_dir -= 1
        else:
            current_dir += 1
        current_dir %= 4

        next_pos = Pos(0, 0)
        match current_dir:
            case 0:
                for s in range(1, steps + 1):
                    next_pos = Pos(current_pos.x, current_pos.y + s)
                    if next_pos in visited:
                        return next_pos.dist()
                    visited.add(next_pos)
            case 1:
                for s in range(1, steps + 1):
                    next_pos = Pos(current_pos.x + s, current_pos.y)
                    if next_pos in visited:
                        return next_pos.dist()
                    visited.add(next_pos)
            case 2:
                for s in range(1, steps + 1):
                    next_pos = Pos(current_pos.x, current_pos.y - s)
                    if next_pos in visited:
                        return next_pos.dist()
                    visited.add(next_pos)
            case 3:
                for s in range(1, steps + 1):
                    next_pos = Pos(current_pos.x - s, current_pos.y)
                    if next_pos in visited:
                        return next_pos.dist()
                    visited.add(next_pos)
        current_pos = next_pos
    raise Exception("FAILED")


def _test():
    def assert_eq(a, b):
        assert a == b, f"{a} != {b}"

    assert_eq(part_1(["R2, L3"]), 5)
    assert_eq(part_1(["R2, R2, R2"]), 2)
    assert_eq(part_1(["R5, L5, R5, R3"]), 12)
    assert_eq(part_2(["R8, R4, R4, R8"]), 4)


def run(fn, year=2016, day=1, part=0):
    start = time.perf_counter_ns()
    res = fn()
    elapsed_ns = time.perf_counter_ns() - start
    print(f"{year}\t{day}\t{part}\t{res}\t{elapsed_ns}")
    return res


if __name__ == "__main__":
    with open("2016-01.input") as f:
        input_file = [line.rstrip("\n") for line in f.readlines()]
    _test()
    run(lambda: part_1(input_file), part=1)
    run(lambda: part_2(input_file), part=2)
