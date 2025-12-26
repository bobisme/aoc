#!/usr/bin/env python

import time
from typing import Generator, Iterable, LiteralString, NamedTuple

Input = list[str] | list[LiteralString]

CONTROL_1 = """\
..@@.@@@@.
@@@.@.@.@@
@@@@@.@.@@
@.@@@@..@.
@@.@@@@.@@
.@@@@@@@.@
.@.@.@.@@@
@.@@@.@@@@
.@@@@@@@@.
@.@.@@@.@.
""".splitlines()

Pos = NamedTuple("Pos", [("i", int), ("j", int)])


def part_1(input: Input):
    def neighbors(pos: Pos, lines: Input) -> Generator[tuple[Pos, str]]:
        for i in (pos.i - 1, pos.i, pos.i + 1):
            if i < 0 or i >= len(lines):
                continue
            for j in (pos.j - 1, pos.j, pos.j + 1):
                if j < 0 or j >= len(lines[0]) or i == pos.i and j == pos.j:
                    continue
                yield Pos(i, j), lines[i][j]

    counts = [[0] * len(input[0]) for _ in range(len(input))]
    for i in range(len(input)):
        for j in range(len(input[0])):
            pos = Pos(i, j)
            if input[i][j] == "@":
                ns = neighbors(pos, input)
                for n, _ in ns:
                    counts[n.i][n.j] += 1
    return sum(
        1
        for i in range(len(counts))
        for j in range(len(counts[0]))
        if input[i][j] == "@" and counts[i][j] < 4
    )


def part_2(input: Input):
    h, w = len(input), len(input[0])

    def neighbors(pos: Pos) -> Generator[Pos]:
        for i in (pos.i - 1, pos.i, pos.i + 1):
            if i < 0 or i >= h:
                continue
            for j in (pos.j - 1, pos.j, pos.j + 1):
                if j < 0 or j >= w or i == pos.i and j == pos.j:
                    continue
                yield Pos(i, j)

    rolls = set(Pos(i, j) for j in range(h) for i in range(w) if input[i][j] == "@")
    counts: dict[Pos, int] = dict((pos, 0) for pos in rolls)

    def change_count(rolls: Iterable[Pos], by: int = 1):
        for pos in rolls:
            ns = neighbors(pos)
            for n in ns:
                if n in counts:
                    counts[n] += by

    change_count(rolls, by=1)

    def get_removable():
        return (pos for pos in rolls if counts[pos] < 4)

    count = 0
    next_count = 1
    while next_count > 0:
        removed = list(get_removable())
        next_count = len(removed)
        count += next_count
        for roll in removed:
            rolls.remove(roll)
        change_count(removed, by=-1)

    return count


def _test():
    def assert_eq(a, b):
        assert a == b, f"{a} != {b}"

    assert_eq(part_1(CONTROL_1), 13)
    assert_eq(part_2(CONTROL_1), 43)


def run(fn, year=2025, day=4, part=0):
    start = time.perf_counter_ns()
    res = fn()
    elapsed_ns = time.perf_counter_ns() - start
    print(f"{year}\t{day}\t{part}\t{res}\t{elapsed_ns}")


if __name__ == "__main__":
    with open("2025-04.input") as f:
        input_file = [line.strip() for line in f.readlines()]
    _test()
    run(lambda: part_1(input_file), part=1)
    run(lambda: part_2(input_file), part=2)
