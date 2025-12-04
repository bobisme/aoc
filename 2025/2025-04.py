#!/usr/bin/env python

from typing import Generator, NamedTuple


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

with open("2025-04.input") as f:
    input_file = [line.strip() for line in f.readlines()]

Pos = NamedTuple("Pos", [("i", int), ("j", int)])


def neighbors(
    pos: Pos, lines: list[str] | list[list[str]]
) -> Generator[tuple[Pos, str]]:
    for i in (pos.i - 1, pos.i, pos.i + 1):
        if i < 0 or i >= len(lines):
            continue
        for j in (pos.j - 1, pos.j, pos.j + 1):
            if j < 0 or j >= len(lines[0]) or i == pos.i and j == pos.j:
                continue
            yield Pos(i, j), lines[i][j]


def part_1(input):
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


def part_2(input: list[str]):
    grid = [[c for c in row] for row in input]

    def step(input: list[list[str]]):
        counts = [[0] * len(input[0]) for _ in range(len(input))]
        for i in range(len(input)):
            for j in range(len(input[0])):
                pos = Pos(i, j)
                if input[i][j] == "@":
                    ns = neighbors(pos, input)
                    for n, _ in ns:
                        counts[n.i][n.j] += 1
        yield from (
            Pos(i, j)
            for i in range(len(counts))
            for j in range(len(counts[0]))
            if input[i][j] == "@" and counts[i][j] < 4
        )

    count = 0
    next_count = 1
    while next_count > 0:
        rolls = list(step(grid))
        next_count = len(rolls)
        count += next_count
        for roll in rolls:
            grid[roll.i][roll.j] = "."

    return count


def _test():
    def assert_eq(a, b):
        assert a == b, f"{a} != {b}"

    assert_eq(part_1(CONTROL_1), 13)
    assert_eq(part_2(CONTROL_1), 43)


if __name__ == "__main__":
    _test()
    print("part 1:", part_1(input_file))
    print("part 2:", part_2(input_file))
