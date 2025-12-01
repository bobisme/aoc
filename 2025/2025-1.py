#!/usr/bin/env python

CONTROL_1 = """\
L68
L30
R48
L5
R60
L55
L1
L99
R14
L82
""".splitlines()

with open("2025-1.input") as f:
    input_file = [line.strip() for line in f.readlines()]


def cycle(i: int) -> int:
    return i % 100


def part_1(input):
    pos = 50
    cross_counts = 0
    turns: list[tuple[str, int]] = []
    for line in input:
        turns.append((line[0], int(line[1:])))
    for turn in turns:
        match turn[0]:
            case "L":
                pos = cycle(pos - turn[1])
            case "R":
                pos = cycle(pos + turn[1])
        if pos == 0:
            cross_counts += 1
    print(cross_counts)


def crosses_and_pos(start: int, turn: int) -> tuple[int, int]:
    turns, next_pos = divmod(start + turn, 100)
    if turn >= 0:
        return turns, next_pos
    # turn < 0
    turns = abs(turns)
    if next_pos == 0:
        turns += 1
    if start == 0:
        turns -= 1
    return turns, next_pos


def part_2(input):
    turns: list[int] = []
    for line in input:
        i = int(line[1:])
        turns.append(i if line[0] == "R" else -i)
    pos = 50
    cross_counts = 0
    for turn in turns:
        crosses, next_pos = crosses_and_pos(pos, turn)
        cross_counts += crosses
        pos = next_pos
    print(cross_counts)


def assert_eq(a, b):
    assert a == b, f"{a} != {b}"


assert_eq(crosses_and_pos(50, 49), (0, 99))
assert_eq(crosses_and_pos(50, 50), (1, 0))
assert_eq(crosses_and_pos(50, -50), (1, 0))
assert_eq(crosses_and_pos(90, 99), (1, 89))
assert_eq(crosses_and_pos(90, 100), (1, 90))
assert_eq(crosses_and_pos(0, 100), (1, 0))
assert_eq(crosses_and_pos(20, -20), (1, 0))
assert_eq(crosses_and_pos(20, -120), (2, 0))
assert_eq(crosses_and_pos(0, -99), (0, 1))
assert_eq(crosses_and_pos(0, -100), (1, 0))
assert_eq(crosses_and_pos(0, -101), (1, 99))
assert_eq(crosses_and_pos(0, -200), (2, 0))
assert_eq(crosses_and_pos(0, -652), (6, 48))

if __name__ == "__main__":
    part_1(input_file)
    part_2(input_file)
