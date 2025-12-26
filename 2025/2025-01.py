#!/usr/bin/env python

import time


def parse(input: list[str]) -> list[int]:
    turns: list[int] = []
    for line in input:
        i = int(line[1:])
        turns.append(i if line[0] == "R" else -i)
    return turns


def part_1(input):
    pos = 50
    cross_counts = 0
    turns = parse(input)
    for turn in turns:
        pos = (pos + turn) % 100
        if pos == 0:
            cross_counts += 1
    return cross_counts


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
    turns = parse(input)
    pos = 50
    cross_counts = 0
    for turn in turns:
        crosses, pos = crosses_and_pos(pos, turn)
        cross_counts += crosses
    return cross_counts


def _test():
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


def run(fn, year=2025, day=1, part=0):
    start = time.perf_counter_ns()
    res = fn()
    elapsed_ns = time.perf_counter_ns() - start
    print(f"{year}\t{day}\t{part}\t{res}\t{elapsed_ns}")


if __name__ == "__main__":
    with open("2025-01.input") as f:
        input_file = [line.strip() for line in f.readlines()]
    _test()
    run(lambda: part_1(input_file), part=1)
    run(lambda: part_2(input_file), part=2)
