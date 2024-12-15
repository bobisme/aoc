#!/usr/bin/env python

from typing import Deque, Generator
from itertools import chain


CONTROL_1 = """\
125 17
""".splitlines()
# 0 1 10 99 999

with open("2024-11.input") as f:
    input_file = [line.strip() for line in f.readlines()]


def blink_slow(stones: Deque[int]):
    i = 0
    while i < len(stones):
        stone = stones[i]
        if stone == 0:
            stones[i] = 1
        elif len(str(stone)) % 2 == 0:
            str_stone = str(stone)
            left = int(str_stone[: len(str_stone) // 2])
            right = int(str_stone[len(str_stone) // 2 :])
            stones[i] = left
            stones.insert(i + 1, right)
            i += 1
        else:
            stones[i] = stone * 2024
        i += 1


def blinks(stones: Deque[int], times=1):
    map: dict[int, list[int]] = {}

    def blink(stone: int) -> list[int]:
        if stone in map:
            return map[stone]
        if stone == 0:
            map[stone] = [1]
            return [1]
        elif len(str(stone)) % 2 == 0:
            str_stone = str(stone)
            left = int(str_stone[: len(str_stone) // 2])
            right = int(str_stone[len(str_stone) // 2 :])
            map[stone] = [left, right]
            return [left, right]
        else:
            map[stone] = [stone * 2024]
            return [stone * 2024]

    def blink_gen(
        stones: list[int], count: int = 1
    ) -> Generator[list[int], None, None]:
        for _ in range(count):
            yield list(chain(*(blink(stone) for stone in stones)))

    # for stone in stones:
    #     sss = [stone]
    #     for _ in range(times):
    #         for s in sss:
    #             if s in map:


def part_1(input):
    stones = Deque(int(x) for x in input[0].split(" "))
    for _ in range(25):
        blink_slow(stones)
    print(len(stones))


def part_2(input):
    stones = Deque(int(x) for x in input[0].split(" "))
    for _ in range(75):
        blink_slow(stones)
        print(stones)
    print(len(stones))


if __name__ == "__main__":
    # part_1(input_file)
    part_2(CONTROL_1)
