#!/usr/bin/env python

from typing import Deque


CONTROL_1 = """\
125 17
""".splitlines()
# 0 1 10 99 999

with open("2024-11.input") as f:
    input_file = [line.strip() for line in f.readlines()]


def blink(stones: Deque[int]):
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


def part_1(input):
    stones = Deque(int(x) for x in input[0].split(" "))
    for _ in range(25):
        blink(stones)
    print(len(stones))


def part_2(input):
    stones = Deque(int(x) for x in input[0].split(" "))
    for _ in range(75):
        blink(stones)
    print(len(stones))


if __name__ == "__main__":
    part_1(input_file)
    part_2(input_file)
