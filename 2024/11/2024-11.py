#!/usr/bin/env python

from typing import DefaultDict, Deque, Generator
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


def blink_fast(stones: Deque[int], times=1):
    old_cache: DefaultDict[int, int] = DefaultDict(int)  # stone -> count
    for stone in stones:
        old_cache[stone] += 1

    for i in range(times):
        new_cache: DefaultDict[int, int] = DefaultDict(int)
        for stone, count in old_cache.items():
            if stone == 0:
                new_cache[1] += count
            elif len(str(stone)) % 2 == 0:
                str_stone = str(stone)
                left = int(str_stone[: len(str_stone) // 2])
                right = int(str_stone[len(str_stone) // 2 :])
                new_cache[left] += count
                new_cache[right] += count
            else:
                new_cache[stone * 2024] += count
        old_cache = new_cache

    return sum(old_cache.values())


def part_1(input):
    stones = Deque(int(x) for x in input[0].split(" "))
    out = blink_fast(stones, times=25)
    print(out)


def part_2(input):
    stones = Deque(int(x) for x in input[0].split(" "))
    out = blink_fast(stones, times=75)
    print(out)


if __name__ == "__main__":
    part_1(input_file)
    part_2(input_file)
