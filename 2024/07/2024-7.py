#!/usr/bin/env python

from collections.abc import Generator, Iterator

CONTROL_1 = """\
190: 10 19
3267: 81 40 27
83: 17 5
156: 15 6
7290: 6 8 6 15
161011: 16 10 13
192: 17 8 14
21037: 9 7 18 13
292: 11 6 16 20
""".splitlines()

with open("2024-7.input") as f:
    input_file = [line.strip() for line in f.readlines()]


def parse(input: Iterator[str]) -> Generator[tuple[int, list[int]]]:
    for line in input:
        total_str, rest = line.split(": ")
        total = int(total_str)
        yield total, [int(x) for x in rest.split(" ")]


def con(a: int, b: int) -> int:
    return int(f"{a}{b}")


def search_1(total: int, nums: list[int]) -> bool:
    last = nums[-1]
    if len(nums) == 2:
        first = nums[0]
        if first + last == total:
            return True
        if first * last == total:
            return True
        return False

    # mul
    if total % last == 0:
        if search_1(total // last, nums[:-1]):
            return True
    # add
    if total - last > 0:
        if search_1(total - last, nums[:-1]):
            return True
    return False


def part_1(input):
    out = 0
    for total, nums in parse(input):
        if search_1(total, nums):
            out += total
    print(out)


def search_2(total: int, nums: list[int]) -> bool:
    last = nums[-1]
    if len(nums) == 2:
        first = nums[0]
        if first + last == total:
            return True
        if first * last == total:
            return True
        if con(first, last) == total:
            return True
        return False

    # mul
    if total % last == 0:
        if search_2(total // last, nums[:-1]):
            return True
    # add
    if total - last > 0:
        if search_2(total - last, nums[:-1]):
            return True
    # concat
    if str(total).endswith(str(last)):
        uncat = str(total).removesuffix(str(last))
        if uncat:
            if search_2(int(uncat), nums[:-1]):
                return True
    return False


def part_2(input):
    out = 0
    for total, nums in parse(input):
        if search_2(total, nums):
            out += total
    print(out)


if __name__ == "__main__":
    part_1(input_file)
    part_2(input_file)
