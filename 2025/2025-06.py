#!/usr/bin/env python

from functools import reduce
import re
from typing import Generator, LiteralString
import timeit

Input = list[str] | list[LiteralString]

CONTROL_1: Input = (
    """\
123 328  51 64 
 45 64  387 23 
  6 98  215 314
*   +   *   +  
""".splitlines()
)

with open("2025-06.input") as f:
    # NOTE: was stripping too much
    input_file = [line.rstrip("\n") for line in f.readlines()]


def part_1(input: Input):
    nums = [list(map(int, re.split(r"\s+", line.strip()))) for line in input[:-1]]
    ops = [
        (int.__add__ if x == "+" else int.__mul__) for x in re.split(r"\s+", input[-1])
    ]

    return sum(
        reduce(ops[col], (nums[row][col] for row in range(len(nums))))
        for col in range(len(nums[0]))
    )


def part_2(input: Input):
    def parse_num_col(rightmost_col: int) -> Generator[tuple[int, int]]:
        "Returns (num, col)."
        for c in range(rightmost_col, -1, -1):
            n = 0
            exp = 0
            for row in input[-2::-1]:
                x = row[c]
                if x == " ":
                    if n > 0:
                        break
                    else:
                        continue
                n += int(x) * 10**exp
                exp += 1
            if n == 0:
                break
            yield (n, c)

    def parse_nums() -> Generator[list[int]]:
        c = len(input[0]) - 1
        while c > 0:
            col_nums = []
            for n, c in parse_num_col(c):
                col_nums.append(n)
            yield col_nums
            c -= 2

    nums = list(reversed(list(parse_nums())))
    ops = [
        (int.__add__ if x == "+" else int.__mul__)
        for x in re.split(r"\s+", input[-1].strip())
    ]
    assert len(nums) == len(ops)

    return sum(reduce(op, col_nums) for op, col_nums in zip(ops, nums))


def _test():
    def assert_eq(a, b):
        assert a == b, f"{a} != {b}"

    assert_eq(part_1(CONTROL_1), 4277556)
    assert_eq(part_2(CONTROL_1), 3263827)


def _bench(fn, count=100):
    return timeit.timeit(fn, number=count) / count * 1_000


if __name__ == "__main__":
    _test()
    print("tests: PASS")
    print("-" * 40)
    print("part_1:", part_1(input_file))
    print("part_2:", part_2(input_file))
    print("-" * 40)
    print(
        "part_1 bench: {:.1f}ms".format(_bench(lambda: part_1(input_file), count=100))
    )
    print(
        "part_2 bench: {:.1f}ms".format(_bench(lambda: part_2(input_file), count=100))
    )
