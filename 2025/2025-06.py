#!/usr/bin/env python

from functools import reduce
import re
import time
from typing import Generator, LiteralString

Input = list[str] | list[LiteralString]

CONTROL_1: Input = (
    """\
123 328  51 64 
 45 64  387 23 
  6 98  215 314
*   +   *   +  
""".splitlines()
)


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


def run(fn, year=2025, day=6, part=0):
    start = time.perf_counter_ns()
    res = fn()
    elapsed_ns = time.perf_counter_ns() - start
    print(f"{year}\t{day}\t{part}\t{res}\t{elapsed_ns}")


if __name__ == "__main__":
    with open("2025-06.input") as f:
        input_file = [line.rstrip("\n") for line in f.readlines()]
    _test()
    run(lambda: part_1(input_file), part=1)
    run(lambda: part_2(input_file), part=2)
