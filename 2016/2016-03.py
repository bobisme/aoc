#!/usr/bin/env python

from typing import LiteralString
import time

Input = list[str] | list[LiteralString]

def part_1(input: Input):
    out = 0
    for line in input:
        nums = tuple(sorted(map(int, (x for x in line.split(' ') if x != ''))))
        if nums[0] + nums[1] > nums[2]:
            out += 1
    return out


def part_2(input: Input):
    out = 0
    tris = []
    for line in input:
        tris.append(tuple(map(int, (x for x in line.split(' ') if x != ''))))
    def gen_tris():
        for row in range(0, len(tris)-1, 3):
            for col in (0, 1, 2):
                yield (
                    tris[row][col],
                    tris[row+1][col],
                    tris[row+2][col],
                )
    for tri in gen_tris():
        ntri = tuple(sorted(tri))
        if ntri[0] + ntri[1] > ntri[2]:
            out += 1
    return out


def run(fn, year=2016, day=3, part=0):
    start = time.perf_counter_ns()
    res = fn()
    elapsed_ns = time.perf_counter_ns() - start
    print(f"{year}\t{day}\t{part}\t{res}\t{elapsed_ns}")
    return res


if __name__ == "__main__":
    with open("2016-03.input") as f:
        input_file = [line.rstrip("\n") for line in f.readlines()]
    run(lambda: part_1(input_file), part=1)
    run(lambda: part_2(input_file), part=2)
