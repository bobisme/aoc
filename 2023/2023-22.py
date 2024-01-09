#!/usr/bin/env python
from dataclasses import dataclass
from typing import NamedTuple
from pprint import pp
from functools import cached_property

import bpy

CONTROL_1 = """\
1,0,1~1,2,1
0,0,2~2,0,2
0,2,3~2,2,3
0,0,4~0,2,4
2,0,5~2,2,5
0,1,6~2,1,6
1,1,8~1,1,9
""".splitlines()

with open("2023-22.input") as f:
    input_file = [line.strip() for line in f.readlines()]


class Vec(NamedTuple("Vec", [("x", int), ("y", int), ("z", int)])):
    def __repr__(self):
        return f"{{{self.x},{self.y},{self.z}}}"


class Brick:
    start: Vec
    end: Vec
    shape: Vec
    offset: Vec

    def __init__(self, start: Vec, end: Vec) -> None:
        self.start = start
        self.end = end
        self.shape = Vec(self.xlen(), self.ylen(), self.zlen())
        # TODO: this will be a problem, should determine if start or end are
        # closer to the ground.
        self.offset = self.start

    def __repr__(self) -> str:
        return f"Brick(shape={self.shape}, offset={self.offset})"

    def xlen(self):
        return abs(self.end.x - self.start.x) + 1

    def ylen(self):
        return abs(self.end.y - self.start.y) + 1

    def zlen(self):
        return abs(self.end.z - self.start.z) + 1


def parse_line(line) -> Brick:
    start, end = line.split("~", 1)
    start = Vec(*(int(x) for x in start.split(",", 2)))
    end = Vec(*(int(x) for x in end.split(",", 2)))
    return Brick(start, end)


def part_1(input):
    for line in input:
        print(line)
    bricks = [parse_line(line) for line in input]
    bricks.sort(key=lambda b: min(b.start.z, b.end.z))
    pp(bricks)
    pp([(x.xlen(), x.ylen(), x.zlen()) for x in bricks])


if __name__ == "__main__":
    part_1(CONTROL_1)
