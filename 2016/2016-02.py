#!/usr/bin/env python

from dataclasses import dataclass
from typing import LiteralString
import time

Input = list[str] | list[LiteralString]

CONTROL_1: Input = """\
ULL
RRDDD
LURDL
UUUUD
""".splitlines()

@dataclass
class Pad:
    NUMS = (
        ('1', '2', '3'),
        ('4', '5', '6'),
        ('7', '8', '9')
    )
    pos = [1, 1]

    def move(self, dir: str):
        match dir:
            case 'U':
                if self.pos[1] == 0:
                    return
                self.pos[1] -= 1
            case 'D':
                if self.pos[1] == 2:
                    return
                self.pos[1] += 1
            case 'L':
                if self.pos[0] == 0:
                    return
                self.pos[0] -= 1
            case 'R':
                if self.pos[0] == 2:
                    return
                self.pos[0] += 1

    def num(self):
        return self.NUMS[self.pos[1]][self.pos[0]]

def part_1(input: Input):
    pad = Pad()
    out = ''
    for line in input:
        for dir in line:
            pad.move(dir)
        out += str(pad.num())
    return out

@dataclass
class Pad2:
    NUMS = (
        ('', '', '1', '', ''),
        ('', '2', '3', '4', ''),
        ('5', '6', '7', '8', '9'),
        ('', 'A', 'B', 'C', ''),
        ('', '', 'D', '', ''),
    )
    pos = [0, 2]

    def move(self, dir: str):
        match dir:
            case 'U':
                next_pos = [self.pos[0], self.pos[1]-1]
                if -1 in next_pos or 5 in next_pos or self.num(pos=next_pos) == '':
                    return
                self.pos[1] -= 1
            case 'D':
                next_pos = [self.pos[0], self.pos[1]+1]
                if -1 in next_pos or 5 in next_pos or self.num(pos=next_pos) == '':
                    return
                self.pos[1] += 1
            case 'L':
                next_pos = [self.pos[0]-1, self.pos[1]]
                if -1 in next_pos or 5 in next_pos or self.num(pos=next_pos) == '':
                    return
                self.pos[0] -= 1
            case 'R':
                next_pos = [self.pos[0]+1, self.pos[1]]
                if -1 in next_pos or 5 in next_pos or self.num(pos=next_pos) == '':
                    return
                self.pos[0] += 1

    def num(self, pos=None):
        if pos is None:
            pos = self.pos
        return self.NUMS[pos[1]][pos[0]]

def part_2(input: Input):
    pad = Pad2()
    out = ''
    for line in input:
        for dir in line:
            pad.move(dir)
        out += str(pad.num())
    return out


def _test():
    def assert_eq(a, b):
        assert a == b, f"{a} != {b}"

    assert_eq(part_1(CONTROL_1), '1985')
    assert_eq(part_2(CONTROL_1), '5DB3')


def run(fn, year=2016, day=2, part=0):
    start = time.perf_counter_ns()
    res = fn()
    elapsed_ns = time.perf_counter_ns() - start
    print(f"{year}\t{day}\t{part}\t{res}\t{elapsed_ns}")
    return res


if __name__ == "__main__":
    with open("2016-02.input") as f:
        input_file = [line.rstrip("\n") for line in f.readlines()]
    _test()
    run(lambda: part_1(input_file), part=1)
    run(lambda: part_2(input_file), part=2)
