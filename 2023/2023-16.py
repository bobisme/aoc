#!/usr/bin/env python
from collections.abc import Iterable
import datetime
from dataclasses import dataclass
from enum import Enum
from typing import Self
from pyrsistent import PSet, s
from rich.progress import track


CONTROL_1 = r""".|...\....
|.-.\.....
.....|-...
........|.
..........
.........\
..../.\\..
.-.-/..|..
.|....-|.\
..//.|....
""".splitlines()

with open("2023-16.input") as f:
    input = [line.strip() for line in f.readlines()]


class Dir(Enum):
    UP = "^"
    RIGHT = ">"
    DOWN = "v"
    LEFT = "<"


@dataclass
class Photon:
    position: tuple[int, int]
    direction: Dir

    def __repr__(self) -> str:
        return f"{self.position} {str(self.direction)}"

    def __hash__(self) -> int:
        return hash((self.position, self.direction))

    def move(self, direction: Dir) -> Self:
        if direction == Dir.UP:
            pos = (self.position[0], self.position[1] - 1)
        elif direction == Dir.RIGHT:
            pos = (self.position[0] + 1, self.position[1])
        elif direction == Dir.DOWN:
            pos = (self.position[0], self.position[1] + 1)
        elif direction == Dir.LEFT:
            pos = (self.position[0] - 1, self.position[1])
        else:
            raise Exception("FUCK")
        return self.__class__(pos, direction)


def count_energized(input):
    return sum(sum(1 for col in row if col == "#") for row in input)


def part_1(input):
    for line in input:
        print(line)
    rows = len(input)
    cols = len(input[0])
    energized = [["." for col in range(cols)] for row in range(rows)]

    q: list[Photon] = [Photon((0, 0), Dir.RIGHT)]

    been_there = set()
    while len(q) > 0:
        p = q.pop()
        i, j = p.position
        if p in been_there:
            continue
        been_there.add(p)
        if not (0 <= i < cols) or not (0 <= j < rows):
            continue
        # energized_at_pos = input[p.position[1]][p.position[0]] == '#'
        energized[p.position[1]][p.position[0]] = "#"
        device = input[p.position[1]][p.position[0]]
        if device == ".":
            q.append(p.move(p.direction))
        elif device == "|":
            if p.direction in (Dir.RIGHT, Dir.LEFT):
                q.append(p.move(Dir.UP))
                q.append(p.move(Dir.DOWN))
            else:
                q.append(p.move(p.direction))
        elif device == "-":
            if p.direction in (Dir.UP, Dir.DOWN):
                q.append(p.move(Dir.LEFT))
                q.append(p.move(Dir.RIGHT))
            else:
                q.append(p.move(p.direction))
        elif device == "/":
            if p.direction == Dir.UP:
                q.append(p.move(Dir.RIGHT))
            elif p.direction == Dir.RIGHT:
                q.append(p.move(Dir.UP))
            elif p.direction == Dir.DOWN:
                q.append(p.move(Dir.LEFT))
            elif p.direction == Dir.LEFT:
                q.append(p.move(Dir.DOWN))
        elif device == "\\":
            if p.direction == Dir.UP:
                q.append(p.move(Dir.LEFT))
            elif p.direction == Dir.RIGHT:
                q.append(p.move(Dir.DOWN))
            elif p.direction == Dir.DOWN:
                q.append(p.move(Dir.RIGHT))
            elif p.direction == Dir.LEFT:
                q.append(p.move(Dir.UP))
        else:
            raise Exception("FUCK")

    print("-" * 40)
    for row in energized:
        print("".join(row))
    print(f"{count_energized(energized)=}")


def energize_with_queue(
    q: list[tuple[Photon, PSet[tuple[int, int]]]],
    cache: dict[Photon, PSet[tuple[int, int]]],
    input: list[str],
    cols: int,
    rows: int,
    p: Photon,
    been_there: set[Photon],
) -> PSet[tuple[int, int]]:
    merged = s()
    while len(q) > 0:
        p, energized = q.pop()
        if p in cache:
            energized = cache[p]
            merged = merged | energized
            continue
        i, j = p.position
        if p in been_there:
            merged = merged | energized
            continue
        been_there.add(p)
        if not (0 <= i < cols) or not (0 <= j < rows):
            merged = merged | energized
            continue
        energized = energized.add(p.position)
        # print_energized(rows, cols, energized)
        # print([1 if x is True else 0 for x in energized])
        device = input[j][i]
        device = input[p.position[1]][p.position[0]]
        next = None
        if device == ".":
            next = p.move(p.direction)
        elif device == "|":
            if p.direction in (Dir.RIGHT, Dir.LEFT):
                next = (p.move(Dir.UP), p.move(Dir.DOWN))
            else:
                next = p.move(p.direction)
        elif device == "-":
            if p.direction in (Dir.UP, Dir.DOWN):
                next = (p.move(Dir.LEFT), p.move(Dir.RIGHT))
            else:
                next = p.move(p.direction)
        elif device == "/":
            if p.direction == Dir.UP:
                next = p.move(Dir.RIGHT)
            elif p.direction == Dir.RIGHT:
                next = p.move(Dir.UP)
            elif p.direction == Dir.DOWN:
                next = p.move(Dir.LEFT)
            elif p.direction == Dir.LEFT:
                next = p.move(Dir.DOWN)
        elif device == "\\":
            if p.direction == Dir.UP:
                next = p.move(Dir.LEFT)
            elif p.direction == Dir.RIGHT:
                next = p.move(Dir.DOWN)
            elif p.direction == Dir.DOWN:
                next = p.move(Dir.RIGHT)
            elif p.direction == Dir.LEFT:
                next = p.move(Dir.UP)
        assert next is not None
        # merged = merged | energized
        if isinstance(next, Iterable):
            q.append((next[0], energized))
            q.append((next[1], energized))
        else:
            q.append((next, energized))
    return merged


def print_energized(rows: int, cols: int, energized: PSet[tuple[int, int]]):
    print("-" * cols)
    for j in range(rows):
        print("".join("#" if (i, j) in energized else "." for i in range(cols)))


def part_2(input):
    for line in input:
        print(line)
    rows = len(input)
    cols = len(input[0])
    print(f"{rows=},{cols=}")
    most_energized = s()
    energized = s()
    q = []
    for i in range(cols):
        q.append(Photon((i, 0), Dir.DOWN))
    for i in range(cols):
        q.append(Photon((i, rows - 1), Dir.UP))
    for j in range(rows):
        q.append(Photon((0, j), Dir.RIGHT))
    for j in range(rows):
        q.append(Photon((cols - 1, j), Dir.LEFT))

    for p in track(q):
        # for p in track([Photon((3, 0), Dir.DOWN)]):
        cache = {}
        res = energize_with_queue([(p, s())], cache, input, cols, rows, p, set())
        print(f"{p=}  \t{len(res)=}")
        if len(res) > len(most_energized):
            most_energized = res
    n = len(most_energized)
    print_energized(rows, cols, most_energized)
    print(f"{n=}")


if __name__ == "__main__":
    start = datetime.datetime.now()
    part_2(input)
    end = datetime.datetime.now()
    print("time:", end - start)
