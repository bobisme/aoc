#!/usr/bin/env python

from typing import Iterator, NamedTuple

from colored import Fore, Style

import sys

sys.setrecursionlimit(10000)

CONTROL_1 = """\
#.#####################
#.......#########...###
#######.#########.#.###
###.....#.>.>.###.#.###
###v#####.#v#.###.#.###
###.>...#.#.#.....#...#
###v###.#.#.#########.#
###...#.#.#.......#...#
#####.#.#.#######.#.###
#.....#.#.#.......#...#
#.#####.#.#.#########v#
#.#...#...#...###...>.#
#.#.#v#######v###.###v#
#...#.>.#...>.>.#.###.#
#####v#.#.###v#.#.###.#
#.....#...#...#.#.#...#
#.#########.###.#.#.###
#...###...#...#...#.###
###.###.#.###v#####v###
#...#...#.#.>.>.#.>.###
#.###.###.#.###.#.#v###
#.....###...###...#...#
#####################.#
""".splitlines()

with open("2023-23.input") as f:
    input_file = [line.strip() for line in f.readlines()]

Pos = NamedTuple("Pos", [("i", int), ("j", int)])


class Field:
    input: list[str]
    rows: int
    cols: int

    def __init__(self, input: list[str]) -> None:
        self.input = input
        self.rows = len(input)
        self.cols = len(input[0])

    def get(self, pos: Pos) -> str:
        return self.input[pos.j][pos.i]

    def in_bounds(self, pos: Pos) -> bool:
        return 0 <= pos.i < self.cols and 0 <= pos.j < self.rows

    def print(self, marks=None, char="O"):
        print(f"{Fore.blue}{'─' * self.cols}{Style.reset}")
        field = [[f"{Fore.black}{c}{Style.reset}" for c in row] for row in self.input]
        if marks:
            for mark in marks:
                if not self.in_bounds(mark):
                    continue
                field[mark.j][mark.i] = char

        for row in field:
            print("".join(row))
        if marks:
            print("length of path =", len(marks) - 1)


def pleft(pos):
    return Pos(pos.i - 1, pos.j)


def pright(pos):
    return Pos(pos.i + 1, pos.j)


def pup(pos):
    return Pos(pos.i, pos.j - 1)


def pdown(pos):
    return Pos(pos.i, pos.j + 1)


def neighbors(field: Field, pos: Pos) -> Iterator[Pos]:
    up, down, left, right = pup(pos), pdown(pos), pleft(pos), pright(pos)

    def all() -> Iterator[Pos]:
        curr = field.get(pos)
        if curr == "^":
            yield up
        elif curr == ">":
            yield right
        elif curr == "<":
            yield left
        elif curr == "v":
            yield down
        else:
            if field.in_bounds(up) and field.get(up) != "v":
                yield up
            if field.in_bounds(left) and field.get(left) != ">":
                yield left
            if field.in_bounds(right) and field.get(right) != "<":
                yield right
            if field.in_bounds(down) and field.get(down) != "^":
                yield down

    for p in all():
        if field.get(p) == "#":
            continue
        yield p


def neighbors2(field: Field, pos: Pos) -> Iterator[Pos]:
    up, down, left, right = pup(pos), pdown(pos), pleft(pos), pright(pos)

    def all() -> Iterator[Pos]:
        if field.in_bounds(up):
            yield up
        if field.in_bounds(left):
            yield left
        if field.in_bounds(right):
            yield right
        if field.in_bounds(down):
            yield down

    for p in all():
        if field.get(p) == "#":
            continue
        yield p


def find_longest(field: Field, start: Pos) -> tuple[int, list[Pos]]:
    best_dist = 0
    best_path = []

    explored = set()

    def inner(pos: Pos, dist: int, path: list[Pos]):
        nonlocal best_dist, best_path
        if pos in explored:
            return
        explored.add(pos)
        new_path = path + [pos]
        ns = [n for n in neighbors2(field, pos) if n not in explored]

        if pos == Pos(field.cols - 2, field.rows - 1) and dist > best_dist:
            best_dist = dist
            best_path = new_path

        for n in ns:
            inner(n, dist + 1, new_path)
        explored.remove(pos)

    inner(start, 0, [])

    return best_dist, best_path


def main(input):
    for line in input:
        print(line)
    field = Field(input)
    start = Pos(1, 0)
    dist, path = find_longest(field, start)
    print(dist, path)
    field.print(marks=path)


if __name__ == "__main__":
    main(CONTROL_1)
