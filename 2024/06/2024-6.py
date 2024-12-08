#!/usr/bin/env python

from collections.abc import Generator, Iterable
from dataclasses import dataclass
from pprint import pp, pformat
from typing import Literal, Optional


CONTROL_1 = """\
....#.....
.........#
..........
..#.......
.......#..
..........
.#..^.....
........#.
#.........
......#...
""".splitlines()

with open("2024-6.input") as f:
    input_file = [line.strip() for line in f.readlines()]

Dir = Literal["^", ">", "v", "<"]


def match_dir(x: str) -> Dir:
    match x:
        case "^" | ">" | "v" | "<":
            return x
        case _:
            raise ValueError(f"Invalid direction: {x}")


@dataclass
class Pos:
    i: int
    j: int

    def __hash__(self) -> int:
        return hash((self.i, self.j))


@dataclass
class Guard:
    pos: Pos
    dir: Dir

    def __repr__(self) -> str:
        return f"{self.pos} {self.dir}"

    def __hash__(self) -> int:
        return hash((self.pos, self.dir))


class Grid:
    def __init__(self, input: Iterable[str]) -> None:
        self.obstacle = None
        self.grid = [list(line) for line in input]

    def __repr__(self) -> str:
        return pformat(self.grid)

    @property
    def height(self):
        return len(self.grid)

    @property
    def width(self):
        return len(self.grid[0])

    def set_obstacle(self, pos: Pos):
        self.obstacle = pos

    def is_obstacle(self, pos: Pos) -> bool:
        if pos == self.obstacle:
            return True
        if self.grid[pos.i][pos.j] == "#":
            return True
        return False


def find_guard(grid: Grid) -> Guard:
    for i in range(grid.height):
        for j in range(grid.width):
            if grid.grid[i][j] in ("^", ">", "v", "<"):
                return Guard(pos=Pos(i=i, j=j), dir=match_dir(grid.grid[i][j]))
    raise Exception("no guard")


class LeftArea(Exception):
    pass


def move(grid: Grid, guard: Guard, dir: Dir) -> Optional[Guard]:
    if dir == "^":
        if guard.pos.i == 0:
            return None
        if grid.is_obstacle(Pos(guard.pos.i - 1, guard.pos.j)):
            return move(grid, guard, ">")
        return Guard(Pos(guard.pos.i - 1, guard.pos.j), "^")
    if dir == ">":
        if guard.pos.j == grid.width - 1:
            return None
        if grid.is_obstacle(Pos(guard.pos.i, guard.pos.j + 1)):
            return move(grid, guard, "v")
        return Guard(Pos(guard.pos.i, guard.pos.j + 1), ">")
    if dir == "v":
        if guard.pos.i == grid.height - 1:
            return None
        if grid.is_obstacle(Pos(guard.pos.i + 1, guard.pos.j)):
            return move(grid, guard, "<")
        return Guard(Pos(guard.pos.i + 1, guard.pos.j), "v")
    if dir == "<":
        if guard.pos.j == 0:
            return None
        if grid.is_obstacle(Pos(guard.pos.i, guard.pos.j - 1)):
            return move(grid, guard, "^")
        return Guard(Pos(guard.pos.i, guard.pos.j - 1), "<")


def get_path(grid: Grid) -> set[Pos]:
    guard = find_guard(grid)
    path = []
    while guard is not None:
        path.append(guard.pos)
        guard = move(grid, guard, guard.dir)
    return set(path)


def part_1(input):
    grid = Grid(input)
    print(len(get_path(grid)))


def obstacle_locations(grid: Grid) -> Generator[Pos, None, None]:
    for i in range(grid.height):
        for j in range(grid.width):
            if grid.grid[i][j] == ".":
                yield Pos(i, j)


def creates_loop(grid: Grid) -> bool:
    guard = find_guard(grid)
    path = set()
    while guard is not None:
        path.add(guard)
        guard = move(grid, guard, guard.dir)
        if guard in path:
            return True
    return False


def part_2(input):
    grid = Grid(input)
    out = 0
    obstacle_locations = get_path(grid)
    for obstacle in obstacle_locations:
        grid.set_obstacle(obstacle)
        if creates_loop(grid):
            out += 1
    print(out)


if __name__ == "__main__":
    part_1(input_file)
    part_2(input_file)
