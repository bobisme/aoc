#!/usr/bin/env python

from collections.abc import Iterable
from dataclasses import dataclass
import sys
from typing import Deque, Literal

CONTROL_1 = """\
###############
#.......#....E#
#.#.###.#.###.#
#.....#.#...#.#
#.###.#####.#.#
#.#.#.......#.#
#.#.#####.###.#
#...........#.#
###.#.#####.#.#
#...#.....#.#.#
#.#.#.###.#.#.#
#.....#...#.#.#
#.###.#.#.#.#.#
#S..#.....#...#
###############
""".splitlines()

CONTROL_2 = """\
#################
#...#...#...#..E#
#.#.#.#.#.#.#.#.#
#.#.#.#...#...#.#
#.#.#.#.###.#.#.#
#...#.#.#.....#.#
#.#.#.#.#.#####.#
#.#...#.#.#.....#
#.#.#####.#.###.#
#.#.#.......#...#
#.#.###.#####.###
#.#.#...#.....#.#
#.#.#.#####.###.#
#.#.#.........#.#
#.#.#.#########.#
#S#.............#
#################
""".splitlines()

with open("2024-16.input") as f:
    input_file = [line.strip() for line in f.readlines()]


@dataclass
class Pos:
    i: int
    j: int

    def __hash__(self) -> int:
        return hash((self.i, self.j))

    def __repr__(self) -> str:
        return f"({self.i},{self.j})"

    def __add__(self, other: "Pos") -> "Pos":
        return Pos(self.i + other.i, self.j + other.j)


class Grid:
    bot: Pos

    def __init__(self, input: Iterable[str]) -> None:
        self.grid = [list(line) for line in input]

    def __repr__(self) -> str:
        return "\n".join("".join(row) for row in self.grid)

    @property
    def height(self):
        return len(self.grid)

    @property
    def width(self):
        return len(self.grid[0])


Dir = Literal["^", ">", "v", "<"]

DIRS = ("^", ">", "v", "<")


def turn_left(dir: Dir) -> Dir:
    match dir:
        case "^":
            return "<"
        case ">":
            return "^"
        case "v":
            return ">"
        case "<":
            return "v"


def turn_right(dir: Dir) -> Dir:
    match dir:
        case "^":
            return ">"
        case ">":
            return "v"
        case "v":
            return "<"
        case "<":
            return "^"


def get_next(pos: Pos, dir: Dir) -> Pos:
    match dir:
        case "^":
            return Pos(pos.i - 1, pos.j)
        case ">":
            return Pos(pos.i, pos.j + 1)
        case "v":
            return Pos(pos.i + 1, pos.j)
        case "<":
            return Pos(pos.i, pos.j - 1)


class Node:
    pos: Pos
    dir: Dir
    cost: int

    def __init__(self, pos: Pos, dir: Dir, cost: int) -> None:
        self.pos = pos
        self.dir = dir
        self.cost = cost

    def __repr__(self) -> str:
        return f"{self.pos} {self.dir} {self.cost}"

    def __hash__(self) -> int:
        return hash((self.pos, self.dir))


def search(grid: Grid, start: Node) -> int:
    q: Deque[Node] = Deque()
    q.append(start)
    lowest = sys.maxsize
    visited: set[tuple[Pos, Dir]] = set()

    while q:
        n = q.popleft()
        # print(n)
        visited.add((n.pos, n.dir))
        if grid.grid[n.pos.i][n.pos.j] == "E":
            if n.cost < lowest:
                lowest = n.cost
                # print("new lower cost", lowest)
            continue
        next_nodes = [
            Node(n.pos, turn_left(n.dir), n.cost + 1000),
            Node(n.pos, turn_right(n.dir), n.cost + 1000),
        ]
        next_pos = get_next(n.pos, n.dir)
        if grid.grid[next_pos.i][next_pos.j] != "#":
            next_nodes.append(Node(next_pos, n.dir, n.cost + 1))
        for nn in next_nodes:
            if (nn.pos, nn.dir) in visited:
                continue
            if nn.cost > lowest:
                continue
            q.append(nn)
    return lowest


def part_1(input):
    grid = Grid(input)
    lowest = search(grid, Node(Pos(grid.width - 2, 1), ">", 0))
    print(lowest)


def part_2(input):
    for line in input:
        print(line)


if __name__ == "__main__":
    part_1(input_file)
    # part_2(input_file)
