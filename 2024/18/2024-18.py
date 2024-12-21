#!/usr/bin/env python

from collections import deque
from dataclasses import dataclass
import re
from typing import Iterable, Optional
import time

DIM = "\033[2m"
BRIGHT = "\033[1m"
RESET = "\033[0m"
YELLOW = "\033[33m"

CONTROL_1 = """\
5,4
4,2
4,5
3,0
2,1
6,3
2,4
1,5
0,6
3,3
2,6
5,1
1,2
5,5
2,5
6,5
1,4
0,4
6,4
1,1
6,1
1,0
0,5
1,6
2,0
""".splitlines()

with open("2024-18.input") as f:
    input_file = [line.strip() for line in f.readlines()]


@dataclass
class Pos:
    x: int
    y: int

    def __repr__(self) -> str:
        return f"({self.x},{self.y})"

    def __hash__(self) -> int:
        return hash((self.x, self.y))


def parse(input: Iterable[str]) -> list[Pos]:
    pattern = re.compile(r"(\d+),(\d+)")
    out = []
    for line in input:
        strx, stry = pattern.findall(line)[0]
        out.append(Pos(int(strx), int(stry)))
    return out


def print_grid(grid: list[list[str]], path: Optional[Iterable[Pos]] = None):
    g = [
        [f"{DIM}{YELLOW}{x}{RESET}" if x == "#" else f"{DIM}{x}{RESET}" for x in row]
        for row in grid
    ]
    if path:
        for p in path:
            g[p.y][p.x] = f"{BRIGHT}O{RESET}"
    for row in g:
        print("".join(row))


def bfs(grid: list[list[str]]):
    width = len(grid[0])
    height = len(grid)

    def neighbors(pos: Pos):
        if pos.x > 0:
            yield Pos(pos.x - 1, pos.y)
        if pos.y > 0:
            yield Pos(pos.x, pos.y - 1)
        if pos.x < width - 1:
            yield Pos(pos.x + 1, pos.y)
        if pos.y < height - 1:
            yield Pos(pos.x, pos.y + 1)

    start = Pos(0, 0)
    visited = {start}
    q = deque([[start]])
    while q:
        path = q.popleft()
        pos = path[-1]

        if pos == Pos(width - 1, height - 1):
            return path
        for n in neighbors(pos):
            if n not in visited and grid[n.y][n.x] != "#":
                visited.add(n)
                q.append(path + [n])
    return None


def part_1(input, byte_count=12):
    positions = parse(input)
    width = max(p.y for p in positions) + 1
    height = max(p.x for p in positions) + 1
    grid = [["." for _ in range(width)] for _ in range(height)]
    for pos in positions[:byte_count]:
        grid[pos.y][pos.x] = "#"
    path = bfs(grid)
    if path is None:
        raise ValueError("could not find path")
    print_grid(grid, path)
    print(len(path) - 1)


def part_2(input):
    positions = parse(input)
    width = max(p.y for p in positions) + 1
    height = max(p.x for p in positions) + 1
    for i in range(1024, len(positions)):
        grid = [["." for _ in range(width)] for _ in range(height)]
        for pos in positions[:i]:
            grid[pos.y][pos.x] = "#"
        path = bfs(grid)
        if path is None:
            print(positions[i - 1])
            break
        else:
            pass
            # print_grid(grid, path)
            # print()


if __name__ == "__main__":
    part_1(input_file, byte_count=1024)
    part_2(input_file)
