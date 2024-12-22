#!/usr/bin/env python

from collections import deque, Counter
import enum
import sys
from typing import Generator, Iterable, NamedTuple

DIM = "\033[2m"
BRIGHT = "\033[1m"
RESET = "\033[0m"
YELLOW = "\033[33m"

CONTROL_1 = """\
###############
#...#...#.....#
#.#.#.#.#.###.#
#S#...#.#.#...#
#######.#.#.###
#######.#.#...#
#######.#.###.#
###..E#...#...#
###.#######.###
#...###...#...#
#.#####.#.###.#
#.#...#.#.#...#
#.#.#.#.#.#.###
#...#...#...###
###############
""".splitlines()

with open("2024-20.input") as f:
    input_file = [line.strip() for line in f.readlines()]

Grid = list[list[str]]

Pos = NamedTuple("Pos", [("i", int), ("j", int)])
Node = NamedTuple("Node", [("pos", Pos), ("cheated", bool), ("cost", int)])
PathNode = NamedTuple(
    "Node", [("pos", Pos), ("cheated", bool), ("cost", int), ("path", list[Pos])]
)


def parse(input) -> tuple[Grid, Pos, Pos]:  # grid, start, end
    grid = [list(line) for line in input]
    start = None
    end = None
    for i, row in enumerate(grid):
        for j, x in enumerate(row):
            if x == "S":
                start = Pos(i, j)
            elif x == "E":
                end = Pos(i, j)
    if start is None:
        raise ValueError("no start")
    if end is None:
        raise ValueError("no end")
    return grid, start, end


def path_char(n: int) -> str:
    chars = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    return chars[n % len(chars)]


def print_grid(grid: Grid, path: Iterable[Pos] | None = None):
    def color(x: str) -> str:
        return f"{DIM}{x}{RESET}" if x not in ("S", "E") else f"{BRIGHT}{x}{RESET}"

    g = [[color(x) for x in row] for row in grid]

    if path:
        for i, p in enumerate(path):
            g[p.i][p.j] = path_char(i)

    for row in g:
        print("".join(row))


def fastest_legit_path(grid: Grid, start: Pos, end: Pos) -> list[Pos]:
    def next_nodes(node: PathNode) -> Generator[PathNode, None, None]:
        p = node.pos
        new_path = node.path + [p]
        if grid[p.i - 1][p.j] != "#":
            yield PathNode(Pos(p.i - 1, p.j), node.cheated, node.cost + 1, new_path)
        if grid[p.i + 1][p.j] != "#":
            yield PathNode(Pos(p.i + 1, p.j), node.cheated, node.cost + 1, new_path)
        if grid[p.i][p.j - 1] != "#":
            yield PathNode(Pos(p.i, p.j - 1), node.cheated, node.cost + 1, new_path)
        if grid[p.i][p.j + 1] != "#":
            yield PathNode(Pos(p.i, p.j + 1), node.cheated, node.cost + 1, new_path)

    visited: set[Pos] = set([start])
    q = deque([PathNode(start, False, 0, [])])
    while q:
        node = q.popleft()
        if node.pos == end:
            return node.path + [end]

        for nn in next_nodes(node):
            if nn.pos in visited:
                continue
            visited.add(nn.pos)
            q.append(nn)

    return []


def cheated_times_dfs2(
    grid: Grid, legit_path: list[Pos], save=100, max_dist=2
) -> Generator[int, None, None]:
    h = len(grid)
    w = len(grid[0])

    legit_times = {p: len(legit_path) - i - 1 for (i, p) in enumerate(legit_path)}

    visited: set[Pos] = set()

    def cheats(pos: Pos, dist: int) -> Generator[tuple[Pos, int], None, None]:
        for di in range(-max_dist, max_dist + 1):
            for dj in range(-max_dist, max_dist + 1):
                new_pos = Pos(pos.i + di, pos.j + dj)
                manhattan = abs(di) + abs(dj)
                if (
                    manhattan > 1
                    and manhattan <= max_dist  # Distance constraints
                    and 0 <= new_pos.i < h
                    and 0 <= new_pos.j < w
                ):
                    if grid[new_pos.i][new_pos.j] != "#" and new_pos not in visited:
                        yield (new_pos, manhattan)

    for i, p in enumerate(legit_path):
        visited.add(p)
        for cheat_pos, dist in cheats(p, i):
            full_dist = i + dist + legit_times[cheat_pos]
            if (len(legit_path) - full_dist) > 100:
                yield full_dist


def part_1(input, save=100):
    grid, start, end = parse(input)
    legit_path = fastest_legit_path(grid, start, end)
    fast_times = cheated_times_dfs2(grid, legit_path, save=save)
    filtered = [x for x in fast_times if (len(legit_path) - x) > save]
    print(len(filtered))


def part_2(input, save=100, dist=2):
    grid, start, end = parse(input)
    legit_path = fastest_legit_path(grid, start, end)
    fast_times = cheated_times_dfs2(grid, legit_path, save=save, max_dist=dist)
    filtered = [
        len(legit_path) - 1 - x for x in fast_times if (len(legit_path) - x) > save
    ]
    print(len(filtered))


if __name__ == "__main__":
    part_1(input_file, save=100)
    part_2(input_file, save=100, dist=20)
