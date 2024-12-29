#!/usr/bin/env python

from collections import defaultdict
from collections.abc import Generator, Iterable
from dataclasses import dataclass
from pprint import pformat
from typing import DefaultDict, Literal, NamedTuple, Optional, TypeGuard
from string import ascii_letters, digits

DIM = "\033[2m"
BRIGHT = "\033[1m"
RESET = "\033[0m"

CONTROL_1 = """\
RRRRIICCFF
RRRRIICCCF
VVRRRCCFFF
VVRCCCJFFF
VVVVCJJCFE
VVIVCCJJEE
VVIIICJJEE
MIIIIIJJEE
MIIISIJEEE
MMMISSJEEE
""".splitlines()

CONTROL_2 = """\
AAAAAA
AAABBA
AAABBA
ABBAAA
ABBAAA
AAAAAA
""".splitlines()

with open("2024-12.input") as f:
    input_file = [line.strip() for line in f.readlines()]


@dataclass
class Pos:
    i: int
    j: int

    def __hash__(self) -> int:
        return hash((self.i, self.j))

    def __repr__(self) -> str:
        return f"({self.i},{self.j})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Pos):
            return NotImplemented
        return (self.i, self.j) == (other.i, other.j)

    def __lt__(self, other: "Pos") -> bool:
        return (self.i, self.j) < (other.i, other.j)


class Grid:
    def __init__(self, input: Iterable[str]) -> None:
        self.grid = [list(line) for line in input]

    def __repr__(self) -> str:
        return pformat(self.grid)

    @property
    def height(self):
        return len(self.grid)

    @property
    def width(self):
        return len(self.grid[0])


def neighbors(pos: Pos, width: int, height: int) -> Generator[Pos, None, None]:
    if pos.i > 0:
        yield Pos(pos.i - 1, pos.j)
    if pos.j > 0:
        yield Pos(pos.i, pos.j - 1)
    if pos.j < width - 1:
        yield Pos(pos.i, pos.j + 1)
    if pos.i < height - 1:
        yield Pos(pos.i + 1, pos.j)


Plot = NamedTuple("Plot", [("area", int), ("perim", int)])


def get_plot_prices(visited: list[list[int]]):
    map = defaultdict(lambda: Plot(0, 0))
    height = len(visited)
    width = len(visited[0])
    for i in range(height):
        for j in range(width):
            key = visited[i][j]
            plot = map[key]
            add_perim = 0
            if i == 0 or i == height - 1:
                add_perim += 1
            if j == 0 or j == width - 1:
                add_perim += 1
            for n in neighbors(Pos(i, j), width, height):
                if visited[n.i][n.j] != key:
                    add_perim += 1
            map[key] = Plot(plot.area + 1, plot.perim + add_perim)
    return map


def map_plot(
    grid: Grid, visited: list[list[Optional[int]]], pos: Pos, id: int
) -> Generator[Pos, None, None]:
    key = grid.grid[pos.i][pos.j]

    def recurse(key: str, pos: Pos):
        if visited[pos.i][pos.j]:
            return
        visited[pos.i][pos.j] = id
        if grid.grid[pos.i][pos.j] == key:
            yield pos
        for n in neighbors(pos, grid.width, grid.height):
            if grid.grid[n.i][n.j] != key:
                continue
            yield from recurse(key, n)

    yield from recurse(key, pos)


def check_no_none(data: list[list[Optional[int]]]) -> TypeGuard[list[list[int]]]:
    return all(all(x is not None for x in sublist) for sublist in data)


def map_plots(grid: Grid) -> list[list[int]]:
    visited: list[list[Optional[int]]] = [
        [None for _ in range(grid.width)] for _ in range(grid.height)
    ]
    id = 1
    for i in range(grid.height):
        for j in range(grid.width):
            plot = list(map_plot(grid, visited, Pos(i, j), id))
            if len(plot) > 0:
                id += 1
    if check_no_none(visited):
        return visited
    raise ValueError("visited contains None value")


def part_1(input):
    grid = Grid(input)
    visited = map_plots(grid)
    prices = get_plot_prices(visited)
    print(sum(plot.area * plot.perim for plot in prices.values()))


def vis_plots(
    perim: Iterable[Pos], width: int, height: int, vertices: Iterable[Pos] | None = None
):
    map = [[f"{DIM}.{RESET}" for _ in range(width + 2)] for _ in range(height + 2)]
    for p in perim:
        map[p.i + 1][p.j + 1] = f"{BRIGHT}#{RESET}"
    if vertices:
        for v in vertices:
            map[v.i + 1][v.j + 1] = "*"
    for row in map:
        print("".join(row))
    print()


def get_plot_tiles(visited: list[list[int]]) -> dict[int, list[int]]:
    map = defaultdict(list)
    height = len(visited)
    width = len(visited[0])
    for i in range(height):
        for j in range(width):
            key = visited[i][j]
            map[key].append(Pos(i, j))
    return map


def get_plot_tiles_for_key(
    visited: list[list[int]], key: int
) -> Generator[Pos, None, None]:
    height = len(visited)
    width = len(visited[0])
    for i in range(height):
        for j in range(width):
            if visited[i][j] == key:
                yield Pos(i, j)


def map_fences(visited: list[list[int]]):
    map = defaultdict(set)
    height = len(visited)
    width = len(visited[0])
    for i in range(height):
        for j in range(width):
            key = visited[i][j]
            pos = Pos(i, j)
            if i == 0:
                map[key].add((pos, Pos(i - 1, j)))
            elif i == height - 1:
                map[key].add((pos, Pos(i + 1, j)))
            elif j == 0:
                map[key].add((pos, Pos(i, j - 1)))
            elif j == width - 1:
                map[key].add((pos, Pos(i, j + 1)))
            for n in neighbors(pos, width, height):
                if visited[n.i][n.j] != key:
                    map[key].add((pos, n))
    return map


def vis_fence(fence: Iterable[tuple[Pos, Pos]], width: int, height: int):
    map = [[f"{DIM}.{RESET}" for _ in range(width + 2)] for _ in range(height + 2)]
    for a, b in fence:
        map[a.i + 1][a.j + 1] = f"{BRIGHT}x{RESET}"
        map[b.i + 1][b.j + 1] = f"{BRIGHT}y{RESET}"
    for row in map:
        print("".join(row))
    print()


def part_2(input):
    grid = Grid(input)
    visited = map_plots(grid)
    all_plot_tiles = get_plot_tiles(visited)
    fence_map = map_fences(visited)
    for zone, fences in fence_map.items():
        # vis_fence(fences, width=len(visited[0]), height=len(visited))
        vis = set()
        to_visit = [x for x in fences if x not in vis]
        while to_visit:
            node = to_visit[0]
            vis.add(node)
            a, b = node
            side = None
            if Pos(b.i - 1, b.j) == a:
                side = "BOTTOM"
            if Pos(b.i + 1, b.j) == a:
                side = "TOP"
            if Pos(b.i, b.j - 1) == a:
                side = "RIGHT"
            if Pos(b.i, b.j + 1) == a:
                side = "LEFT"
            if side is None:
                raise ValueError("nah")
            if side in ("TOP", "BOTTOM"):
                j = a.j + 1
                # scan right
                while True:
                    match side:
                        case "TOP":
                            next = (Pos(a.i, j), Pos(a.i - 1, j))
                        case "BOTTOM":
                            next = (Pos(a.i, j), Pos(a.i + 1, j))
                    if next in fences:
                        fences.remove(next)
                        j += 1
                        continue
                    break
                # scan left
                j = a.j - 1
                while True:
                    match side:
                        case "TOP":
                            next = (Pos(a.i, j), Pos(a.i - 1, j))
                        case "BOTTOM":
                            next = (Pos(a.i, j), Pos(a.i + 1, j))
                    if next in fences:
                        fences.remove(next)
                        j -= 1
                        continue
                    break
            else:
                # scan down
                i = a.i + 1
                while True:
                    match side:
                        case "LEFT":
                            next = (Pos(i, a.j), Pos(i, a.j - 1))
                        case "RIGHT":
                            next = (Pos(i, a.j), Pos(i, a.j + 1))
                    if next in fences:
                        fences.remove(next)
                        i += 1
                        continue
                    break
                # scan up
                i = a.i - 1
                while True:
                    match side:
                        case "LEFT":
                            next = (Pos(i, a.j), Pos(i, a.j - 1))
                        case "RIGHT":
                            next = (Pos(i, a.j), Pos(i, a.j + 1))
                    if next in fences:
                        fences.remove(next)
                        i -= 1
                        continue
                    break
            to_visit = [x for x in fences if x not in vis]
    out = 0
    for key in all_plot_tiles.keys():
        area = len(all_plot_tiles[key])
        sides = len(fence_map[key])
        out += area * sides
    print(out)


if __name__ == "__main__":
    part_1(input_file)
    part_2(input_file)
