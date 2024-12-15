#!/usr/bin/env python

from collections import defaultdict
from collections.abc import Generator, Iterable
from dataclasses import dataclass
from pprint import pformat
from typing import DefaultDict, Literal, NamedTuple, Optional, TypeGuard
from string import ascii_letters, digits

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


def map_perimeters(visited: list[list[int]]):
    map = defaultdict(set)
    height = len(visited)
    width = len(visited[0])
    for i in range(height):
        for j in range(width):
            key = visited[i][j]
            if i == 0 or i == height - 1:
                map[key].add(Pos(i, j))
            elif j == 0 or j == width - 1:
                map[key].add(Pos(i, j))
            else:
                for n in neighbors(Pos(i, j), width, height):
                    if visited[n.i][n.j] != key:
                        map[key].add(Pos(i, j))
    return map


def visualize_perimiters(perimeters: dict[int, set[Pos]], width: int, height: int):
    vis = [["." for _ in range(width)] for _ in range(height)]
    codes = (digits + ascii_letters) * 20
    for x, set_ in perimeters.items():
        for s in set_:
            vis[s.i][s.j] = codes[x]
    for row in vis:
        print("".join(row))


def get_outer_perimiter(
    visited: list[list[int]], key: int, perimeter: set[Pos]
) -> set[Pos]:
    height = len(visited)
    width = len(visited[0])
    outer_perimeter = set()

    def ns(pos: Pos) -> Generator[Pos, None, None]:
        for i in range(pos.i - 1, pos.i + 2):
            for j in range(pos.j - 1, pos.j + 2):
                if i == pos.i and j == pos.j:
                    continue
                yield Pos(i, j)

    for pos in perimeter:
        for n in ns(pos):
            if n.i < 0 or n.i >= height or n.j < 0 or n.j >= width:
                outer_perimeter.add(n)
            else:
                if visited[n.i][n.j] != key:
                    outer_perimeter.add(n)

    return outer_perimeter


Dir = Literal["x", "^", ">", "v", "<"]


def walk_perimeter(outer_perimeter: set[Pos]) -> list[tuple[Pos, Dir]]:
    turns = 0
    visited_perimeter = set()
    path: list[tuple[Pos, Dir]] = []
    while len(visited_perimeter) < len(outer_perimeter):
        # print(len(outer_perimeter), len(visited_perimeter))
        pos: Pos = list(outer_perimeter - visited_perimeter)[0]
        dir: Dir = "x"
        while True:
            directions: dict[Dir, Pos] = dict(
                (
                    ("^", Pos(pos.i - 1, pos.j)),
                    ("<", Pos(pos.i, pos.j - 1)),
                    ("v", Pos(pos.i + 1, pos.j)),
                    (">", Pos(pos.i, pos.j + 1)),
                )
            )
            for key in ("<", "^", ">", "v"):
                if (
                    directions[key] not in outer_perimeter
                    or directions[key] in visited_perimeter
                ):
                    del directions[key]
            visited_perimeter.add(pos)
            path.append((pos, dir))
            if len(directions) == 0:
                break
            if dir == "x":
                dir, _ = list(directions.items())[0]
            else:
                straight = directions.get(dir)
                if straight is None:
                    turns += 1
                    dir, pos = list(directions.items())[0]
                else:
                    pos = straight

    return path[1:]


def vis_walked_perimeter(path: list[tuple[Pos, Dir]], width: int, height: int):
    map = [["." for _ in range(width + 2)] for _ in range(height + 2)]
    for p, d in path:
        map[p.i + 1][p.j + 1] = d
    for row in map:
        print("".join(row))
    print()


def count_turns(path: list[tuple[Pos, Dir]]) -> int:
    turns = 0
    prev_dir = None
    for _, dir in path:
        if prev_dir is None:
            prev_dir = dir
        else:
            if prev_dir != dir:
                turns += 1
                prev_dir = dir
    if path[-1][1] != path[0][1]:
        turns += 1
    return turns


def part_2(input):
    print()
    grid = Grid(input)
    visited = map_plots(grid)
    perimeters = map_perimeters(visited)
    visualize_perimiters(perimeters, grid.width, grid.height)
    print()
    for key, perimeter in perimeters.items():
        outer_perimeter = get_outer_perimiter(visited, key, perimeter)
        # print(outer_perimeter)
        # print(count_turns(outer_perimeter))
        path = walk_perimeter(outer_perimeter)
        vis_walked_perimeter(path, grid.width, grid.height)
        turns = count_turns(path)
        print(key, turns)


if __name__ == "__main__":
    part_1(input_file)
    part_2(CONTROL_1)
