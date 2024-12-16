#!/usr/bin/env python

from collections import defaultdict
from collections.abc import Generator, Iterable
from dataclasses import dataclass
from pprint import pformat
from typing import DefaultDict, Literal, NamedTuple, Optional, TypeGuard
from string import ascii_letters, digits

DIM = "\033[2m"  # Dim/dark text
BRIGHT = "\033[1m"  # Bright/bold text
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

ALL_DIRS = ("^", ">", "v", "<")


def get_next_pos(pos: Pos, dir: Dir) -> Pos:
    if dir == "x":
        return pos
    if dir == "^":
        return Pos(pos.i - 1, pos.j)
    if dir == ">":
        return Pos(pos.i, pos.j + 1)
    if dir == "v":
        return Pos(pos.i + 1, pos.j)
    if dir == "<":
        return Pos(pos.i, pos.j - 1)


def opposite(dir: Dir) -> Dir:
    if dir == "x":
        return "x"
    if dir == "^":
        return "v"
    if dir == ">":
        return "<"
    if dir == "v":
        return "^"
    if dir == "<":
        return ">"


def right_turn(dir: Dir) -> Dir:
    if dir == "x":
        return "x"
    if dir == "^":
        return ">"
    if dir == ">":
        return "v"
    if dir == "v":
        return "<"
    if dir == "<":
        return "^"


def left_turn(dir: Dir) -> Dir:
    if dir == "x":
        return "x"
    if dir == "^":
        return "<"
    if dir == ">":
        return "^"
    if dir == "v":
        return ">"
    if dir == "<":
        return "v"


def is_dead_end(outer_perimeter: set[Pos], pos: Pos, dir: Dir) -> bool:
    next_dirs: Iterable[Dir] = (d for d in ("<", "^", ">", "v") if d != opposite(d))
    if any(get_next_pos(pos, x) in outer_perimeter for x in next_dirs):
        return False
    return True


def walk_perimeter(outer_perimeter: set[Pos]) -> tuple[list[tuple[Pos, Dir]], int]:
    visited_perimeter = set()
    turns = 1

    def can_turn_left(pos: Pos, dir: Dir):
        return get_next_pos(pos, left_turn(dir)) in outer_perimeter

    def can_turn_right(pos: Pos, dir: Dir):
        return get_next_pos(pos, right_turn(dir)) in outer_perimeter

    lowest_i = sorted(p.i for p in outer_perimeter)[0]
    lowest_j = sorted(p.j for p in [p for p in outer_perimeter if p.i == lowest_i])[0]
    start_pos = Pos(lowest_i, lowest_j)
    start_dir: Dir = ">"
    next_pos = get_next_pos(start_pos, start_dir)
    next_dir = start_dir

    def travel(pos: Pos, dir: Dir) -> tuple[list[tuple[Pos, Dir]], int]:
        if pos == start_pos:
            return [], 0
        visited_perimeter.add(pos)
        path: list[tuple[Pos, Dir]] = [(pos, dir)]
        turns = 0
        is_straight_ok = get_next_pos(pos, dir) in outer_perimeter
        if is_straight_ok:
            p, t = travel(get_next_pos(pos, dir), dir)
            path += p
            turns += t
        is_left_ok = can_turn_left(pos, dir)
        if is_left_ok:
            turns += 1
            p, t = travel(get_next_pos(pos, left_turn(dir)), left_turn(dir))
            path += p
            turns += t
        is_right_ok = can_turn_right(pos, dir)
        if is_right_ok:
            turns += 1
            p, t = travel(get_next_pos(pos, right_turn(dir)), right_turn(dir))
            path += p
            turns += t
        if not any((is_straight_ok, is_left_ok, is_right_ok)):
            turns += 3
        return path, turns

    p, turns = travel(next_pos, next_dir)
    path = [(start_pos, start_dir)] + p
    return path, turns + 1


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


def vis_walked_perimeter(path: list[tuple[Pos, Dir]], width: int, height: int):
    map = [[f"{DIM}.{RESET}" for _ in range(width + 2)] for _ in range(height + 2)]
    for p, d in path:
        map[p.i + 1][p.j + 1] = f"{BRIGHT}{d}{RESET}"
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


def map_vertices(
    visited: list[list[int]], key: int, perimeter: Iterable[Pos]
) -> list[Pos]:
    height = len(visited)
    width = len(visited[0])

    def hit(pos: Pos) -> bool:
        if pos.i < 0 or pos.i >= height or pos.j < 0 or pos.j >= width:
            return False
        if visited[pos.i][pos.j] == key:
            return True
        return False

    vertices = []

    for pos in perimeter:
        corners = (
            Pos(pos.i - 1, pos.j - 1),
            Pos(pos.i - 1, pos.j + 1),
            Pos(pos.i + 1, pos.j - 1),
            Pos(pos.i + 1, pos.j + 1),
        )
        up = Pos(pos.i - 1, pos.j)
        down = Pos(pos.i + 1, pos.j)
        left = Pos(pos.i, pos.j - 1)
        right = Pos(pos.i, pos.j + 1)
        sides = ((up, left), (up, right), (down, left), (down, right))
        for i, corner in enumerate(corners):
            if hit(corner):
                continue
            # print("corner", corner)
            if not (hit(sides[i][0]) ^ hit(sides[i][1])):
                vertices.append(corner)
    return vertices


def part_2(input):
    print()
    grid = Grid(input)
    visited = map_plots(grid)
    print()
    out = 0
    all_plot_tiles = get_plot_tiles(visited)
    for key, plot_tiles in all_plot_tiles.items():
        plot_tiles = list(get_plot_tiles_for_key(visited, key))
        verts = map_vertices(visited, key, plot_tiles)
        out += len(plot_tiles) * len(verts)
        vis_plots(plot_tiles, grid.width, grid.height, vertices=verts)
        print(len(verts))
    print(out)


if __name__ == "__main__":
    part_1(input_file)
    part_2(input_file)
