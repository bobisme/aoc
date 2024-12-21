#!/usr/bin/env python

from collections import deque
from collections.abc import Iterable
import heapq
from dataclasses import dataclass
from pprint import pp
import sys
from typing import DefaultDict, Deque, Literal

DIM = "\033[2m"
BRIGHT = "\033[1m"
RESET = "\033[0m"

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


State = tuple[Pos, Dir]


class Node:
    pos: Pos
    dir: Dir
    cost: int

    def __init__(self, pos: Pos, dir: Dir, cost: int) -> None:
        self.pos = pos
        self.dir = dir
        self.cost = cost

    @property
    def state(self) -> State:
        return (self.pos, self.dir)

    def __repr__(self) -> str:
        return f"{self.pos} {self.dir} {self.cost}"

    def __hash__(self) -> int:
        return hash((self.pos, self.dir))

    def __lt__(self, other: "Node") -> bool:
        return self.cost < other.cost


PrevMap = dict[State, set[State]]


def dijkstra(grid: Grid, start: Node) -> tuple[int, PrevMap]:
    distances = {(start.pos, start.dir): start.cost}  # (pos, dir) -> cost
    pq = [(start.cost, start)]
    prev_map: DefaultDict[State, set[State]] = DefaultDict(set)
    min_end_cost = sys.maxsize

    while pq:
        curr_cost, curr_node = heapq.heappop(pq)

        if grid.grid[curr_node.pos.i][curr_node.pos.j] == "E":
            if curr_cost < min_end_cost:
                min_end_cost = curr_cost
            continue

        if curr_cost > distances[(curr_node.pos, curr_node.dir)]:
            continue

        # Generate next possible moves
        next_moves: list[tuple[Dir, int]] = [
            (turn_left(curr_node.dir), 1001),  # Left turn cost
            (turn_right(curr_node.dir), 1001),  # Right turn cost
            (curr_node.dir, 1),  # Forward cost
        ]

        for next_dir, move_cost in next_moves:
            next_pos = get_next(curr_node.pos, next_dir)
            if grid.grid[next_pos.i][next_pos.j] != "#":
                next_cost = curr_cost + move_cost
                next_state = (next_pos, next_dir)

                if next_state not in distances or next_cost <= distances[next_state]:
                    next_node = Node(next_pos, next_dir, next_cost)
                    if next_state not in distances or next_cost < distances[next_state]:
                        distances[next_state] = next_cost
                        prev_map[next_state] = set()

                    distances[next_state] = next_cost
                    prev_map[next_state].add(curr_node.state)
                    heapq.heappush(pq, (next_cost, next_node))

    return min_end_cost, prev_map


def part_1(input):
    grid = Grid(input)
    lowest, _ = dijkstra(grid, Node(Pos(grid.width - 2, 1), ">", 0))
    print(lowest)


def map_paths(end_pos: Pos, prev_map: PrevMap) -> set[Pos]:
    positions = {end_pos}
    end_state = next(k for (k, v) in prev_map.items() if k[0] == end_pos)
    state = end_state
    q = deque([end_state])
    v = set()
    while q:
        state = q.popleft()
        v.add(state)
        positions.add(state[0])
        prev_states = prev_map[state]
        for prev_state in prev_states:
            if prev_state in v:
                continue
            q.append(prev_state)
    return positions


def print_path_positions(grid: Grid, positions: Iterable[Pos]):
    g = [[f"{DIM}{x}{RESET}" for x in row] for row in grid.grid]
    for p in positions:
        g[p.i][p.j] = f"{BRIGHT}O{RESET}"
    for row in g:
        print("".join(row))


def part_2(input):
    grid = Grid(input)
    lowest, prev_map = dijkstra(grid, Node(Pos(grid.width - 2, 1), ">", 0))
    end_pos = Pos(1, grid.width - 2)
    positions = map_paths(end_pos, prev_map)
    print_path_positions(grid, positions)
    print(len(positions))


if __name__ == "__main__":
    part_1(input_file)
    part_2(input_file)
