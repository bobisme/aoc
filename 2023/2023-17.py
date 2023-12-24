#!/usr/bin/env python
from collections.abc import Generator
from dataclasses import dataclass, field
from typing import (
    Any,
    DefaultDict,
    Deque,
    Dict,
    Iterator,
    Literal,
    NamedTuple,
    Optional,
    Self,
    Tuple,
)
import heapq
from enum import CONFORM, CONTINUOUS, UNIQUE, Enum, IntEnum, ReprEnum, auto, verify
from colored import Fore, Style
from itertools import islice, pairwise
import pyrsistent
from pyrsistent import PSet
from pprint import pp

BIG_NUM = 1_000_000

# Each city block is marked by a single digit that represents the amount of
# heat loss if the crucible enters that block.
# The crucible can move at most three blocks in a single direction before it
# must turn 90 degrees left or right. The crucible also can't reverse
# direction; after entering each city block, it may only turn left, continue
# straight, or turn right.
CONTROL_1 = """\
2413432311323
3215453535623
3255245654254
3446585845452
4546657867536
1438598798454
4457876987766
3637877979653
4654967986887
4564679986453
1224686865563
2546548887735
4322674655533
""".splitlines()

STRAIGHTS = 3

with open("2023-17.input") as f:
    input = [line.strip() for line in f.readlines()]

Pos = NamedTuple("Pos", [("i", int), ("j", int)])
Pos.__repr__ = lambda x: f"({x.i}, {x.j})"


def dist(a: Pos, b: Pos) -> int:
    return abs(b.i - a.i) + abs(b.j - a.j)


Input = tuple[tuple[int, ...], ...]


@verify(UNIQUE)
class D(str, ReprEnum):
    UP = "^"
    DOWN = "v"
    LEFT = "<"
    RIGHT = ">"
    INVALID = "X"

    @classmethod
    def detect(cls, prev: Pos, curr: Pos) -> str:
        if curr.i - prev.i == 1:
            return cls.RIGHT
        if curr.i - prev.i == -1:
            return cls.LEFT
        if curr.j - prev.j == 1:
            return cls.DOWN
        if curr.j - prev.j == -1:
            return cls.UP
        return cls.INVALID


def print_input(input: Input):
    for line in input:
        print("".join(str(x) for x in line))


def up(coord) -> Pos:
    return Pos(coord.i, coord.j - 1)


def right(coord) -> Pos:
    return Pos(coord.i + 1, coord.j)


def down(coord) -> Pos:
    return Pos(coord.i, coord.j + 1)


def left(coord) -> Pos:
    return Pos(coord.i - 1, coord.j)


def in_bounds(rows: int, cols: int, coord: Pos) -> bool:
    return 0 <= coord.i < cols and 0 <= coord.j < rows


def expand_coord(input: Input, coord: Pos) -> list[Pos]:
    rows = len(input)
    cols = len(input[0])
    next_dirs = [right(coord), down(coord), left(coord), up(coord)]
    next_dirs = [d for d in next_dirs if in_bounds(rows, cols, d)]
    return next_dirs


def retrace_path(prev, start: Pos):
    yield start
    n = prev[start.j][start.i]
    while n != Pos(-1, -1):
        yield n
        n = prev[n.j][n.i]


def back_track(prev: list[list[Pos]], curr: Pos, count=3) -> list[Pos]:
    rev = [curr]
    p = prev[curr.j][curr.i]
    i = count
    while p != Pos(-1, -1) and i > 0:
        rev.append(p)
        curr = p
        p = prev[curr.j][curr.i]
        i -= 1
    rev.reverse()
    return rev


def all_straight(coords: list[Pos]) -> bool:
    first = coords[0]
    return all(x.i == first.i for x in coords[1:]) or all(
        x.j == first.j for x in coords[1:]
    )


assert all_straight([Pos(1, 4), Pos(1, 3), Pos(1, 2), Pos(1, 1)])


def visualize_path(input, path: list[Pos]):
    print(f"{Fore.blue}{'-'*len(input[0])}{Style.reset}")
    pathviz: list[list[str]] = [
        [f"{Fore.black}{c}{Style.reset}" for c in row] for row in input
    ]
    p = Pos(0, 0)
    for n in path[1:]:
        dir = D.detect(p, n)
        pathviz[n.j][n.i] = dir
        p = n
    for row in pathviz:
        print("".join(row))
    print(f"{Fore.blue}{'-'*len(input[0])}{Style.reset}")


@dataclass
class HeapItem:
    total_heat: int
    coord: Pos
    input_dir: Optional[D] = None
    v: int = 0

    def __lt__(self, other):
        return self.total_heat < other.total_heat


def neighbors(coord: Pos) -> list[Pos]:
    return [right(coord), down(coord), left(coord), up(coord)]


def expand_to_depth(
    input: Input,
    unvisited: set[Pos],
    prev: list[Pos],
    curr: Pos,
    rem: int,
    depth: int,
) -> Generator[list[Pos], Any, Any]:
    rows = len(input)
    cols = len(input[0])
    dir = D.detect(prev[-1], curr)

    def is_ok(dir):
        return in_bounds(rows, cols, dir) and dir in unvisited

    neighbors = right(curr), down(curr), left(curr), up(curr)
    neighbors = [d for d in neighbors if is_ok(d)]
    if depth <= 0:
        yield [curr]
        return
    for nbr in neighbors:
        if nbr in prev:
            continue
        next_dir = D.detect(curr, nbr)
        next_rem = rem - 1 if next_dir == dir else STRAIGHTS
        if next_rem <= 0:
            continue
        for sub in expand_to_depth(
            input, unvisited, prev + [curr], nbr, next_rem, depth - 1
        ):
            yield [curr] + sub


def eval_path(input: Input, path: list[Pos]) -> int:
    return sum(input[c.j][c.i] for c in path)


NodeKey = Tuple[Optional[str], Pos, int]


@dataclass
class Node:
    coord: Pos
    cumulative_heat: int
    prev: Optional[Self] = None
    times_same_dir: int = 0
    visited: bool = False
    in_dir: str = D.INVALID
    out: set[Self] = field(default_factory=set)
    v: int = 0

    def __repr__(self) -> str:
        return (
            f"{self.in_dir}{self.coord}"
            f" out:{[x.coord for x in self.out]}"
            f" heat:{self.cumulative_heat} same:{self.times_same_dir}"
            # f" visited:{'Y' if self.visited else 'N'}"
        )

    def __hash__(self):
        return hash(self.key())

    def __lt__(self, other):
        return self.cumulative_heat < other.cumulative_heat

    def key(self) -> NodeKey:
        return (self.in_dir, self.coord, self.times_same_dir)

    def dir_from(self, coord: Optional[Pos]) -> str:
        if not coord:
            return D.INVALID
        return D.detect(coord, self.coord)


class Q:
    q_set: set[NodeKey]
    q: list[Node]
    versions: dict[NodeKey, int]

    def __init__(self):
        self.versions = dict()
        self.q_set = set()
        self.q = []

    def __contains__(self, key: NodeKey):
        return key in self.q_set

    def __bool__(self) -> bool:
        return len(self.q) > 0

    def __len__(self) -> int:
        return len(self.q)

    def expected_version(self, node: Node) -> int:
        return self.versions.get(node.key(), 0)

    def has_latest_version(self, node: Node) -> bool:
        return self.expected_version(node) == node.v

    def increment_version(self, node: Node):
        next_v = self.expected_version(node) + 1
        node.v = next_v
        self.versions[node.key()] = next_v

    def pop(self) -> Optional[Node]:
        # node = heapq.heappop(self.q)
        # self.q_set.remove(node)
        # return node
        while self.q:
            node = heapq.heappop(self.q)
            if not node:
                return
            expected = self.expected_version(node)
            if node.v != expected:
                continue
            try:
                self.q_set.remove(node.key())
            except Exception:
                pass
            return node

    def push(self, node: Node):
        # self.versions[node.key()] = 0
        self.q_set.add(node.key())
        heapq.heappush(self.q, node)

    def update(self, node: Node):
        self.increment_version(node)
        self.push(node)

    def reheap(self):
        heapq.heapify(self.q)


def retrace(
    prev_nodes: dict[NodeKey, Node], node: Node, limit=1_000
) -> Generator[Pos, Any, Any]:
    i = 1
    yield node.coord
    while node.prev and i <= limit:
        yield node.prev.coord
        node = prev_nodes[node.key()]
        i += 1


_done = set()


def populate_graph(
    rows: int, cols: int, nodes: dict[NodeKey, Node], explored: PSet[Node], curr: Node
):
    # if curr.key() in _cache_pop_graph:
    #     return _cache_pop_graph[curr.key()]
    if curr.key() in _done:
        return
    next_coords = [
        right(curr.coord),
        down(curr.coord),
        left(curr.coord),
        up(curr.coord),
    ]
    next_coords = [x for x in next_coords if in_bounds(rows, cols, x)]
    neighbors = []
    for next in next_coords:
        next_dir = D.detect(curr.coord, next)
        same = 0
        if next_dir == curr.in_dir:
            same = curr.times_same_dir + 1
        if same > STRAIGHTS:
            continue
        next_node = Node(next, BIG_NUM, in_dir=next_dir, times_same_dir=same)
        if next_node.key() in nodes:
            next_node = nodes[next_node.key()]
        neighbors.append(next_node)
    neighbors = [x for x in neighbors if x not in explored]
    # neighbors = [x for x in neighbors if x not in curr.out]
    # out = []
    for next in neighbors:
        # print(next)
        curr.out.add(next)
        nodes[next.key()] = next
        populate_graph(rows, cols, nodes, explored.add(curr), next)
    # _cache_pop_graph[curr.key()] = (curr, out)
    _done.add(curr.key())


def blue_line(length: int):
    print(f"{Fore.blue}{'━'*length}{Style.reset}")


# def part_1_3(input):
#     rows = len(input)
#     cols = len(input[0])
#     input = tuple(tuple(int(x) for x in line) for line in input)
#     blue_line(cols)
#     print_input(input)
#     blue_line(cols)
#     q = Q()
#     start = Node(Coord(0, 0), cumulative_heat=0)
#     nodes: dict[NodeKey, Node] = {
#         start.key(): start,
#     }
#     pop_visited: set[NodeKey] = set()
#     populate_graph(rows, cols, nodes, pyrsistent.s(), start)
#     nodes_to_q = set()
#
#     def q_nodes(n: Node):
#         nodes_to_q.add(n)
#         for o in n.out:
#             if o not in nodes_to_q:
#                 q_nodes(o)
#
#     q_nodes(start)
#     for node in nodes_to_q:
#         q.push(node)
#     last_node = None
#     prev_nodes: dict[NodeKey, Node] = {}
#     print(f"POPULATED {len(pop_visited)}")
#     # for n in pop_visited:
#     #     if n[2] == 3:
#     #         print(n)
#
#     pp(q.q[:3])
#     checked = 0
#     while q:
#         node = q.pop()
#         checked += 1
#         # if checked > 20:
#         #     last_node = node
#         #     break
#         # print(len(q), q.q[:3])
#         # print(f"{node=}")
#         if node is None:
#             break
#         curr = node.coord
#         next_coords = [up(curr), right(curr), down(curr), left(curr)]
#         neighbors = node.out
#         # neighbors = [x for x in neighbors if x.key() in q]
#         neighbors = [x for x in neighbors if not x.visited]
#         for nbr in neighbors:
#             # print(node, "->", nbr)
#             edge_heat = input[nbr.coord.j][nbr.coord.i]
#             tmp_heat = node.cumulative_heat + edge_heat
#             # print(f"{{{node=}}}->{{{nbr=}}} {edge_heat=}")
#             if tmp_heat < nbr.cumulative_heat:
#                 # print("lower")
#                 # nbr.times_same_dir = times_same_dir
#                 nbr.cumulative_heat = tmp_heat
#                 # TODO: use update to push another node
#                 # q.update(nbr)
#                 # q.increment_version(nbr)
#                 nbr.prev = node
#                 prev_nodes[nbr.key()] = node
#                 if nbr.coord == Coord(cols - 1, rows - 1):
#                     last_node = nbr
#         # TODO: eliminate
#         q.reheap()
#         node.visited = True
#     print("checked", checked)
#     if last_node is None:
#         raise Exception("FUCK, checked", checked)
#     print(f"{last_node.cumulative_heat=}")
#     path = list(retrace(prev_nodes, last_node))
#     path.reverse()
#     print(path)
#     # print(f"{len(path)}")
#     visualize_path(input, path)
#     total_heat_loss = sum(input[n.j][n.i] for n in path)
#     return total_heat_loss


class Field:
    def __init__(self, input: tuple[tuple[int, ...], ...]):
        self.input = input
        self.rows = len(input)
        self.cols = len(input[0])

    def in_bounds(self, pos: Pos) -> bool:
        return 0 <= pos.i < self.cols and 0 <= pos.j < self.rows

    def get(self, pos: Pos) -> int:
        return self.input[pos.j][pos.i]


def part_1_4(input):
    input = tuple(tuple(int(x) for x in line) for line in input)
    field = Field(input)
    blue_line(field.cols)
    print_input(input)
    blue_line(field.cols)

    def next_nodes(in_dir: D, pos: Pos) -> Iterator[tuple[Pos, D, int]]:
        if in_dir == D.RIGHT or in_dir == D.LEFT:
            up_heat = 0
            for i in range(1, 4):
                p = Pos(pos.i, pos.j - i)
                if not field.in_bounds(p):
                    break
                up_heat += field.get(p)
                yield p, D.UP, up_heat
            down_heat = 0
            for i in range(1, 4):
                p = Pos(pos.i, pos.j + i)
                if not field.in_bounds(p):
                    break
                down_heat += field.get(p)
                yield p, D.DOWN, down_heat
        if in_dir == D.UP or in_dir == D.DOWN:
            left_heat = 0
            for i in range(1, 4):
                p = Pos(pos.i - i, pos.j)
                if not field.in_bounds(p):
                    break
                left_heat += field.get(p)
                yield p, D.LEFT, left_heat
            right_heat = 0
            for i in range(1, 4):
                p = Pos(pos.i + i, pos.j)
                if not field.in_bounds(p):
                    break
                right_heat += field.get(p)
                yield p, D.RIGHT, right_heat

    discovered = set()

    @dataclass
    class Node:
        pos: Pos
        in_dir: D
        heat: int

        def __hash__(self):
            return hash((self.pos, self.in_dir))

    # def dfs():
    #     start = Pos(0, 0)
    #     end = Pos(field.cols - 1, field.rows - 1)
    #     adj_list = DefaultDict(set)
    #     stack = [
    #         (Node(start, D.RIGHT, field.get(start)), {start}, field.get(start)),
    #         (Node(start, D.DOWN, field.get(start)), {start}, field.get(start)),
    #     ]
    #     lowest_terminal = BIG_NUM
    #     while stack:
    #         node, path, path_heat = stack.pop()
    #         if path_heat > lowest_terminal:
    #             continue
    #         if node.pos == end:
    #             summed = sum(field.get(p) for p in path)
    #             if path_heat < lowest_terminal:
    #                 lowest_terminal = path_heat
    #             print("terminal", summed, path_heat, "lowest:", lowest_terminal)
    #             visualize_path(field.input, list(path))
    #             continue
    #         if node not in discovered:
    #             discovered.add(node)
    #             for p, d, h in next_nodes(node.in_dir, node.pos):
    #                 if p in path:
    #                     continue
    #                 n = Node(p, d, h)
    #                 adj_list[node].add(n)
    #                 next_path = path.copy()
    #                 path.add(p)
    #                 stack.append((n, next_path, path_heat + h))
    #             discovered.remove(node)
    #
    # expanse = dfs()
    def expand(field):
        # TODO: iterate through each cell and throw its possible neighbors
        # into the adjacency list.
        adj_list = DefaultDict(set)

    # def dijkstra():
    #     start = Pos(0, 0)
    #     end = Pos(field.cols - 1, field.rows - 1)
    #     expanded = set()
    #     heat = [[BIG_NUM for c in row] for row in field.input]
    #     heat[start.j][start.i] = 0
    #     prev = {}
    #     frontier = []
    #     frontier.append((start, D.RIGHT, field.get(start), [start], field.get(start)))
    #     frontier.append((start, D.DOWN, field.get(start), [start], field.get(start)))
    #     while frontier:
    #         frontier.sort(key=lambda x: -heat[x[0].j][x[0].i])
    #         pos, in_dir, heat_, path, total_heat = frontier.pop()
    #         if pos == end:
    #             print("end")
    #             return
    #         expanded.add((pos, in_dir))
    #         for n, d, h in next_nodes(in_dir, pos):
    #             if n not in expanded and n not in frontier:
    #                 frontier.append(n)
    #             elif n in frontier with higher cost:
    #                 # frontier.
    #                 # replace existing node with n
    #             else:
    #                 panic!
    #             alt = heat[pos.j][pos.i] + field.get(n)
    #             if alt < heat[n.j][n.i]:
    #                 heat[n.j][n.i] = alt
    #                 prev[n] = pos
    #     return heat, prev
    #
    # heat, prev = dijkstra()

    def retrace(prev: dict[Pos, Pos], end: Pos):
        yield end
        n = prev[end]
        while n != Pos(-1, -1):
            yield n
            n = prev[n]

    # pp(heat)
    # print(list(retrace(prev, Pos(field.cols - 1, field.rows - 1))))

    return 0


def main(input):
    print(part_1_4(input))


if __name__ == "__main__":
    main(CONTROL_1)
