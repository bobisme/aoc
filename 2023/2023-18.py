#!/usr/bin/env python
from dataclasses import dataclass
import enum
from typing import Any, Generator, NamedTuple, Self
from pprint import pp
from itertools import pairwise

CONTROL_1 = """\
R 6 (#70c710)
D 5 (#0dc571)
L 2 (#5713f0)
D 2 (#d2c081)
R 2 (#59c680)
D 2 (#411b91)
L 5 (#8ceee2)
U 2 (#caa173)
L 1 (#1b58a2)
U 2 (#caa171)
R 2 (#7807d2)
U 3 (#a77fa3)
L 2 (#015232)
U 2 (#7a21e3)
""".splitlines()

with open("2023-18.input") as f:
    input = [line.strip() for line in f.readlines()]

BIG_NUM = 696969


class D(str, enum.ReprEnum):
    U = "U"
    R = "R"
    D = "D"
    L = "L"

    @classmethod
    def from_(cls, x: str) -> str:
        if x == "0":
            return cls.R
        if x == "1":
            return cls.D
        if x == "2":
            return cls.L
        if x == "3":
            return cls.U
        raise Exception("unreachable")


Point = NamedTuple("Point", [("x", int), ("y", int)])


@dataclass
class Instruction:
    dir: str
    steps: int

    @classmethod
    def parse(cls, line) -> Self:
        d, steps, color = line.split(" ", 2)
        return cls(d, int(steps))

    @classmethod
    def parse2(cls, line) -> Self:
        _, _, color = line.split(" ", 2)
        color = color.strip("# ()")
        steps = int(color[:5], 16)
        d = D.from_(color[5])
        return cls(d, steps)


def lerp(a: Point, b: Point) -> Generator[Point, Any, Any]:
    if a.x == b.x:
        step = 1 if a.y < b.y else -1
        for y in range(a.y, b.y, step):
            yield Point(a.x, y)
    if a.y == b.y:
        step = 1 if a.x < b.x else -1
        for x in range(a.x, b.x, step):
            yield Point(x, a.y)


def calc_area(points: list[Point]):
    S = sum((a.y + b.y) * (a.x - b.x) for a, b in pairwise(points))
    # print(f"{S=}")
    return abs(S) // 2


def calc_edge_mass(points: list[Point]):
    S = sum((abs(b.y - a.y) + abs(b.x - a.x)) for a, b in pairwise(points))
    # print(f"{S=}")
    return S


def bounding_box(points: list[Point]) -> tuple[Point, Point]:
    min_x = BIG_NUM
    min_y = BIG_NUM
    max_x = -BIG_NUM
    max_y = -BIG_NUM
    for p in points:
        min_x = min(p.x, min_x)
        min_y = min(p.y, min_y)
        max_x = max(p.x, max_x)
        max_y = max(p.y, max_y)
    return Point(min_x, min_y), Point(max_x, max_y)


def make_field(bounds):
    min_p, max_p = bounds
    rows = max_p.y - min_p.y + 1
    cols = max_p.x - min_p.x + 1
    field = [["." for _ in range(cols)] for _ in range(rows)]
    return field


def viz_edge(field, bounds, points: list[Point]) -> list[list[str]]:
    min_p, max_p = bounds
    y_off = -min_p.y
    x_off = -min_p.x
    for a, b in pairwise(points):
        # print(a, b)
        for p in lerp(a, b):
            # print(p)
            field[p.y + y_off][p.x + x_off] = "#"
    # edges = sum(1 for r in range(rows) for c in range(cols) if field[r][c] == "#")
    return field


def in_bounds(rows: int, cols: int, point: Point) -> bool:
    return 0 <= point.x < cols and 0 <= point.y < rows


def flood(field, bounds, point) -> int:
    rows = len(field)
    cols = len(field[0])
    min_p, _ = bounds
    y_off = -min_p.y
    x_off = -min_p.x
    point = Point(point.x + x_off, point.y + y_off)
    q = []
    q.append(point)
    count = 0
    while q:
        p = q.pop()
        neighbors = (
            Point(p.x + 1, p.y),
            Point(p.x - 1, p.y),
            Point(p.x, p.y + 1),
            Point(p.x, p.y - 1),
            Point(p.x + 1, p.y + 1),
            Point(p.x - 1, p.y + 1),
            Point(p.x + 1, p.y - 1),
            Point(p.x - 1, p.y - 1),
        )
        for n in neighbors:
            if not in_bounds(rows, cols, n):
                continue
            try:
                char = field[n.y][n.x]
            except:
                print(n, x_off, y_off, n.x + x_off, n.y + y_off, rows, cols)
                raise
            if char == "#":
                continue
            field[n.y][n.x] = "#"
            q.append(n)
        count += 1
    return count


def count_hashes(field):
    rows = len(field)
    cols = len(field[0])
    return sum(1 for r in range(rows) for c in range(cols) if field[r][c] == "#")


def main(input):
    instructions = [Instruction.parse(line) for line in input]
    points = []
    curr = Point(0, 0)
    points.append(curr)
    for inst in instructions:
        next = None
        if inst.dir == D.U:
            next = Point(curr.x, curr.y - (inst.steps))
        elif inst.dir == D.R:
            next = Point(curr.x + inst.steps, curr.y)
        elif inst.dir == D.D:
            next = Point(curr.x, curr.y + inst.steps)
        elif inst.dir == D.L:
            next = Point(curr.x - (inst.steps), curr.y)
        if not next:
            raise Exception("oh no")
        points.append(next)
        curr = next
    bounds = bounding_box(points)
    field = make_field(bounds)
    viz_edge(field, bounds, points)
    min_p, max_p = bounds
    center = Point(
        (max_p.x - min_p.x) // 2 + min_p.x, (max_p.y - min_p.y) // 2 + min_p.y
    )
    print(f"{center=}")
    flood_count = flood(field, bounds, center)
    print(f"{flood_count=}")
    for row in field:
        print("".join(row))
    # pp(points)

    hashes = count_hashes(field)
    # diff = area_ - edges
    print(f"{hashes=}")
    # area_ = area(points)


def part_2(input):
    instructions = [Instruction.parse2(line) for line in input]
    # instructions = [Instruction.parse(line) for line in input]
    pp(instructions)
    points = []
    curr = Point(0, 0)
    points.append(curr)
    for inst in instructions:
        next = None
        if inst.dir == D.U:
            next = Point(curr.x, curr.y - inst.steps)
        elif inst.dir == D.R:
            next = Point(curr.x + inst.steps, curr.y)
        elif inst.dir == D.D:
            next = Point(curr.x, curr.y + inst.steps)
        elif inst.dir == D.L:
            next = Point(curr.x - inst.steps, curr.y)
        if not next:
            raise Exception("oh no")
        points.append(next)
        curr = next
    pp(points)
    # field = make_field(bounds)
    # viz_edge(field, bounds, points)
    # min_p, max_p = bounds
    # center = Point(
    #     (max_p.x - min_p.x) // 2 + min_p.x, (max_p.y - min_p.y) // 2 + min_p.y
    # )
    # print(f"{center=}")
    # flood_count = flood(field, bounds, center)
    # print(f"{flood_count=}")
    # for row in field:
    #     print("".join(row))
    # # pp(points)
    #
    # hashes = count_hashes(field)
    # # diff = area_ - edges
    # print(f"{hashes=}")
    area = calc_area(points)
    edge_mass = calc_edge_mass(points)
    total = area + edge_mass // 2 + 1
    print(f"{area=} {edge_mass=} {total=}")
    # pass


if __name__ == "__main__":
    part_2(input)
