#!/usr/bin/env python

from collections import deque
from dataclasses import dataclass
from functools import cache
import itertools
import time
from typing import LiteralString, NamedTuple

Input = list[str] | list[LiteralString]

CONTROL_1: Input = (
    """\
7,1
11,1
11,7
9,7
9,5
2,5
2,3
7,3
""".splitlines()
)

Pos = NamedTuple("Pos", [("x", int), ("y", int)])


@dataclass
class Line:
    a: Pos
    b: Pos

    def __post_init__(self):
        # Re-order line for efficient intersection checks.
        if self.a.x == self.b.x:  # vertical: order by y
            if self.a.y > self.b.y:
                self.a, self.b = self.b, self.a
        else:  # horizontal
            if self.a.x > self.b.x:
                self.a, self.b = self.b, self.a


def area(a: Pos, b: Pos) -> int:
    return (abs(b.x - a.x) + 1) * (abs(b.y - a.y) + 1)


def get_areas(positions: list[Pos]) -> list[tuple[int, tuple[int, int]]]:
    areas = []
    for i in range(len(positions) - 1):
        a = positions[i]
        for j in range(i + 1, len(positions)):
            b = positions[j]
            areas.append((area(a, b), (i, j)))
    return areas


def part_1(input: Input):
    positions = [Pos(*map(int, line.split(","))) for line in input]
    areas = get_areas(positions)
    return max(areas, key=lambda x: x[0])[0]


VOID = -1
EMPTY = 0
RED = 1
GREEN = 2


def gen_lines(positions: list[Pos]):
    "Generator for all line segments."
    for i in range(len(positions) - 1):
        a = positions[i]
        b = positions[i + 1]
        yield Line(a, b)
    yield Line(positions[-1], positions[0])


def positions_to_grid(positions: list[Pos]):
    max_x = max(p.x for p in positions)
    max_y = max(p.y for p in positions)
    grid = [[-1 for _ in range(max_x + 3)] for _ in range(max_y + 3)]
    for p in positions:
        grid[p.y + 1][p.x + 1] = RED
    for line in gen_lines(positions):
        x_dir = 1 if line.b.x >= line.a.x else -1
        for x in range(line.a.x, line.b.x + x_dir, x_dir):
            y_dir = 1 if line.b.y >= line.a.y else -1
            for y in range(line.a.y, line.b.y + y_dir, y_dir):
                if grid[y + 1][x + 1] <= EMPTY:
                    grid[y + 1][x + 1] = GREEN

    return grid


def print_grid(grid: list[list[int]]):
    for row in grid:
        print(
            "".join(
                "." if cell <= EMPTY else ("#" if cell == RED else "X") for cell in row
            )
        )


def compress_positions(positions: list[Pos]) -> list[Pos]:
    """
    Return a new list of positions that compress the space between points
    which maps to the original list of positions.
    """
    compressed_positions = positions.copy()
    pos_by_x = sorted(enumerate(positions), key=lambda x: x[1].x)
    pos_by_y = sorted(enumerate(positions), key=lambda x: x[1].y)

    last_x = pos_by_x[0][1].x
    offset = 0
    for i, p in pos_by_x:
        if p.x == last_x:
            compressed_positions[i] = Pos(p.x - last_x + offset, p.y)
            continue
        last_x = p.x
        offset += 2
        compressed_positions[i] = Pos(p.x - last_x + offset, p.y)

    last_y = pos_by_y[0][1].y
    offset = 0
    for i, p in pos_by_y:
        if p.y == last_y:
            compressed_positions[i] = Pos(
                compressed_positions[i].x, p.y - last_y + offset
            )
            continue
        last_y = p.y
        offset += 2
        compressed_positions[i] = Pos(compressed_positions[i].x, p.y - last_y + offset)
    return compressed_positions


def lines_intersect(l1: Line, l2: Line) -> bool:
    if l1.b.x < l2.a.x or l2.b.x < l1.a.x:
        return False
    if l1.b.y < l2.a.y or l2.b.y < l1.a.y:
        return False
    return True


def part_2_check_borders(input: Input):
    positions = [Pos(*map(int, line.split(","))) for line in input]
    areas = get_areas(positions)
    areas.sort(key=lambda x: -x[0])
    lines = list(gen_lines(positions))

    def ray_intersects(start: Pos, line: Line) -> bool:
        "Check the ray cast from the given point to the right."
        # vertical
        if line.a.x == line.b.x:
            return (line.a.y <= start.y <= line.b.y) and (start.x <= line.a.x)

        # horizontal
        return (start.y == line.a.y) and (start.x <= line.b.x)

    @cache
    def is_point_inside(p: Pos) -> bool:
        "Check if given point is interior to the polygon."
        count = sum(1 for line in lines if ray_intersects(p, line))
        return count % 2 != 0

    def get_check_point(a: Pos, b: Pos) -> Pos:
        "Return `Pos` that is one step closer from `a` to `b`."
        return Pos(
            x=a.x + (1 if b.x > a.x else (-1 if b.x < a.x else 0)),
            y=a.y + (1 if b.y > a.y else (-1 if b.y < a.y else 0)),
        )

    def check_border_intersections():
        for _idx, (area, (i, j)) in enumerate(areas):
            a = positions[i]
            b = positions[j]
            # if (idx + 1) % 1000 == 0:
            #     print(f"checked {idx+1}/{len(areas)}")
            check_bounds = (
                get_check_point(a, b),
                get_check_point(b, a),
            )
            min_x = min(p.x for p in check_bounds)
            min_y = min(p.y for p in check_bounds)
            max_x = max(p.x for p in check_bounds)
            max_y = max(p.y for p in check_bounds)

            # Approach: if any point is internal and none of the borders intersect
            # other lines, we're good.
            if not is_point_inside(check_bounds[0]):
                continue
            borders = (
                Line(Pos(min_x, min_y), Pos(max_x, min_y)),
                Line(Pos(max_x, min_y), Pos(max_x, max_y)),
                Line(Pos(max_x, max_y), Pos(min_x, max_y)),
                Line(Pos(min_x, max_y), Pos(min_x, min_y)),
            )
            if not any(
                lines_intersect(border, line) for border in borders for line in lines
            ):
                return area

    # return check_border_positions() # 9.0s
    return check_border_intersections()  # 2.8s


def part_2_fill_and_check(input: Input, print_=False):
    positions = [Pos(*map(int, line.split(","))) for line in input]
    areas = get_areas(positions)
    areas.sort(key=lambda x: -x[0])
    compressed_positions = compress_positions(positions)
    # compressed_lines = list(gen_lines(compressed_positions))
    grid = positions_to_grid(compressed_positions)

    def get_check_point(a: Pos, b: Pos) -> Pos:
        "Return `Pos` that is one step closer from `a` to `b`."
        return Pos(
            x=a.x + (1 if b.x > a.x else (-1 if b.x < a.x else 0)),
            y=a.y + (1 if b.y > a.y else (-1 if b.y < a.y else 0)),
        )

    def is_point_inside(p: Pos) -> bool:
        "Check if given point is interior to the polygon."
        return grid[p.y + 1][p.x + 1] == -1

    def flood_fill(start: Pos, grid: list[list[int]]):
        "Flood fill the grid outside the polygon. Everything else is inside."
        q = deque()
        q.appendleft(start)
        while (p := q.popleft()) is not None:
            if not (0 <= p.x < len(grid[0]) and 0 <= p.y < len(grid)):
                return
            if grid[p.y][p.x] > -1:
                return
            grid[p.y][p.x] = EMPTY
            q.append(Pos(p.x - 1, p.y - 1))
            q.append(Pos(p.x - 1, p.y))
            q.append(Pos(p.x - 1, p.y + 1))
            q.append(Pos(p.x, p.y - 1))
            q.append(Pos(p.x, p.y + 1))
            q.append(Pos(p.x + 1, p.y - 1))
            q.append(Pos(p.x + 1, p.y))
            q.append(Pos(p.x + 1, p.y + 1))

    flood_fill(Pos(0, 0), grid)
    if print_:
        print_grid(grid)

    for _idx, (area, (i, j)) in enumerate(areas):
        a = compressed_positions[i]
        b = compressed_positions[j]
        # if (idx + 1) % 1000 == 0:
        #     print(f"checked {idx+1}/{len(areas)}")
        check_bounds = (
            get_check_point(a, b),
            get_check_point(b, a),
        )
        min_x = min(p.x for p in check_bounds)
        min_y = min(p.y for p in check_bounds)
        max_x = max(p.x for p in check_bounds)
        max_y = max(p.y for p in check_bounds)
        check_points = itertools.chain(
            (Pos(x, min_y) for x in range(min_x, max_x + 1)),
            (Pos(x, max_y) for x in range(min_x, max_x + 1)),
            (Pos(min_x, y) for y in range(min_y + 1, max_y)),
            (Pos(max_x, y) for y in range(min_y + 1, max_y)),
        )
        if all(is_point_inside(p) for p in check_points):
            return area


part_2 = part_2_check_borders


def _test():
    def assert_eq(a, b):
        assert a == b, f"{a} != {b}"

    assert_eq(part_1(CONTROL_1), 50)
    assert_eq(part_2(CONTROL_1), 24)


def run(fn, year=2025, day=9, part=0):
    start = time.perf_counter_ns()
    res = fn()
    elapsed_ns = time.perf_counter_ns() - start
    print(f"{year}\t{day}\t{part}\t{res}\t{elapsed_ns}")


if __name__ == "__main__":
    with open("2025-09.input") as f:
        input_file = [line.rstrip("\n") for line in f.readlines()]
    _test()
    run(lambda: part_1(input_file), part=1)
    run(lambda: part_2(input_file), part=2)
