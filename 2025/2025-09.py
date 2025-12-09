#!/usr/bin/env python

from collections import deque
from functools import cache
import itertools
from typing import LiteralString, NamedTuple
import timeit

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

with open("2025-09.input") as f:
    input_file = [line.rstrip("\n") for line in f.readlines()]

Pos = NamedTuple("Pos", [("x", int), ("y", int)])
Line = NamedTuple("Line", [("a", Pos), ("b", Pos)])


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


def lines(positions: list[Pos]):
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
    for line in lines(positions):
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


def part_2_check_borders(input: Input):
    positions = [Pos(*map(int, line.split(","))) for line in input]
    areas = get_areas(positions)
    areas.sort(key=lambda x: -x[0])
    compressed_positions = compress_positions(positions)

    def intersects(ray: Pos, line: Line) -> bool:
        "Check the ray cast from the given point to the right."
        # vertical line
        if line.a.x == line.b.x:
            line_a, line_b = line.a, line.b
            if line_a.y > line_b.y:
                line_a, line_b = line_b, line_a
            return (line_a.y <= ray.y <= line_b.y) and ray.x <= line.a.x
        # horizontal
        line_a, line_b = line.a, line.b
        if line_a.x > line_b.x:
            line_a, line_b = line_b, line_a
        return ray.y == line_a.y and ray.x <= line_b.x

    @cache
    def is_point_inside(p: Pos) -> bool:
        "Check if given point is interior to the polygon."
        intersections = [
            line for line in lines(compressed_positions) if intersects(p, line)
        ]
        intersect_count = len(intersections)
        return intersect_count % 2 != 0

    def get_check_point(a: Pos, b: Pos) -> Pos:
        "Return `Pos` that is one step closer from `a` to `b`."
        return Pos(
            x=a.x + (1 if b.x > a.x else (-1 if b.x < a.x else 0)),
            y=a.y + (1 if b.y > a.y else (-1 if b.y < a.y else 0)),
        )

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
        # check all points on the border
        check_points = itertools.chain(
            (Pos(x, min_y) for x in range(min_x, max_x + 1)),
            (Pos(x, max_y) for x in range(min_x, max_x + 1)),
            (Pos(min_x, y) for y in range(min_y + 1, max_y)),
            (Pos(max_x, y) for y in range(min_y + 1, max_y)),
        )
        if all(is_point_inside(p) for p in check_points):
            return area

    return 0


# this ends up being 2x faster than _check_borders
def part_2_fill_and_check(input: Input, print_=False):
    positions = [Pos(*map(int, line.split(","))) for line in input]
    areas = get_areas(positions)
    areas.sort(key=lambda x: -x[0])
    compressed_positions = compress_positions(positions)
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


part_2 = part_2_fill_and_check


def _test():
    def assert_eq(a, b):
        assert a == b, f"{a} != {b}"

    assert_eq(part_1(CONTROL_1), 50)
    assert_eq(part_2(CONTROL_1, print_=True), 24)


def _bench(fn, count=100):
    return timeit.timeit(fn, number=count) / count * 1_000


if __name__ == "__main__":
    _test()
    print("tests: PASS")
    print("-" * 40)
    print("part_1:", part_1(input_file))
    print("part_2:", part_2(input_file))
    print("-" * 40)
    print("part_1 bench: {:.1f}ms".format(_bench(lambda: part_1(input_file), count=1)))
    print("part_2 bench: {:.1f}ms".format(_bench(lambda: part_2(input_file), count=1)))
