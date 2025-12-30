#!/usr/bin/env python


import bisect
from collections import deque
from dataclasses import dataclass, field
import sys
from typing import Generator, Iterable, Iterator, LiteralString
import itertools
import time

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


# NOTE: @dataclass(slots=True) is way faster than NamedTuple!
@dataclass(slots=True)
class Pos:
    x: int
    y: int


@dataclass(slots=True)
class Line:
    a: Pos
    b: Pos

    def __post_init__(self):
        # Re-order line for efficient intersection checks.
        if self.a.x == self.b.x:
            # vertical: order by y
            if self.a.y > self.b.y:
                self.a, self.b = self.b, self.a
        else:  # horizontal
            if self.a.x > self.b.x:
                self.a, self.b = self.b, self.a


def area(a: Pos, b: Pos) -> int:
    return (abs(b.x - a.x) + 1) * (abs(b.y - a.y) + 1)


def get_areas(positions: list[Pos]) -> Iterable[tuple[int, tuple[int, int]]]:
    "Yields `(area, (pos_i, pos_j))`."
    for i in range(len(positions) - 1):
        a = positions[i]
        for j in range(i + 1, len(positions)):
            b = positions[j]
            yield area(a, b), (i, j)


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


def print_grid(grid: list[list[int]], file=sys.stdout):
    for row in grid:
        print(
            "".join(
                "." if cell <= EMPTY else ("#" if cell == RED else "X") for cell in row
            ),
            file=file,
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
    # areas = sorted(get_areas(positions), reverse=True)
    areas = sorted(get_areas(positions), key=lambda x: x[0], reverse=True)
    lines = list(gen_lines(positions))

    def lines_intersect(l1: Line, l2: Line) -> bool:
        return (l1.b.x >= l2.a.x and l2.b.x >= l1.a.x) and (
            l1.b.y >= l2.a.y and l2.b.y >= l1.a.y
        )

    def get_check_point(a: Pos, b: Pos) -> Pos:
        "Return `Pos` that is one step closer from `a` to `b`."
        return Pos(
            x=a.x + (1 if b.x > a.x else (-1 if b.x < a.x else 0)),
            y=a.y + (1 if b.y > a.y else (-1 if b.y < a.y else 0)),
        )

    def check_border_intersections(areas):
        for area, (i, j) in areas:
            a = positions[i]
            b = positions[j]
            check_bounds = (
                get_check_point(a, b),
                get_check_point(b, a),
            )
            min_x = min(p.x for p in check_bounds)
            min_y = min(p.y for p in check_bounds)
            max_x = max(p.x for p in check_bounds)
            max_y = max(p.y for p in check_bounds)

            # Approach: if any point is internal and none of the borders
            # intersect other lines, we're good. Internal check removed because
            # input doesn't need it.
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
        assert not "UNREACHABLE"

    return check_border_intersections(areas)


def part_2_fill_and_check(input: Input, print_=False):
    positions = [Pos(*map(int, line.split(","))) for line in input]
    areas = sorted(get_areas(positions), reverse=True)
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


@dataclass(slots=True)
class IntervalNode:
    """Node for binary interval tree."""

    range: range
    left: "IntervalNode | None" = None
    right: "IntervalNode | None" = None
    overlap_by_start: list[range] = field(default_factory=list)
    overlap_by_end: list[range] = field(default_factory=list)
    # cached values
    center: int = field(init=False)
    lo: int = field(init=False, repr=False)
    hi: int = field(init=False, repr=False)

    def __post_init__(self):
        r = self.range
        self.center = r.start + (r.stop - r.start) // 2
        self.lo = self.range.start
        self.hi = self.range.stop - 1

    def find(self, y: int) -> range | None:
        "Return first range that contains y."
        if y in self.range:
            return self.range
        if y < self.center:
            for r in self.overlap_by_start:
                if r.start > y:
                    break
                if y in r:
                    return r
            return self.left.find(y) if self.left else None
        else:
            for r in self.overlap_by_end:
                if r.stop <= y:
                    break
                if y in r:
                    return r
            return self.right.find(y) if self.right else None

    def add(self, r: range):
        if self.center not in r:
            if r.stop <= self.center:
                if self.left is None:
                    self.left = IntervalNode(r)
                else:
                    self.left.add(r)
            else:
                if self.right is None:
                    self.right = IntervalNode(r)
                else:
                    self.right.add(r)
            return

        bisect.insort(self.overlap_by_start, r, key=lambda r: r.start)
        bisect.insort(self.overlap_by_end, r, key=lambda r: r.stop)
        self.lo = min(self.lo, r.start)
        self.hi = max(self.hi, r.stop - 1)


@dataclass(slots=True)
class Candidate:
    pos: Pos
    r: range


@dataclass(slots=True)
class Candidates:
    # ordered list
    ys: list[int] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.ys)

    def __iter__(self, /) -> Iterator[int]:
        return iter(self.ys)

    def ordered_pairs(self) -> Iterator[tuple[int, int]]:
        # assert len(self) % 2 == 0, f"not even number of ys: {len(self)}"
        iterator = iter(self)
        return zip(iterator, iterator)

    def ordered_ranges(self) -> Iterator[range]:
        for y0, y1 in self.ordered_pairs():
            yield range(y0, y1 + 1)

    def toggle(self, y: int):
        idx = bisect.bisect_left(self.ys, y)
        if idx < len(self.ys) and self.ys[idx] == y:
            del self.ys[idx]
        else:
            self.ys.insert(idx, y)


def intersect_ranges(r1: range, r2: range) -> range | None:
    if r1.stop - 1 < r2.start or r1.start > r2.stop - 1:
        return None
    else:
        return range(max(r1.start, r2.start), min(r1.stop, r2.stop))


def part_2_sweep_line_interval_tree(input: Input):
    """
    https://www.wikiwand.com/en/articles/Sweep_line_algorithm
    """

    def prune_candidates(
        candidates: list[Candidate], interval_tree: IntervalNode
    ) -> Generator[Candidate]:
        for candidate in candidates:
            r = interval_tree.find(candidate.pos.y)
            if not r:
                continue
            intersection = intersect_ranges(candidate.r, r)
            if not intersection:
                continue
            candidate.r = intersection
            yield candidate

    positions = [Pos(*map(int, line.split(","))) for line in input]
    xy_ordered = sorted(positions, key=lambda p: (p.x, p.y))
    candidates: list[Candidate] = []
    left_candidates = Candidates()
    largest_area = 0

    pos_iter = iter(xy_ordered)
    for a, b in zip(pos_iter, pos_iter):
        assert a.x == b.x  # on same vertical
        left_candidates.toggle(a.y)
        left_candidates.toggle(b.y)

        ranges = left_candidates.ordered_ranges()
        next_range = next(ranges, None)
        if next_range is None:
            break
        interval_tree = IntervalNode(next_range)
        for r in ranges:
            interval_tree.add(r)

        for candidate in candidates:
            for y in (a.y, b.y):
                if y in candidate.r:
                    largest_area = max(largest_area, area(candidate.pos, Pos(a.x, y)))

        candidates = list(prune_candidates(candidates, interval_tree))

        for y in (a.y, b.y):
            containing_range = interval_tree.find(y)
            if containing_range:
                candidates.append(Candidate(pos=Pos(a.x, y), r=containing_range))
    return largest_area


def _test():
    def assert_eq(a, b):
        assert a == b, f"{a} != {b}"

    assert_eq(part_1(CONTROL_1), 50)
    assert_eq(part_2_check_borders(CONTROL_1), 24)
    assert_eq(part_2_sweep_line_interval_tree(CONTROL_1), 24)


def run(fn, year=2025, day=9, part=0):
    start = time.perf_counter_ns()
    res = fn()
    elapsed_ns = time.perf_counter_ns() - start
    print(f"{year}\t{day}\t{part}\t{res}\t{elapsed_ns}")


if __name__ == "__main__":
    with open("2025-09.input") as f:
        input_file = [line.rstrip("\n") for line in f.readlines()]
    _test()

    # positions = [Pos(*map(int, line.split(","))) for line in input_file]
    # compressed = compress_positions(positions)
    # grid = positions_to_grid(compressed)
    # with open("2025-09.output", "w") as f:
    #     print_grid(grid, file=f)

    run(lambda: part_1(input_file), part=1)
    # run(lambda: part_2_check_borders(input_file), part=2)
    run(lambda: part_2_sweep_line_interval_tree(input_file), part=2)
    # run(lambda: part_2_sweep_line_interval_list(input_file), part=2)
