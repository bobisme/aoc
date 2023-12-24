#!/usr/bin/env python

from dataclasses import dataclass
from typing import Deque, Iterator, NamedTuple
from functools import cache, wraps

from colored import Fore, Style

CONTROL_1 = """\
...........
.....###.#.
.###.##..#.
..#.#...#..
....#.#....
.##..S####.
.##..#...#.
.......##..
.##.#.####.
.##..##.##.
...........
""".splitlines()

with open("2023-21.input") as f:
    input_file = [line.strip() for line in f.readlines()]


Pos = NamedTuple("Pos", [("i", int), ("j", int)])
Pos.__add__ = lambda a, b: Pos(a.i + b.i, a.j + b.j)
# Pos.__radd__ = lambda a, b: Pos(a.i + b.i, a.j + b.j)


def find_start(input):
    for j in range(len(input)):
        for i in range(len(input[0])):
            if input[j][i] == "S":
                yield Pos(i, j)


@dataclass
class Node:
    pos: Pos
    is_terminal: bool


StrMatrix = list[list[str]]


def neighbors(input: list[str], pos: Pos) -> Iterator[Pos]:
    rows = len(input)
    cols = len(input[0])

    def all() -> Iterator[Pos]:
        yield Pos(pos.i - 1, pos.j)
        yield Pos(pos.i + 1, pos.j)
        yield Pos(pos.i, pos.j - 1)
        yield Pos(pos.i, pos.j + 1)

    for p in all():
        if not (0 <= p.i < cols and 0 <= p.j < rows):
            continue
        if input[p.j][p.i] == "#":
            continue
        yield p


def traverse(
    input: list[str],
    marks: StrMatrix,
    discovered: set[tuple[Pos, int]],
    pos: Pos,
    rem: int,
):
    if rem == 0:
        marks[pos.j][pos.i] = "O"
        return
    discovered.add((pos, rem))

    for n in neighbors(input, pos):
        if rem == 1 and marks[pos.j][pos.i] == "O":
            continue
        if (n, rem - 1) in discovered:
            continue
        traverse(input, marks, discovered, n, rem - 1)


class Garden:
    input: list[str]
    rows: int
    cols: int

    def __init__(self, input: list[str]) -> None:
        self.input = input
        self.rows = len(input)
        self.cols = len(input[0])

    def get(self, pos: Pos) -> str:
        return self.input[pos.j % self.rows][pos.i % self.cols]


def neighbors_part_2(input: Garden, pos: Pos) -> Iterator[Pos]:
    def all() -> Iterator[Pos]:
        yield Pos(pos.i - 1, pos.j)
        yield Pos(pos.i + 1, pos.j)
        yield Pos(pos.i, pos.j - 1)
        yield Pos(pos.i, pos.j + 1)

    for p in all():
        if input.get(p) == "#":
            continue
        yield p


_traverse_part_2_cache = {}


# def traverse_in_bounds(
#     garden: Garden,
#     start: Pos,
#     steps: int,
# ) -> set[Pos]:
#     cached = _traverse_part_2_cache.get((start, steps))
#     if cached is not None:
#         return cached
#     marks = set()
#     q = Deque()
#     q.append((start, steps))
#     explored = set()
#     explored.add(start)
#     while q:
#         pos, rem = q.popleft()
#         explored.add(pos)
#         if rem % 2 == 0:
#             marks.add(pos)
#         if rem <= 0:
#             continue
#
#         for n in neighbors_part_2(garden, pos):
#             if not in_bounds(garden.rows, garden.cols, pos):
#                 continue
#             if n in explored:
#                 continue
#             q.append((n, rem - 1))
#     return marks


def traverse_in_bounds(
    garden: Garden,
    start: Pos,
    steps: int,
) -> set[Pos]:
    if steps < 0:
        return set()

    marks = set()
    discovered = set()

    @cache
    def inner(pos: Pos, rem: int, mark: bool):
        if pos in discovered:
            return
        discovered.add(pos)
        if mark:
            marks.add(pos)
        if rem > 0:
            for n in neighbors_part_2(garden, pos):
                if not in_bounds(garden.rows, garden.cols, n):
                    continue
                inner(n, rem - 1, not mark)
        discovered.remove(pos)

    inner(start, steps, steps % 2 == 0)
    return marks


def expected_marks(steps: int) -> int:
    return (steps + 1) ** 2


def outer_bounds(steps: int, start: Pos) -> Iterator[Pos]:
    for i in range(steps):
        j = steps - i
        yield Pos(start.i + i, start.j - j)
    for i in range(-steps, 0):
        j = -steps - i
        yield Pos(start.i + i, start.j - j)
    for i in range(steps + 1):
        j = i - steps
        yield Pos(start.i + i, start.j - j)
    for i in range(-steps + 1, 0):
        j = i - -steps
        yield Pos(start.i + i, start.j - j)


def edge_points(rows: int, cols: int, start: Pos) -> Iterator[tuple[Pos, Pos]]:
    """return relative spot position, relative block"""
    yield Pos(0, start.j), Pos(1, 0)
    yield Pos(cols - 1, start.j), Pos(-1, 0)
    yield Pos(start.i, 0), Pos(0, -1)
    yield Pos(start.i, rows - 1), Pos(0, 1)


def corner_points(rows: int, cols: int) -> Iterator[tuple[Pos, Pos]]:
    """return relative spot position, relative block"""
    yield Pos(0, 0), Pos(1, -1)
    yield Pos(cols - 1, 0), Pos(-1, -1)
    yield Pos(cols - 1, rows - 1), Pos(-1, 1)
    yield Pos(0, rows - 1), Pos(1, 1)


def in_bounds(rows: int, cols: int, pos: Pos) -> bool:
    return 0 <= pos.i < cols and 0 <= pos.j < rows


def rocks_at_bounds(garden: Garden, steps: int, start: Pos) -> int:
    n = 0
    for b in outer_bounds(steps, start):
        if garden.get(b) == "#":
            n += 1
    return n


_rocks_at_bounds_within_block_cache: dict[tuple[Pos, int], int] = dict()


def rocks_at_bounds_within_block(garden: Garden, steps: int, start: Pos) -> int:
    if steps >= 2 * garden.rows:
        return 0
    cached = _rocks_at_bounds_within_block_cache.get((start, steps))
    if cached is not None:
        return cached
    n = 0
    for b in outer_bounds(steps, start):
        if not in_bounds(garden.rows, garden.cols, b):
            continue
        if garden.get(b) == "#":
            n += 1
    _rocks_at_bounds_within_block_cache[(start, steps)] = n
    return n


def part_1(input):
    for line in input:
        print(line)
    start = next(find_start(input))
    rows = len(input)
    cols = len(input[0])
    print(f"{Fore.blue}{'─' * cols}{Style.reset}")
    marks = [[f"{Fore.black}.{Style.reset}" for _ in range(cols)] for _ in range(rows)]
    discovered = set()
    traverse(input, marks, discovered, start, 64)
    for row in marks:
        print("".join(row))
    n_marks = 0
    for row in marks:
        for col in row:
            if col == "O":
                n_marks += 1
    print(f"{n_marks=}")


def vis_field(garden: Garden, steps: int, start: Pos, marks=None, char="O"):
    rows = garden.rows
    cols = garden.cols
    print(f"{Fore.blue}{'─' * garden.cols}{Style.reset}")
    field = [[f"{Fore.black}{c}{Style.reset}" for c in row] for row in garden.input]
    if marks:
        for mark in marks:
            if not in_bounds(rows, cols, mark):
                continue
            field[mark.j][mark.i] = char
    # if in_bounds(garden.rows, garden.cols, start):
    #     field[start.j][start.i] = "S"
    # for s in range(steps % 2, steps + 1, 2):
    #     for b in outer_bounds(s, start):
    #         if not in_bounds(rows, cols, b):
    #             continue
    #         field[b.j % rows][b.i % cols] = "O"
    #         if garden.get(b) == "#":
    #             field[b.j % rows][b.i % cols] = f"{Fore.black}#{Style.reset}"
    # print(f"{rocks_at_bounds_within_block(garden, steps, start)=}")

    for row in field:
        print("".join(row))


def spot_to_block(rows: int, cols: int, pos: Pos) -> Pos:
    return Pos(pos.i // cols, pos.j // rows)


_count_fill_cache = {}


def count_fill(garden: Garden, start: Pos, steps: int) -> int:
    cached = _count_fill_cache.get((start, steps))
    if cached is not None:
        return cached
    rows = garden.rows
    cols = garden.cols
    count = 0
    for s in range(steps % 2, steps + 1, 2):
        for b in outer_bounds(s, start):
            if not in_bounds(rows, cols, b):
                continue
            if garden.get(b) == "#":
                continue
            count += 1
    _count_fill_cache[(start, steps)] = count
    return count


_count_rocks_in_full_block_cache = {}


def count_spots_in_full_block(garden, start) -> int:
    # cached = _count_rocks_in_full_block_cache.get(start)
    # if cached is not None:
    #     return cached
    width = garden.cols
    steps = int(1.5 * width)
    out = len(traverse_in_bounds(garden, start, steps))
    _count_rocks_in_full_block_cache[start] = out
    return out


def count_spots_in_full_blocks(garden, start, steps) -> int:
    width = garden.cols
    level = (steps + width + 1) // (2 * width)
    print(f"full blocks level {level} for {steps} steps")
    if level < 1:
        return 0
    n_blocks = level**2 + (level - 1) ** 2
    # n_blocks = num_full_blocks(garden, steps)
    out = n_blocks * count_spots_in_full_block(garden, start)
    print(f"count {n_blocks} full blocks = {out}")
    return out


def count_spots_in_a_points(garden, start, steps: int) -> int:
    # Once we're out of the starting block, there are always 4 or 8 point blocks
    # at the frontiers.
    width = garden.cols
    half_width = (width - 1) // 2
    if steps <= half_width:
        return 0
    s = (steps - half_width - 1) % width
    if s >= width - 1:
        return 0
    count = 0
    for p, _ in edge_points(garden.rows, garden.cols, start):
        # count += rocks_at_bounds_within_block(garden, s, p)
        c = len(traverse_in_bounds(garden, p, s))
        count += c
        # count += len(marks)
        # count += count_fill(garden, p, s)
        print(f"counting A point from {p} for {s} steps = {c}")
    return count


def count_spots_in_b_points(garden, start, steps: int) -> int:
    width = garden.cols
    half_width = (width - 1) // 2
    if steps <= width + half_width:
        return 0
    s = (steps - half_width - 1 + width) % (2 * width)
    if s >= 2 * width - 1:
        return 0
    count = 0
    for p, _ in edge_points(garden.rows, garden.cols, start):
        # count += rocks_at_bounds_within_block(garden, s, p)
        count += len(traverse_in_bounds(garden, p, s))
        # count += count_fill(garden, p, s)
    print(f"counting 4 B points {s} steps = {count}")
    return count


def count_spots_in_a_diags(garden, steps: int) -> int:
    # Once we're greater than the width of the square, we will always have
    # at least 1 diagonal A per corner, odd numbers only
    width = garden.cols
    if steps <= width:
        return 0
    diags_per_corner = ((steps - (width + 1)) // width // 2) * 2 + 1
    count = 0
    s = (steps - width - 1) % (2 * width)
    if s >= width - 1:
        return 0
    for p, _ in corner_points(garden.rows, garden.cols):
        # count += diags_per_corner * rocks_at_bounds_within_block(garden, s, p)
        count += diags_per_corner * len(traverse_in_bounds(garden, p, s))
        # count += diags_per_corner * count_fill(garden, p, s)
    print(f"counting {diags_per_corner} A diags for {s} steps = {count}")
    return count


def count_spots_in_b_diags(garden, steps: int) -> int:
    # Once we're greater than the 2*width of the square, we will always have
    # at least 2 diagonals B per corner, even numbers only
    width = garden.cols
    if steps <= 2 * width:
        return 0
    count = 0
    diags_per_corner = ((steps - 1) // width // 2) * 2
    s = (steps - 1) % (2 * width)
    if s >= width - 1:
        return 0
    for p, _ in corner_points(garden.rows, garden.cols):
        # count += diags_per_corner * rocks_at_bounds_within_block(garden, s, p)
        count += diags_per_corner * len(traverse_in_bounds(garden, p, s))
        # count += diags_per_corner * count_fill(garden, p, s)
    print(f"counting {diags_per_corner} B diags per corner for {s} steps = {count}")
    return count


def count_spots_the_hard_way(garden: Garden, start: Pos, steps: int) -> int:
    count = 0
    width = garden.cols
    if steps < 1.5 * width:
        count += len(traverse_in_bounds(garden, start, steps))
    # count += rocks_at_bounds_within_block(garden, steps, start)
    # count += count_spots_in_full_blocks(garden, start, steps)
    # count += count_spots_in_a_points(garden, start, steps)
    # count += count_spots_in_b_points(garden, start, steps)
    # count += count_spots_in_a_diags(garden, steps)
    # count += count_spots_in_b_diags(garden, steps)
    return count


def cache_traverse_beyond(f):
    cache = {}

    @wraps(f)
    def wrapper(garden: Garden, start: Pos, steps: int) -> set[Pos]:
        cached = cache.get((start, steps))
        if cached is not None:
            return cached
        out = f(garden, start, steps)
        cache[(start, steps)] = out
        return out

    return wrapper


def traverse_beyond(
    garden: Garden,
    start: Pos,
    steps: int,
) -> set[Pos]:
    if steps < 0:
        return set()

    marks = set()
    discovered = set()

    @cache
    def inner(pos: Pos, rem: int, mark: bool):
        if pos in discovered:
            return
        discovered.add(pos)
        if mark:
            marks.add(pos)
        if rem > 0:
            for n in neighbors_part_2(garden, pos):
                inner(n, rem - 1, not mark)
        discovered.remove(pos)

    inner(start, steps, steps % 2 == 0)
    return marks

    vis_field(garden, steps, start, marks=discovered, char="D")


TARGET_STEPS = 26_501_365


def part_2(input):
    for line in input:
        print(line)
    width = len(input)
    half_width = (width - 1) // 2
    start = next(find_start(input))
    garden = Garden(input)
    print(f"{Fore.blue}{'─' * garden.cols}{Style.reset}")
    # steps = 64
    results = {}
    for steps in (half_width, width):
        marks = traverse_in_bounds(garden, start, steps)
        # vis_field(garden, steps, start, marks)
        results[steps] = len(marks)
    diff = results[width] - results[half_width]
    print(f"diamond = {results[half_width]} corners = {diff}")
    # steps to radius of complete tesselation = half_width + level * width
    levels = (TARGET_STEPS - half_width) // width + 1
    # levels = 2
    diamond_parts = levels**2 + (levels - 1) ** 2
    corner_parts = 2 * (levels - 1) * levels
    print(f"{diamond_parts=} {corner_parts=}")
    diamond_marks = diamond_parts * results[half_width]
    corner_marks = corner_parts * diff
    print(f"{diamond_marks=} {corner_marks=} stops = {diamond_marks+corner_marks}")
    # expected = expected_marks(steps)
    # rocks_easy = sum(rocks_at_bounds(garden, i, start) for i in range(1, steps + 1))
    # total_easy = expected - rocks_easy
    # vis_field(garden, steps, start)
    # a = rocks_at_bounds(garden, steps, start)
    # total_hard = expected - rocks_hard
    # print("width", len(input))


if __name__ == "__main__":
    x = """\
    .......0......
    """
    part_2(input_file)
