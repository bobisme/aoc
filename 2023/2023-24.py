#!/usr/bin/env python

from typing import Optional
import vector
from sympy import solve
from sympy.abc import x, y, z
from itertools import permutations, pairwise
from functools import cache

vector.register_awkward()

CONTROL_1 = """\
19, 13, 30 @ -2,  1, -2
18, 19, 22 @ -1, -1, -2
20, 25, 34 @ -2, -2, -4
12, 31, 28 @ -1, -2, -1
20, 19, 15 @  1, -5, -3
""".splitlines()

TEST_AREA_1 = (7, 27)
TEST_AREA_2 = (200000000000000, 400000000000000)

with open("2023-24.input") as f:
    input_file = [line.strip() for line in f.readlines()]


def parse_line(line):
    pos, vel = line.split("@", 1)
    pos = [int(x.strip()) for x in pos.split(", ", 2)]
    vel = [int(x.strip()) for x in vel.split(", ", 2)]
    return [
        dict(x=pos[0], y=pos[1], z=pos[2]),
        dict(x=vel[0], y=vel[1], z=vel[2]),
    ]


def intersect(pos1, pos2, vel1, vel2) -> tuple[Optional[float], Optional[float]]:
    slope1 = vel1.y / vel1.x
    slope2 = vel2.y / vel2.x
    f1 = slope1 * (x - pos1.x) + pos1.y - y
    f2 = slope2 * (x - pos2.x) + pos2.y - y
    sol = solve([f1, f2], [x, y], dict=True)
    if not sol:
        return None, None
    return sol[0][x], sol[0][y]


def is_in_test_area(
    area: tuple[int, int], x: Optional[float], y: Optional[float]
) -> bool:
    if x is None or y is None:
        return False
    lower, upper = area
    return lower <= x <= upper and lower <= y <= upper


def is_in_future(pos, vel, x, y) -> bool:
    x_in_future = False
    y_in_future = False
    if vel.x >= 0:
        x_in_future = x > pos.x
    else:
        x_in_future = x < pos.x
    if vel.y >= 0:
        y_in_future = y > pos.y
    else:
        y_in_future = y < pos.y
    return x_in_future and y_in_future


def main(input):
    for line in input:
        print(line)
    vecs = vector.awk([parse_line(line) for line in input])
    # x, y = intersect(vecs[0][0], vecs[1][0], vecs[0][1], vecs[1][1])
    # print(x, y)
    # print("in area", is_in_test_area(TEST_AREA_1, x, y))
    intersections = list(
        intersect(pv1[0], pv2[0], pv1[1], pv2[1])
        for pv1, pv2 in permutations(vecs, r=2)
    )
    total = sum(
        1 if is_in_test_area(TEST_AREA_1, *intr) else 0 for intr in intersections
    )
    # TODO: filter by `is_in_future(pos, vel, x, y)`
    print(total)


if __name__ == "__main__":
    main(CONTROL_1)
