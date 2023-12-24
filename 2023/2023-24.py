#!/usr/bin/env python

import math
from typing import Optional
import vector
from itertools import permutations, pairwise, combinations
from functools import cache
from rich.progress import Progress
from numba import njit

vector.register_awkward()

CONTROL_1 = """\
19, 13, 30 @ -2,  1, -2
18, 19, 22 @ -1, -1, -2
20, 25, 34 @ -2, -2, -4
12, 31, 28 @ -1, -2, -1
20, 19, 15 @  1, -5, -3
""".splitlines()

TEST_AREA_1 = (7, 27)
TEST_AREA_2 = (200_000_000_000_000, 400_000_000_000_000)

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


# @njit
# def intersect(pos1, pos2, vel1, vel2):
#     # Ensure division by zero is handled
#     if vel1.x == 0 or vel2.x == 0:
#         return None, None
#
#     # Calculate slopes
#     slope1 = vel1.y / vel1.x
#     slope2 = vel2.y / vel2.x
#
#     # Handle parallel lines
#     if slope1 == slope2:
#         return None, None
#
#     # Calculate y-intercepts
#     y_intercept1 = pos1.y - slope1 * pos1.x
#     y_intercept2 = pos2.y - slope2 * pos2.x
#
#     # Solve for x
#     x = (y_intercept2 - y_intercept1) / (slope1 - slope2)
#
#     # Solve for y using either of the original equations
#     y = slope1 * x + y_intercept1
#
#     return x, y


def intersect_rays(pos1, pos2, vel1, vel2):
    # denom is the determinant
    denom = vel1.x * vel2.y - vel1.y * vel2.x
    if denom == 0:
        return None, None

    t = (pos2[0] - pos1[0]) * vel2[1] - (pos2[1] - pos1[1]) * vel2[0]
    s = (pos2[0] - pos1[0]) * vel1[1] - (pos2[1] - pos1[1]) * vel1[0]

    t /= denom
    s /= denom

    if t >= 0 and s >= 0:
        return pos1[0] + t * vel1[0], pos1[1] + t * vel1[1]
    else:
        return None


@njit
def is_in_test_area(
    area: tuple[int, int], x: Optional[float], y: Optional[float]
) -> bool:
    if x is None or y is None:
        return False
    lower, upper = area
    return lower <= x <= upper and lower <= y <= upper


@njit
def is_in_future(pos, vel, x, y) -> bool:
    x_in_future = False
    y_in_future = False
    if vel.x >= 0:
        x_in_future = x >= pos.x
    else:
        x_in_future = x <= pos.x
    if vel.y >= 0:
        y_in_future = y >= pos.y
    else:
        y_in_future = y <= pos.y
    return x_in_future and y_in_future


def main(input, test_area):
    for line in input:
        print(line)
    vecs = vector.awk([parse_line(line) for line in input])
    # print(vecs.type)
    # return
    n_intersections = 0
    n_inter_in_test = 0
    total = 0
    with Progress() as progress:
        n = len(vecs)
        k = 2
        # ops = math.factorial(n) // (math.factorial(k) * math.factorial(len(vecs) - 2))
        ops = math.factorial(n) // math.factorial(n - k)
        print("operations:", ops)
        task = progress.add_task("operating", total=ops)
        for pv1, pv2 in permutations(vecs, r=2):
            progress.update(task, advance=1)
            intx, inty = intersect(pv1[0], pv2[0], pv1[1], pv2[1])
            if intx is None or inty is None:
                continue
            n_intersections += 1
            if not is_in_test_area(test_area, intx, inty):
                continue
            n_inter_in_test += 1
            p1, v1 = pv1
            p2, v2 = pv2
            if not is_in_future(p1, v1, intx, inty) or is_in_future(p2, v2, intx, inty):
                continue
            total += 1
    # intersections = list(
    #     intersect(pv1[0], pv2[0], pv1[1], pv2[1])
    #     for pv1, pv2 in permutations(vecs, r=2)
    # )
    # total = sum(
    #     1 if is_in_test_area(TEST_AREA_1, *intr) else 0 for intr in intersections
    # )
    # TODO: filter by `is_in_future(pos, vel, x, y)`
    print(
        f"total intersections {n_intersections} in test area {test_area} {n_inter_in_test} in the future = {total}"
    )


if __name__ == "__main__":
    main(input_file, TEST_AREA_2)
