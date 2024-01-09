#!/usr/bin/env python

import datetime
import math
from typing import Optional
from itertools import combinations
from rich.progress import Progress
from numba import njit
import numpy as np
import sympy as sp
from pprint import pp

sp.init_printing()

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


def parse_2d(input):
    def parse_2d_line(line):
        pos, vel = line.split("@", 1)
        yield from (int(x.strip()) for x in pos.split(", ", 2))
        yield from (int(x.strip()) for x in vel.split(", ", 2))

    def parse_all(input):
        for line in input:
            yield from parse_2d_line(line)

    return np.array(list(parse_all(input))).reshape(-1, 2, 3)[:, :, :2]


def parse_3d(input):
    def parse_line(line):
        pos, vel = line.split("@", 1)
        yield from (int(x.strip()) for x in pos.split(", ", 2))
        yield from (int(x.strip()) for x in vel.split(", ", 2))

    def parse_all(input):
        for line in input:
            yield from parse_line(line)

    return np.array(list(parse_all(input)), dtype=np.int64).reshape(-1, 2, 3)


def get_pos_at_time(pos, vel, t):
    return pos + t * vel


def intersect_2d_rays(P1, P2, V1, V2):
    # print(f"checking intersection of {P1}->{V1} and {P2}->{V2}")
    # Assuming P1, P2, D1, D2 are given as numpy arrays
    # P1, P2 are the starting points and D1, D2 are the direction vectors of the rays

    # Solve the system of equations: P1 + t * D1 = P2 + s * D2
    A = np.array([V1, -V2]).T
    B = P2 - P1

    # Check if the matrix A is singular (parallel lines)
    if np.linalg.det(A) == 0:
        # print("✗✗✗: determinant is 0, lines are parallel")
        return None, None

    # Solve for t and s
    t, s = np.linalg.solve(A, B)

    # Check if the intersection is on both rays (t and s should be non-negative)
    if t >= 0 and s >= 0:
        # print("---: t and s >= 0, in the future", t, s)
        return P1 + t * V1
        # return t, s
    else:
        # print("✗✗✗: t or s < 0, not in the future", t, s)
        return None, None


def closest_points(P1, V1, P2, V2):
    E = P2 - P1
    Vcross = np.cross(V1, V2)
    denom = Vcross @ Vcross
    # Solve for t and s
    t = (np.cross(E, V2) @ Vcross) / denom
    s = (np.cross(E, V1) @ Vcross) / denom

    # Check if the intersection is on both rays (t and s should be non-negative)
    if t >= 0 and s >= 0:
        # print("---: t and s >= 0, in the future", t, s)
        return P1 + t * V1, P2 + s * V2
        # return t, s
    else:
        # print("✗✗✗: t or s < 0, not in the future", t, s)
        return None, None


def points_same(P1, P2) -> bool:
    if P1 is None or P2 is None:
        return False
    return (abs(P1 - P2) < 0.0001).all()


@njit
def is_in_test_area(
    area: tuple[int, int], x: Optional[float], y: Optional[float]
) -> bool:
    if x is None or y is None:
        return False
    lower, upper = area
    return lower <= x <= upper and lower <= y <= upper


def part_1(input, test_area):
    for line in input:
        print(line)
    # vecs = vector.awk([parse_2(line) for line in input])
    vecs = parse_2d(input)
    # print(vecs.type)
    # return
    n_intersections = 0
    n_inter_in_test = 0
    with Progress() as progress:
        n = len(vecs)
        k = 2
        ops = math.factorial(n) // (math.factorial(k) * math.factorial(len(vecs) - 2))
        # ops = math.factorial(n) // math.factorial(n - k)
        print("operations:", ops)
        task = progress.add_task("operating", total=ops)
        # for pv1, pv2 in permutations(vecs, r=2):
        for pv1, pv2 in combinations(vecs, r=2):
            progress.update(task, advance=1)
            intx, inty = intersect_2d_rays(pv1[0], pv2[0], pv1[1], pv2[1])
            if intx is None or inty is None:
                continue
            n_intersections += 1
            if not is_in_test_area(test_area, intx, inty):
                # print("✗✗✗: not in test area", intx, inty)
                continue
            # print("✓✓✓: within test area", intx, inty)
            n_inter_in_test += 1
    print(
        f"total intersections {n_intersections} in test area {test_area} = {n_inter_in_test}"
    )


def part_2(input):
    vecs = parse_3d(input)

    t1, t2, t3, s = sp.symbols("t1 t2 t3 s")
    V_x, V_y, V_z, P_x, P_y, P_z = sp.symbols("Vx Vy Vz Px Py Pz")
    p0_x, p0_y, p0_z = sp.symbols("p0x p0y p0z")
    p1_x, p1_y, p1_z = sp.symbols("p1x p1y p1z")
    p2_x, p2_y, p2_z = sp.symbols("p2x p2y p2z")
    v0_x, v0_y, v0_z = sp.symbols("v0x v0y v0z")
    v1_x, v1_y, v1_z = sp.symbols("v1x v1y v1z")
    v2_x, v2_y, v2_z = sp.symbols("v2x v2y v2z")
    p0 = sp.Matrix([p0_x, p0_y, p0_z])
    p1 = sp.Matrix([p1_x, p1_y, p1_z])
    p2 = sp.Matrix([p2_x, p2_y, p2_z])
    v0 = sp.Matrix([v0_x, v0_y, v0_z])
    v1 = sp.Matrix([v1_x, v1_y, v1_z])
    v2 = sp.Matrix([v2_x, v2_y, v2_z])

    # Define the unknown vectors for the intersecting line
    V = sp.Matrix([V_x, V_y, V_z])
    P = sp.Matrix([P_x, P_y, P_z])
    eq1 = P + V * t1 - (p0 + v0 * t1)
    eq2 = P + V * t2 - (p1 + v1 * t2)
    eq3 = P + V * t3 - (p2 + v2 * t3)
    equations = [x for eq in (eq1, eq2, eq3) for x in eq]
    pp(equations)

    def get_right_side(eq, t, idxs: tuple[int, int]):
        t_left = sp.solve(eq[idxs[0]], t)[0]
        t_right = sp.solve(eq[idxs[1]], t)[0]
        eq_t = sp.Eq(t_left, t_right)
        _, d_left = sp.fraction(t_left)
        _, d_right = sp.fraction(t_right)
        eq_t2 = sp.expand(sp.Eq(d_left * d_right * t_left, d_left * d_right * t_right))
        left = V[idxs[0]] * P[idxs[1]] - V[idxs[1]] * P[idxs[0]]
        right = sp.solve(eq_t2, (left,))[0]
        return right

    eq_set = []
    for pair in ((0, 1), (1, 2), (0, 2)):
        left = get_right_side(eq1, t1, pair)
        right = get_right_side(eq2, t2, pair)
        eq_set.append(left - right)
        right = get_right_side(eq3, t3, pair)
        eq_set.append(left - right)
    pp(eq_set)

    sp_vecs = [v0, v1, v2]
    sp_poss = [p0, p1, p2]

    substituted = []
    for eq in eq_set:
        for idx in (0, 1, 2):
            for coord in (0, 1, 2):
                eq = eq.subs(sp_poss[idx][coord], vecs[idx][0][coord])
                eq = eq.subs(sp_vecs[idx][coord], vecs[idx][1][coord])
        substituted.append(eq)
    # pp(substituted)
    sol = sp.solve(substituted, (V_x, V_y, V_z, P_x, P_y, P_z))
    print(sol)
    answer = sol.get(P_x) + sol.get(P_y) + sol.get(P_z)
    print("pos sum = ", answer)
    return


if __name__ == "__main__":
    start = datetime.datetime.now()
    part_2(input_file)
    print(f"part 2 finished in {datetime.datetime.now() - start}")
