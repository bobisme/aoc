#!/usr/bin/env python

import math
from typing import Optional
import vector
from itertools import permutations, pairwise, combinations
from functools import cache
from rich.progress import Progress
from numba import njit
import numpy as np
import sympy as sp
from z3 import sat, Int, Optimize, IntVector
from scipy.optimize import minimize
from pprint import pp
import pulp

vector.register_awkward()
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
    # intersections = list(
    #     intersect(pv1[0], pv2[0], pv1[1], pv2[1])
    #     for pv1, pv2 in permutations(vecs, r=2)
    # )
    # total = sum(
    #     1 if is_in_test_area(TEST_AREA_1, *intr) else 0 for intr in intersections
    # )
    # TODO: filter by `is_in_future(pos, vel, x, y)`
    print(
        f"total intersections {n_intersections} in test area {test_area} = {n_inter_in_test}"
    )


#
# Objective function
# def objective(p, v, p1, v1, t1, p2, v2, t2):
def objective(p, v, p1, v1, p2, v2):
    points = closest_points(p, v, p1, v1)
    a = 69696969
    if points[0] is not None and points[1] is not None:
        a = np.sum((points[0] - points[1]) ** 2)
    points = closest_points(p, v, p2, v2)
    b = 69696969
    if points[0] is not None and points[1] is not None:
        b = np.sum((points[0] - points[1]) ** 2)
    # term1 = p - p1 + t1 * (v - v1)
    # term2 = p - p2 + t2 * (v - v2)
    return np.sum(a**2) + np.sum(b**2)


# Gradient of the objective function
def gradient(p, v, p1, v1, t1, p2, v2, t2):
    grad_p = 2 * ((p - p1 + t1 * (v - v1)) + (p - p2 + t2 * (v - v2)))
    grad_v = 2 * (t1 * (p - p1 + t1 * (v - v1)) + t2 * (p - p2 + t2 * (v - v2)))
    return grad_p, grad_v


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
        # print()
        # sp.pprint(t_left)
        # print()
        # sp.pprint(t_right)
        eq_t = sp.Eq(t_left, t_right)
        # print()
        # sp.pprint(eq_t)
        _, d_left = sp.fraction(t_left)
        _, d_right = sp.fraction(t_right)
        eq_t2 = sp.expand(sp.Eq(d_left * d_right * t_left, d_left * d_right * t_right))
        # print()
        # sp.pprint(eq_t2)
        left = V[idxs[0]] * P[idxs[1]] - V[idxs[1]] * P[idxs[0]]
        right = sp.solve(eq_t2, (left,))[0]
        # print()
        # sp.pprint(sp.Eq(left, right))
        return right
        eq_t1 = [(a_x - P_x) * (V_y - u_y), (V_x - u_x) * (a_y - P_y)]
        expanded = [sp.expand(x) for x in eq_t1]
        print(expanded)
        print(sp.simplify(expanded[0] - expanded[1]))
        right = sp.solve(expanded[0] - expanded[1], (left,))[0]
        print(left, "==", sp.factor(right))
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

    def subs_v(vec, eq, vec_idx: int, coord: int):
        return eq.subs(vec[coord], vecs[vec_idx][1][coord])

    substituted = []
    for eq in eq_set:
        for idx in (0, 1, 2):
            for coord in (0, 1, 2):
                eq = eq.subs(sp_poss[idx][coord], vecs[idx][0][coord])
                eq = eq.subs(sp_vecs[idx][coord], vecs[idx][1][coord])
        substituted.append(eq)
    pp(substituted)
    # print(sp.simplify(eq_t1))
    # print(sp.solve((eq1[0] - eq1[1]), (t1,)))
    # print("sympy solve")
    sol = sp.solve(substituted, (V_x, V_y, V_z, P_x, P_y, P_z))
    print(sol)
    answer = sol.get(P_x) + sol.get(P_y) + sol.get(P_z)
    print("pos sum = ", answer)
    return

    # def line_to_line_dist(p1, v1, p2, v2):
    #     perpendicular = np.cross(v1, v2)
    #     perp_norm = np.linalg.norm(perpendicular)
    #     if abs(perp_norm - 0.0) < 0.001:
    #         return 1e100  # parallel
    #     # print(f"{v_cross=}")
    #     out = np.linalg.norm(perpendicular @ (p2 - p1)) / perp_norm
    #     # print(f"{out=}")
    #     return out
    #
    # def point_to_line_d2(point, p, v):
    #     return np.linalg.norm(np.cross(v, p - point)) ** 2 / np.linalg.norm(v) ** 2
    #
    # def objective(args_array):
    #     point = args_array[:3]
    #     vel = args_array[3:6]
    #     t = args_array[6:]
    #     # print(f"{vel=}")
    #     errs = [
    #         ((vecs[i][0] + vecs[i][1] * t[i]) - (point + vel * t[i]))
    #         for i in range(len(t))
    #     ]
    #     # distances = [
    #     #     line_to_line_dist(point, vel, vecs[i][0], vecs[i][1]) ** 2 for i in range(2)
    #     # ]
    #     # pp(distances)
    #     return sum(x.dot(x) for x in errs)
    #
    # init = np.concatenate(vecs[0])
    # ts = np.ones(len(vecs[:]))
    # init = np.concatenate([init, ts])
    # # init = np.concatenate([np.array([0, 0, 0, 1, 1, 1]), np.zeros(len(vecs))])
    # bounds = [(1, None)] * 3 + [(None, None)] * 3 + [(1, None)] * len(ts)
    # pp(bounds)
    # # pp(init[:3])
    # # pp(init[3:6])
    # # pp(init[6:])
    # # - 'Nelder-Mead' (see here): `optimize.minimize-neldermead`
    # # - 'Powell'      (see here): `optimize.minimize-powell`
    # # - 'CG'          (see here): `optimize.minimize-cg`
    # # - 'BFGS'        (see here): `optimize.minimize-bfgs`
    # # - 'Newton-CG'   (see here): `optimize.minimize-newtoncg`
    # # - 'L-BFGS-B'    (see here): `optimize.minimize-lbfgsb`
    # # - 'TNC'         (see here): `optimize.minimize-tnc`
    # # - 'COBYLA'      (see here): `optimize.minimize-cobyla`
    # # - 'SLSQP'       (see here): `optimize.minimize-slsqp`
    # # - 'trust-constr'(see here): `optimize.minimize-trustconstr`
    # # - 'dogleg'      (see here): `optimize.minimize-dogleg`
    # # - 'trust-ncg'   (see here): `optimize.minimize-trustncg`
    # # - 'trust-exact' (see here): `optimize.minimize-trustexact`
    # # - 'trust-krylov' (see here): `optimize.minimize-trustkrylov`
    # out = minimize(
    #     objective,
    #     init,
    #     method="SLSQP",
    #     # bounds=bounds,
    #     options=dict(maxiter=100_000),
    # )
    # if not out.success:
    #     print("FAILED:", out.message)
    #     pp(out)
    # else:
    #     pp(out)
    #     x = out.x.astype("long")
    #     pp(x[:3])
    #     pp(x[3:6])
    #     pp(x[6:])
    #     print("summed =", sum(x[:3]))
    # prob = pulp.LpProblem("IntersectingLines", pulp.LpMinimize)
    # d_x = pulp.LpVariable("d_x", cat="Integer")
    # d_y = pulp.LpVariable("d_y", cat="Integer")
    # d_z = pulp.LpVariable("d_z", cat="Integer")
    # p_x = pulp.LpVariable("p_x", cat="Integer")
    # p_y = pulp.LpVariable("p_y", cat="Integer")
    # p_z = pulp.LpVariable("p_z", cat="Integer")
    # t1 = pulp.LpVariable("t1", cat="Integer")
    # t2 = pulp.LpVariable("t2", cat="Integer")
    # t3 = pulp.LpVariable("t3", cat="Integer")
    # prob += d_x * t1 + p_x == vecs[0][1][0] * t1 + vecs[0][0][0]
    # prob += d_y * t1 + p_y == vecs[0][1][1] * t1 + vecs[0][0][1]
    # prob += d_z * t1 + p_z == vecs[0][1][2] * t1 + vecs[0][0][2]
    #
    # prob += 0
    # print("solving")
    # prob.solve()
    # print("Status:", pulp.LpStatus[prob.status])
    # print(
    #     "Intersection Point: (",
    #     pulp.value(d_x),
    #     ",",
    #     pulp.value(d_y),
    #     ",",
    #     pulp.value(d_z),
    #     ")",
    # )
    # print(
    #     "Direction of Intersecting Line: (",
    #     pulp.value(p_x),
    #     ",",
    #     pulp.value(p_y),
    #     ",",
    #     pulp.value(p_z),
    #     ")",
    # )
    # opt = Optimize()
    # P = IntVector("P", 3)
    # V = IntVector("V", 3)
    # for i in range(0, 4):
    #     t = Int(f"t_{i}")
    #     for j in range(3):
    #         opt.add(t > 0)
    #         opt.add(P[j] + t * V[j] == vecs[i][0][j] + t * vecs[i][1][j])
    # print("model built, checking satisfiability")
    # if not opt.check() == sat:
    #     print("UNSATISFIABLE!")
    #     return
    # print("SATISFIED")
    # model = opt.model()
    # print("pos sum = ", sum(model[P[i]].as_long() for i in range(3)))


if __name__ == "__main__":
    part_2(input_file)
