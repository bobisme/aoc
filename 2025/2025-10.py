#!/usr/bin/env python

import math
from dataclasses import dataclass
from functools import wraps
from itertools import chain, combinations, islice, product
import os
from typing import Callable, Generator, Iterable, LiteralString
import timeit
from textwrap import indent

DEBUG = bool(os.getenv("DEBUG", False))
EPSILON = 1e-9

EXPECTED = [
    (118, [11, 25, 0, 6, 19, 20, 4, 21, 2, 6, 4]),
    (83, [8, 10, 19, 13, 1, 2, 23, 7]),
    (144, [8, 22, 0, 18, 0, 14, 16, 15, 27, 15, 0, 6, 3]),
    (59, [7, 3, 5, 0, 11, 9, 4, 11, 8, 1]),
    (173, [16, 109, 11, 3, 7, 10, 17]),
    (239, [159, 16, 3, 5, 3, 15, 16, 1, 19, 1, 1]),
    (59, [16, 0, 27, 1, 15, 0]),
    (175, [2, 5, 147, 0, 21, 0]),
    (128, [23, 12, 17, 35, 3, 0, 23, 0, 10, 5]),
    (54, [14, 0, 2, 19, 0, 7, 8, 4]),
    (225, [19, 22, 0, 8, 2, 165, 2, 7]),
    (170, [148, 10, 12]),
    (227, [19, 130, 1, 19, 6, 12, 0, 12, 17, 11]),
    (267, [0, 7, 18, 9, 5, 200, 9, 19]),
    (130, [10, 13, 4, 32, 1, 29, 13, 8, 20]),
    (54, [7, 6, 13, 17, 11]),
    (40, [20, 7, 12, 1]),
    (266, [15, 159, 7, 18, 19, 19, 13, 16]),
    (289, [18, 16, 13, 8, 0, 18, 15, 20, 181]),
    (74, [16, 19, 0, 19, 0, 3, 10, 7]),
    (58, [9, 3, 19, 13, 11, 3]),
    (77, [1, 1, 1, 2, 19, 5, 14, 20, 9, 5]),
    (224, [15, 178, 3, 0, 19, 3, 6]),
    (65, [16, 6, 4, 1, 18, 3, 7, 10]),
    (69, [17, 14, 15, 9, 14]),
    (216, [14, 172, 6, 9, 15]),
    (176, [0, 23, 131, 19, 3]),
    (227, [20, 7, 0, 7, 11, 150, 3, 1, 4, 18, 2, 3, 1]),
    (46, [8, 19, 5, 14]),
    (164, [0, 8, 4, 10, 11, 7, 124]),
    (156, [10, 21, 17, 15, 0, 0, 0, 1, 90, 2]),
    (203, [16, 9, 13, 165]),
    (162, [84, 1, 34, 24, 10, 9, 0]),
    (167, [15, 20, 118, 14]),
    (50, [0, 7, 16, 2, 6, 3, 16]),
    (181, [1, 13, 20, 14, 129, 4]),
    (74, [20, 14, 17, 10, 10, 3]),
    (64, [9, 8, 15, 3, 0, 14, 15]),
    (79, [12, 20, 3, 13, 2, 9, 12, 8]),
    (71, [21, 1, 3, 0, 24, 11, 11]),
    (12, [7, 2, 2, 1]),
    (33, [8, 9, 7, 9]),
    (79, [15, 13, 9, 9, 7, 20, 6]),
    (58, [0, 13, 13, 4, 7, 0, 14, 2, 5]),
    (157, [19, 24, 21, 0, 8, 1, 5, 2, 23, 29, 7, 18]),
    (326, [3, 26, 25, 20, 140, 0, 9, 0, 9, 35, 18, 20, 21]),
    (59, [11, 11, 1, 7, 14, 15]),
    (164, [112, 14, 2, 15, 10, 11]),
    (207, [12, 0, 1, 114, 3, 2, 19, 3, 53]),
    (96, [7, 7, 12, 6, 30, 28, 1, 0, 5]),
    (83, [19, 20, 8, 0, 14, 8, 14]),
    (69, [1, 15, 12, 12, 0, 4, 25]),
    (74, [8, 5, 16, 4, 14, 8, 9, 6, 4]),
    (122, [0, 14, 9, 6, 13, 7, 18, 9, 28, 10, 8]),
    (27, [9, 1, 1, 16, 0]),
    (79, [23, 1, 2, 0, 5, 15, 5, 16, 12, 0]),
    (332, [186, 16, 20, 23, 10, 16, 7, 0, 16, 0, 21, 17, 0]),
    (94, [19, 9, 13, 12, 0, 18, 5, 18]),
    (102, [0, 12, 8, 10, 4, 0, 14, 12, 10, 0, 4, 28]),
    (100, [0, 20, 11, 1, 3, 7, 2, 10, 31, 15]),
    (54, [6, 2, 19, 0, 5, 12, 10]),
    (32, [10, 0, 17, 5]),
    (72, [2, 8, 17, 14, 1, 0, 17, 13]),
    (41, [0, 17, 16, 0, 8, 0]),
    (165, [3, 2, 6, 12, 3, 116, 7, 16]),
    (78, [20, 19, 20, 3, 16]),
    (74, [14, 20, 3, 1, 16, 20]),
    (242, [31, 0, 166, 3, 18, 24]),
    (270, [19, 19, 16, 8, 183, 13, 12]),
    (50, [9, 0, 11, 16, 8, 6]),
    (227, [12, 19, 10, 12, 174]),
    (27, [8, 11, 8]),
    (72, [20, 5, 8, 2, 17, 20]),
    (55, [19, 4, 10, 3, 4, 0, 5, 5, 0, 5]),
    (167, [9, 13, 15, 11, 100, 2, 17]),
    (80, [24, 16, 10, 11, 0, 19]),
    (56, [5, 2, 2, 3, 8, 2, 15, 12, 4, 2, 1]),
    (52, [7, 5, 10, 12, 18, 0]),
    (267, [7, 17, 9, 20, 14, 0, 10, 190]),
    (213, [10, 173, 10, 0, 8, 9, 3]),
    (60, [3, 0, 2, 2, 10, 20, 5, 9, 9]),
    (47, [8, 32, 3, 0, 4]),
    (28, [1, 9, 0, 8, 0, 10]),
    (177, [1, 0, 114, 0, 28, 17, 17]),
    (212, [0, 1, 16, 157, 14, 24]),
    (115, [8, 16, 1, 13, 18, 3, 0, 9, 19, 17, 11]),
    (71, [0, 6, 1, 0, 15, 0, 15, 6, 6, 3, 19]),
    (47, [19, 13, 15]),
    (104, [19, 30, 15, 1, 2, 34, 3, 0]),
    (61, [4, 13, 18, 0, 11, 7, 8]),
    (150, [8, 142]),
    (41, [5, 14, 1, 0, 21]),
    (181, [7, 4, 131, 3, 0, 18, 6, 0, 12]),
    (214, [5, 19, 0, 157, 13, 0, 20]),
    (83, [4, 0, 12, 11, 7, 13, 0, 6, 8, 11, 11]),
    (201, [4, 6, 20, 0, 1, 10, 6, 151, 3]),
    (186, [15, 19, 7, 105, 12, 1, 14, 13]),
    (2, [0, 2]),
    (46, [15, 13, 18]),
    (66, [6, 3, 1, 14, 4, 6, 5, 1, 11, 15]),
    (18, [1, 5, 4, 8]),
    (139, [0, 4, 112, 12, 10, 1]),
    (214, [20, 13, 17, 10, 129, 7, 18]),
    (80, [1, 14, 5, 0, 6, 7, 12, 18, 17]),
    (194, [188, 6]),
    (73, [16, 8, 0, 11, 0, 6, 2, 0, 10, 8, 12]),
    (50, [17, 1, 20, 12]),
    (77, [6, 1, 25, 11, 3, 31]),
    (85, [15, 11, 15, 6, 11, 5, 16, 6]),
    (76, [18, 4, 20, 16, 0, 1, 9, 8]),
    (175, [19, 121, 3, 11, 0, 15, 6]),
    (101, [22, 20, 2, 2, 25, 3, 0, 11, 16]),
    (52, [0, 11, 15, 10, 0, 16]),
    (46, [3, 14, 12, 3, 14]),
    (186, [137, 12, 8, 19, 10, 0, 0]),
    (66, [16, 2, 10, 0, 22, 0, 13, 1, 2]),
    (94, [0, 19, 12, 8, 13, 10, 10, 5, 17]),
    (18, [8, 8, 2]),
    (137, [16, 121]),
    (70, [14, 2, 12, 20, 0, 2, 19, 1]),
    (94, [20, 7, 20, 7, 14, 12, 4, 10]),
    (222, [13, 153, 7, 0, 9, 0, 19, 21]),
    (61, [2, 13, 4, 8, 16, 16, 2, 0]),
    (59, [20, 1, 20, 18]),
    (44, [20, 14, 10]),
    (221, [13, 4, 190, 0, 2, 7, 5]),
    (81, [10, 15, 16, 12, 2, 15, 0, 3, 6, 2]),
    (71, [14, 11, 20, 5, 15, 6]),
    (90, [11, 8, 0, 13, 18, 23, 9, 8]),
    (69, [6, 7, 8, 9, 2, 4, 6, 10, 17]),
    (50, [9, 0, 1, 0, 20, 14, 6]),
    (60, [14, 7, 1, 0, 14, 18, 0, 6]),
    (96, [0, 5, 13, 14, 10, 16, 1, 8, 7, 4, 10, 8]),
    (68, [17, 19, 7, 13, 12]),
    (80, [15, 5, 15, 0, 13, 0, 5, 10, 12, 5]),
    (16, [11, 4, 1]),
    (55, [17, 11, 12, 15]),
    (124, [0, 0, 7, 3, 21, 17, 7, 20, 25, 24]),
    (69, [19, 9, 14, 0, 6, 19, 2]),
    (102, [19, 9, 23, 12, 0, 11, 0, 7, 1, 20]),
    (58, [8, 9, 4, 8, 19, 10]),
    (10, [7, 1, 2]),
    (62, [16, 9, 9, 17, 11]),
    (111, [17, 3, 12, 8, 18, 10, 18, 13, 12]),
    (50, [8, 4, 27, 0, 11]),
    (20, [5, 8, 4, 3]),
    (215, [186, 16, 13]),
    (68, [8, 15, 5, 19, 14, 7]),
    (54, [15, 9, 20, 10]),
    (30, [9, 0, 0, 4, 0, 9, 8]),
    (110, [7, 2, 17, 18, 22, 4, 1, 0, 13, 2, 13, 11]),
    (47, [17, 18, 9, 3, 0]),
    (38, [10, 14, 14]),
    (8, [6, 2]),
    (45, [10, 13, 7, 3, 12]),
    (41, [14, 3, 20, 0, 4]),
    (188, [172, 3, 13, 0]),
    (73, [14, 20, 9, 18, 7, 1, 4]),
    (79, [14, 15, 25, 6, 19, 0]),
    (224, [0, 17, 1, 14, 144, 25, 12, 11, 0]),
    (29, [4, 7, 5, 13]),
    (135, [35, 10, 3, 12, 19, 18, 0, 9, 10, 6, 1, 12]),
    (108, [7, 20, 9, 13, 6, 19, 16, 0, 12, 6]),
    (42, [7, 12, 1, 19, 3]),
    (108, [4, 0, 9, 18, 10, 0, 12, 16, 18, 21, 0, 0, 0]),
    (130, [14, 14, 17, 19, 10, 8, 19, 9, 20]),
    (44, [19, 2, 5, 0, 18]),
    (29, [10, 0, 14, 3, 2]),
    (228, [14, 0, 19, 16, 139, 4, 16, 19, 1]),
    (21, [13, 8, 0]),
    (192, [1, 156, 7, 17, 8, 3]),
    (5, [1, 0, 4]),
    (228, [18, 0, 11, 13, 15, 0, 9, 162]),
    (175, [17, 17, 16, 3, 20, 102]),
    (110, [20, 0, 11, 0, 19, 12, 18, 0, 30]),
]

Num = Fraction
ZERO: Num = Fraction(0)
ONE: Num = Fraction(1)
BIG: Num = Fraction(10**10)


def bench(count=1):
    def wrapper(fn):
        @wraps(fn)
        def inner(*args, **kwargs):
            t = timeit.timeit(lambda: fn(*args, **kwargs), number=count) / count * 1_000
            print(f"{fn.__name__} bench: {t:.1f}ms")
            return fn(*args, **kwargs)

        return inner

    return wrapper


Input = list[str] | list[LiteralString]


def _btoa(b: int, len: int) -> str:
    fmtstr = f"{{:0{len}b}}"
    return fmtstr.format(b)


def is_whole(n: float) -> bool:
    return math.isclose(n, round(n), abs_tol=1e-6)


def fmtn(n: float) -> str:
    return str(round(n)) if is_whole(n) else f"{n:.1f}"


def is_zero(n: float) -> bool:
    return math.isclose(n, 0, abs_tol=1e-6)


@dataclass
class Machine:
    target_diagram: tuple[bool, ...]
    buttons: tuple[tuple[int, ...], ...]
    joltage: tuple[int, ...]

    def __post_init__(self):
        # self.light_diagram = [False for _ in self.target_diagram]
        self.target_int = 0
        for x in self.target_diagram:
            self.target_int = (self.target_int << 1) + int(x)
        self.light_diagram = 0
        self.button_ints = self._get_button_as_ints()
        self.total_joltage = [0 for _ in range(len(self.joltage))]

    def __repr__(self) -> str:
        schematics = (f"({','.join(str(x) for x in s)})" for s in self.buttons)
        return f"[{''.join('#' if x == '1' else '.' for x in _btoa(self.light_diagram, len(self.target_diagram)))}] {' '.join(schematics)} {{{','.join(str(x) for x in self.joltage)}}}"

    def clone(self) -> "Machine":
        m = Machine(self.target_diagram, self.buttons, self.joltage)
        m.light_diagram = self.light_diagram
        return m

    def _get_button_as_ints(self) -> list[int]:
        out = []
        max_ = len(self.target_diagram) - 1
        for b in self.buttons:
            b_out = 0
            for n in b:
                b_out += 1 << (max_ - n)
            out.append(b_out)
        return out

    def press(self, i: int):
        self.light_diagram ^= self.button_ints[i]
        for x in self.buttons[i]:
            self.total_joltage[x] += 1
        return self.light_diagram


CONTROL_1: Input = (
    """\
[.##.] (3) (1,3) (2) (2,3) (0,2) (0,1) {3,5,4,7}
[...#.] (0,2,3,4) (2,3) (0,4) (0,1,2) (1,2,3,4) {7,5,12,7,2}
[.###.#] (0,1,2,3,4) (0,3,4) (0,1,2,4,5) (1,2) {10,11,11,5,10,5}
""".splitlines()
)


@dataclass
class Range:
    start: int = 0
    stop: int = int(10e6)

    def __and__(self, other: "Range") -> "Range":
        return Range(max(self.start, other.start), min(self.stop, other.stop))

    def __iter__(self):
        return chain(range(self.start, self.stop))


with open("2025-10.input") as f:
    input_file = [line.rstrip("\n") for line in f.readlines()]


def parse(input: Input) -> list[Machine]:
    machines = []
    for line in input:
        parts = line.split(" ")
        diagram = tuple(x == "#" for x in parts[0].strip("[]"))
        schematics = tuple(
            tuple(map(int, part.strip("()").split(","))) for part in parts[1:-1]
        )
        joltage = tuple(map(int, parts[-1].strip("{}").split(",")))
        machines.append(Machine(diagram, schematics, joltage))
    return machines


@bench()
def part_1(input: Input):
    machines = parse(input)

    def search(machine: Machine) -> int:
        for count in range(1, len(machine.button_ints) + 1):
            for buttons in combinations(range(len(machine.button_ints)), count):
                m = machine.clone()
                for b in buttons:
                    m.press(b)
                if m.light_diagram == m.target_int:
                    return count
        assert False

    return sum(search(m) for m in machines)


def argmax(
    nums: Iterable[float],
    # apply function if given
    fn: Callable[[float], float] = lambda x: x,
    # only return in bounds if given
    bounds: range | None = None,
) -> int:
    "Return index of max value."
    max_ = (0, 0.0)
    if bounds is not None:
        max_ = max(
            islice(enumerate(nums), bounds.start, bounds.stop, bounds.step),
            key=lambda x: fn(x[1]),
        )
    else:
        max_ = max(enumerate(nums), key=lambda x: fn(x[1]))
    return max_[0]


def fmt_addsub(n: float, var: str | None = None) -> str:
    if var is None:
        return f"{'+' if n > 0 else '-'} {fmtn(abs(n))}"
    return f"{'+' if n > 0 else '-'} {fmtn(abs(n)) if abs(n) != 1 else ''}{var}"


def fmtsum(nums: list[float], vars: list[str]) -> str:
    assert len(nums) == len(vars)
    nonzero = list(x for x in zip(nums, vars) if not is_zero(x[0]))
    if not nonzero:
        return ""
    (x, label) = nonzero[0]
    out = f"{'-' if x < 0 else ''}{fmtn(abs(x)) if abs(x) != 1 else ''}{label}"
    for x, label in nonzero[1:]:
        out += f" {fmt_addsub(x, var=label)}"
    return out


@dataclass
class Ineq:
    const: float
    coeffs: tuple[float, ...]
    vars: tuple[int, ...]
    ineq: str = "<="

    def __post_init__(self):
        if (
            all(x <= 0 for x in self.coeffs)
            and sum(1 for c in self.coeffs if not is_zero(c)) == 1
        ):
            # if self.const < 0:
            self.invert()
        if self.ineq == ">=" and self.const < 0:
            self.const = 0

    def __repr__(self) -> str:
        if any(not is_zero(x) for x in self.coeffs):
            out = fmtsum(list(self.coeffs), [f"x_{v+1}" for v in self.vars])
            out += f"{self.ineq} {fmtn(self.const)}"
            return out
        return f"{0} == {0}".format(fmtn(self.const))

    def invert(self):
        self.coeffs = tuple(-x for x in self.coeffs)
        self.const *= -1
        self.ineq = ">=" if self.ineq == "<=" else "<="

    def is_equality(self) -> bool:
        return all(is_zero(x) for x in self.coeffs)

    def get_range(self, *args: float, coeff_i=-1, max_=int(1e6)) -> Range:
        # If iterating on x1, x2, and x3, we fix x1 and x2, passing them in
        # here as args, returning the feasible range for x3.
        assert len(args) == len(self.coeffs) - 1

        if coeff_i < 0:
            coeff_i += len(self.coeffs)

        rhs = self.const
        target_coeff = self.coeffs[coeff_i]
        if is_zero(target_coeff):
            return Range(0, 0)

        arg_idx = 0
        for i, coeff in enumerate(self.coeffs):
            if i == coeff_i:
                continue
            rhs -= coeff * args[arg_idx]
            arg_idx += 1
        val = rhs / target_coeff
        is_upper_bound = self.ineq == "<="

        if target_coeff < 0:
            is_upper_bound = not is_upper_bound

        if is_upper_bound:
            limit = math.floor(val + EPSILON)
            return Range(stop=min(limit, max_) + 1)
        limit = math.ceil(val - EPSILON)
        return Range(start=limit)


@dataclass
class Fn:
    const: float
    coeffs: tuple[float, ...]
    free: tuple[int, ...]
    lhs: str | None = None

    def __post_init__(self):
        assert len(self.free) == len(self.coeffs)

    def __repr__(self) -> str:
        out = f"{self.lhs} = " if self.lhs else ""
        if any(not is_zero(x) for x in self.coeffs):
            out += fmtsum(list(self.coeffs), [f"x_{i+1}" for i in self.free])
            if not is_zero(self.const):
                out += f" {fmt_addsub(self.const)}"
            return out
        return out + fmtn(self.const)

    def __call__(self, vars: list[float]) -> float:
        right = sum(self.coeffs[i] * v for i, v in enumerate(vars))
        return self.const + right

    def bounds(self) -> str:
        cs = [(i, c) for (i, c) in enumerate(self.coeffs) if not is_zero(c)]
        if not cs:
            return f"{self.const} == {self.const}"

        s = f"{-self.const} <="

        def sign(x) -> str:
            return "+" if x < 0 else "-"

        s += f" {-cs[0][1] if cs[0][1] >= 1 else ''}x_{cs[0][0]+1}"
        for i, c in cs[1:]:
            s += f" {sign(c)} {abs(c) if abs(c) > 1 else ''}x_{i+1}"
        return s


# @dataclass
# class SumFn:
#     const: float
#     coeffs: tuple[float, ...]
#     free: tuple[int, ...]
#
#     def __repr__(self) -> str:
#         s = f"Σ = {self.const}"
#         for i, c in enumerate(self.coeffs):
#             if c == 0.0:
#                 continue
#             s += f" {'+' if c > 0 else '-'} {abs(c) if abs(c) > 1 else ''}x_{i+1}"
#         return s
#
#     def __call__(self, free_vars: list[float]) -> float:
#         total = self.const
#         for i, v in enumerate(free_vars):
#             total += self.coeffs[self.free[i]] * v
#         return total


@dataclass
class Matrix:
    data: list[list[float]]
    col_headers: list[str] | None = None

    def __post_init__(self):
        self.n_rows = len(self.data)
        self.n_cols = len(self.data[0])

    def __repr__(self) -> str:
        out = ""
        if self.col_headers:
            out += "\t".join(self.col_headers) + "\n"
        out += "\n".join(("\t".join(map(fmtn, row)) for row in self.data))
        return out

    def latex(self) -> str:
        if self.n_rows == 0 or self.n_cols == 0:
            return r"\begin{bmatrix}\end{bmatrix}"
        rows = []
        for row in self.data:
            cells = map(fmtn, row)
            rows.append(" & ".join(cells))
        body = " \\\\\n".join(rows)
        return f"\\begin{{bmatrix}}\n{body}\n\\end{{bmatrix}}"

    def __getitem__(self, row_col: tuple[int, int]) -> float:
        row, col = row_col
        return self.data[row][col]

    def __setitem__(self, row_col: tuple[int, int], val: float):
        row, col = row_col
        self.data[row][col] = val

    @staticmethod
    def zeroes(n_rows: int, n_cols: int) -> "Matrix":
        return Matrix([[0.0 for _ in range(n_cols)] for _ in range(n_rows)])

    @staticmethod
    def from_machine(machine: Machine) -> "Matrix":
        data = []
        for row_i, joltage in enumerate(machine.joltage):
            row = []
            for b in machine.buttons:
                row.append(1.0 if row_i in b else 0.0)
            row.append(joltage)
            data.append(row)
        col_headers = [f"x_{i+1}" for i in range(len(machine.buttons))]
        col_headers.append("J")
        return Matrix(data, col_headers=col_headers)

    def log(self, msg):
        if not DEBUG:
            return
        print(msg)
        print(self)

    def transpose(self) -> "Matrix":
        out = Matrix.zeroes(n_rows=self.n_cols, n_cols=self.n_rows)
        for i, row in enumerate(self.data):
            for j, col in enumerate(row):
                out[j, i] = col
        return out

    def col(self, i: int) -> tuple[float, ...]:
        return tuple(row[i] for row in self.data)

    def swap_rows(self, r1: int, r2: int):
        self.data[r1], self.data[r2] = self.data[r2], self.data[r1]

    def g_eliminate(self):
        "Gaussian elimination."
        pivot_row = 0
        pivot_col = 0

        for pivot_col in range(self.n_cols):
            if pivot_row >= self.n_rows:
                break
            # find row with largest abs val in the pivot column
            max_row_i = argmax(
                self.col(pivot_col), fn=abs, bounds=range(pivot_row, self.n_rows)
            )
            if is_zero(self[max_row_i, pivot_col]):
                # if it's zero, the whole column is zeroed; go to the next col
                continue
            self.log(f"find pivot in row {max_row_i} for column {pivot_col}")
            if pivot_row != max_row_i:
                # swap rows so row with max abs col is the pivot row
                self.swap_rows(pivot_row, max_row_i)
                self.log(f"swap rows {pivot_row} and {max_row_i}")
            for row_i in range(pivot_row + 1, self.n_rows):
                # for every row below the pivot...
                # calculate a fraction multiple to apply to other rows
                frac = self[row_i, pivot_col] / self[pivot_row, pivot_col]
                # the rest of the lower part of the col will become 0
                self[row_i, pivot_col] = 0.0
                # subtract the fraction from the rest of the current row
                for col_i in range(pivot_col + 1, self.n_cols):
                    self[row_i, col_i] -= self[pivot_row, col_i] * frac
            self.log(f"eliminate column {pivot_col}")
            pivot_row += 1

    def gj_eliminate(self):
        "Gauss-Jordan elimination."
        pivot_row = 0
        for pivot_col in range(self.n_cols - 1):
            if pivot_row >= self.n_rows:
                continue
            if is_zero(self[pivot_row, pivot_col]):
                continue
            # for col_i in range(self.n_cols - 1, pivot_col, -1):
            for col_i in reversed(range(pivot_col, self.n_cols)):
                self[pivot_row, col_i] /= self[pivot_row, pivot_col]
            for row_i in range(pivot_row):
                for col_i in reversed(range(pivot_col, self.n_cols)):
                    self[row_i, col_i] -= (
                        self[row_i, pivot_col] * self[pivot_row, col_i]
                    )
            self.log(f"eliminate row {pivot_row} column {pivot_col}")
            pivot_row += 1

    def bareiss(self):
        prev = 1
        row = 0

        for col in range(self.n_cols):
            # print(f"bar {row=} {col=}")
            # find pivot
            pivot_row = None
            for r in range(row, self.n_rows):
                if not is_zero(self[r, col]):
                    pivot_row = r
                    break
            if pivot_row is None:
                continue

            if pivot_row != row:
                self.swap_rows(pivot_row, row)

            pivot = self[row, col]

            for i in range(row + 1, self.n_rows):
                aik = self[i, col]
                if is_zero(aik):
                    continue
                for j in range(col + 1, self.n_cols):
                    self[i, j] = (pivot * self[i, j] - aik * self[row, j]) // prev
                self[i, col] = 0

            # NOTE: this feels like it's missing, not sure though
            if pivot < 0:
                for j in range(col, self.n_cols):
                    self[row, j] *= -1

            prev = pivot
            row += 1
            if row == self.n_rows:
                break
            # print(self)

    def eliminate(self):
        self.g_eliminate()
        self.gj_eliminate()

    def back_substitute(self) -> list[float] | None:
        # if self.n_rows != (self.n_cols - 1):
        #     return None
        solution = [0.0 for _ in range(self.n_cols)]
        end = min(self.n_rows, self.n_cols)
        for row_i in reversed(range(end)):
            sum = 0.0
            for row_j in reversed(range(row_i, self.n_rows)):
                sum += solution[row_j] * self[row_i, row_j]
            div = self[row_i, row_i]
            if not is_zero(div):
                solution[row_i] = (self[row_i, self.n_cols - 1] - sum) / div
            else:
                solution[row_i] = 0
        return solution

    def sort(self, skip_last_row=False):
        if skip_last_row:
            self.data.sort(
                key=lambda row: [
                    abs(x) if i < (self.n_rows - 1) else -100000000
                    for (i, x) in enumerate(row)
                ],
                reverse=True,
            )
        else:
            self.data.sort(key=lambda row: [abs(x) for x in row], reverse=True)


@dataclass
class Funcs:
    fns: dict[int, Fn]
    free: tuple[int, ...]
    var_count: int

    @staticmethod
    def from_matrix(matrix: Matrix) -> "Funcs":
        row_i = 0
        col_i = 0
        fns = {}
        free = tuple(
            i
            for i in range(matrix.n_cols - 1)
            if sum(1 for x in matrix.col(i) if not is_zero(x)) > 1
        )
        while row_i < matrix.n_rows:
            if col_i > matrix.n_cols - 2:
                break
            if is_zero(matrix[row_i, col_i]):
                col_i += 1
                continue
            coefficients = []
            for _ in range(col_i + 1):
                coefficients.append(0)
            for col_j in range(col_i + 1, matrix.n_cols - 1):
                coefficients.append(matrix[row_i, col_j])
            fns[col_i] = Fn(
                matrix[row_i, matrix.n_cols - 1],
                tuple(-c for i, c in enumerate(coefficients) if i in free),
                free,
                lhs=f"x_{col_i+1}",
            )
            row_i += 1
            col_i += 1
        return Funcs(
            fns,
            tuple(i for i in range(matrix.n_cols - 1) if i not in fns),
            var_count=matrix.n_cols - 1,
        )

    def __post_init__(self):
        self._bounds = list(self.gen_bounds())

    def __repr__(self) -> str:
        s = "Dependent vars:\n" + indent(
            "\n".join(repr(fn) for fn in self.fns.values()), "  "
        )
        if self.free:
            free_vars = ", ".join(f"x_{i+1}" for i in self.free)
            s += f"\nFree vars: {free_vars}"
        return s

    def eval(self, free_vars: list[float]) -> list[float]:
        assert len(free_vars) == len(self.free)
        vals = [0.0 for _ in range(self.var_count)]
        for i, v in enumerate(free_vars):
            vals[self.free[i]] = v
        for i in reversed(range(self.var_count)):
            if i not in self.fns:
                continue
            fn = self.fns[i]
            vals[i] = fn(free_vars)
        return vals

    def get_sum_fn(self) -> Fn:
        const = sum(fn.const for fn in self.fns.values())
        xs = [1.0 for _ in range(len(self.free))]
        for var_i in range(self.var_count):
            fn = self.fns.get(var_i)
            if fn is None:
                continue
            for ci, _ in enumerate(self.free):
                xs[ci] += fn.coeffs[ci]
        return Fn(const, tuple(xs), self.free, lhs="C")

    def gen_bounds(self) -> Generator[Callable[[list[float]], bool]]:
        for fn in self.fns.values():
            if is_zero(sum(fn.coeffs)):
                continue

            def in_bounds(free_vars: list[float]) -> bool:
                total = 0
                for i, v in enumerate(free_vars):
                    total += fn.coeffs[i] * v
                return total <= fn.const

            yield in_bounds

    def is_in_bounds(self, *free_vars: list[float]) -> bool:
        return all(b(*free_vars) for b in self._bounds)


@dataclass
class System:
    inequalities: list[Ineq]
    goal: Fn
    maximize: bool = False

    def __repr__(self) -> str:
        out = f"Objective: {'max' if self.maximize else 'min'} {self.goal}:\n"
        out += "Constraints:\n"
        out += indent("\n".join(map(repr, self.inequalities)), "  ")
        return out

    @property
    def free(self) -> tuple[int, ...]:
        return self.goal.free

    @staticmethod
    def from_funcs(fns: Funcs) -> "System":
        ineqs = list(
            filter(
                lambda ineq: not ineq.is_equality(),
                (
                    Ineq(
                        const=fn.const,
                        coeffs=tuple(-x for x in fn.coeffs),
                        vars=fns.free,
                    )
                    for fn in fns.fns.values()
                    if any(not is_zero(c) for c in fn.coeffs)
                    # and not (all(c < 0 for c in fn.coeffs) and fn.const >= 0)
                ),
            )
        )
        sum_fn = fns.get_sum_fn()
        return System(ineqs, sum_fn)

    def ranges(self, max_=100000) -> tuple[list[Range], list[Callable[[float], Range]]]:
        rs = []
        zero_args = [0.0] * (len(self.free) - 1)
        for i, _ in enumerate(self.free):
            r = Range(stop=100_000)
            for ineq in self.inequalities:
                if not (
                    not is_zero(ineq.coeffs[i])
                    and all(is_zero(x) for (ci, x) in enumerate(ineq.coeffs) if ci != i)
                ):
                    continue
                r &= ineq.get_range(*zero_args, coeff_i=i)
            if r.stop >= 100_000:
                r.stop = max_ + 1
            rs.append(r)
        fns = []
        for ineq in self.inequalities:
            var_count = sum(1 for c in ineq.coeffs if not is_zero(c))
            if var_count <= 1:
                continue
            fns.append(ineq.get_range)
        return rs, fns

    def answer(self) -> float | None:
        if all(is_zero(x) for x in self.goal.coeffs):
            return self.goal.const


@dataclass
class Tableau:
    MAX_STEPS = 100
    mat: Matrix

    def __post_init__(self):
        self.n_cols = self.mat.n_cols
        self.n_rows = self.mat.n_rows
        self.min = float("inf")

    def __repr__(self) -> str:
        return repr(self.mat)

    @staticmethod
    def from_system(system: System) -> "Tableau":
        n = len(system.free)
        m = len(system.inequalities)

        t_count = 0

        # cols:  [x_1 .. x_n | s_1 .. s_m | RHS]
        mat = Matrix.zeroes(n_rows=m + 1, n_cols=n + m + 1)
        mat.col_headers = (
            [f"x_{i+1}" for i in system.free]
            + [f"s_{i+1}" for i in range(len(system.inequalities))]
            + ["RHS"]
        )

        goal_const = system.goal.const
        goal_coeffs = list(system.goal.coeffs)

        # constraint rows
        for i, ineq in enumerate(system.inequalities):
            if ineq.is_equality():
                continue
            # skip x >= 0
            if sum(1 for c in ineq.coeffs if not is_zero(c)) == 1 and is_zero(
                ineq.const
            ):
                continue
            if ineq.ineq == "<=":
                # ineq.coeffs is length n
                for ci, coef in enumerate(ineq.coeffs):
                    mat[i, ci] = coef
                mat[i, n + i] = 1  # slack s_i
                mat[i, -1] = ineq.const
            else:
                if is_zero(ineq.const):
                    continue
                for ci, coef in enumerate(ineq.coeffs):
                    # mat[i, ci] = coef
                    # goal_coeffs[ci] *= -1
                    mat[i, ci] = coef
                    goal_coeffs[ci] = 1
                    t_count += 1
                    mat.col_headers[ci] = f"t_{t_count}"
                mat[i, n + i] = 1  # slack s_i
                mat[i, -1] = ineq.const - 1
                goal_const += 1

        # objective row (row m)
        for ci, _ in enumerate(system.free):
            mat[m, ci] = goal_coeffs[ci]
        # slacks in obj row = 0
        mat[m, -1] = -goal_const
        return Tableau(mat)

    @property
    def data(self) -> list[list[float]]:
        return self.mat.data

    def __getitem__(self, row_col: tuple[int, int]) -> float:
        return self.mat[row_col]

    def __setitem__(self, row_col: tuple[int, int], val: float):
        self.mat[row_col] = val

    def has_solution(self):
        return all(x >= 0 for x in self.data[-1][:-1])

    def get_pivot_col(self) -> int | None:
        for j, val in enumerate(self.data[-1][:-1]):
            if val < 0:
                return j
        return None

    def get_pivot_row(self, col_i: int) -> int:
        options = [
            (row_i, self.mat[row_i, -1] / self.mat[row_i, col_i])
            for row_i in range(self.mat.n_rows - 1)
            if self.mat[row_i, col_i] > 0
        ]
        if not options:
            raise RuntimeError("unbounded")
        min_ratio = min(options, key=lambda x: x[1])[1]
        return next((i, x) for (i, x) in options if x == min_ratio)[0]

    def step_gauss(self):
        pivot_col = self.get_pivot_col()
        if pivot_col is None:
            raise StopIteration
        pivot_row = self.get_pivot_row(pivot_col)

        # Divide pivot row out
        div = self[pivot_row, pivot_col]
        for col_i in range(self.n_cols):
            self[pivot_row, col_i] /= div

        # Eliminate other rows
        for row_i in (i for i in range(self.n_rows) if i != pivot_row):
            mul = self[row_i, pivot_col]
            for col_i in range(self.n_cols):
                self[row_i, col_i] -= self[pivot_row, col_i] * mul

    def step_bareiss(self):
        pivot_col = self.get_pivot_col()
        if pivot_col is None:
            raise StopIteration
        pivot_row = self.get_pivot_row(pivot_col)

        pivot = self[pivot_row, pivot_col]

        # multiply whole self by the pivot
        for i in range(self.n_rows):
            for j in range(self.n_cols):
                self[i, j] *= pivot

        # eliminate pivot column
        for i in filter(lambda x: x != pivot_row, range(self.n_rows)):
            self[i, pivot_col] = 0

        # normalize using greatest common divisor of the row
        gcd = math.gcd(*(int(x) for x in self.data[pivot_row] if not is_zero(x)))
        if gcd != 0:
            for j in range(self.n_cols):
                self[pivot_row, j] /= gcd

    def back_substitute(self) -> list[float] | None:
        # if self.n_rows != (self.n_cols - 1):
        #     return None
        solution = [0.0 for _ in range(self.n_cols)]
        for row_i in reversed(range(self.n_rows - 1)):
            if all(is_zero(x) for x in self.data[row_i][:-1]):
                continue
            sum = 0.0
            col_i = next(
                i for i in range(self.n_cols - 1) if not is_zero(self[row_i, i])
            )
            for row_j in reversed(range(row_i, self.n_rows - 1)):
                sum += solution[row_j] * self[row_i, row_j]
            div = self[row_i, row_i]
            if not is_zero(div):
                solution[row_i] = (self[row_i, self.n_cols - 1] - sum) / div
            else:
                solution[row_i] = 0
        return solution

    def simplex(self):
        i = 0
        while i < self.MAX_STEPS:
            i += 1
            if self.has_solution():
                return -self[-1, -1]
            try:
                self.step_gauss()
            except StopIteration:
                return self.min
        raise RuntimeError("Exceeded max steps")

        # zero "nonbasic" vars
        for col_i in range(self.n_cols - 1):
            col = self.mat.col(col_i)
            if sum(1 for c in col if not is_zero(c)) == 1:
                continue
            for row_i in range(self.n_rows):
                self[row_i, col_i] = 0

        self.mat.sort(skip_last_row=True)
        print(self.mat)
        return self.back_substitute()


# def branch_and_bound(system: System, prev_val: list[float] = []):
#     print(f"{prev_val=}")
#     tableau = Tableau.from_system(system)
#     out = tableau.simplex()
#     if is_whole(out):
#         return out
#     if out in prev_val:
#         return None
#     print(f"got {out}, which isn't an integer")
#     lower_ineqs = system.inequalities.copy()
#     if prev_val:
#         lower_ineqs.pop()
#     lower_bound = Ineq(
#         math.floor(out), system.goal.coeffs, vars=system.goal.free, ineq="<="
#     )
#     print(f"searching lower bound: {lower_bound}")
#     lower_ineqs.append(lower_bound)
#     lower = System(lower_ineqs, system.goal)
#     try:
#         lower_out = branch_and_bound(lower, prev_val=prev_val + [out])
#         if lower_out is not None:
#             return lower_out
#     except Exception as e:
#         print(f"nothing in the lower bound: {e}")
#     upper_ineqs = system.inequalities.copy()
#     if prev_val:
#         upper_ineqs.pop()
#     upper_bound = Ineq(
#         math.ceil(out), system.goal.coeffs, vars=system.goal.free, ineq=">="
#     )
#     print(f"searching upper bound: {upper_bound}")
#     upper_ineqs.append(upper_bound)
#     upper = System(upper_ineqs, system.goal)
#     try:
#         upper_out = branch_and_bound(upper, prev_val=prev_val + [out])
#         if upper_out is not None:
#             return upper_out
#     except Exception as e:
#         print(f"nothing in the upper bound: {e}")
#     raise RuntimeError("no solution found")


def lazy(m: Machine, fns: Funcs, system: System):
    print("Lazy mode activated")
    min_ = float("inf")
    ins = None
    max_joltage = max(m.joltage)
    ranges, range_fns = system.ranges(max_=max_joltage)
    print(f"{ranges=}, {len(range_fns)=}")

    def check(ins: list[float], min_: float) -> float:
        if not fns.is_in_bounds(ins):
            return min_
        evaluated = fns.eval(ins)
        if not all(x >= 0 for x in evaluated):
            return min_
        if not all(is_whole(x) for x in evaluated):
            return min_
        # Validate
        check_m = m.clone()
        for i, count in enumerate(evaluated):
            for _ in range(round(count)):
                check_m.press(i)
        if tuple(check_m.total_joltage) != check_m.joltage:
            print(
                f"Invalid total joltage: {check_m.total_joltage} != {check_m.joltage}"
            )
            return min_

        s = sum(evaluated)
        if s >= 1 and s < min_ and is_whole(s):
            return s
        return min_

    for combos in product(*ranges[:-1]):
        final_range = ranges[-1]
        # print(f"{final_range=}")
        partial_input = list(map(float, combos))
        for rfn in range_fns[:1]:
            final_range &= rfn(*partial_input)
        # print(f"{final_range=}")
        print(f"{partial_input=} {ranges=} {final_range=}")
        for x in final_range:
            input = partial_input + [float(x)]
            minn = check(input, min_)
            if minn < min_:
                min_ = minn
                ins = input

    print(f"{min_=} {ins=}")
    return min_


@bench()
def part_2(input: Input):
    machines = parse(input)
    out = 0
    mins = []

    def capture(x: float):
        print(f"got {x}")
        assert x < float("inf")
        mins.append(x)

    # for i, m in enumerate(machines[5:6]): # funky
    for i, m in enumerate(machines):
        print(f"\nMACHINE {i+1:03d} of {len(machines)}")
        # print(f"{m}")
        # print("-" * 40)
        mat = Matrix.from_machine(m)
        print(mat)
        print()
        mat.eliminate()
        # mat.bareiss()
        # mat.gj_eliminate()
        # print(mat)
        fns = Funcs.from_matrix(mat)
        # bs_sol = mat.back_substitute()
        # if bs_sol:
        #     print("solution:", mat.back_substitute())
        #     out += sum(bs_sol)
        #     continue
        # print(fns)
        # if len(fns.free) >= 3:
        #     print(fns)
        #     print(fns.get_sum_fn())
        # for fn in fns.fns.values():
        #     print(fn.bounds())
        # sum_fn = fns.get_sum_fn()
        # print(sum_fn)
        system = System.from_funcs(fns)
        if (a := system.answer()) is not None:
            capture(a)
            continue
        print(fns)
        print(system)
        tableau = Tableau.from_system(system)
        # print("Tableau")
        sim_res = tableau.simplex()
        # print(tableau)
        if is_whole(sim_res):
            capture(sim_res)
        else:
            capture(lazy(m, fns, system))
        # print("Tableau")
        # print(tableau)
        # print(sim_res)
        # capture(sim_res)
        # capture(branch_and_bound(system))
        continue
        capture(lazy(m, fns, system))
        continue

        tableau = system.to_tableau()
        print(tableau)
        assert sol is not None
        print(sol)
        sum_fn = fns.get_sum_fn()
        val = sum_fn(sol[: len(fns.free)])
        assert val < float("inf")
        if not (val < float("inf")):
            val = lazy(m, fns, system)
        assert val < float("inf")
        print(val)
        mins.append(val)
        continue
        print(tableau)
        # exact sum
        # if all(x == 0 for x in sum_fn.coeffs) and is_whole(sum_fn.const):
        #     out += round(sum_fn.const)
        #     # print(f"{sum_fn.const=}")
        #     continue
        # print(sum_fn)
        min_ = 10**20
        max_joltage = max(m.joltage)
        for comb in product(range(max_joltage + 1), repeat=len(fns.free)):
            if sum(comb) > max_joltage:
                continue
            fcomb = [float(x) for x in comb]
            # print(f"{comb}")
            if not fns.is_in_bounds(fcomb):
                # print("not in bounds")
                continue
            evaluated = fns.eval(fcomb)
            if not all(x >= 0 for x in evaluated):
                # print("not above zero", evaluated)
                continue
            # if not all(is_whole(x) for x in evaluated):
            #     # print("not above zero", evaluated)
            #     continue
            # print(evaluated, sum(evaluated))
            s = sum(evaluated)
            # if not is_whole(s):
            #     continue
            # s = sum_fn(fcomb)
            # print(f"{s=}, {min_=}")
            if s >= 1 and s < min_:
                min_ = round(s)
        assert min_ < 10**20, f"{m}\n{mat}\n{fns}\n{sum_fn}"
        # print(f"{min_=}")
        out += min_
        # print(sum(fns.eval([1, 0])))
        # mat.sort()
        # print(mat)

    print(mins)
    out = sum(mins)
    print("PART 2!", out)
    return round(out)


def _test():
    def assert_eq(a, b):
        assert a == b, f"{a} != {b}"

    assert_eq(part_1(CONTROL_1), 7)
    assert_eq(part_2(CONTROL_1), 33)
    # 18963 is too high
    # 18957 ??? not right
    # 18296 ??? not right
    assert_eq(part_2(input_file), 18960)


if __name__ == "__main__":
    _test()
    print("tests: PASS")
    print("-" * 40)
    print("part_1:", part_1(input_file))
    print("part_2:", part_2(input_file))
