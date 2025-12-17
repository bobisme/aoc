#!/usr/bin/env python

from fractions import Fraction
import math
from dataclasses import dataclass
from functools import reduce
from itertools import chain, combinations, islice
import os
from typing import Callable, Generator, Iterable, LiteralString
import timeit
from textwrap import indent

DEBUG = bool(os.getenv("DEBUG", False))


def debug(*args, **kwargs):
    if DEBUG:
        print(*args, **kwargs)


EPSILON = 1e-9

# Num = Fraction
# ZERO: Num = Fraction(0)
# ONE: Num = Fraction(1)
# BIG: Num = Fraction(10**10)
Num = float
ZERO: Num = 0
ONE: Num = 1
BIG: Num = 10.0**10


Input = list[str] | list[LiteralString]


def _btoa(b: int, len: int) -> str:
    fmtstr = f"{{:0{len}b}}"
    return fmtstr.format(b)


def is_whole(n: float | Fraction) -> bool:
    if isinstance(n, Fraction):
        return n.denominator == 1
    return math.isclose(n, round(n), abs_tol=1e-6)


def fmtn(n: float | Fraction) -> str:
    if isinstance(n, Fraction):
        return str(n)
    return str(round(n)) if is_whole(n) else f"{n:.3f}"


def is_zero(n: float | Fraction) -> bool:
    if isinstance(n, Fraction):
        return n == 0
    return math.isclose(n, 0, abs_tol=1e-6)


def forcefrac(n: float | Fraction) -> Fraction:
    assert isinstance(n, Fraction)
    return n


def decimals(n: Num) -> float:
    "given 1.234, return 0.234"
    return float(n) - math.floor(n)


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
    nums: Iterable[Num],
    # apply function if given
    fn: Callable[[Num], Num] = lambda x: x,
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


def fmt_addsub(n: Num, var: str | None = None) -> str:
    if var is None:
        return f"{'+' if n > 0 else '-'} {fmtn(abs(n))}"
    return f"{'+' if n > 0 else '-'} {fmtn(abs(n)) if abs(n) != 1 else ''}{var}"


def fmtsum(nums: list[Num], vars: list[str]) -> str:
    assert len(nums) == len(vars), f"lens must match: {nums=} {vars=}"
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
    const: Num
    coeffs: tuple[Num, ...]
    vars: tuple[int, ...]
    ineq: str = "<="

    def __post_init__(self):
        assert len(self.coeffs) == len(self.vars)
        if all(x <= 0 for x in self.coeffs) and self.var_count() == 1:
            self.invert()

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

    def var_count(self) -> int:
        return sum(1 for c in self.coeffs if not is_zero(c))

    def is_redundant(self) -> bool:
        return self.var_count() == 1 and self.ineq == ">=" and is_zero(self.const)


@dataclass
class Fn:
    const: Num
    coeffs: tuple[Num, ...]
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

    def __call__(self, vars: list[Num]) -> Num:
        right = sum(self.coeffs[i] * v for i, v in enumerate(vars))
        return self.const + right


@dataclass
class Matrix:
    data: list[list[Num]]
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

    def __getitem__(self, row_col: tuple[int, int]) -> Num:
        row, col = row_col
        return self.data[row][col]

    def __setitem__(self, row_col: tuple[int, int], val: Num):
        row, col = row_col
        self.data[row][col] = val

    @staticmethod
    def zeroes(n_rows: int, n_cols: int) -> "Matrix":
        return Matrix([[ZERO for _ in range(n_cols)] for _ in range(n_rows)])

    @staticmethod
    def from_machine(machine: Machine) -> "Matrix":
        data: list[list[Num]] = []
        for row_i, joltage in enumerate(machine.joltage):
            row: list[Num] = []
            for b in machine.buttons:
                row.append(ONE if row_i in b else ZERO)
            row.append(Num(joltage))
            data.append(row)
        col_headers = [f"x_{i+1}" for i in range(len(machine.buttons))]
        col_headers.append("J")
        return Matrix(data, col_headers=col_headers)

    def log(self, msg):
        if not DEBUG:
            return
        debug(msg)
        debug(self)

    def transpose(self) -> "Matrix":
        out = Matrix.zeroes(n_rows=self.n_cols, n_cols=self.n_rows)
        for i, row in enumerate(self.data):
            for j, col in enumerate(row):
                out[j, i] = col
        return out

    def col(self, i: int) -> tuple[Num, ...]:
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
                self[row_i, pivot_col] = ZERO
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

    def eliminate(self):
        self.g_eliminate()
        self.gj_eliminate()

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
            free,
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

    def eval(self, free_vars: list[Num]) -> list[Num]:
        assert len(free_vars) == len(self.free)
        vals = [ZERO for _ in range(self.var_count)]
        for i, v in enumerate(free_vars):
            vals[self.free[i]] = Num(v)
        for i in reversed(range(self.var_count)):
            if i not in self.fns:
                continue
            fn = self.fns[i]
            vals[i] = fn(free_vars)
        return vals

    def get_sum_fn(self) -> Fn:
        const = reduce(lambda a, b: a + b, (fn.const for fn in self.fns.values()))
        xs = [ONE for _ in range(len(self.free))]
        for var_i in range(self.var_count):
            fn = self.fns.get(var_i)
            if fn is None:
                continue
            for ci, _ in enumerate(self.free):
                xs[ci] += fn.coeffs[ci]
        return Fn(const, tuple(xs), self.free, lhs="C")

    def gen_bounds(self) -> Generator[Callable[[list[int]], bool]]:
        for fn in self.fns.values():
            if is_zero(sum(fn.coeffs)):
                continue

            def in_bounds(free_vars: list[int]) -> bool:
                total = 0
                for i, v in enumerate(free_vars):
                    total += fn.coeffs[i] * v
                return total <= fn.const

            yield in_bounds

    def is_in_bounds(self, *free_vars: list[int]) -> bool:
        return all(b(*free_vars) for b in self._bounds)


@dataclass
class System:
    inequalities: list[Ineq]
    goal: Fn
    maximize: bool = False
    fns: Funcs | None = None

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
                lambda ineq: not ineq.is_equality() and not ineq.is_redundant(),
                (
                    Ineq(
                        const=fn.const,
                        coeffs=tuple(-x for x in fn.coeffs),
                        vars=fn.free,
                    )
                    for fn in fns.fns.values()
                    if any(not is_zero(c) for c in fn.coeffs)
                ),
            )
        )
        sum_fn = fns.get_sum_fn()
        return System(ineqs, sum_fn, fns=fns)

    def answer(self) -> Num | None:
        if all(is_zero(x) for x in self.goal.coeffs):
            return self.goal.const


@dataclass
class Tableau:
    MAX_STEPS = 100
    mat: Matrix
    basis: list[str]
    free: tuple[int, ...]

    def __post_init__(self):
        self.n_cols = self.mat.n_cols
        self.n_rows = self.mat.n_rows
        self.min = BIG

    def __repr__(self) -> str:
        assert self.mat.col_headers is not None
        data = (
            [[""] + self.mat.col_headers]
            + [
                [self.basis[i]] + list(map(fmtn, row))
                for i, row in enumerate(self.mat.data[:-1])
            ]
            + [[""] + list(map(fmtn, self.mat.data[-1]))]
        )
        return "\n".join(("\t".join(row) for row in data))

    @staticmethod
    def from_system(system: System) -> "Tableau":
        n = len(system.free)
        m = len(system.inequalities)

        # cols:  [x_1 .. x_n | s_1 .. s_m | RHS]
        mat = Matrix.zeroes(n_rows=m + 1, n_cols=n + m + 1)
        basis = [f"s_{i+1}" for i in range(len(system.inequalities))]
        mat.col_headers = [f"x_{i+1}" for i in system.free] + basis.copy() + ["RHS"]

        for ci, coef in enumerate(system.goal.coeffs):
            mat[m, ci] = coef
        mat[m, -1] = -system.goal.const

        for i, ineq in enumerate(system.inequalities):
            multiplier = 1
            if ineq.ineq == ">=":
                multiplier = -1

            for ci, coef in enumerate(ineq.coeffs):
                mat[i, ci] = coef * multiplier

            mat[i, n + i] = ONE  # Slack is always +1
            mat[i, -1] = ineq.const * multiplier

        return Tableau(mat, basis, free=system.free)

    @property
    def data(self) -> list[list[Num]]:
        return self.mat.data

    def __getitem__(self, row_col: tuple[int, int]) -> Num:
        return self.mat[row_col]

    def __setitem__(self, row_col: tuple[int, int], val: Num):
        self.mat[row_col] = val

    def is_optimal(self):
        return all(x > -EPSILON for x in self.data[-1][:-1])

    def is_feasible(self):
        return all(x > -EPSILON for x in self.mat.col(-1)[:-1])

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

    def pivot(self, pivot_row: int, pivot_col: int):
        assert self.mat.col_headers is not None
        a, b = self.basis[pivot_row], self.mat.col_headers[pivot_col]
        self.basis[pivot_row] = b
        debug(f"pivot ({pivot_row}, {pivot_col}) / {a}<->{b}")
        # Divide pivot row out
        div = self[pivot_row, pivot_col]
        for col_i in range(self.n_cols):
            self[pivot_row, col_i] /= div
        # Eliminate other rows
        for row_i in (i for i in range(self.n_rows) if i != pivot_row):
            mul = self[row_i, pivot_col]
            for col_i in range(self.n_cols):
                self[row_i, col_i] -= self[pivot_row, col_i] * mul
        debug(self)

    def step_dual(self):
        # find row with most negative RHS
        pivot_row: int | None = None
        min_rhs = -EPSILON
        for i in range(self.n_rows - 1):
            if self[i, -1] < min_rhs:
                min_rhs = self[i, -1]
                pivot_row = i

        if pivot_row is None:
            return

        # find negative coefficient that results in the least steep change
        pivot_col = None
        max_ratio = -BIG

        for j in range(self.n_cols - 1):
            val = self[pivot_row, j]
            if val >= -EPSILON:
                continue
            ratio = self[-1, j] / val
            if ratio > max_ratio:
                max_ratio = ratio
                pivot_col = j
        # no negative coefficients? Infeasible!
        if pivot_col is None:
            raise RuntimeError("system is infeasible")
        self.pivot(pivot_row, pivot_col)

    def step_primal(self):
        # find column with the most negative objective value
        pivot_col = None
        min_val = -EPSILON
        for j in range(self.n_cols - 1):
            if self[-1, j] < min_val:
                min_val = self[-1, j]
                pivot_col = j
        if pivot_col is None:
            return

        # pick row that results in minimum positive ratio
        pivot_row = None
        min_ratio = BIG
        for i in range(self.n_rows - 1):
            val = self[i, pivot_col]
            if val <= EPSILON:
                continue
            ratio = self[i, -1] / val
            if ratio < min_ratio:
                min_ratio = ratio
                pivot_row = i
        if pivot_row is None:
            raise RuntimeError("system is unbounded")
        self.pivot(pivot_row, pivot_col)

    def vals(self) -> list[Num]:
        self.zero_nonbasic()
        out = [ZERO] * len(self.free)
        for i in range(len(self.free)):
            for row_i in range(self.n_rows - 1):
                if not is_zero(self[row_i, i]):
                    out[i] = self[row_i, -1] / self[row_i, i]
                    break
        return out

    def solve(self):
        debug("SOLVING:")
        debug(self)
        steps = 0

        while True:
            while not self.is_feasible():
                steps += 1
                if steps > self.MAX_STEPS:
                    raise RuntimeError("max steps, phase 1")
                self.step_dual()

            debug("tableau is feasible")
            # debug(self)

            while not self.is_optimal():
                steps += 1
                if steps > self.MAX_STEPS:
                    debug(self)
                    raise RuntimeError("max steps, phase 2")
                self.step_primal()

            debug("tableau is optimal")
            # debug(self)

            if self.is_feasible():
                break
            debug("... but it's not feasible")

        debug()
        debug(self)
        return -self[-1, -1]

    def zero_nonbasic(self):
        for col_i in range(self.n_cols - 1):
            col = self.mat.col(col_i)
            if sum(1 for c in col if not is_zero(c)) == 1:
                continue
            for row_i in range(self.n_rows):
                self[row_i, col_i] = ZERO


def branch_and_bound(system: System) -> Num:
    def inner(branch_sys: System, best: Num, upper_bound: Num) -> Num:
        debug(branch_sys)
        debug(f"last constraint: {branch_sys.inequalities[-1]}")

        tableau = Tableau.from_system(branch_sys)
        try:
            solution = tableau.solve()  # returns optimal val which may not be integer
        except RuntimeError:  # if infeasible
            return best
        if solution >= best:
            return best
        # returns the computed values of the free variables
        solution_vals = tableau.vals()
        assert system.fns
        full_vals = system.fns.eval(solution_vals)
        if all(is_whole(x) for x in full_vals):
            if is_whole(solution) and solution < best:
                return solution
            return best

        # find branch point
        branch_i = min(
            ((i, decimals(n)) for (i, n) in enumerate(full_vals) if not is_whole(n)),
            key=lambda x: abs(x[1]),
        )[0]
        v = full_vals[branch_i]
        const_offset = 0
        is_dependent = branch_i in system.fns.fns
        coeffs = tuple(ONE if i == branch_i else ZERO for i in branch_sys.free)
        if is_dependent:
            v = full_vals[branch_i]
            fn = system.fns.fns[branch_i]
            const_offset = fn.const
            coeffs = tuple(fn.coeffs)

        # upper
        upper_constraint = Ineq(
            const=Num(math.ceil(v)) - const_offset,
            coeffs=coeffs,
            vars=system.free,
            ineq=">=",
        )
        upper_ineqs = branch_sys.inequalities + [upper_constraint]
        upper_sys = System(inequalities=upper_ineqs, goal=branch_sys.goal)
        # lower
        lower_constraint = Ineq(
            const=Num(math.floor(v)) - const_offset,
            coeffs=coeffs,
            vars=system.free,
            ineq="<=",
        )
        lower_ineqs = branch_sys.inequalities + [lower_constraint]
        lower_sys = System(inequalities=lower_ineqs, goal=branch_sys.goal)

        best = inner(upper_sys, best, upper_bound)
        best = inner(lower_sys, best, upper_bound)
        return best

    return inner(system, BIG, BIG)


def part_2(input: Input):
    machines = parse(input)
    mins: list[int | float | Num] = [0] * len(input)
    stats = {
        "exact_answers": [],
        "simplex_solves": [],
        "searches": [],
    }

    def capture(i: int, x: Num | float | int):
        assert x < BIG
        assert x > EPSILON
        assert is_whole(x), f"{x} is not whole"
        debug(f"got {x}")
        mins[i] = round(x)

    for i, m in islice(enumerate(machines), 0, 10000):
        mat = Matrix.from_machine(m)
        mat.eliminate()
        fns = Funcs.from_matrix(mat)
        system = System.from_funcs(fns)
        if (a := system.answer()) is not None:
            stats["exact_answers"].append(i)
            capture(i, a)
            continue
        b_and_b = branch_and_bound(system)
        capture(i, b_and_b)

    debug(mins)
    out = sum(mins)
    return round(out)


def _test():
    def assert_eq(a, b):
        assert a == b, f"{a} != {b}"

    assert_eq(part_1(CONTROL_1), 7)
    assert_eq(part_2(CONTROL_1), 33)


def _bench(fn, count=100):
    return timeit.timeit(fn, number=count) / count * 1_000


if __name__ == "__main__":
    print("-" * 40)
    _test()
    print("tests: PASS")
    debug("-" * 40)
    print("part_1:", part_1(input_file))
    print("part_2:", part_2(input_file))
    print("part_1 bench: {:.1f}ms".format(_bench(lambda: part_1(input_file), count=1)))
    print("part_2 bench: {:.1f}ms".format(_bench(lambda: part_2(input_file), count=1)))
