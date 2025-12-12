#!/usr/bin/env python

import math
from dataclasses import dataclass
from functools import wraps
from itertools import combinations, islice, product
import os
from typing import Callable, Generator, Iterable, LiteralString
import timeit

DEBUG = bool(os.getenv("DEBUG", False))


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
        return self.light_diagram


CONTROL_1: Input = (
    """\
[.##.] (3) (1,3) (2) (2,3) (0,2) (0,1) {3,5,4,7}
[...#.] (0,2,3,4) (2,3) (0,4) (0,1,2) (1,2,3,4) {7,5,12,7,2}
[.###.#] (0,1,2,3,4) (0,3,4) (0,1,2,4,5) (1,2) {10,11,11,5,10,5}
""".splitlines()
)

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
    nonzero = list(x for x in zip(nums, vars) if x[0] != 0)
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

    # def __post_init__(self):
    #     if all(x <= 0 for x in self.coeffs):
    #         self.invert()

    def __repr__(self) -> str:
        if any(x != 0 for x in self.coeffs):
            out = fmtsum(list(self.coeffs), [f"x_{v+1}" for v in self.vars])
            out += f"{self.ineq} {fmtn(self.const)}"
            return out
        return f"{0} == {0}".format(fmtn(self.const))

    def invert(self):
        self.coeffs = tuple(-x for x in self.coeffs)
        self.const *= -1
        self.ineq = ">=" if self.ineq == "<=" else "<="

    def is_equality(self) -> bool:
        return all(x == 0 for x in self.coeffs)


@dataclass
class Fn:
    const: float
    coeffs: tuple[float, ...]

    def __repr__(self) -> str:
        if any(x != 0 for x in self.coeffs):
            out = fmtsum(
                [-x for x in self.coeffs], [f"x_{i+1}" for i in range(len(self.coeffs))]
            )
            c = self.const
            if c != 0:
                out += f" {fmt_addsub(c)}"
            return out
        return fmtn(self.const)

    def __call__(self, vars: list[float]) -> float:
        right = sum(self.coeffs[i] * v for i, v in enumerate(vars))
        return self.const - right

    def bounds(self) -> str:
        cs = [(i, c) for (i, c) in enumerate(self.coeffs) if c != 0]
        if not cs:
            return f"{self.const} == {self.const}"

        s = f"{-self.const} <="

        def sign(x) -> str:
            return "+" if x < 0 else "-"

        s += f" {-cs[0][1] if cs[0][1] >= 1 else ''}x_{cs[0][0]+1}"
        for i, c in cs[1:]:
            s += f" {sign(c)} {abs(c) if abs(c) > 1 else ''}x_{i+1}"
        return s


@dataclass
class SumFn:
    const: float
    coeffs: tuple[float, ...]
    free: tuple[int, ...]

    def __repr__(self) -> str:
        s = f"Σ = {self.const}"
        for i, c in enumerate(self.coeffs):
            if c == 0.0:
                continue
            s += f" {'+' if c > 0 else '-'} {abs(c) if abs(c) > 1 else ''}x_{i+1}"
        return s

    def __call__(self, free_vars: list[float]) -> float:
        total = self.const
        for i, v in enumerate(free_vars):
            total += self.coeffs[self.free[i]] * v
        return total


def fmtn(n: float) -> str:
    return str(round(n)) if is_whole(n) else f"{n:.1f}"


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
            if self[max_row_i, pivot_col] == 0.0:
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
                # frac = self[row_i, pivot_col] / self[pivot_row, pivot_col]
                frac = math.gcd(*self.data[row_i])
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
            if self[pivot_row, pivot_col] == 0.0:
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
            if div != 0.0:
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
    count: int

    @staticmethod
    def from_matrix(matrix: Matrix) -> "Funcs":
        row_i = 0
        col_i = 0
        fns = {}
        # # loop
        while row_i < matrix.n_rows:
            if col_i > matrix.n_cols - 2:
                break
            if matrix[row_i, col_i] == 0:
                col_i += 1
                continue
            coefficients = []
            for _ in range(col_i + 1):
                coefficients.append(0)
            for col_j in range(col_i + 1, matrix.n_cols - 1):
                coefficients.append(matrix[row_i, col_j])
            fns[col_i] = Fn(matrix[row_i, matrix.n_cols - 1], tuple(coefficients))
            row_i += 1
            col_i += 1
        return Funcs(
            fns,
            tuple(i for i in range(matrix.n_cols - 1) if i not in fns),
            count=matrix.n_cols - 1,
        )

    def __post_init__(self):
        self._bounds = list(self.gen_bounds())

    def __repr__(self) -> str:
        s = "\n".join((f"x_{i+1} = {fn}") for i, fn in self.fns.items())
        if self.free:
            free_vars = ", ".join(f"x_{i+1}" for i in self.free)
            s += f"\nfree: {free_vars}"
        return s

    def eval(self, free_vars: list[float]) -> list[float]:
        assert len(free_vars) == len(self.free)
        vals = [0.0 for _ in range(self.count)]
        for i, v in enumerate(free_vars):
            vals[self.free[i]] = v
        for i in reversed(range(self.count)):
            if i not in self.fns:
                continue
            fn = self.fns[i]
            vals[i] = fn(vals)
        return vals

    def get_sum_fn(self) -> SumFn:
        const = sum(fn.const for fn in self.fns.values())
        xs = [0.0 for _ in range(self.count)]
        for i in range(self.count):
            fn = self.fns.get(i)
            if fn is None:
                xs[i] += 1
                continue
            for j in range(self.count):
                xs[j] += -fn.coeffs[j]
        return SumFn(const, tuple(xs), self.free)

    def gen_bounds(self) -> Generator[Callable[[list[float]], bool]]:
        for fn in self.fns.values():
            if sum(fn.coeffs) == 0:
                continue

            def in_bounds(free_vars: list[float]) -> bool:
                total = 0
                for i, v in enumerate(free_vars):
                    total += fn.coeffs[self.free[i]] * v
                return total <= fn.const

            yield in_bounds

    def is_in_bounds(self, *free_vars: list[float]) -> bool:
        return all(b(*free_vars) for b in self._bounds)


@dataclass
class System:
    inequalities: list[Ineq]
    goal: SumFn
    maximize: bool = False

    def __repr__(self) -> str:
        out = f"{'max' if self.maximize else 'min'} {self.goal}:\n"
        out += "\n".join(map(repr, self.inequalities))
        return out

    @property
    def free(self) -> tuple[int, ...]:
        return self.goal.free

    @staticmethod
    def from_funcs(fns: Funcs) -> "System":
        ineqs = [
            Ineq(
                const=fn.const,
                coeffs=tuple(fn.coeffs[i] for i in fns.free),
                vars=fns.free,
            )
            for fn in fns.fns.values()
            if any(fn.coeffs[i] != 0 for i in fns.free)
            and not (all(fn.coeffs[i] < 0 for i in fns.free) and fn.const >= 0)
        ]
        for i, ineq in enumerate(ineqs):
            mul = 1
            if not is_whole(ineq.const):
                mul *= 1 / (ineq.const - round(ineq.const))
            for c in ineq.coeffs:
                if not is_whole(c):
                    mul *= 1 / (c - round(c))
            if mul == 1:
                continue
            ineqs[i] = Ineq(
                const=ineq.const * mul,
                coeffs=tuple(c * mul for c in ineq.coeffs),
                vars=fns.free,
            )

        sum_fn = fns.get_sum_fn()

        # Don't int-ify the sum
        # if is_whole(sum_fn.const):
        #     sum_fn = SumFn(sum_fn.const, tuple(sum_fn.coeffs), free=fns.free)
        # else:
        #     mul = 1 / (sum_fn.const - round(sum_fn.const))
        #     sum_fn = SumFn(
        #         mul * sum_fn.const, tuple(mul * x for x in sum_fn.coeffs), free=fns.free
        #     )
        # mul = 1
        # for c in sum_fn.coeffs:
        #     if not is_whole(c):
        #         mul *= 1 / (c - round(c))
        # if mul != 1:
        #     sum_fn = SumFn(
        #         mul * sum_fn.const, tuple(mul * x for x in sum_fn.coeffs), free=fns.free
        #     )

        return System(ineqs, sum_fn)

    def answer(self) -> float | None:
        if all(x == 0 for x in self.goal.coeffs):
            return self.goal.const

    def to_tableau(self) -> Matrix:
        # vars = (
        #     [f"y_{i+1}" for i in range(len(self.inequalities))]
        #     + [f"x_{i+1}" for i in self.free]
        #     + ["RHS"]
        # )
        # mat = Matrix.zeroes(n_rows=len(vars), n_cols=len(self.free) + 1)
        # for row_i, ineq in enumerate(self.inequalities):
        #     for col_i, coef in enumerate(ineq.coeffs):
        #         mat[row_i, col_i] = coef
        #     mat[row_i, -1] = -ineq.const
        # for col_i, coef_i in enumerate(self.free):
        #     mat[-1, col_i] = -self.goal.coeffs[coef_i]
        # mat = mat.transpose()
        # for i, _ in enumerate(self.free):
        #     mat[i, len(self.inequalities) + i] = 1
        # mat[-1, -1] = self.goal.const
        # mat.col_headers = vars
        # return mat
        n = len(self.free)
        m = len(self.inequalities)

        # cols:  [x_1 .. x_n | s_1 .. s_m | RHS]
        mat = Matrix.zeroes(n_rows=m + 1, n_cols=n + m + 1)

        # constraint rows
        for i, ineq in enumerate(self.inequalities):
            # ineq.coeffs is length n
            for j, coef in enumerate(ineq.coeffs):
                mat[i, j] = coef
            mat[i, n + i] = 1  # slack s_i
            mat[i, -1] = ineq.const

        # objective row (row m)
        for j in range(n):
            mat[m, j] = self.goal.coeffs[self.free[j]]
        # slacks in obj row = 0
        mat[m, -1] = -self.goal.const
        return mat


def simplex(tableau: Matrix):
    min_ = float("inf")

    def has_solution():
        return all(x >= 0 for x in tableau.data[-1][:-1])

    def get_pivot_col() -> int | None:
        for j, val in enumerate(tableau.data[-1][:-1]):
            if val < 0:
                return j
        return None

    def get_pivot_row(col_i: int) -> int:
        options = [
            (row_i, tableau[row_i, -1] / tableau[row_i, col_i])
            for row_i in range(tableau.n_rows - 1)
            if tableau[row_i, col_i] > 0
        ]
        if not options:
            raise RuntimeError("unbounded")
        min_ratio = min(options, key=lambda x: x[1])[1]
        return next((i, x) for (i, x) in options if x == min_ratio)[0]

    def step() -> bool:
        pivot_col = get_pivot_col()
        if pivot_col is None:
            return False
        pivot_row = get_pivot_row(pivot_col)

        # Divide pivot row out
        div = tableau[pivot_row, pivot_col]
        # div = math.gcd(*(tableau.data[pivot_row]))
        for col_i in range(tableau.n_cols):
            tableau[pivot_row, col_i] /= div

        # Eliminate other rows
        for row_i in (i for i in range(tableau.n_rows) if i != pivot_row):
            mul = tableau[row_i, pivot_col]
            for col_i in range(tableau.n_cols):
                tableau[row_i, col_i] -= tableau[pivot_row, col_i] * mul
        return True

    def back_substitute() -> list[float] | None:
        # if self.n_rows != (self.n_cols - 1):
        #     return None
        solution = [0.0 for _ in range(tableau.n_cols)]
        for row_i in reversed(range(tableau.n_rows - 1)):
            if all(x == 0 for x in tableau.data[row_i][:-1]):
                continue
            sum = 0.0
            col_i = next(i for i in range(tableau.n_cols - 1) if tableau[row_i, i] != 0)
            for row_j in reversed(range(row_i, tableau.n_rows - 1)):
                sum += solution[row_j] * tableau[row_i, row_j]
            div = tableau[row_i, row_i]
            if div != 0.0:
                solution[row_i] = (tableau[row_i, tableau.n_cols - 1] - sum) / div
            else:
                solution[row_i] = 0
        return solution

    step()
    for _ in range(100):
        if has_solution():
            sum_ = tableau[-1, -1]
            if is_whole(sum_) and sum_ < min_:  # and sum_ >= 0:
                break
        if not step():
            break

    print("-----")
    print(tableau)

    # zero "nonbasic" vars
    for col_i in range(tableau.n_cols - 1):
        col = tableau.col(col_i)
        if sum(1 for c in col if c != 0) == 1:
            continue
        for row_i in range(tableau.n_rows):
            tableau[row_i, col_i] = 0

    tableau.sort(skip_last_row=True)

    print("----- sorted")
    print(tableau)

    return back_substitute()


def is_whole(n: float) -> bool:
    x = n - round(n)
    return abs(x) < 0.0001


def lazy(m: Machine, fns: Funcs):
    print("Lazy mode activated")
    min_ = float("inf")
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
    return min_


@bench()
def part_2(input: Input):
    machines = parse(input)
    out = 0
    mins = []
    # for i, m in enumerate(machines[5:6]): # funky
    for i, m in enumerate(machines):
        print(f"\nMACHINE {i+1:03d} of {len(machines)}")
        # print(f"{m}")
        # print("-" * 40)
        mat = Matrix.from_machine(m)
        # print(mat)
        mat.eliminate()
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
            mins.append(a)
            continue
        print(fns)
        print(system)

        tableau = system.to_tableau()
        print(tableau)
        sol = simplex(tableau)
        assert sol is not None
        print(sol)
        sum_fn = fns.get_sum_fn()
        val = sum_fn(sol[: len(fns.free)])
        assert val < float("inf")
        if not (val < float("inf")):
            val = lazy(m, fns)
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
    # assert_eq(part_2(CONTROL_1), 33)
    # 18963 is too high
    # 18957 ??? not right


if __name__ == "__main__":
    _test()
    print("tests: PASS")
    print("-" * 40)
    print("part_1:", part_1(input_file))
    print("part_2:", part_2(input_file))
