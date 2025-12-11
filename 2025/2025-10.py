#!/usr/bin/env python

from dataclasses import dataclass
from functools import wraps
from itertools import combinations, islice, product
import pprint
from typing import Callable, Generator, Iterable, LiteralString
import timeit


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


@dataclass
class Fn:
    const: float
    coeffs: tuple[float, ...]

    def __repr__(self) -> str:
        s = f"{self.const}"
        for i, c in enumerate(self.coeffs):
            if c == 0.0:
                continue
            s += f" {'+' if c < 0 else '-'} {abs(c) if abs(c) > 1 else ''}x_{i+1}"
        return s

    def __call__(self, vars: list[float]) -> float:
        right = sum(self.coeffs[i] * v for i, v in enumerate(vars))
        return self.const - right

    def bounds(self) -> str:
        cs = [(i, c) for (i, c) in enumerate(self.coeffs) if c != 0]
        if not cs:
            return f"{self.const} == {self.const}"

        s = f"{-self.const} <="

        def sign(x) -> str:
            return "+" if c < 0 else "-"

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


@dataclass
class Funcs:
    fns: dict[int, Fn]
    free: tuple[int, ...]
    count: int

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
class Matrix:
    data: list[list[float]]

    def __post_init__(self):
        self.n_rows = len(self.data)
        self.n_cols = len(self.data[0])

    def __repr__(self) -> str:
        return "\n".join(("\t".join((f"{x:.1f}" for x in row)) for row in self.data))
        return pprint.pformat(self.data)

    def __getitem__(self, row_col: tuple[int, int]) -> float:
        row, col = row_col
        return self.data[row][col]

    def __setitem__(self, row_col: tuple[int, int], val: float):
        row, col = row_col
        self.data[row][col] = val

    @staticmethod
    def from_machine(machine: Machine) -> "Matrix":
        data = []
        for row_i, joltage in enumerate(machine.joltage):
            row = []
            for b in machine.buttons:
                row.append(1.0 if row_i in b else 0.0)
            row.append(joltage)
            data.append(row)
        return Matrix(data)

    def col(self, i: int) -> tuple[float, ...]:
        return tuple(row[i] for row in self.data)

    def swap_rows(self, r1: int, r2: int):
        self.data[r1], self.data[r2] = self.data[r2], self.data[r1]

    def eliminate(self) -> Generator[str]:
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
            yield f"find pivot in row {max_row_i} for column {pivot_col}"
            if pivot_row != max_row_i:
                # swap rows so row with max abs col is the pivot row
                self.swap_rows(pivot_row, max_row_i)
                yield f"swap rows {pivot_row} and {max_row_i} "
            for row_i in range(pivot_row + 1, self.n_rows):
                # for every row below the pivot...
                # calculate a fraction multiple to apply to other rows
                frac = self[row_i, pivot_col] / self[pivot_row, pivot_col]
                # the rest of the lower part of the col will become 0
                self[row_i, pivot_col] = 0.0
                # subtract the fraction from the rest of the current row
                for col_i in range(pivot_col + 1, self.n_cols):
                    self[row_i, col_i] -= self[pivot_row, col_i] * frac
            yield f"eliminate column {pivot_col}"
            pivot_row += 1

    def reduce(self) -> Generator[str]:
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
            yield f"eliminate row {pivot_row} column {pivot_col}"
            pivot_row += 1

    def back_substitute(self) -> list[float] | None:
        if self.n_rows != (self.n_cols - 1):
            return None
        solution = [0.0 for _ in range(self.n_cols)]
        end = min(self.n_rows, self.n_cols)
        for row_i in reversed(range(end)):
            sum = 0.0
            for row_j in reversed(range(row_i, self.n_rows)):
                sum += solution[row_j] * self[row_i, row_j]  # this might be a bug...
            div = self[row_i, row_i]
            if div != 0.0:
                solution[row_i] = (self[row_i, self.n_cols - 1] - sum) / div
            else:
                solution[row_i] = 0
        return solution

    def functions(self) -> Funcs:
        row_i = 0
        col_i = 0
        fns = {}
        # # loop
        while row_i < self.n_rows:
            if col_i > self.n_cols - 2:
                break
            if self[row_i, col_i] == 0:
                col_i += 1
                continue
            coefficients = []
            for _ in range(col_i + 1):
                coefficients.append(0)
            for col_j in range(col_i + 1, self.n_cols - 1):
                coefficients.append(self[row_i, col_j])
            fns[col_i] = Fn(self[row_i, self.n_cols - 1], tuple(coefficients))
            row_i += 1
            col_i += 1
        return Funcs(
            fns,
            tuple(i for i in range(self.n_cols - 1) if i not in fns),
            count=self.n_cols - 1,
        )

    def sort(self):
        self.data.sort(key=lambda row: [-abs(x) for x in row])


def is_whole(n: float) -> bool:
    x = n - round(n)
    return abs(x) < 0.0001


@bench()
def part_2(input: Input):
    machines = parse(input)
    out = 0
    # for i, m in enumerate(machines[5:6]): # funky
    for i, m in enumerate(machines):
        print(f"machine {i+1:03d} of {len(machines)}")
        # print(f"{m}")
        # print("-" * 40)
        mat = Matrix.from_machine(m)
        # print(f"{mat}")
        for op in mat.eliminate():
            # print("reduce:", op)
            # print(mat)
            pass
        for op in mat.reduce():
            # print("eliminate:", op)
            # print(mat)
            pass
        fns = mat.functions()
        bs_sol = mat.back_substitute()
        if bs_sol:
            print("solution:", mat.back_substitute())
            out += sum(bs_sol)
            continue
        # print(fns)
        # if len(fns.free) >= 3:
        #     print(fns)
        #     print(fns.get_sum_fn())
        # for fn in fns.fns.values():
        #     print(fn.bounds())
        sum_fn = fns.get_sum_fn()
        # print(sum_fn)
        # exact sum
        if all(x == 0 for x in sum_fn.coeffs) and is_whole(sum_fn.const):
            out += round(sum_fn.const)
            # print(f"{sum_fn.const=}")
            continue
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

    print("PART 2!", out)
    return round(out)


def _test():
    def assert_eq(a, b):
        assert a == b, f"{a} != {b}"

    assert_eq(part_1(CONTROL_1), 7)
    assert_eq(part_2(CONTROL_1), 33)
    # 18963 is too high
    # 18957 ??? not right


if __name__ == "__main__":
    _test()
    print("tests: PASS")
    print("-" * 40)
    print("part_1:", part_1(input_file))
    print("part_2:", part_2(input_file))
