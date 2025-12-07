#!/usr/bin/env python
"""
This is a rewrite of my 2025-02, omitting the initial
solutions and focusing on the math-based approach I worked
out on paper.

This introduces the range [1, 10^4000] as an additional challenge.
This is mostly an exercise in formalizing my approach,
optimizing, fixing assumptions based on the challenge
inputs, and generalizing.
"""

from functools import cache
import math
import sys
from typing import Generator, LiteralString
import timeit

# Let's use ridiculously large numbers.
sys.set_int_max_str_digits(10_000)

Input = list[str] | list[LiteralString]

CONTROL_1: Input = (
    """\
11-22,95-115,998-1012,1188511880-1188511890,222220-222224,1698522-1698528,446443-446449,38593856-38593862,565653-565659,824824821-824824827,2121212118-2121212124
""".splitlines()
)

with open("2025-02.input") as f:
    input_file = [line.rstrip("\n") for line in f.readlines()]


def parse(input: Input) -> list[range]:
    out = []
    for r in input[0].split(","):
        start, end = map(int, r.split("-"))
        out.append(range(start, end + 1))
    return out


def digits(n: int) -> int:
    """
    Let d = ⌊log₁₀(n)⌋ + 1 be the number of decimal digits.
    """
    return math.floor(math.log10(n)) + 1


def generating_constant(num_decimals: int, period: int) -> int:
    """
    Let p be a period such that p | d (p divides d).

    This returns a constant K that generates the repeating
    pattern for a given length d and period p. This ends up
    being the sum of a geometric series:

        K_dp = ∑_{k=0}^{q-1} 10^{k·p}

             = (10^d - 1) / (10^p - 1)
    """
    return (10**num_decimals - 1) // (10**period - 1)


@cache
def proper_divisors(d: int) -> list[int]:
    """
    Yield all proper divisors k of d, 1 <= k < d.
    """
    assert d > 1

    def unordered():
        yield 1
        for i in range(2, int(math.sqrt(d)) + 1):
            if d % i == 0:
                yield i
                if (b := d // i) != i:
                    yield b

    return sorted(unordered())


def split_ranges(ranges: list[range]) -> Generator[tuple[range, int]]:
    "Splits ranges by number of decimals. Yields (range, num_decimals)."
    for r in ranges:
        start_d = digits(r.start)
        end_d = digits(r.stop - 1)
        if start_d == end_d:
            yield r, start_d
            continue
        yield range(r.start, 10**start_d), start_d
        for d in range(start_d + 1, end_d):
            yield range(10 ** (d - 1), 10**d), d
        yield range(10 ** (end_d - 1), r.stop), end_d


class DPSet:
    """
    Let R(d,p) denote the set of all d-digit integers formed
    by repeating a p-digit pattern `c`.

    R(d,p) is the set of all numbers of period p with d
    digits where p is a proper divisor of d.

      R(d,p) = { c·K_dp | c ∈ ℕ, 10^(p-1) ≤ c < 10^p }

    Example: R(6,2) = {101010, 111111, 121212, ..., 989898, 999999}
    Example: R(6,3) = {100100, 101101, 102102, ..., 998998, 999999}
    """

    d: int
    p: int
    gen_constant: int
    _coeff_bounds: tuple[int, int] | None = None

    def __init__(self, num_decimals, period, bounds=None):
        assert num_decimals > 0
        assert period > 0
        assert num_decimals % period == 0

        self.d = num_decimals
        self.p = period
        self.gen_constant = generating_constant(num_decimals, period)
        if bounds is not None:
            self._coeff_bounds = self._shrink_bounds(bounds)

    def _shrink_bounds(self, r: range) -> tuple[int, int]:
        """
        Shrink bounds [A, B] to the subset [A', B'] ⊆ R(d,p).

        Maps the range bounds in the target space to the coefficient space 'c':
            c_min = ⌈A / K_dp⌉
            c_max = ⌊B / K_dp⌋

        Returns range [c_min, c_max]

        Full bounds can be calculated by [c_min·K_dp, c_max·K_dp],
        returned by `.bounds()`.
        """
        # math.ceil, math.floor fail for gigantic numbers.
        c_min, rem = divmod(r.start, self.gen_constant)
        if rem != 0:
            c_min += 1
        c_max = (r.stop - 1) // self.gen_constant
        return (c_min, c_max)

    @property
    def bounds(self) -> range | None:
        if self._coeff_bounds is None:
            return None
        c_min, c_max = self._coeff_bounds
        return range(c_min * self.gen_constant, c_max * self.gen_constant + 1)

    def sum(self) -> int:
        """
        This produces the sum of all elements within the
        bounds of the set.

          T(d,p) = ∑_{c=c_min}^{c_max} c·K_dp

                 = K_dp·∑{c=c_min}^{c_max} c

        Given

          ∑_{i=1}^n i = n·(n + 1) / 2,

          ∑_{i=a}^b i = (b·(b + 1) - (a - 1)·a) / 2

        Which can reduce to

          ∑_{i=a}^b i = (a + b) * (b - a + 1) / 2

        It follows that

          T(d,p) = K_dp·((c_min + c_max) * (c_max - c_min + 1)) / 2
        """
        assert self._coeff_bounds is not None, "infinity"

        c_min, c_max = self._coeff_bounds
        # Can't do len(range) for super large numbers.
        if c_max < c_min:
            return 0

        sum_coeff = (c_min + c_max) * (c_max - c_min + 1) // 2
        return sum_coeff * self.gen_constant


class DSet:
    """
    R(d) is the union of all sets R(d,p) where `p` is a
    proper divisor of `d`.
    """

    d: int
    bounds: range | None = None

    def __init__(self, d, bounds=None):
        self.d = d
        self.bounds = bounds

    def sum(self) -> int:
        """
        Sum over the union of all sets R(d,p).

        Note that sets will fully contain other sets that
        have periods that are proper divisors of p.
        R(d,q) ⊂ R(d,p) where q | p (e.g. R(12,2) ⊂ R(12,4) and R(12,1) ⊂ R(12,4).

        Also note that sets with periods that have common
        denominators will overlap (e.g. R(24,8) ∩ R(24,12) = R(24,4)).

        To account for this, in addition to T(d,p) as
        defined in `DPSet.sum`, let E(d,p) be the sum of
        elements that have minimal (exact) period p.

        The relationship is cumulative over the divisors:

          T(d,p) = ∑_{k|p} E(d,k)

        By inverting this, the exact sum for period p can be
        found by subtracting the exact sums of its divisors:

          E(d,p) = T(d,p) - ∑_{k|p, k<p} E(d,k)

        The final solution for the sum of all elements in
        R(d) is then the sum of exact parts for all proper
        divisors:

          ∑_{p|d, p<d} E(d,p)

        Given E(d,p) is called recursively and repeatedly, memoizing E
        results in very efficient computation.
        """

        @cache
        def exact_sum(p: int) -> int:
            if p == 1:
                return DPSet(self.d, p, bounds=self.bounds).sum()
            return DPSet(self.d, p, bounds=self.bounds).sum() - sum(
                exact_sum(div_p) for div_p in proper_divisors(p)
            )

        return sum(exact_sum(div) for div in proper_divisors(self.d))


class PeriodicNumbers:
    """
    R is the set of all periodic integers.
    It is the disjoint union of D(d) for all d > 1.
    """

    bounds: range | None = None

    def __init__(self, bounds) -> None:
        self.bounds = bounds

    def sum(self):
        """
        Using bounds [A, B], let d_min = digits(A), let
        d_max = digits(B), then calculate the exact sums of
        all sets of R(d) for all `d` in [d_min, d_max].
        """
        assert self.bounds is not None, "infinity"
        return sum(DSet(d, bounds=r).sum() for r, d in split_ranges([self.bounds]))


def part_1(input: Input):
    return sum(
        DPSet(d, d // 2, bounds=r).sum()
        for r, d in split_ranges(parse(input))
        if d % 2 == 0
    )


def part_2(input: Input):
    return sum(PeriodicNumbers(bounds=r).sum() for r in parse(input))


def _test():
    def assert_eq(a, b):
        assert a == b, f"{a} != {b}"

    assert_eq(part_1(CONTROL_1), 1227775554)
    assert_eq(part_2(CONTROL_1), 4174379265)


def _bench(fn, count=100):
    return timeit.timeit(fn, number=count) / count * 1_000


if __name__ == "__main__":
    _test()
    print("tests: PASS")
    print("-" * 40)
    print("part_1:", part_1(input_file))
    print("part_2:", part_2(input_file))
    print("-" * 40)
    print("part_1 bench: {:.2f}ms".format(_bench(lambda: part_1(input_file))))
    print("part_2 bench: {:.2f}ms".format(_bench(lambda: part_2(input_file))))

    # Let's throw some big numbers at it, just to check.
    # 1-10^4000 is ~1.37s on my 6yo machine.
    big_range = f"1-{10**4000}"
    print(
        "part_2 hardcore: {:.1f}ms".format(_bench(lambda: part_2([big_range]), count=1))
    )
