#!/usr/bin/env python

from dataclasses import dataclass
import math
import re
from typing import Generator, NamedTuple


CONTROL_1 = """\
11-22,95-115,998-1012,1188511880-1188511890,222220-222224,1698522-1698528,446443-446449,38593856-38593862,565653-565659,824824821-824824827,2121212118-2121212124
""".splitlines()

with open("2025-2.input") as f:
    input_file = [line.strip() for line in f.readlines()]

Range = NamedTuple("Range", [("start", int), ("end", int)])


def parse(input: list[str]) -> list[Range]:
    out = []
    for r in input[0].split(","):
        start, end = map(int, r.split("-"))
        out.append(Range(start=start, end=end))
    return out


def decimals(i: int):
    return math.floor(math.log10(i)) + 1


# original solution
def part_1_stringy(input):
    ranges = parse(input)
    invalid_count = 0
    for r in ranges:
        for i in range(r.start, r.end + 1):
            s = str(i)
            n = len(s)
            if n % 2 != 0:
                continue
            if s[: n // 2] == s[n // 2 :]:
                invalid_count += i
    return invalid_count


def decompose_symmetric_ranges(r: Range) -> Generator[Range]:
    min_d, max_d = decimals(r.start), decimals(r.end)
    if min_d == max_d:
        if min_d % 2 == 0:
            yield r
        return
    for d in range(min_d, max_d + 1):
        if d % 2 != 0:
            continue
        if d == min_d:
            yield Range(r.start, 1 * 10**d - 1)
        elif d == max_d:
            yield Range(1 * 10 ** (d - 1), r.end)
        else:
            yield Range(1 * 10 ** (d - 1), 1 * 10**d - 1)


def part_1(input):
    ranges = parse(input)
    invalid_count = 0
    for full_range in ranges:
        for r in decompose_symmetric_ranges(full_range):
            # start_d = decimals(r.start)
            # end_d = decimals(r.end)
            for i in range(r.start, r.end + 1):
                n = decimals(i)
                if n % 2 != 0:
                    continue
                zeroes = 10 ** (n // 2)
                left = i // zeroes
                right = i - (left * zeroes)
                if left == right:
                    invalid_count += i
    return invalid_count


def part_2_with_re(input):
    ranges = parse(input)
    invalid_count = 0
    for r in ranges:
        for i in range(r.start, r.end + 1):
            if re.match(r"^(\d+)(\1)+$", str(i)):
                invalid_count += i
    return invalid_count


# slower than re: 2.29s vs 1.66s
def part_2_without_re(input):
    def pat_fills(s: str, pat: str) -> bool:
        for start in range(len(pat), len(s), len(pat)):
            if s[start : start + len(pat)] != pat:
                return False
        return True

    def has_repeats(s: str) -> bool:
        n = len(s)
        for pat_len in range(1, n // 2 + 1):
            if n % pat_len != 0:
                continue
            pat = s[:pat_len]
            if pat_fills(s, pat):
                return True
        return False

    ranges = parse(input)
    invalid_count = 0
    for r in ranges:
        for i in range(r.start, r.end + 1):
            if has_repeats(str(i)):
                invalid_count += i
    return invalid_count


# pass 3: using math
# NOTE: my math proof language sucks.
#
# Let's say given a number, n which exists in set Nat\{0}
# d is the number of decimal digits in that number
# p is some period (number of decimals) such that p is a divisor of d
# min_p(p) returns the smallest number with p decimals (p = 3: min_p = 100)
# max_p(p) returns the largest number with p decimals (p = 3: max_p = 999)
# let q = d / p
# S(d, p) is the set of periodic numbers such that
# there exists c in range [min_p(p), max_p(p)]
# x is in S(d, p) such that x satisfies polynomial x = c*10^(q*p) + c*10^((q-1)*p) + ... + c*10^(0*p)
#
# Example: S(6, 2) = {101010, 111111, 121212, ..., 989898, 999999}
# Example: S(6, 3) = {100100, 101101, 102102, ..., 998998, 999999}


def polynomial_unit(num_decimals: int, period: int) -> int:
    # 10^(n*p) + 10^((n-1)*p) ... + 10^(0*p)
    # assert(num_decimals % period == 0)
    out = 1
    for _ in range(num_decimals // period - 1):
        out = out * 10**period + 1
    return out


@dataclass
class PSet:
    d: int  # number of decimals
    p: int  # period
    unit: int
    div: int
    bounds: Range | None = None

    def __init__(self, num_decimals, period, bounds=None):
        assert num_decimals > 0
        assert period > 0
        assert num_decimals % period == 0

        self.d = num_decimals
        self.p = period
        self.unit = polynomial_unit(num_decimals, period)
        divisor = 1
        for _ in range(num_decimals // period - 1):
            divisor *= 10**period
        self.div = divisor
        if bounds is not None:
            self.bounds = self._contract_bounds(bounds)

    def _contract_bounds(self, r: Range) -> Range:
        "Contract bounds to elements actually in the set."
        if (x := (r.start // self.div) * self.unit) >= r.start:
            start = x
        else:
            start = (r.start // self.div + 1) * self.unit

        if (x := (r.end // self.div) * self.unit) <= r.end:
            end = x
        else:
            end = (r.end // self.div - 1) * self.unit
        return Range(start, end)

    def contains(self, n: int) -> bool:
        if self.bounds is None or self.bounds.start <= n <= self.bounds.end:
            return (n // self.div * self.unit) == n
        return False

    def sum_set(self) -> int:
        assert self.bounds is not None  # would be infinite otherwise

        if self.bounds.end < self.bounds.start:
            return 0
        lo = self.bounds.start // self.div
        hi = self.bounds.end // self.div
        # Thanks, Gauss
        sum_coeff = ((hi * (hi + 1)) - ((lo - 1) * lo)) // 2
        # multiply by pre-computed polynomial where c = 1
        return sum_coeff * self.unit


def divisors(n: int) -> list[int]:
    "Yield all divisors of `n`. Omit 1 and n."

    def unordered():
        for i in range(2, int(math.sqrt(n)) + 1):
            if n % i == 0:
                yield i
                if (b := n // i) != i:
                    yield b

    return sorted(unordered())


def greatest_divisors(divs: list[int]) -> Generator[int]:
    # If a divisor has other divisors, those smaller divisors will include all
    # patterns generated by the greater one.
    for i, a in enumerate(divs):
        if any(x for x in divs[i + 1 :] if x % a == 0):
            continue
        yield a


def split_ranges(ranges: list[Range]) -> Generator[tuple[Range, int]]:
    "Splits ranges by number of decimals. Yields (range, num_decimals)."
    for r in ranges:
        start_d = decimals(r.start)
        end_d = decimals(r.end)
        if start_d == end_d:
            yield r, start_d
            continue
        yield Range(r.start, 10**start_d - 1), start_d
        for d in range(start_d + 1, end_d):
            yield Range(10 ** (d - 1), 10**d - 1), d
        yield Range(10 ** (end_d - 1), r.end), end_d


def part_2_math(input):
    out = 0
    for r, d in split_ranges(parse(input)):
        periods = list(greatest_divisors(divisors(d)))
        range_sum = sum(PSet(d, p, bounds=r).sum_set() for p in periods)
        # Special case p=1 since all elements of S(d, 1) exist in all other S(d, p)
        one_sum = PSet(d, 1, bounds=r).sum_set()
        range_sum += one_sum - (one_sum * len(periods))
        out += range_sum

    return out


part_2 = part_2_math


def _test():
    def assert_eq(a, b):
        assert a == b, f"{a} != {b}"

    assert_eq(part_1(CONTROL_1), 1227775554)
    assert_eq(part_2(CONTROL_1), 4174379265)
    assert_eq(part_2_without_re(CONTROL_1), 4174379265)

    assert_eq(
        list(decompose_symmetric_ranges(Range(22, 4444))),
        [Range(22, 99), Range(1000, 4444)],
    )
    # divisors
    assert_eq(
        divisors(2 * 3 * 4 * 5), [2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 24, 30, 40, 60]
    )
    assert_eq(
        list(greatest_divisors([2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 24, 30, 40, 60])),
        [24, 40, 60],
    )
    # polynomial_unit
    assert_eq(polynomial_unit(12, 3), 1001001001)  # 4 periods of 3
    assert_eq(polynomial_unit(5, 1), 11111)  # 5 periods of 1
    # split_ranges
    assert_eq(
        list(split_ranges([Range(20, 3000), Range(400, 500)])),
        [
            (Range(20, 99), 2),
            (Range(100, 999), 3),
            (Range(1000, 3000), 4),
            (Range(400, 500), 3),
        ],
    )

    # PSet.contains
    s = PSet(6, 3)
    assert_eq(s.contains(123123), True)
    # PSet.bounds
    s = PSet(6, 3, bounds=Range(123456, 765432))
    assert_eq(s.bounds, Range(124124, 764764))
    s = PSet(6, 3, bounds=Range(123123, 456456))
    assert_eq(s.bounds, Range(123123, 456456))
    # Pset.sum_set
    s = PSet(2, 1, Range(11, 22))
    assert_eq(s.sum_set(), 11 + 22)
    s = PSet(3, 1, Range(100, 115))
    assert_eq(s.sum_set(), 111)
    s = PSet(4, 1, Range(1000, 1010))
    assert_eq(s.sum_set(), 0)


def _bench(fn, count=100):
    import timeit

    return timeit.timeit(fn, number=count) / count * 1_000


if __name__ == "__main__":
    _test()
    print("tests: PASS")
    print("-" * 40)
    print("part_1:", part_1(input_file))
    print("part_2:", part_2(input_file))
    print("-" * 40)
    print("part_1 bench: {:.1f}ms".format(_bench(lambda: part_1(input_file), count=10)))
    print("part_2 bench: {:.1f}ms".format(_bench(lambda: part_2(input_file), count=1)))
