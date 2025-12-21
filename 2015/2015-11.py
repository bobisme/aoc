#!/usr/bin/env python

from dataclasses import dataclass
from typing import LiteralString
import time


def bench(fn):
    def inner(*args, **kwargs):
        start = time.perf_counter()
        res = fn(*args, **kwargs)
        t_ms = (time.perf_counter() - start) * 1000
        print(f"{fn.__name__} = {res} in {t_ms:.2f}ms")
        return res

    return inner


Input = list[str] | list[LiteralString]

CONTROL_1: Input = (
    """\
""".splitlines()
)

with open("2015-11.input") as f:
    input_file = [line.rstrip("\n") for line in f.readlines()]

A = ord("a")
BANNED = (ord("i") - A, ord("o") - A, ord("l") - A)


@dataclass
class RevOrd:
    data: list[int]

    @staticmethod
    def from_str(s: str) -> "RevOrd":
        return RevOrd([ord(x) - A for x in reversed(s)])

    def to_str(self) -> str:
        return "".join(chr(x + A) for x in reversed(self.data))

    def incr(self):
        chars = self.data
        carry, chars[0] = divmod(chars[0] + 1, 26)
        i = 1
        while carry > 0:
            c = chars[i]
            carry, chars[i] = divmod(c + carry, 26)
            i += 1
        if carry > 0:
            chars.append(0)

    def has_straight(self) -> bool:
        for i in range(len(self.data) - 2):
            if (
                self.data[i] == self.data[i + 1] + 1
                and self.data[i + 1] == self.data[i + 2] + 1
            ):
                return True
        return False

    def is_valid(self) -> bool:
        if not self.has_straight():
            return False
        for letter in BANNED:
            if letter in self.data:
                return False
        letter = get_pair(self.data)
        if letter is None:
            return False
        return get_pair(self.data, exclude=letter) is not None


def get_pair(s: str | list[int], exclude: str | int | None = None) -> str | int | None:
    for i in range(len(s) - 1):
        if s[i] == exclude:
            continue
        if s[i] == s[i + 1]:
            return s[i]
    return None


@bench
def part_1(input: Input):
    s = input[0]
    data = RevOrd.from_str(s)
    while True:
        data.incr()
        if data.is_valid():
            return data.to_str()


part_2 = part_1


def _test():
    def assert_eq(a, b):
        assert a == b, f"{a} != {b}"

    assert_eq(part_1(["abcdefgh"]), "abcdffaa")


if __name__ == "__main__":
    _test()
    print("tests: PASS")
    print("-" * 40)
    res = part_1(input_file)
    part_2([res])
