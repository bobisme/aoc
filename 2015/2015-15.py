#!/usr/bin/env python

import functools
from typing import Generator, LiteralString, NamedTuple
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
Butterscotch: capacity -1, durability -2, flavor 6, texture 3, calories 8
Cinnamon: capacity 2, durability 3, flavor -2, texture -1, calories 3
""".splitlines()
)

with open("2015-15.input") as f:
    input_file = [line.rstrip("\n") for line in f.readlines()]


class Ingredient(NamedTuple):
    capacity: int
    durability: int
    flavor: int
    texture: int
    calories: int

    def add(self, other: "Ingredient") -> "Ingredient":
        return Ingredient(
            self.capacity + other.capacity,
            self.durability + other.durability,
            self.flavor + other.flavor,
            self.texture + other.texture,
            self.calories + other.calories,
        )

    def mul(self, x: int) -> "Ingredient":
        return Ingredient(
            self.capacity * x,
            self.durability * x,
            self.flavor * x,
            self.texture * x,
            self.calories * x,
        )

    def clamp(self) -> "Ingredient":
        return Ingredient(
            max(self.capacity, 0),
            max(self.durability, 0),
            max(self.flavor, 0),
            max(self.texture, 0),
            max(self.calories, 0),
        )

    def prop_product(self) -> int:
        return self.capacity * self.durability * self.flavor * self.texture


def parse(input: Input) -> Generator[tuple[str, Ingredient]]:
    for line in input:
        ingredient, rest = line.split(": ", maxsplit=1)
        parts = rest.split((", "))
        props = {x[0]: int(x[1]) for x in map(lambda x: x.split(" "), parts)}
        yield ingredient, Ingredient(**props)


def gen_iter(ingredients, rem: int) -> Generator[tuple[int, ...]]:
    match len(ingredients):
        case 0:
            raise Exception("Unreachable")
        case 1:
            yield (rem,)
        case _:
            for i in range(rem + 1):
                for subset in gen_iter(ingredients[1:], rem - i):
                    yield (i,) + subset


@bench
def part_1(input: Input):
    def brute_force(ingredients: list[tuple[str, Ingredient]]) -> int:
        best = 0
        for counts in gen_iter(ingredients, 100):
            assert sum(counts) == 100, counts
            scaled = [
                ing.mul(mul) for (ing, mul) in zip((x[1] for x in ingredients), counts)
            ]
            summed = functools.reduce(lambda a, b: a.add(b), scaled).clamp()
            if (sub_best := summed.prop_product()) > best:
                best = sub_best

        return best

    ingredients = list(parse(input))
    return brute_force(ingredients)


@bench
def part_2(input: Input):
    def brute_force(ingredients: list[tuple[str, Ingredient]]) -> int:
        best = 0
        for counts in gen_iter(ingredients, 100):
            assert sum(counts) == 100, counts
            scaled = [
                ing.mul(mul) for (ing, mul) in zip((x[1] for x in ingredients), counts)
            ]
            summed = functools.reduce(lambda a, b: a.add(b), scaled).clamp()
            if summed.calories != 500:
                continue
            if (sub_best := summed.prop_product()) > best:
                best = sub_best

        return best

    ingredients = list(parse(input))
    return brute_force(ingredients)


def _test():
    def assert_eq(a, b):
        assert a == b, f"{a} != {b}"

    assert_eq(part_1(CONTROL_1), 62842880)
    assert_eq(part_2(CONTROL_1), 57600000)


if __name__ == "__main__":
    _test()
    print("tests: PASS")
    print("-" * 40)
    part_1(input_file)
    part_2(input_file)
