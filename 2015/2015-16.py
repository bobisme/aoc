#!/usr/bin/env python

from collections.abc import Callable, Generator, Iterable
import time


TARGET = {
    "children": 3,
    "cats": 7,
    "samoyeds": 2,
    "pomeranians": 3,
    "akitas": 0,
    "vizslas": 0,
    "goldfish": 5,
    "trees": 3,
    "cars": 2,
    "perfumes": 1,
}


def parse(input: list[str]) -> Generator[dict[str, int]]:
    for line in input:
        _, counts = line.split(": ", maxsplit=1)
        yield {
            s[0]: int(s[1]) for s in (prop.split(": ") for prop in counts.split(", "))
        }


def argmin(
    sues: Iterable[dict[str, int]], score_fn: Callable[[dict[str, int]], float]
) -> int:
    idx, _ = min(
        ((i, score_fn(sue)) for (i, sue) in enumerate(sues)), key=lambda x: x[1]
    )
    return idx


def mean_squared_error(errors: Iterable[int]) -> float:
    count = 0
    total = 0
    for err in errors:
        count += 1
        total += err * err
    if count == 0:
        return 0
    return total / count


def part_1(input: list[str]):
    def score_sue(sue: dict[str, int]) -> float:
        errors = []
        for k, v in TARGET.items():
            if k in sue:
                errors.append(v - sue[k])
            else:
                errors.append(2)
        return mean_squared_error(errors)

    sues = list(parse(input))
    return argmin(sues, score_sue) + 1


def part_2(input: list[str]):
    BIG_ERR = 1

    def score_sue(sue: dict[str, int]) -> float:
        def errors():
            for k, target_val in TARGET.items():
                if k in sue:
                    sue_val = sue[k]
                    if k in ("cats", "trees"):
                        if sue_val <= target_val:
                            yield BIG_ERR
                        continue
                    if k in ("pomeranians", "goldfish"):
                        if sue_val >= target_val:
                            yield BIG_ERR
                        continue
                    yield target_val - sue_val

        return mean_squared_error(errors())

    sues = list(parse(input))
    return argmin(sues, score_sue) + 1


def run(fn, year=2015, day=16, part=0):
    start = time.perf_counter_ns()
    res = fn()
    elapsed_ns = time.perf_counter_ns() - start
    print(f"{year}\t{day}\t{part}\t{res}\t{elapsed_ns}")
    return res


if __name__ == "__main__":
    with open("2015-16.input") as f:
        input_file = [line.rstrip("\n") for line in f.readlines()]
    run(lambda: part_1(input_file), part=1)
    run(lambda: part_2(input_file), part=2)
