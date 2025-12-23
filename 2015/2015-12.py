#!/usr/bin/env python

import json
from typing import Any, Generator, LiteralString
import time


Input = list[str] | list[LiteralString]


def part_1(input: Input):
    data = json.loads(input[0])

    def get_nums(data: dict[str, Any] | list[Any]) -> Generator[int]:
        if isinstance(data, dict):
            for val in data.values():
                if isinstance(val, int):
                    yield val
                else:
                    yield from get_nums(val)
        elif isinstance(data, list):
            assert isinstance(data, list)
            for val in data:
                if isinstance(val, int):
                    yield val
                else:
                    yield from get_nums(val)

    return sum(get_nums(data))


def part_2(input: Input):
    data = json.loads(input[0])

    def get_nums(data: dict[str, Any] | list[Any]) -> Generator[int]:
        if isinstance(data, dict):
            if "red" in data.values():
                return
            for val in data.values():
                if isinstance(val, int):
                    yield val
                else:
                    yield from get_nums(val)
        elif isinstance(data, list):
            assert isinstance(data, list)
            for val in data:
                if isinstance(val, int):
                    yield val
                else:
                    yield from get_nums(val)

    return sum(get_nums(data))


def run(fn, year=2015, day=12, part=0):
    start = time.perf_counter_ns()
    res = fn()
    elapsed_ns = time.perf_counter_ns() - start
    print(f"{year}\t{day}\t{part}\t{res}\t{elapsed_ns}")
    return res


if __name__ == "__main__":
    with open("2015-12.input") as f:
        input_file = [line.rstrip("\n") for line in f.readlines()]
    run(lambda: part_1(input_file), part=1)
    run(lambda: part_2(input_file), part=2)
