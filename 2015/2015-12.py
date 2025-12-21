#!/usr/bin/env python

import json
from typing import Any, Generator, LiteralString
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

with open("2015-12.input") as f:
    input_file = [line.rstrip("\n") for line in f.readlines()]


@bench
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


@bench
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


if __name__ == "__main__":
    part_1(input_file)
    part_2(input_file)
