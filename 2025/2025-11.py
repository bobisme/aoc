#!/usr/bin/env python

from functools import cache, reduce
from typing import Iterable, LiteralString
import timeit

Input = list[str] | list[LiteralString]

CONTROL_1: Input = (
    """\
aaa: you hhh
you: bbb ccc
bbb: ddd eee
ccc: ddd eee fff
ddd: ggg
eee: out
fff: out
ggg: out
hhh: ccc fff iii
iii: out
""".splitlines()
)

CONTROL_2: Input = (
    """\
svr: aaa bbb
aaa: fft
fft: ccc
bbb: tty
tty: ccc
ccc: ddd eee
ddd: hub
hub: fff
eee: dac
dac: fff
fff: ggg hhh
ggg: out
hhh: out
""".splitlines()
)

with open("2025-11.input") as f:
    input_file = [line.rstrip("\n") for line in f.readlines()]

Graph = dict[str, set[str]]


def parse(input: Input) -> Graph:
    g = {}
    for line in input:
        src, rest = line.split(": ")
        g[src] = {dst for dst in rest.split(" ")}
    return g


def dfs_count(graph: Graph, src: str, dst: str) -> int:
    @cache
    def inner(src: str):
        if src == dst:
            return 1
        if src == "out":
            return 0
        return sum(inner(next_node) for next_node in graph[src])

    return inner(src)


def part_1(input: Input):
    g = parse(input)
    return dfs_count(g, "you", "out")


def part_2(input: Input):
    g = parse(input)

    svr_to_dac = dfs_count(g, "svr", "dac")
    svr_to_fft = dfs_count(g, "svr", "fft")
    fft_to_dac = dfs_count(g, "fft", "dac")
    dac_to_fft = dfs_count(g, "dac", "fft")
    dac_to_out = dfs_count(g, "dac", "out")
    fft_to_out = dfs_count(g, "fft", "out")

    def prod(x: Iterable[int]) -> int:
        return reduce(lambda a, b: a * b, x, 1)

    return sum(
        (
            prod((svr_to_fft, fft_to_dac, dac_to_out)),
            prod((svr_to_dac, dac_to_fft, fft_to_out)),
        )
    )


def _test():
    def assert_eq(a, b):
        assert a == b, f"{a} != {b}"

    assert_eq(part_1(CONTROL_1), 5)
    assert_eq(part_2(CONTROL_2), 2)


def _bench(fn, count=100):
    return timeit.timeit(fn, number=count) / count * 1_000


if __name__ == "__main__":
    _test()
    print("tests: PASS")
    print("-" * 40)
    print("part_1:", part_1(input_file))
    print("part_2:", part_2(input_file))
    print("-" * 40)
    print("part_1 bench: {:.1f}ms".format(_bench(lambda: part_1(input_file), count=1)))
    print("part_2 bench: {:.1f}ms".format(_bench(lambda: part_2(input_file), count=1)))
