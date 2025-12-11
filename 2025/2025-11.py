#!/usr/bin/env python

from collections import deque
from typing import DefaultDict, LiteralString
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

with open("2025-11.input") as f:
    input_file = [line.rstrip("\n") for line in f.readlines()]

Graph = dict[str, set[str]]


def parse(input: Input) -> Graph:
    g = {}
    for line in input:
        src, rest = line.split(": ")
        g[src] = {dst for dst in rest.split(" ")}
    return g


def part_1(input: Input):
    g = parse(input)
    q = deque()
    q.append(("you", set()))
    visited = set()
    path_count = 0
    while q:
        node, path = q.popleft()
        if node == "out":
            path_count += 1
            continue
        if node in visited:
            continue
        next_path = path | {node}
        for next_node in g[node]:
            if next_node in next_path:
                continue
            q.append((next_node, next_path))

    return path_count


def part_2(input: Input):
    # for line in input:
    #     print(line)
    return 0


def _test():
    def assert_eq(a, b):
        assert a == b, f"{a} != {b}"

    assert_eq(part_1(CONTROL_1), 5)
    # assert_eq(part_2(CONTROL_1), 0)


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
