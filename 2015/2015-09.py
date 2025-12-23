#!/usr/bin/env python

from pprint import pformat
from typing import DefaultDict, LiteralString
import time


Input = list[str] | list[LiteralString]

CONTROL_1: Input = (
    """\
London to Dublin = 464
London to Belfast = 518
Dublin to Belfast = 141
""".splitlines()
)


class Graph:
    nodes: set[str]
    edges: dict[tuple[str, str], int]
    adj: DefaultDict[str, list[str]]

    def __init__(self) -> None:
        self.nodes = set()
        self.edges = {}
        self.adj = DefaultDict(list)

    def __repr__(self) -> str:
        return pformat(self.edges)

    def __getitem__(self, key: tuple[str, str]) -> int:
        a, b = key
        if a > b:
            a, b = b, a
        return self.edges[(a, b)]

    def add(self, a: str, b: str, dist: int):
        self.nodes.add(a)
        self.nodes.add(b)
        if a > b:
            a, b = b, a
        self.edges[(a, b)] = dist
        self.adj[a].append(b)
        self.adj[b].append(a)

    def nearest(self, node: str, remaining: set[str]) -> tuple[str, int]:
        assert remaining, "remaining is empty"
        best_dist = 10**10
        best_node = node
        for n in remaining:
            if n == node:
                continue
            if (dist := self[node, n]) < best_dist:
                best_dist = dist
                best_node = n
        return best_node, best_dist

    def farthest(self, node: str, remaining: set[str]) -> tuple[str, int]:
        assert remaining, "remaining is empty"
        best_dist = -1
        best_node = node
        for n in remaining:
            if n == node:
                continue
            if (dist := self[node, n]) > best_dist:
                best_dist = dist
                best_node = n
        return best_node, best_dist


def parse(input: Input) -> Graph:
    G = Graph()
    for line in input:
        start, rest = line.split(" to ")
        dest, dist = rest.split(" = ")
        dist = int(dist)
        G.add(start, dest, dist)
    return G


def part_1(input: Input):
    G = parse(input)

    def greedy(start: str):
        remaining = G.nodes.copy()
        node = start
        while len(remaining) > 1:
            remaining.remove(node)
            node, dist = G.nearest(node, remaining)
            yield node, dist

    shortest = 10**10
    for start in G.nodes:
        path = [(start, 0)] + list(greedy(start))
        if (total := sum(x[1] for x in path)) < shortest:
            shortest = total

    return shortest


def part_2(input: Input):
    G = parse(input)

    def greedy(start: str):
        remaining = G.nodes.copy()
        node = start
        while len(remaining) > 1:
            remaining.remove(node)
            node, dist = G.farthest(node, remaining)
            yield node, dist

    farthest = -1
    for start in G.nodes:
        path = [(start, 0)] + list(greedy(start))
        if (total := sum(x[1] for x in path)) > farthest:
            farthest = total

    return farthest


def _test():
    def assert_eq(a, b):
        assert a == b, f"{a} != {b}"

    assert_eq(part_1(CONTROL_1), 605)


def run(fn, year=2015, day=9, part=0):
    start = time.perf_counter_ns()
    res = fn()
    elapsed_ns = time.perf_counter_ns() - start
    print(f"{year}\t{day}\t{part}\t{res}\t{elapsed_ns}")
    return res


if __name__ == "__main__":
    with open("2015-09.input") as f:
        input_file = [line.rstrip("\n") for line in f.readlines()]
    _test()
    run(lambda: part_1(input_file), part=1)
    run(lambda: part_2(input_file), part=2)
