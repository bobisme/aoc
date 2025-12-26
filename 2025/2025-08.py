#!/usr/bin/env python

from dataclasses import dataclass
import math
from typing import DefaultDict, LiteralString, NamedTuple
import time

Input = list[str] | list[LiteralString]

CONTROL_1: Input = (
    """\
162,817,812
57,618,57
906,360,560
592,479,940
352,342,300
466,668,158
542,29,236
431,825,988
739,650,466
52,470,668
216,146,977
819,987,18
117,168,530
805,96,715
346,949,466
970,615,88
941,993,340
862,61,35
984,92,344
425,690,689
""".splitlines()
)


Box = NamedTuple("Box", [("x", int), ("y", int), ("z", int)])

Distances = list[tuple[float, tuple[int, int]]]


def get_distances(boxes: list[Box]) -> Distances:
    distances = []
    for i in range(len(boxes) - 1):
        for j in range(i + 1, len(boxes)):
            distances.append((math.dist(boxes[i], boxes[j]), (i, j)))
    distances.sort(key=lambda x: x[0])
    return distances


def closest_boxes(
    distances: Distances, connections: set[tuple[int, int]], offset=0
) -> tuple[tuple[int, int], int]:
    for idx in range(offset, len(distances)):
        _, (i, j) = distances[idx]
        if (i, j) in connections:
            continue
        return (i, j), idx + 1
    raise ValueError("no more connections possible")


@dataclass
class Circuits:
    def __init__(self, count: int) -> None:
        self.parent = list(range(count))
        self.size = [1] * count
        self.connections = set()

    def get_root(self, id: int) -> int:
        if self.parent[id] == id:
            return id
        self.parent[id] = self.get_root(self.parent[id])
        return self.parent[id]

    def merge(self, x_id: int, y_id: int) -> int:
        x_id = self.get_root(x_id)
        y_id = self.get_root(y_id)
        if x_id == y_id:
            return x_id
        if self.size[x_id] < self.size[y_id]:
            x_id, y_id = y_id, x_id
        self.parent[y_id] = x_id
        self.size[x_id] += self.size[y_id]
        self.size[y_id] = 0
        return x_id

    def connect(self, x_id: int, y_id: int) -> int:
        if x_id > y_id:
            x_id, y_id = y_id, x_id
        self.connections.add((x_id, y_id))
        return self.merge(x_id, y_id)


def part_1(input: Input, max_conn_count=1_000):
    boxes = [Box(*map(int, line.split(","))) for line in input]
    distances = get_distances(boxes)
    circuits = Circuits(len(boxes))

    offset = 0
    for _ in range(max_conn_count):
        (i, j), offset = closest_boxes(distances, circuits.connections, offset=offset)
        circuits.connect(i, j)

    return math.prod(c for c in sorted(circuits.size, reverse=True)[:3])


def part_2(input: Input):
    """
    Continue connecting the closest unconnected pairs of junction boxes
    together until they're all in the same circuit. What do you get if you
    multiply together the X coordinates of the last two junction boxes you need
    to connect?
    """

    boxes = [Box(*map(int, line.split(","))) for line in input]
    distances = get_distances(boxes)
    circuits = Circuits(len(boxes))

    offset = 0
    while True:
        connection, offset = closest_boxes(
            distances, circuits.connections, offset=offset
        )
        root_id = circuits.connect(*connection)
        i, j = connection
        if circuits.size[root_id] >= len(boxes):
            return boxes[i].x * boxes[j].x


def _test():
    def assert_eq(a, b):
        assert a == b, f"{a} != {b}"

    assert_eq(part_1(CONTROL_1, max_conn_count=10), 40)
    assert_eq(part_2(CONTROL_1), 25272)


def run(fn, year=2025, day=8, part=0):
    start = time.perf_counter_ns()
    res = fn()
    elapsed_ns = time.perf_counter_ns() - start
    print(f"{year}\t{day}\t{part}\t{res}\t{elapsed_ns}")


if __name__ == "__main__":
    with open("2025-08.input") as f:
        input_file = [line.rstrip("\n") for line in f.readlines()]
    _test()
    run(lambda: part_1(input_file), part=1)
    run(lambda: part_2(input_file), part=2)
