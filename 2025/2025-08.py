#!/usr/bin/env python

import math
from typing import DefaultDict, Generator, LiteralString, NamedTuple
import timeit

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

with open("2025-08.input") as f:
    input_file = [line.rstrip("\n") for line in f.readlines()]

Box = NamedTuple("Box", [("x", int), ("y", int), ("z", int)])


def part_1(input: Input, max_conn_count=1_000):
    boxes = [Box(*map(int, line.split(","))) for line in input]
    connections: DefaultDict[int, set[int]] = DefaultDict(set)

    def calculate_distances():
        map = {}
        for i in range(len(boxes) - 1):
            for j in range(i + 1, len(boxes)):
                map[(i, j)] = math.dist(boxes[i], boxes[j])
        return map

    distances = calculate_distances()
    by_distance = list(
        sorted(
            ((dist, (i, j)) for ((i, j), dist) in distances.items()), key=lambda x: x[0]
        )
    )

    def closest_boxes() -> tuple[int, int]:
        for _, (i, j) in by_distance:
            if i in connections and j in connections[i]:
                continue
            return (i, j)
        raise ValueError("crap")

    def connect(i: int, j: int):
        # print(f"connecting: {boxes[i]} and {boxes[j]}")
        connections[i].add(j)
        connections[j].add(i)

    for _ in range(max_conn_count):
        (i, j) = closest_boxes()
        connect(i, j)

    def get_circuit(box_i: int) -> set[int]:
        circuit = {box_i}

        def expand(i: int):
            for other_box in connections[i]:
                if other_box in circuit:
                    continue
                circuit.add(other_box)
                expand(other_box)

        expand(box_i)
        return circuit

    checked = set()
    circuits = []
    for i in range(len(boxes)):
        if i in checked:
            continue
        circuit = get_circuit(i)
        circuits.append(circuit)
        checked |= circuit

    circuits.sort(key=lambda x: -len(x))

    return math.prod(len(c) for c in circuits[:3])


def part_2(input: Input):
    boxes = [Box(*map(int, line.split(","))) for line in input]
    connections: DefaultDict[int, set[int]] = DefaultDict(set)

    def calculate_distances():
        map = {}
        for i in range(len(boxes) - 1):
            for j in range(i + 1, len(boxes)):
                map[(i, j)] = math.dist(boxes[i], boxes[j])
        return map

    distances = calculate_distances()
    by_distance = list(
        sorted(
            ((dist, (i, j)) for ((i, j), dist) in distances.items()), key=lambda x: x[0]
        )
    )

    def closest_boxes() -> tuple[int, int]:
        for _, (i, j) in by_distance:
            if (
                i in connections
                and j in connections[i]
                or j in connections
                and i in connections[j]
            ):
                continue
            return (i, j)
        raise ValueError("no more boxes to connect")

    def connect(i: int, j: int):
        # print(f"connecting: {boxes[i]} and {boxes[j]}")
        connections[i].add(j)
        connections[j].add(i)

    def get_circuit(box_i: int) -> set[int]:
        circuit = {box_i}

        def expand(i: int):
            for other_box in connections[i]:
                if other_box in circuit:
                    continue
                circuit.add(other_box)
                expand(other_box)

        expand(box_i)
        return circuit

    last_connection = (0, 0)
    while True:
        (i, j) = closest_boxes()
        if (i, j) == (0, 0):
            break
        if len(get_circuit(i)) >= len(boxes):
            break
        last_connection = (i, j)
        connect(i, j)
    return boxes[last_connection[0]].x * boxes[last_connection[1]].x


def _test():
    def assert_eq(a, b):
        assert a == b, f"{a} != {b}"

    assert_eq(part_1(CONTROL_1, max_conn_count=10), 40)
    assert_eq(part_2(CONTROL_1), 25272)


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
