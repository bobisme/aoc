#!/usr/bin/env python

import math
from typing import DefaultDict, LiteralString, NamedTuple
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

Distances = list[tuple[float, tuple[int, int]]]


class Connections:
    map: DefaultDict[int, set[int]]
    # full circuits
    circuit_map: dict[int, set[int]]
    # map of set id (pointer) to set of box index which point to that set
    id_map: dict[int, set[int]]

    def __init__(self, boxes: list[Box]):
        self.map = DefaultDict(set)
        self.circuit_map = {i: {i} for i in range(len(boxes))}
        self.id_map = {id(s): {i} for i, s in self.circuit_map.items()}

    def __contains__(self, x) -> bool:
        return x in self.map

    def __getitem__(self, i: int) -> set[int]:
        return self.map[i]

    # TODO: This is broken. Doesn't work for p1, but works for p2.
    def merge_circuits(self, i: int, j: int):
        i_circuit = self.circuit_map[i]
        j_circuit = self.circuit_map[j]
        i_id, j_id = id(i_circuit), id(j_circuit)
        if i_id == j_id:
            return
        i_circuit |= j_circuit
        for other_i in self.id_map[j_id]:
            self.circuit_map[other_i] = i_circuit
        self.id_map[i_id].add(j)

    def connect(self, i: int, j: int):
        self.map[i].add(j)
        self.map[j].add(i)
        self.merge_circuits(i, j)

    def full_circuit(self, box_i: int) -> set[int]:
        circuit = {box_i}

        def expand(i: int):
            for other_box in self.map[i]:
                if other_box in circuit:
                    continue
                circuit.add(other_box)
                expand(other_box)

        expand(box_i)
        return circuit


def get_distances(boxes: list[Box]) -> Distances:
    distances = []
    for i in range(len(boxes) - 1):
        for j in range(i + 1, len(boxes)):
            distances.append((math.dist(boxes[i], boxes[j]), (i, j)))
    distances.sort(key=lambda x: x[0])
    return distances


def closest_boxes(
    distances: Distances, connections: Connections, offset=0
) -> tuple[tuple[int, int], int]:
    for idx in range(offset, len(distances)):
        _, (i, j) = distances[idx]
        if i in connections and j in connections[i]:
            continue
        return (i, j), idx
    raise ValueError("no more connections possible")


def part_1(input: Input, max_conn_count=1_000):
    boxes = [Box(*map(int, line.split(","))) for line in input]
    connections = Connections(boxes)
    distances = get_distances(boxes)

    offset = 0
    for _ in range(max_conn_count):
        (i, j), offset = closest_boxes(distances, connections, offset=offset)
        connections.connect(i, j)

    checked = set()
    circuits = []
    for i in range(len(boxes)):
        if i in checked:
            continue
        circuit = connections.full_circuit(i)
        circuits.append(circuit)
        checked |= circuit

    circuits.sort(key=lambda x: -len(x))

    return math.prod(len(c) for c in circuits[:3])


def part_2(input: Input):
    boxes = [Box(*map(int, line.split(","))) for line in input]
    connections = Connections(boxes)
    distances = get_distances(boxes)

    connection = (0, 0)
    offset = 0
    while True:
        connection, offset = closest_boxes(distances, connections, offset=offset)
        connections.connect(*connection)
        if len(connections.circuit_map[connection[0]]) >= len(boxes):
            break
    return boxes[connection[0]].x * boxes[connection[1]].x


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
    boxes = [Box(*map(int, line.split(","))) for line in input_file]
    print("-" * 40)
    print("part_1:", part_1(input_file))
    print("part_2:", part_2(input_file))
    print("-" * 40)
    print("part_1 bench: {:.1f}ms".format(_bench(lambda: part_1(input_file), count=1)))
    print("part_2 bench: {:.1f}ms".format(_bench(lambda: part_2(input_file), count=1)))
