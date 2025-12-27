#!/usr/bin/env python

from dataclasses import dataclass
from typing import LiteralString, NamedTuple
import heapq
import math
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


def prim(boxes: list[Box]) -> list[tuple[int, int]]:
    """Prim's algorithm from Wikipedia."""
    cheapest_cost = {i: float("inf") for i in range(len(boxes))}
    cheapest_edge: dict[int, tuple[int, int] | None] = {
        i: None for i in range(len(boxes))
    }
    explored = set()
    unexplored = set(range(len(boxes)))
    cheapest_cost[0] = 0

    while unexplored:
        current_box_id = min(unexplored, key=lambda x: cheapest_cost[x])
        current_box = boxes[current_box_id]
        unexplored.remove(current_box_id)
        explored.add(current_box_id)

        for other_id in (i for i in range(len(boxes)) if i != current_box_id):
            other = boxes[other_id]
            if (
                other_id in unexplored
                and (dist := math.dist(current_box, other)) < cheapest_cost[other_id]
            ):
                cheapest_cost[other_id] = dist
                cheapest_edge[other_id] = (current_box_id, other_id)

    edges = []
    for i, _ in enumerate(boxes):
        if (edge := cheapest_edge[i]) is not None:
            edges.append(edge)
    return edges


def part_2_prim(input: Input) -> int:
    """173ms vs 605ms of previous part 2."""

    boxes = [Box(*map(int, line.split(","))) for line in input]
    connections = prim(boxes)
    connections.sort(key=lambda x: math.dist(boxes[x[0]], boxes[x[1]]))
    a_id, b_id = connections[-1:][0]
    a, b = boxes[a_id], boxes[b_id]
    return a.x * b.x


NodeId = int


class KDNode(NamedTuple):
    box_id: int
    box: Box
    left: NodeId | None = None
    right: NodeId | None = None


@dataclass
class KDHeapNode:
    dist: float
    node_id: NodeId

    def __lt__(self, other: "KDHeapNode"):
        return self.dist > other.dist

    def __gt__(self, other: "KDHeapNode"):
        return self.dist < other.dist


class KDTree:
    nodes: list[KDNode]

    def __init__(self, boxes: list[Box]):
        self.nodes = []
        indexed = list(enumerate(boxes))

        def build(indexed: list[tuple[int, Box]], depth: int) -> NodeId | None:
            if not indexed:
                return None
            axis = depth % 3
            indexed.sort(key=lambda x: x[1][axis])
            mid = len(indexed) // 2
            node_id = len(self.nodes)
            self.nodes.append(KDNode(*indexed[mid]))
            node = KDNode(
                box_id=indexed[mid][0],
                box=indexed[mid][1],
                left=build(indexed[:mid], depth + 1),
                right=build(indexed[mid + 1 :], depth + 1),
            )
            self.nodes[node_id] = node
            return node_id

        build(indexed, 0)

    def k_nearest(
        self, k: int, query: Box, exclude: set[NodeId] | None = None
    ) -> list[NodeId]:
        exclude = exclude or set()
        heap = []

        def search(node_id: NodeId | None, depth: int):
            nonlocal heap

            if node_id is None:
                return

            node = self.nodes[node_id]
            dist = math.dist(query, node.box)

            if dist > 0 and node_id not in exclude:
                if len(heap) < k:
                    heapq.heappush(heap, KDHeapNode(dist, node_id))
                elif dist < heap[0].dist:
                    heapq.heapreplace(heap, KDHeapNode(dist, node_id))

            axis = depth % 3
            diff = query[axis] - node.box[axis]
            near, far = (node.left, node.right) if diff < 0 else (node.right, node.left)
            search(near, depth + 1)

            if not heap or abs(diff) < heap[0].dist:
                search(far, depth + 1)

        search(0, 0)
        return list(n.node_id for n in sorted(heap))


class PrimNode(NamedTuple):
    dist: float
    from_box_id: int
    to_box_id: int


def kdprim(boxes: list[Box], tree: KDTree) -> list[tuple[int, int]]:
    visited = [False] * len(boxes)
    heap = [PrimNode(0, -1, 0)]
    edges = []
    box_to_node = {node.box_id: i for i, node in enumerate(tree.nodes)}
    explored = set()

    while heap and len(edges) < len(boxes) - 1:
        _, from_id, to_id = heapq.heappop(heap)
        if visited[to_id]:
            continue
        visited[to_id] = True
        explored.add(box_to_node[to_id])
        if from_id >= 0:
            edges.append((from_id, to_id))

        for node_id in tree.k_nearest(5, boxes[to_id], exclude=explored):
            other_id = tree.nodes[node_id].box_id
            if not visited[other_id]:
                d = math.dist(boxes[to_id], boxes[other_id])
                heapq.heappush(heap, PrimNode(d, to_id, other_id))
    return edges


def part_2_kdprim(input: Input) -> int:
    """Feeds Prim's from a KDTree. 69ms... nice"""
    boxes = [Box(*map(int, line.split(","))) for line in input]
    tree = KDTree(boxes)
    connections = kdprim(boxes, tree)
    connections.sort(key=lambda x: math.dist(boxes[x[0]], boxes[x[1]]))
    a_id, b_id = connections[-1:][0]
    a, b = boxes[a_id], boxes[b_id]
    return a.x * b.x


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
    # run(lambda: part_2(input_file), part=2)
    # run(lambda: part_2_prim(input_file), part=3)
    run(lambda: part_2_kdprim(input_file), part=2)
