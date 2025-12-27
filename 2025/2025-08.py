#!/usr/bin/env python

from collections import deque
from dataclasses import dataclass, field
from typing import Counter, Iterable, LiteralString, NamedTuple, Self
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


################################################################################
# The functions and classes below are not part of original implementation. Just
# for learning purposes.
################################################################################


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


type BoxId = int
type NodeId = int


class KDNode(NamedTuple):
    box_id: int
    box: Box
    left: NodeId | None = None
    right: NodeId | None = None


@dataclass(slots=True)
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
        self, k: int, query: Box, exclude: set[BoxId] | None = None
    ) -> list[BoxId]:
        exclude = exclude or set()
        heap = []

        def search(node_id: NodeId | None, depth: int):
            nonlocal heap

            if node_id is None:
                return

            node = self.nodes[node_id]

            if node.box_id not in exclude:
                dist = math.dist(query, node.box)
                if dist > 0:
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
        return list(self.nodes[n.node_id].box_id for n in sorted(heap))

    def nearest(self, query: Box, exclude: set[BoxId] | None = None) -> BoxId | None:
        node_list = self.k_nearest(1, query, exclude)
        if node_list:
            return node_list[0]

    def traverse(self) -> Iterable[BoxId]:
        def visit(node_id: NodeId | None) -> Iterable[BoxId]:
            if node_id is None:
                return
            node = self.nodes[node_id]
            yield from visit(node.left)
            yield node.box_id
            yield from visit(node.right)

        yield from visit(0)


def kdprim(boxes: list[Box], tree: KDTree) -> list[tuple[int, int]]:
    visited = [False] * len(boxes)
    heap = [(0.0, -1, 0)]
    edges: list[tuple[BoxId, BoxId]] = []
    explored: set[BoxId] = set()

    while heap and len(edges) < len(boxes) - 1:
        _, from_id, to_id = heapq.heappop(heap)
        if visited[to_id]:
            continue
        visited[to_id] = True
        explored.add(to_id)
        if from_id >= 0:
            edges.append((from_id, to_id))

        for other_id in tree.k_nearest(5, boxes[to_id], exclude=explored):
            if not visited[other_id]:
                d = math.dist(boxes[to_id], boxes[other_id])
                heapq.heappush(heap, (d, to_id, other_id))
    return edges


def part_1_kd(input: Input, max_conn_count=1_000):
    """Use KDTree to reduce search space."""
    boxes = [Box(*map(int, line.split(","))) for line in input]
    circuits = Circuits(len(boxes))
    edges: list[tuple[BoxId, BoxId]] = []
    tree = KDTree(boxes)
    seen: set[tuple[BoxId, BoxId]] = set()
    for i in range(len(boxes)):
        for j in tree.k_nearest(5, boxes[i]):
            edge = (min(i, j), max(j, i))
            if edge not in seen:
                seen.add(edge)
                edges.append((i, j))

    edges.sort(key=lambda x: math.dist(boxes[x[0]], boxes[x[1]]))
    for i, j in edges[:max_conn_count]:
        circuits.connect(i, j)

    return math.prod(c for c in sorted(circuits.size, reverse=True)[:3])


def part_2_kdprim(input: Input) -> int:
    """Feeds Prim's from a KDTree"""
    boxes = [Box(*map(int, line.split(","))) for line in input]
    tree = KDTree(boxes)
    connections = kdprim(boxes, tree)
    connections.sort(key=lambda x: math.dist(boxes[x[0]], boxes[x[1]]))
    a_id, b_id = connections[-1:][0]
    a, b = boxes[a_id], boxes[b_id]
    return a.x * b.x


@dataclass(slots=True)
class P3:
    x: float
    y: float
    z: float

    def fields(self) -> tuple[float, float, float]:
        return (self.x, self.y, self.z)

    def __add__(self, other: "P3") -> "P3":
        return P3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: "P3") -> "P3":
        return P3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, val: float) -> "P3":
        return P3(self.x * val, self.y * val, self.z * val)

    def __truediv__(self, val: float) -> "P3":
        return P3(self.x / val, self.y / val, self.z / val)

    def cross(self, other: "P3") -> "P3":
        return P3(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )

    def dot(self, other: "P3") -> float:
        return self.x * other.x + self.y * other.y + self.z * other.z

    def mag_sq(self) -> float:
        return self.x**2 + self.y**2 + self.z**2


@dataclass(slots=True)
class Face:
    triangle: tuple[P3, P3, P3]
    ids: tuple[BoxId, BoxId, BoxId]

    def __post_init__(self) -> None:
        t = sorted(enumerate(self.triangle), key=lambda x: self.ids[x[0]])
        self.triangle = (t[0][1], t[1][1], t[2][1])
        self.ids = (
            self.ids[t[0][0]],
            self.ids[t[1][0]],
            self.ids[t[2][0]],
        )

    def __hash__(self) -> int:
        return hash(self.ids)

    def __eq__(self, other: object, /) -> bool:
        return self.ids == getattr(other, "ids")


FACE_IDXS = (
    (0, 1, 2),
    (0, 1, 3),
    (2, 3, 0),
    (2, 3, 1),
)


def det3(m):
    """3x3 determinant via Rule of Sarrus"""
    return (
        m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
    )


def det4(m):
    """4x4 determinant via expansion by first row"""
    return (
        m[0][0] * det3([r[1:] for r in m[1:]])
        - m[0][1] * det3([[r[0]] + r[2:] for r in m[1:]])
        + m[0][2] * det3([[r[0], r[1], r[3]] for r in m[1:]])
        - m[0][3] * det3([r[:3] for r in m[1:]])
    )


def minor(M, col):
    return det4([[r[j] for j in range(5) if j != col] for r in M])


@dataclass(slots=True)
class Tetrahedron:
    points: tuple[P3, P3, P3, P3]
    box_ids: tuple[BoxId, BoxId, BoxId, BoxId]

    circumcenter: P3 = field(init=False)
    radius: float = field(init=False)

    def __post_init__(self):
        self.circumcenter, self.radius = self._calculate_circumsphere()

    def __hash__(self) -> int:
        return hash(self.box_ids)

    # def circumsphere(self) -> tuple[P3, float]:
    #     """
    #     Returns circumcenter, radius.
    #     https://mathworld.wolfram.com/Circumsphere.html
    #     """
    #     D = [[p.x**2 + p.y**2 + p.z**2, p.x, p.y, p.z, 1] for p in self.points]
    #     a = minor(D, 0)
    #     Dx = minor(D, 1)
    #     Dy = minor(D, 2)
    #     Dz = minor(D, 3)
    #
    #     cx = Dx / (2 * a)
    #     cy = -Dy / (2 * a)
    #     cz = Dz / (2 * a)
    #
    #     center = P3(cx, cy, cz)
    #     radius = math.dist(center.fields(), self.points[0].fields())
    #     return center, radius
    def _calculate_circumsphere(self) -> tuple[P3, float]:
        """
        Optimized vector algebra approach.
        Avoids 16 calls to determinant functions.
        """
        p0, p1, p2, p3 = self.points

        # Vectors relative to p0
        a = p1 - p0
        b = p2 - p0
        c = p3 - p0

        # Cross products
        axb = a.cross(b)
        bxc = b.cross(c)
        cxa = c.cross(a)

        # Denominator = 2 * (a . (b x c))
        # Note: scalar triple product is the volume of the parallelepiped
        denom = 2 * a.dot(bxc)

        if abs(denom) < 1e-9:
            # Degenerate tetra (coplanar points). Handle gracefully or raise.
            # Returning large radius acts as a safeguard.
            return P3(0.0, 0.0, 0.0), float("inf")

        # Vector formula for circumcenter relative to p0
        # ( |a|^2 (b x c) + |b|^2 (c x a) + |c|^2 (a x b) ) / denom
        v = (bxc * a.mag_sq() + cxa * b.mag_sq() + axb * c.mag_sq()) / denom

        center = p0 + v
        radius = math.dist(center.fields(), self.points[0].fields())
        # return center, v.mag_sq()
        return center, radius

    def is_point_inside_circumsphere(self, p: P3) -> bool:
        dist = math.dist(self.circumcenter.fields(), p.fields())
        return dist < self.radius

    def faces(self) -> Iterable[Face]:
        return (
            Face(
                (self.points[i], self.points[j], self.points[k]),
                (self.box_ids[i], self.box_ids[j], self.box_ids[k]),
            )
            for (i, j, k) in FACE_IDXS
        )

    def has_face(self, face: Face) -> bool:
        return any(face == f for f in self.faces())


def triangulate(boxes: list[Box]):
    """Bowyer-Watson algorithm of Delaunay Triangulation from Wikipedia."""
    points = [P3(*map(float, box)) for box in boxes]
    super_tetra = Tetrahedron(
        (
            P3(-200000.0, -200000.0, -200000.0),
            P3(400000.0, -200000.0, -200000.0),
            P3(50000.0, 400000.0, -200000.0),
            P3(-50000.0, -50000.0, 400000.0),
        ),
        (-1, -2, -3, -4),
    )
    mesh: set[Tetrahedron] = {super_tetra}
    for i, point in enumerate(points):
        print(i)
        bad_tetrahedra: set[Tetrahedron] = set()

        # find all tetrahedra that are no longer valid due to the insertion
        for tet in mesh:
            if tet.is_point_inside_circumsphere(point):
                bad_tetrahedra.add(tet)

        # find boundary of the polyhedral hole
        face_count = Counter(face for tet in bad_tetrahedra for face in tet.faces())
        polyhedral_boundary: list[Face] = [f for f, c in face_count.items() if c == 1]
        # for tet in bad_tetrahedra:
        #     for face in tet.faces():
        #         # if edge is not shared by any other tet in bad_tetrahedra, add
        #         # face to polyhedron
        #         is_shared = any(
        #             other is not tet and face in other.faces()
        #             for other in bad_tetrahedra
        #         )
        #         if not is_shared:
        #             polyhedral_boundary.append(face)

        # remove bad tets from mesh
        mesh -= bad_tetrahedra

        # re-triangulate the polyhedral hole
        for face in polyhedral_boundary:
            new_tet = Tetrahedron(
                points=((point,) + face.triangle),
                box_ids=((i,) + face.ids),
            )
            mesh.add(new_tet)

    # if tet contains vert from original super_tetra, remove tet from mesh
    # super_tetra point ids are all negative
    return {tet for tet in mesh if all(i >= 0 for i in tet.box_ids)}


@dataclass(slots=True)
class Mesh:
    tets: set[Tetrahedron] = field(default_factory=lambda: set())
    face_map: dict[Face, list[Tetrahedron]] = field(default_factory=lambda: {})

    def __contains__(self, tet: Tetrahedron) -> bool:
        return tet in self.tets

    def add(self, tet: Tetrahedron):
        self.tets.add(tet)

        for face in tet.faces():
            self.face_map.setdefault(face, []).append(tet)

    def remove(self, tet: Tetrahedron):
        self.tets.remove(tet)
        for face in tet.faces():
            if face in self.face_map:
                self.face_map[face].remove(tet)
                if not self.face_map[face]:
                    del self.face_map[face]


def triangulate_2(boxes: list[Box]):
    tree = KDTree(boxes)
    sorted_indices = list(tree.traverse())
    original_indices = sorted_indices.copy()
    points = [P3(*map(float, boxes[i])) for i in sorted_indices]
    super_points = (
        P3(-200000.0, -200000.0, -200000.0),
        P3(400000.0, -200000.0, -200000.0),
        P3(50000.0, 400000.0, -200000.0),
        P3(-50000.0, -50000.0, 400000.0),
    )
    super_tet = Tetrahedron(super_points, (-1, -2, -3, -4))

    mesh = Mesh()
    mesh.add(super_tet)

    last_tet = super_tet

    for i, point in enumerate(points):
        print(i)
        bad_tetrahedra = set()
        start_tet = last_tet
        if not start_tet.is_point_inside_circumsphere(point):
            q = deque([start_tet])
            visited = {start_tet}
            found = None
            search_limit = 50
            while q and search_limit > 0:
                curr = q.popleft()
                if curr.is_point_inside_circumsphere(point):
                    found = curr
                    break
                search_limit -= 1

                # add neighbors to queue
                for face in curr.faces():
                    neighbors = mesh.face_map.get(face, [])
                    for n in neighbors:
                        if n is not curr and n not in visited:
                            visited.add(n)
                            q.append(n)

            # if walk failed, scan everything
            if not found:
                for tet in mesh.tets:
                    if tet.is_point_inside_circumsphere(point):
                        found = tet
                        break
            start_tet = found

        assert start_tet is not None

        # expand cavity
        q_cavity = deque([start_tet])
        bad_tetrahedra.add(start_tet)

        while q_cavity:
            curr = q_cavity.popleft()
            for face in curr.faces():
                for n in mesh.face_map.get(face, []):
                    if n is not curr and n not in bad_tetrahedra:
                        if n.is_point_inside_circumsphere(point):
                            bad_tetrahedra.add(n)
                            q_cavity.append(n)
        # re-triangulate cavity
        boundary_faces: list[Face] = []
        for tet in bad_tetrahedra:
            for face in tet.faces():
                is_boundary = True
                for n in mesh.face_map.get(face, []):
                    if n is not tet and n in bad_tetrahedra:
                        is_boundary = False
                        break
                if is_boundary:
                    boundary_faces.append(face)

        # remove bad tets
        for tet in bad_tetrahedra:
            mesh.remove(tet)

        # create new tets connected to the point
        new_tets = []
        real_box_id = original_indices[i]

        for face in boundary_faces:
            new_tet = Tetrahedron(
                points=((point,) + face.triangle),
                box_ids=((real_box_id,) + face.ids),
            )
            new_tets.append(new_tet)
            mesh.add(new_tet)

        if new_tets:
            last_tet = new_tets[0]

    return {tet for tet in mesh.tets if all(i >= 0 for i in tet.box_ids)}


def part_1_triangulation(input: Input, max_conn_count=1_000):
    boxes = [Box(*map(int, line.split(","))) for line in input]
    mesh = triangulate_2(boxes)
    print(mesh)


def _test():
    def assert_eq(a, b):
        assert a == b, f"{a} != {b}"

    assert_eq(part_1(CONTROL_1, max_conn_count=10), 40)
    assert_eq(part_1_kd(CONTROL_1, max_conn_count=10), 40)
    # assert_eq(part_1_triangulation(CONTROL_1, max_conn_count=10), 40)
    assert_eq(part_2(CONTROL_1), 25272)
    assert_eq(part_2_prim(CONTROL_1), 25272)
    assert_eq(part_2_kdprim(CONTROL_1), 25272)


def run(fn, year=2025, day=8, part=0):
    start = time.perf_counter_ns()
    res = fn()
    elapsed_ns = time.perf_counter_ns() - start
    print(f"{year}\t{day}\t{part}\t{res}\t{elapsed_ns}")


if __name__ == "__main__":
    with open("2025-08.input") as f:
        input_file = [line.rstrip("\n") for line in f.readlines()]
    _test()
    # run(lambda: part_1(input_file), part=1)
    run(lambda: part_1_kd(input_file), part=1)
    # run(lambda: part_1_triangulation(input_file), part=1)
    # run(lambda: part_2(input_file), part=2)
    # run(lambda: part_2_prim(input_file), part=3)
    run(lambda: part_2_kdprim(input_file), part=2)
