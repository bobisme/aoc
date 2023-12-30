#!/usr/bin/env python
from collections import defaultdict
from pprint import pp
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from rich.progress import track

import graphviz


CONTROL_1 = """\
jqt: rhn xhk nvd
rsh: frs pzl lsr
xhk: hfx
cmg: qnr nvd lhk bvb
rhn: xhk bvb hfx
bvb: xhk hfx
pzl: lsr hfx nvd
qnr: nvd
ntq: jqt hfx bvb xhk
nvd: lhk
lsr: lhk
rzs: qnr cmg lsr rsh
frs: qnr lhk lsr
""".splitlines()

with open("2023-25.input") as f:
    input_file = [line.strip() for line in f.readlines()]


def parse_line(line):
    left, right = line.split(": ")
    right = set(right.split(" "))
    return left, right


def edge_key(u: str, v: str) -> tuple[str, str]:
    if u < v:
        return (u, v)
    return (v, u)


@dataclass
class Graph:
    adjacency_list: dict[str, set[str]] = field(default_factory=dict)
    edges: dict[tuple[str, str], int] = field(default_factory=dict)

    def copy(self) -> "Graph":
        new = Graph()
        new.adjacency_list = dict((k, v.copy()) for k, v in self.adjacency_list.items())
        new.edges = self.edges.copy()
        return new

    def add_edge(self, u: str, v: str) -> None:
        """Add an edge between vertices u and v."""
        if u not in self.adjacency_list:
            self.adjacency_list[u] = set()
        if v not in self.adjacency_list:
            self.adjacency_list[v] = set()
        self.adjacency_list[u].add(v)
        self.adjacency_list[v].add(u)
        self.edges[edge_key(u, v)] = 1

    def vertex_count(self) -> int:
        """Return the number of vertices in the graph."""
        return len(self.adjacency_list)

    def edge_count(self) -> int:
        """Return the number of edges in the graph."""
        return sum(self.edges.values())

    def random_edge(self) -> tuple[str, str]:
        return random.choice(list(self.edges.keys()))

    def vert_exists(self, vert: str) -> bool:
        if vert in self.adjacency_list:
            return True
        for connections in self.adjacency_list.values():
            if vert in connections:
                return True
        return False

    def tightest_to(self, v: str) -> Optional[str]:
        neighbors = list(self.adjacency_list[v])
        if not neighbors:
            return None
        return max(neighbors, key=lambda u: self.edges[edge_key(u, v)])

    def merge_vertices(self, u: str, v: str) -> None:
        """Merge the vertices u and v into a single vertex."""
        if u not in self.adjacency_list or v not in self.adjacency_list:
            return  # Edge does not exist

        # Merge the neighbors of u and v, excluding u and v themselves
        u_neighbors = self.adjacency_list[u]
        v_neighbors = self.adjacency_list[v]
        new_neighbors = set(u_neighbors | v_neighbors) - {u, v}

        new_weights = defaultdict(int)
        for un in u_neighbors:
            key = edge_key(u, un)
            new_weights[un] = self.edges[key]
        for vn in v_neighbors:
            key = edge_key(v, vn)
            new_weights[vn] += self.edges[key]
        for un in u_neighbors:
            del self.edges[edge_key(u, un)]
        for vn in v_neighbors:
            key = edge_key(v, vn)
            if key in self.edges:
                del self.edges[key]

        # Choose a new name for the merged vertex
        new_vertex = u + "_" + v

        for vert, w in new_weights.items():
            self.edges[edge_key(new_vertex, vert)] = w

        # Remove u and v from the graph
        del self.adjacency_list[u]
        del self.adjacency_list[v]

        # Update the adjacency list for the remaining vertices
        for neighbors in self.adjacency_list.values():
            if u in neighbors:
                neighbors.remove(u)
                neighbors.add(new_vertex)
            if v in neighbors:
                neighbors.remove(v)
                neighbors.add(new_vertex)

        # Add the new merged vertex to the graph
        self.adjacency_list[new_vertex] = new_neighbors


def count_keys(key: str) -> int:
    return len(key.split("_"))


def cut_size(cut: tuple[str, set[str]]) -> int:
    return count_keys(cut[0]) + sum(count_keys(x) for x in cut[1])


def cut_balance(cut: tuple[str, set[str]]) -> int:
    return abs(count_keys(cut[0]) - sum(count_keys(x) for x in cut[1]))


def contract(graph: Graph):
    G = graph.copy()
    while G.vertex_count() > 2:
        u, v = G.random_edge()
        G.merge_vertices(u, v)
    return G


def stoer_wagner(graph: Graph):
    g_size = graph.vertex_count()
    global_min_cut = None
    global_min_cut_w = 1_000_000_000
    verts = list(graph.adjacency_list.keys())
    verts.sort()

    exclude = set()
    # cuts = []

    def inner(vert):
        G = graph.copy()

        def min_cut_phase(vert=None):
            g_vert_count = G.vertex_count()
            # if vert is None or vert not in G.adjacency_list:
            vert = max(G.adjacency_list.keys(), key=lambda x: len(G.adjacency_list[x]))
            last, next_last = vert, vert
            for _ in range(1, g_vert_count):
                tightest = G.tightest_to(last)
                if tightest is None:
                    break
                next_last = last
                last = tightest
            last_cut = (next_last, last)
            weight = G.edges[edge_key(*last_cut)]
            G.merge_vertices(last, next_last)
            return last_cut, weight, last

        last_cut, weight, last = None, 0, None
        while G.vertex_count() > 1:
            last_cut, weight, last = min_cut_phase(vert)
            # vert = None
        return last_cut, weight, last

    for vert_idx in track(range(g_size)):
        # cut, weight = inner(verts[vert_idx])
        cut, weight, last = inner(None)
        print(f"{cut=} {weight=}")
        if global_min_cut is None or weight < global_min_cut_w:
            global_min_cut = cut
            global_min_cut_w = weight

    # pp(cuts)
    return global_min_cut, global_min_cut_w


def part_1(input):
    # for line in input:
    #     print(line)
    adj_list = dict(parse_line(line) for line in input)
    # pp(adj_list)
    # dot = graphviz.Digraph(format="png", graph_attr={"layout": "neato"})
    # for left, right in adj_list.items():
    #     for r in right:
    #         dot.edge(left, r, dir="none")
    # dot.save("2023-25.dot")
    # dot.render("2023-25", format="png")
    # print(dot)
    graph = Graph()
    for vert, connections in adj_list.items():
        for conn in connections:
            graph.add_edge(vert, conn)

    # print(f"{init_vert_count=} {graph.edge_count()=}")
    # print(f"{graph.most_tightly_connected_vertex()=}")
    min_cut, weight = stoer_wagner(graph)
    print(f"{min_cut=} {weight=}")
    a, b = count_keys(min_cut[0]), count_keys(min_cut[1])
    print(f"{a=} {b=} {a*b=}")
    # print(f"{graph.vertex_count()=} {graph.edge_count()=}")
    # print(graph.adjacency_list)


if __name__ == "__main__":
    part_1(input_file)
