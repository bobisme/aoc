#!/usr/bin/env python
from collections import defaultdict, namedtuple
from dataclasses import dataclass
from itertools import groupby, pairwise
from typing import NamedTuple, Optional, Union, Self
import sys
from pprint import pp
from interval import interval

from graphviz import Digraph
import humanize

CONTROL_1 = """\
seeds: 79 14 55 13

seed-to-soil map:
50 98 2
52 50 48

soil-to-fertilizer map:
0 15 37
37 52 2
39 0 15

fertilizer-to-water map:
49 53 8
0 11 42
42 0 7
57 7 4

water-to-light map:
88 18 7
18 25 70

light-to-temperature map:
45 77 23
81 45 19
68 64 13

temperature-to-humidity map:
0 69 1
1 0 69

humidity-to-location map:
60 56 37
56 93 4
""".splitlines()

with open("2023-5.input") as f:
    input = [line.strip() for line in f.readlines()]

INFINITY = "∞"

PROGRESSION = (
    "seed",
    "soil",
    "fertilizer",
    "water",
    "light",
    "temperature",
    "humidity",
    "location",
)


MAP_PROGRESSION: dict[str, str] = dict(pairwise(PROGRESSION))
# ((src, des), group_idx) -> (in, out)
MinCache = dict[tuple[tuple[str, str], int], tuple[int, int]]
MIN_CACHE: MinCache = {}


def range_repr(r):
    end = r.stop - 1
    if end >= sys.maxsize >> 1:
        end = INFINITY
    return f"{r.start}–{end}"


@dataclass
class Range:
    inner: range

    @property
    def start(self):
        return self.inner.start

    @property
    def stop(self):
        return self.inner.stop

    def __hash__(self):
        return hash((self.inner.start, self.inner.stop))

    def __repr__(self):
        start = humanize.intcomma(self.start)
        end = self.stop - 1
        if end >= sys.maxsize >> 1:
            end = INFINITY
        else:
            end = humanize.intcomma(end)
        return f"{start}–{end}"

    def __len__(self):
        return len(self.inner)

    def __and__(self, other) -> Optional[Self]:
        r = range_overlap(self, other)
        if len(r) == 0:
            return None
        return r

    def __sub__(self, other: Self) -> tuple[Optional[Self], Optional[Self]]:
        overlap = range_overlap(self, other)
        if len(overlap) == 0:
            return None, None
        a, b = self, other
        r1 = Range(range(a.start, b.start))
        if len(r1) == 0:
            r1 = None
        r2 = Range(range(b.stop, a.stop))
        if len(r2) == 0:
            r2 = None
        return r1, r2

    def contains(self, x: int) -> bool:
        return self.start <= x < self.stop

    @classmethod
    def from_(cls, start: int, stop: int) -> Self:
        return cls(range(start, stop))

    def interval(self) -> interval:
        return interval[self.start, self.stop - 1]


def range_overlap(a: Union[range, Range], b: Union[range, Range]) -> Range:
    return Range.from_(max(a.start, b.start), min(a.stop, b.stop))


_ina, _inb = Range(range(0, 101)), Range(range(25, 76))
assert _ina - _inb == (Range(range(0, 25)), Range(range(76, 101)))
_ina, _inb = Range(range(0, 101)), Range(range(25, 120))
assert _ina - _inb == (Range(range(0, 25)), None)
_ina, _inb = Range(range(25, 101)), Range(range(0, 30))
assert _ina - _inb == (None, Range(range(30, 101)))
_ina, _inb = Range(range(25, 30)), Range(range(20, 35))
assert _ina - _inb == (None, None)


@dataclass
class Ranges:
    src: list[Range]
    dst: list[Range]


@dataclass
class RangeMap:
    dst: Range
    src: Range

    def __hash__(self):
        return hash((self.dst, self.src))

    def __repr__(self):
        return f"{self.src} -> {self.dst}"

    @staticmethod
    def parse(line: str):
        dest, src, len_ = [int(x) for x in line.split(" ", 2)]
        return RangeMap(
            dst=Range(range(dest, dest + len_ + 1)),
            src=Range(range(src, src + len_ + 1)),
        )

    def map(self, x: int) -> Optional[int]:
        if not self.src.contains(x):
            return None
        return x - self.src.start + self.dst.start


@dataclass
class MapGroup:
    src: str
    dest: str
    range_maps: list[RangeMap]

    def __hash__(self):
        return hash((self.src, self.dest))

    def map(self, x: int):
        for range_map in self.range_maps:
            mapped = range_map.map(x)
            if mapped is not None:
                return mapped
        return x


def parse_ranges_line(line: str) -> tuple[Range, Range]:
    fro, to, n = [int(x) for x in line.split(" ")]
    return Range.from_(fro, fro + n), Range.from_(to, to + n)


assert parse_ranges_line("50 98 2") == (Range(range(50, 52)), Range(range(98, 100)))


def parse_range_name(line: str) -> tuple[str, str]:
    a, b = line.replace(" map:", "").split("-to-", 1)
    return a, b


assert parse_range_name("soil-to-fertilizer") == ("soil", "fertilizer")


def parse_ranges_in_map(input: list[str]) -> tuple[str, Ranges]:
    from_, to = parse_range_name(input[0])
    srcs = []
    dsts = []
    for line in input[1:]:
        if not line:
            break
        src, dst = parse_ranges_line(line)
        srcs.append(src)
        dsts.append(dst)
    return from_, Ranges(srcs, dsts)


# assert parse_ranges_in_map("""\
# light-to-temperature map:
# 45 77 23
# 81 45 19
# 68 64 13
# """) == ("light", Ranges([45, ]))


def parse_maps(input: list[str], index: int) -> MapGroup:
    name = input[index].replace(" map:", "")
    i = 1
    range_maps = []
    while index + i < len(input) and input[index + i]:
        line = input[index + i]
        range_maps.append(RangeMap.parse(line))
        i += 1
    src_name, dest_name = name.split("-to-")
    return MapGroup(src=src_name, dest=dest_name, range_maps=range_maps)


def to_location(x: int, maps: dict[tuple[str, str], MapGroup]) -> int:
    for src, dst in MAP_PROGRESSION.items():
        next = maps[(src, dst)].map(x)
        # print(f"{x} {src} -> {next} {dst}", file=stderr)
        x = next
    return x


def range_map_lowest(
    map_group: MapGroup, min_inputs: MinCache, map_key: tuple[str, str], x: int
) -> tuple[int, bool]:
    """\
    -> (out, is_from_cache)
    """
    for i, range_map in enumerate(map_group.range_maps):
        next = range_map.map(x)
        if next is None:
            continue
        # print(f"{x} is valid for range_map", file=stderr)
        cached = min_inputs.get((map_key, i))
        if cached is not None:
            lowest_in, lowest_out = cached
            if x >= lowest_in:
                # print(f"{x} too high", file=stderr)
                return lowest_out, True
        # print(f"{x} is lower than cached", file=stderr)
        min_inputs[map_key, i] = (x, next)
        return next, False
    return x, False


def to_lowest_location(
    x: int,
    maps: dict[tuple[str, str], MapGroup],
    cache: MinCache,
) -> Optional[int]:
    for src, dst in MAP_PROGRESSION.items():
        map_key = (src, dst)
        map_group = maps[map_key]
        next, is_from_cache = range_map_lowest(map_group, cache, map_key, x)
        # if next is None:
        #     print(f"{x} {src} -> too high", file=stderr)
        #     return None
        # print(f"{x} {src} -> {next} {dst}")
        x = next
    return x


def main(input):
    maps = {}
    seeds = [int(x) for x in input[0].replace("seeds: ", "").split(" ")]
    print("seeds =", seeds)
    i = 2
    while i < len(input):
        group = parse_maps(input, i)
        maps[(group.src, group.dest)] = group
        # print("maps = ", name, m, file=stderr)
        i += len(group.range_maps) + 2
    # pp(maps)
    lowest = None
    for seed in seeds:
        location = to_location(seed, maps)
        if lowest is None:
            lowest = location
        else:
            lowest = min(lowest, location)
    print(lowest)


def parse(input: list[str]) -> tuple[list[Range], dict[str, Ranges]]:
    map_groups = {}
    s = [int(x) for x in input[0].replace("seeds: ", "").split(" ")]
    seeds_ranges = []
    for i in range(0, len(s), 2):
        seeds_ranges.append(Range.from_(s[i], s[i] + s[i + 1]))
    # pp(seeds_ranges)
    # exit(0)
    i = 2

    while i < len(input):
        group = parse_maps(input, i)
        # map_groups[group.src] = group
        src_ranges = [rm.src for rm in group.range_maps]
        dst_ranges = [rm.dst for rm in group.range_maps]
        ranges = Ranges(src_ranges, dst_ranges)
        map_groups[group.src] = ranges
        # print("maps = ", name, m)
        i += len(group.range_maps) + 2
    return seeds_ranges, map_groups


def parse2(input: list[str]) -> tuple[list[Range], dict[str, Ranges]]:
    s = [int(x) for x in input[0].replace("seeds: ", "").split(" ")]
    seeds_ranges = []
    for i in range(0, len(s), 2):
        seeds_ranges.append(Range.from_(s[i], s[i] + s[i + 1]))
    ranges = {}
    i = 2
    while i < len(input):
        stage, r = parse_ranges_in_map(input[i:])
        ranges[stage] = r
        i += len(r.src) + 2
    return seeds_ranges, ranges


Child = NamedTuple(
    "Child", [("sub_range", Range), ("linked_range", Optional[Range]), ("label", str)]
)


def map_layer(
    src_range: Range, dst_ranges: list[Range], label: str = ""
) -> tuple[list[Child], list[Child]]:
    mapped: list[Child] = []
    unmapped: list[Child] = []
    for dst_range in dst_ranges:
        overlap = src_range & dst_range
        if overlap is None:
            continue
        mapped.append(Child(overlap, dst_range, label))
        a, b = src_range - overlap
        if a:
            unmapped.append(Child(a, None, label))
        if b:
            unmapped.append(Child(b, None, label))
    for u in unmapped:
        print("unmapped", u)
    return mapped, unmapped


def map_layer_r(
    src_range: Range, dst_stage: str, map_groups
) -> tuple[list[Child], list[Child]]:
    mapped, unmapped = map_layer(src_range, map_groups[dst_stage].src, label=dst_stage)
    for child in unmapped:
        dst_stage = MAP_PROGRESSION[dst_stage]
        if dst_stage == "location":
            return mapped, unmapped
        m, unmapped = map_layer_r(child.sub_range, dst_stage, map_groups)
        mapped.extend(m)
    return mapped, unmapped


Node = NamedTuple("Node", [("id", str), ("label", str), ("rank", str)])
Edge = NamedTuple("Edge", [("id1", str), ("id2", str), ("range", Range)])


def main2(input):
    dot = Digraph(
        format="svg",
        graph_attr=dict(
            rankdir="LR",
            fontname="Berkeley Mono Trial, FantasqueSansM Nerd Font",
        ),
        node_attr=dict(
            shape="none",
            fontname="Berkeley Mono Trial, FantasqueSansM Nerd Font",
        ),
        edge_attr=dict(
            fontname="Berkeley Mono Trial, FantasqueSansM Nerd Font",
            fontsize="11",
        ),
    )
    nodes: set[Node] = set()
    edges: set[Edge] = set()
    seeds_ranges, map_groups = parse2(input)
    # for stage, group in map_groups.items():
    #     dst_stage = MAP_PROGRESSION[stage]
    #     for i in range(len(group.src)):
    #         id1 = node_id(stage, group.src[i])
    #         id2 = node_id(dst_stage, group.dst[i])
    #         nodes.add(Node(id1, node_label(stage, group.src[i]), rank=stage))
    #         nodes.add(Node(id2, node_label(dst_stage, group.dst[i]), rank=stage))
    #         dot.edge(id1, id2)
    ####
    for in_range in seeds_ranges[:1]:
        for dst_stage in PROGRESSION[:-1]:
            parent_id = node_id("input", in_range)
            nodes.add(Node(parent_id, node_label("input", in_range), rank="input"))
            children, unmapped = map_layer_r(in_range, dst_stage, map_groups)
            children.extend(unmapped)
            for child in children:
                print(child)
                if child.linked_range is None:
                    child_id = node_id("location", child.sub_range)
                    child_label = node_label("location", child.sub_range)
                    rank = "location"
                else:
                    child_id = node_id(child.label, child.linked_range)
                    child_label = node_label(child.label, child.linked_range)
                    rank = child.label
                nodes.add(Node(child_id, child_label, rank=rank))
                edges.add(Edge(parent_id, child_id, child.sub_range))
    # KNOWN RANGE TO SUB RANGE
    enum_prog = list(enumerate(PROGRESSION))
    for in_stage_i, in_stage in enum_prog[:-1]:
        for in_range in map_groups[in_stage].dst[:1]:
            print("mapping", in_stage, in_range)
            for dst_stage in PROGRESSION[in_stage_i + 1 : -1]:
                print("to", dst_stage)
                parent_id = node_id(in_stage, in_range)
                nodes.add(
                    Node(parent_id, node_label(in_stage, in_range), rank=in_stage)
                )
                children, unmapped = map_layer_r(in_range, dst_stage, map_groups)
                children.extend(unmapped)
                for child in children:
                    print("    ", child)
                    if child.linked_range is None:
                        child_id = node_id("location", child.sub_range)
                        child_label = node_label("location", child.sub_range)
                        rank = "location"
                    else:
                        child_id = node_id(child.label, child.linked_range)
                        child_label = node_label(child.label, child.linked_range)
                        rank = child.label
                    nodes.add(Node(child_id, child_label, rank=rank))
                    dot.edge(parent_id, child_id, label=edge_label(child.sub_range))

    for edge in sorted(edges, key=lambda x: x.range.start):
        dot.edge(edge.id1, edge.id2, label=edge_label(edge.range))

    for stage, node_group in groupby(sorted(nodes), key=lambda x: x.rank):
        # print(group, node_group)
        with dot.subgraph(name=stage, graph_attr=dict(rank="same")) as s:
            for node in sorted(node_group):
                s.node(name=node.id, label=node.label)
    dot.render("2023-5")


def node_id(type_, range_) -> str:
    # if range_ is None:
    #     return f"{type_},None"
    return f"{type_},{range_.start},{range_.stop-1}"


def edge_label(range_: Range) -> str:
    return f"{humanize.intcomma(range_.start)}\n{humanize.intcomma(range_.stop-1)}"


def node_label(type_, range_):
    # if range_ is None:
    #     start = "None"
    #     stop = "None"
    # else:
    start = humanize.intcomma(range_.start)
    stop = humanize.intcomma(range_.stop - 1)
    return f"""<<table border="0" cellborder="1" cellpadding="10" cellspacing="0">
        <tr>
            <td bgcolor="black" valign="middle"><font color="white"><b>{type_.upper()}</b></font></td>
            <td cellpadding="0" border="0">
                <table border="0" cellpadding="2" cellspacing="0">
                <tr><td border="1" align="right">{start}</td></tr>
                <tr><td border="1" align="right">{stop}</td></tr>
                </table>
            </td>
        </tr>
    </table>>"""


def range_gap(a: Union[range, Range], b: Union[range, Range]):
    return range(min(a.stop, b.stop), max(a.start, b.start))


def find_gaps(ranges, type_):
    gaps = defaultdict(list)
    lowest_src = None
    highest_src = None
    for i, range_ in enumerate(ranges[:-1]):
        if lowest_src is None or range_.start < lowest_src:
            lowest_src = range_.start
        if highest_src is None or range_.end > highest_src:
            highest_src = range_.end
        gap = range_gap(range_, ranges[i + 1])
        if len(gap) == 0:
            continue
        gaps[type_].append(gap)
    if lowest_src is not None:
        zero_range = range(0, lowest_src)
        if len(zero_range) > 0:
            gaps[type_].append(zero_range)
    if highest_src is not None:
        max_range = range(highest_src + 1, sys.maxsize)
        if len(max_range) > 0:
            gaps[type_].append(max_range)
    return gaps


def find_overlap(
    src_i: int, dst_i: int, map_groups: dict[tuple[str, str], MapGroup], src_ranges=None
):
    if dst_i >= len(PROGRESSION) - 1:
        src = PROGRESSION[src_i]
        dst = PROGRESSION[-1]
        dst_range_maps = map_groups[PROGRESSION[-2], PROGRESSION[-1]].range_maps
        dst_ranges = [x.dst for x in dst_range_maps]
    elif dst_i >= len(PROGRESSION):
        return None
    else:
        src = PROGRESSION[src_i]
        dst = PROGRESSION[dst_i]
        dst_range_maps = map_groups[dst, PROGRESSION[dst_i + 1]].range_maps
        dst_ranges = [x.src for x in dst_range_maps]
    src_range_maps = map_groups[src, PROGRESSION[src_i + 1]].range_maps
    if src_ranges is None:
        src_ranges = [x.dst for x in src_range_maps]
    for src_range in src_ranges:
        for dst_range in dst_ranges:
            overlap = range_overlap(src_range, dst_range)
            if len(overlap) == 0:
                if dst_i >= len(PROGRESSION) - 1:
                    continue
                yield from find_overlap(
                    src_i, dst_i + 1, map_groups, src_ranges=src_ranges
                )
                continue
            uid_1 = node_id(src, src_range)
            uid_2 = node_id(dst, dst_range)
            yield ("n", src, uid_1, node_label(src, src_range))
            yield ("n", dst, uid_2, node_label(dst, dst_range))
            yield ("e", uid_1, uid_2, f"{overlap.start}-{overlap.stop-1}")


def build_dag(map_groups: dict[tuple[str, str], MapGroup]):
    dot = Digraph(format="png", graph_attr={"rankdir": "LR", "ordering": "out"})
    map_groups_list = list(map_groups.items())
    nodes = defaultdict(set)
    edges: set[tuple[str, str, str]] = set()
    for (src_type, dst_type), map_group in map_groups_list:
        for range_map in map_group.range_maps:
            uid_1 = node_id(src_type, range_map.src)
            uid_2 = node_id(dst_type, range_map.dst)
            nodes[src_type].add((uid_1, node_label(src_type, range_map.src)))
            nodes[dst_type].add((uid_2, node_label(dst_type, range_map.dst)))
            edges.add((uid_1, uid_2, ""))
    for i, src in enumerate(PROGRESSION[:-1]):
        graph_elements = find_overlap(i, i + 1, map_groups)
        for element in graph_elements:
            if element[0] == "n":
                _, type_, uid, label = element
                nodes[type_].add((uid, label))
            elif element[0] == "e":
                _, uid_1, uid_2, label = element
                edges.add((uid_1, uid_2, label))

    for uid_1, uid_2, label in edges:
        dot.edge(uid_1, uid_2, label=label)
    for k, nodes in nodes.items():
        with dot.subgraph(graph_attr={"rank": "same"}) as s:
            for uid, label in sorted(set(nodes)):
                s.node(name=uid, label=label, shape="record")

    print(dot)


def diff(a: interval, b: interval) -> interval:
    return (
        interval[min(a.sup, b.sup), min(a.inf, b.sup)]
        | interval[max(a.sup, b.inf), max(a.inf, b.inf)]
    )


def main3(input):
    seeds_ranges, map_groups = parse2(input)
    a = seeds_ranges[0].interval()
    seed_groups = map_groups["seed"]
    c = seed_groups.src[0].interval()
    c_off = seed_groups.dst[0].start - seed_groups.src[0].start
    d = seed_groups.src[1].interval()
    d_off = seed_groups.dst[1].start - seed_groups.src[1].start
    print(c_off, d_off)
    x = (a & c) + c_off
    y = (a & d) + d_off
    z = diff(a, c)
    w = diff(a, d)
    print(f"{x=} {y=} {z=} {w=}")


if __name__ == "__main__":
    main3(CONTROL_1)
