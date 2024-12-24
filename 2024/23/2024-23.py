#!/usr/bin/env python

from itertools import combinations
from typing import DefaultDict, Iterable


CONTROL_1 = """\
kh-tc
qp-kh
de-cg
ka-co
yn-aq
qp-ub
cg-tb
vc-aq
tb-ka
wh-tc
yn-cg
kh-ub
ta-co
de-co
tc-td
tb-wq
wh-td
ta-ka
td-qp
aq-cg
wq-ub
ub-vc
de-ta
wq-aq
wq-vc
wh-yn
ka-de
kh-ta
co-tc
wh-qp
tb-vc
td-yn
""".splitlines()

with open("2024-23.input") as f:
    input_file = [line.strip() for line in f.readlines()]


def get_adjacency_list(input: Iterable[str]):
    adjlist: DefaultDict[str, set[str]] = DefaultDict(set)
    for line in input:
        left, right = line.split("-")
        adjlist[left].add(right)
        adjlist[right].add(left)
    return adjlist


def part_1(input):
    adjlist = get_adjacency_list(input)
    interconnected: set[tuple[str, ...]] = set()
    count = 0
    for a, others in adjlist.items():
        for b, c in combinations(others, 2):
            if c in adjlist[b]:
                interconnected.add(tuple(sorted([a, b, c])))
    for a, b, c in interconnected:
        if a[0] == "t" or b[0] == "t" or c[0] == "t":
            count += 1
    print(count)


def part_2(input):
    for line in input:
        print(line)


if __name__ == "__main__":
    part_1(input_file)
    # part_2(input_file)
