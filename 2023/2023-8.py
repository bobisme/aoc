#!/usr/bin/env python

from dataclasses import dataclass
from itertools import cycle, islice
from typing import Iterable, Literal, Optional, OrderedDict, Self
from pprint import pp
import re

import graphviz
from progress.bar import PixelBar

CONTROL_1 = """\
RL

AAA = (BBB, CCC)
BBB = (DDD, EEE)
CCC = (ZZZ, GGG)
DDD = (DDD, DDD)
EEE = (EEE, EEE)
GGG = (GGG, GGG)
ZZZ = (ZZZ, ZZZ)
""".splitlines()
CONTROL_2 = """\
LLR

AAA = (BBB, BBB)
BBB = (AAA, ZZZ)
ZZZ = (ZZZ, ZZZ)
""".splitlines()

with open("2023-8.input") as f:
    input = [line.strip() for line in f.readlines()]


@dataclass
class Node:
    label: str
    left: Optional[str]
    right: Optional[str]

    def get(self, dir=Literal["R", "L"]) -> Optional[str]:
        if dir == "R":
            return self.right
        if dir == "L":
            return self.left


def parse_node(line: str) -> Node:
    found = re.findall(r"([A-Z]{3})\s+=\s+\(([A-Z]{3}), ([A-Z]{3})\)", line)[0]
    label = found[0]
    left = found[1] if label != found[1] else None
    right = found[2] if label != found[2] else None
    node = Node(label=found[0], left=left, right=right)
    return node


def parse(input) -> tuple[Iterable[str], dict[str, Node]]:
    directions = cycle(c for c in input[0])
    nodes = (parse_node(line) for line in input[2:])
    nodes = OrderedDict((n.label, n) for n in nodes)
    return directions, nodes


def main(input):
    directions, nodes = parse(input)
    _, root = next(iter(nodes.items()))
    node = root
    step_count = 0
    bar = PixelBar("Traversing", max=10_000_000)
    while node.label != "ZZZ":
        direction = next(directions)
        step_count += 1
        bar.next()
        node_label = node.get(direction)
        if node_label is None:
            break
        node = nodes[node_label]
    bar.finish()
    print(step_count)


if __name__ == "__main__":
    main(input)
