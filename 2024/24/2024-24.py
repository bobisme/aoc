#!/usr/bin/env python

from dataclasses import dataclass
from pprint import pp
import re
from typing import Iterable, Literal, NamedTuple, Optional


CONTROL_1 = """\
x00: 1
x01: 1
x02: 1
y00: 0
y01: 1
y02: 0

x00 AND y00 -> z00
x01 XOR y01 -> z01
x02 OR y02 -> z02
""".splitlines()

CONTROL_2 = """\
x00: 1
x01: 0
x02: 1
x03: 1
x04: 0
y00: 1
y01: 1
y02: 1
y03: 1
y04: 1

ntg XOR fgs -> mjb
y02 OR x01 -> tnw
kwq OR kpj -> z05
x00 OR x03 -> fst
tgd XOR rvg -> z01
vdt OR tnw -> bfw
bfw AND frj -> z10
ffh OR nrd -> bqk
y00 AND y03 -> djm
y03 OR y00 -> psh
bqk OR frj -> z08
tnw OR fst -> frj
gnj AND tgd -> z11
bfw XOR mjb -> z00
x03 OR x00 -> vdt
gnj AND wpb -> z02
x04 AND y00 -> kjc
djm OR pbm -> qhw
nrd AND vdt -> hwm
kjc AND fst -> rvg
y04 OR y02 -> fgs
y01 AND x02 -> pbm
ntg OR kjc -> kwq
psh XOR fgs -> tgd
qhw XOR tgd -> z09
pbm OR djm -> kpj
x03 XOR y03 -> ffh
x00 XOR y04 -> ntg
bfw OR bqk -> z06
nrd XOR fgs -> wpb
frj XOR qhw -> z04
bqk OR frj -> z07
y03 OR x01 -> nrd
hwm AND bqk -> z03
tgd XOR rvg -> z12
tnw OR pbm -> gnj
""".splitlines()

with open("2024-24.input") as f:
    input_file = [line.strip() for line in f.readlines()]

Op = Literal["AND", "OR", "XOR"]


@dataclass
class Node:
    ins: list["Node"]
    op: Optional[Op]
    key: str
    out: Optional[int] = None

    def __repr__(self) -> str:
        if self.ins:
            return f"{self.key} {self.op if self.op else ''} {self.ins}"
        return self.key

    def operate(self):
        if not self.op:
            raise ValueError("no operator")
        if self.ins[0].out is None:
            self.ins[0].out = self.ins[0].operate()
        if self.ins[1].out is None:
            self.ins[1].out = self.ins[1].operate()
        if self.ins[0].out is None or self.ins[1].out is None:
            raise ValueError("can't")
        match self.op:
            case "AND":
                return self.ins[0].out & self.ins[1].out
            case "OR":
                return self.ins[0].out | self.ins[1].out
            case "XOR":
                return self.ins[0].out ^ self.ins[1].out


def parse(input: Iterable[str]):
    mode = 0
    init_vals = {}
    node_map = {}
    pattern = re.compile(r"(\w+) (XOR|AND|OR) (\w+) -> (\w+)")
    for line in input:
        if not line:
            mode = 1
            continue
        if mode == 0:
            key, val = line.split(": ")
            init_vals[key] = int(val)
        else:
            k1, op, k2, k3 = pattern.findall(line)[0]
            ins = []
            if k1 in node_map:
                ins.append(node_map[k1])
            else:
                k1_node = Node(ins=[], op=None, key=k1)
                node_map[k1] = k1_node
                ins.append(k1_node)
            if k2 in node_map:
                ins.append(node_map[k2])
            else:
                k2_node = Node(ins=[], op=None, key=k2)
                node_map[k2] = k2_node
                ins.append(k2_node)
            if k3 in node_map:
                node = node_map[k3]
                node.ins += ins
                node.op = op
            else:
                node = Node(ins, op, k3)
                node_map[k3] = node

    for k, v in init_vals.items():
        node_map[k].out = v

    return node_map


def part_1(input):
    nodes = parse(input)
    max_z: str = max(nodes.keys())
    z_count = int(max_z.lstrip("z")) + 1
    out = 0
    for i in range(z_count):
        key = f"z{i:0>2}"
        op = nodes[key].operate()
        out |= op << i
    # print(bin(out))
    print(out)


def part_2(input):
    for line in input:
        print(line)


if __name__ == "__main__":
    part_1(input_file)
    # part_2(input_file)
