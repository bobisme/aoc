#!/usr/bin/env python

from dataclasses import dataclass
import itertools
from pprint import pp
import random
import re
from typing import DefaultDict, Generator, Iterable, Literal, NamedTuple, Optional


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

CONTROL_3 = """\
x00: 0
x01: 1
x02: 0
x03: 1
x04: 0
x05: 1
y00: 0
y01: 0
y02: 1
y03: 1
y04: 0
y05: 1

x00 AND y00 -> z05
x01 AND y01 -> z02
x02 AND y02 -> z01
x03 AND y03 -> z03
x04 AND y04 -> z04
x05 AND y05 -> z00
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


Pair = tuple[str, str]
Vals = dict[str, int]
Ops = dict[Pair, list[tuple[str, Op]]]
OutToIns = dict[str, Pair]


def parse_2(input: Iterable[str]):
    pattern = re.compile(r"(\w+) (XOR|AND|OR) (\w+) -> (\w+)")
    mode = 0
    init_vals: Vals = {}
    ops: Ops = DefaultDict(list)
    out_to_ins: OutToIns = {}
    for line in input:
        if not line:
            mode = 1
            continue
        if mode == 0:
            ins, val = line.split(": ")
            init_vals[ins] = int(val)
        else:
            in1, op, in2, out = pattern.findall(line)[0]
            ins = tuple(sorted((in1, in2)))
            if ins == ("ggg", "wdq"):
                print(in1, op, in2, out)
            ops[ins].append((out, op))
            out_to_ins[out] = ins
    return init_vals, ops, out_to_ins


def write_to_dot_file(
    out_to_ins: OutToIns, ops: Ops, vals: Vals, filename: str
) -> None:
    with open(filename, "w", encoding="utf-8") as f:
        f.write("digraph G {\n")
        for child in sorted(out_to_ins.keys(), reverse=True):
            parents = out_to_ins[child]
            if child in vals:
                child = f"{child}|{vals[child]}"
            op_label = get_op(ops, parents, child)
            for parent in parents:
                if parent in vals:
                    parent = f"{parent}|{vals[parent]}"
                f.write(f'    "{parent}" -> "{child}" [label="{op_label}"];\n')
        f.write("}\n")


def get_op(ops: Ops, ins: Pair, out: str) -> Op:
    for out2, op in ops[ins]:
        if out2 == out:
            return op
    raise ValueError("no ops")


def swap_wires(out_to_ins: OutToIns, w1: str, w2: str):
    p1 = out_to_ins[w1]
    p2 = out_to_ins[w2]
    out_to_ins[w1] = p2
    out_to_ins[w2] = p1


def operate(out_to_ins: OutToIns, ops: Ops, vals: Vals, output: str) -> int:
    ins = out_to_ins[output]
    if vals.get(ins[0]) is None:
        vals[ins[0]] = operate(out_to_ins, ops, vals, ins[0])
    if vals.get(ins[1]) is None:
        vals[ins[1]] = operate(out_to_ins, ops, vals, ins[1])
    op = get_op(ops, ins, output)
    match op:
        case "AND":
            return vals[ins[0]] & vals[ins[1]]
        case "OR":
            return vals[ins[0]] | vals[ins[1]]
        case "XOR":
            return vals[ins[0]] ^ vals[ins[1]]


# https://www.wikiwand.com/en/articles/Disjoint-set_data_structure
class UnionFind:
    def __init__(self, nodes) -> None:
        self.parent = {x: x for x in nodes}

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        px, py = self.find(x), self.find(y)
        if px == py:
            return False  # cycle
        self.parent[px] = py
        return True


def has_cycle(out_to_ins: OutToIns):
    all_nodes = set(out_to_ins.keys()) | {
        x for pair in out_to_ins.values() for x in pair
    }
    union_find = UnionFind(all_nodes)
    seen_edges = set()

    for node in out_to_ins:
        for connected in out_to_ins[node]:
            edge = tuple(sorted((node, connected)))
            if edge in seen_edges:
                continue
            seen_edges.add(edge)

            if not union_find.union(node, connected):
                return True
    return False


def get_val(vals: Vals, prefix: str) -> int:
    max_char: str = max(k for k in vals.keys() if k.startswith(prefix))
    char_count = int(max_char.lstrip(prefix)) + 1
    out = 0
    for i in range(char_count):
        key = f"{prefix}{i:0>2}"
        out |= vals[key] << i
    return out


def operate_zs(out_to_ins: OutToIns, ops: Ops, vals: Vals) -> int:
    max_z: str = max(out_to_ins.keys())
    z_count = int(max_z.lstrip("z")) + 1
    out = 0
    for i in range(z_count):
        key = f"z{i:0>2}"
        op = operate(out_to_ins, ops, vals, key)
        out |= op << i
    return out


def get_non_overlapping_pairs(elements, n=4):
    """Return combinations of n disjoint pairs from the provided elements."""
    combos = list(itertools.combinations(elements, 2))

    for group in itertools.combinations(combos, n):
        used = set()
        is_valid = True
        for pair in group:
            if pair[0] in used or pair[1] in used:
                is_valid = False
                break
            used.update(pair)
        if is_valid:
            yield group


# S = A ^ B ^ Cin
# Cout = (A & B) | (Cin & (A ^ B))
def map_bits(
    out_to_ins: OutToIns, ops: Ops, i: int, max_z="z45"
) -> Generator[str, None, None]:
    x = f"x{i:0>2}"
    y = f"y{i:0>2}"
    z = f"z{i:0>2}"
    ins_to_z = out_to_ins[z]
    z_op = get_op(ops, ins_to_z, z)
    if i == 0:
        # only S path
        if ins_to_z != (x, y) and z_op != "XOR":
            print("BAD WIRE", z)
            yield z
    elif z == max_z:
        # only Cout path
        if not z_op == "OR":
            print("BAD WIRE", z)
            yield z
    else:
        if z_op != "XOR":
            print("BAD WIRE", z)
            yield z
        else:
            # one in is the carry, one is the result of x ^ y
            # print("ins to z", ins_to_z)
            a, b = ins_to_z
            ins_to_a = out_to_ins[a]
            ins_to_b = out_to_ins[b]
            if ins_to_a == (x, y):
                S = a
                S_ins = ins_to_a
                C = b
                C_ins = ins_to_b
            elif ins_to_b == (x, y):
                S = b
                S_ins = ins_to_b
                C = a
                C_ins = ins_to_a
            else:
                raise ValueError("not cool")
            if get_op(ops, S_ins, S) != "XOR":
                print("BAD WIRE, S", S)
                yield S
            elif get_op(ops, C_ins, C) == "XOR":
                print("BAD WIRE, C", C)
                yield C
            # now we check C_ins
            a, b = C_ins
            if a in out_to_ins:
                ins_to_a = out_to_ins[a]
                op = get_op(ops, ins_to_a, a)
                if op == "XOR":
                    yield a
            if b in out_to_ins:
                ins_to_b = out_to_ins[b]
                op = get_op(ops, ins_to_a, a)
                if op == "XOR":
                    yield a


def part_2(input, swaps=2):
    input_pattern = re.compile(r"[xy]\d\d")

    def is_input(s: str):
        return bool(input_pattern.match(s))

    init_vals, ops, out_to_ins = parse_2(input)

    x = get_val(init_vals, "x")
    y = get_val(init_vals, "y")
    expected = x + y

    max_z: str = max(out_to_ins.keys())
    z_count = int(max_z.lstrip("z")) + 1
    print("map bits")
    bad = []
    for i in range(z_count):
        bad.extend(list(map_bits(out_to_ins, ops, i, max_z=max_z)))

    write_to_dot_file(out_to_ins, ops, init_vals, "2024-24.dot")
    print(f"{x=}, {y=}, {expected=}, {max_z=}")
    wires_to_swap = []
    for out in out_to_ins.keys():
        parents = out_to_ins[out]
        op = get_op(ops, parents, out)
        if out == max_z:
            if op != "OR":
                print(f"BAD: {parents[0]} {op} {parents[1]} -> {out}")
                wires_to_swap.append(out)
        elif out.startswith("z"):
            if op != "XOR":
                print(f"BAD: {parents[0]} {op} {parents[1]} -> {out}")
                wires_to_swap.append(out)
        elif not is_input(parents[0]) and not is_input(parents[1]):
            if op not in ("AND", "OR"):
                print(f"BAD: {parents[0]} {op} {parents[1]} -> {out}")
                wires_to_swap.append(out)
    print(wires_to_swap)
    print(bad)
    set_of_bad = set(wires_to_swap) | set(bad)
    if len(set_of_bad) == 8:
        print(",".join(sorted(set_of_bad)))
    else:
        print("no solution")


if __name__ == "__main__":
    part_1(input_file)
    part_2(input_file, swaps=2)
