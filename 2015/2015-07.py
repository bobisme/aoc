#!/usr/bin/env python

from typing import (
    Callable,
    DefaultDict,
    LiteralString,
    NamedTuple,
    OrderedDict,
)
import time


def bench(fn):
    def inner(*args, **kwargs):
        start = time.perf_counter()
        res = fn(*args, **kwargs)
        t_ms = (time.perf_counter() - start) * 1000
        print(f"{fn.__name__} = {res} in {t_ms:.2f}ms")
        return res

    return inner


Input = list[str] | list[LiteralString]

CONTROL_1: Input = (
    """\
123 -> x
456 -> y
x AND y -> d
x OR y -> e
x LSHIFT 2 -> f
y RSHIFT 2 -> g
NOT x -> h
NOT y -> i
""".splitlines()
)

with open("2015-07.input") as f:
    input_file = [line.rstrip("\n") for line in f.readlines()]

MASK = 0xFFFF


class Reg(str):
    pass


class Registers(DefaultDict[Reg, int]):
    def __repr__(self) -> str:
        out = ""
        for key in sorted(self.keys()):
            out += f"{key}: {self[key]}\n"
        return out


def register(fn):
    def inner(registers: Registers, inputs: list[Reg | int], out: Reg):
        dereg = [registers[x] if isinstance(x, Reg) else x for x in inputs]
        registers[out] = fn(*dereg) & MASK

    inner.__name__ = fn.__name__

    return inner


@register
def set_(a: int):
    return a


@register
def not_(a: int):
    return ~a


@register
def and_(a: int, b: int):
    return a & b


@register
def or_(a: int, b: int):
    return a | b


@register
def lshift(a: int, b: int):
    return a << b


@register
def rshift(a: int, b: int):
    return a >> b


def reg_or_int(s: str) -> Reg | int:
    try:
        return int(s)
    except ValueError:
        return Reg(s)


class Op(NamedTuple):
    fn: Callable[[Registers, list[Reg | int], Reg], None]
    inputs: list[Reg | int]
    out: Reg

    def __repr__(self) -> str:
        return f"{self.fn.__name__} {' '.join(map(str, self.inputs))} -> {self.out}"


def parse(input: Input):
    for line in input:
        inp, _, out = line.rsplit(" ", 2)
        parts = inp.split(" ")
        match len(parts):
            case 1:
                yield Op(set_, [reg_or_int(parts[0])], Reg(out))
                # set_(registers, reg_or_int(parts[0]), Reg(out))
            case 2:
                yield Op(not_, [reg_or_int(parts[1])], Reg(out))
                # not_(registers, reg_or_int(parts[1]), Reg(out))
            case 3:
                match parts[1]:
                    case "AND":
                        # and_(registers, reg_or_int(parts[0]), reg_or_int(parts[2]), Reg(out))
                        yield Op(
                            and_,
                            [reg_or_int(parts[0]), reg_or_int(parts[2])],
                            Reg(out),
                        )
                    case "OR":
                        # or_( registers, reg_or_int(parts[0]), reg_or_int(parts[2]), Reg(out),)
                        yield Op(
                            or_,
                            [reg_or_int(parts[0]), reg_or_int(parts[2])],
                            Reg(out),
                        )
                    case "LSHIFT":
                        # lshift( registers, reg_or_int(parts[0]), reg_or_int(parts[2]), Reg(out),)
                        yield Op(
                            lshift,
                            [reg_or_int(parts[0]), reg_or_int(parts[2])],
                            Reg(out),
                        )
                    case "RSHIFT":
                        # rshift( registers, reg_or_int(parts[0]), reg_or_int(parts[2]), Reg(out),)
                        yield Op(
                            rshift,
                            [reg_or_int(parts[0]), reg_or_int(parts[2])],
                            Reg(out),
                        )
                    case _:
                        raise ValueError("what")
            case _:
                raise ValueError("what")


def dfs_topo_sort(ops: list[Op]) -> list[Reg]:
    in_to_outs: OrderedDict[Reg, set[Reg]] = OrderedDict()
    all_regs = set()
    for op in ops:
        all_regs.add(op.out)
        for inp in op.inputs:
            if isinstance(inp, Reg):
                if inp not in in_to_outs:
                    in_to_outs[inp] = {op.out}
                else:
                    in_to_outs[inp].add(op.out)

    explored = set()
    temp = set()
    out = []

    def inner(reg: Reg, depth: int):
        nonlocal explored, out
        if reg in explored:
            return
        assert reg not in temp

        temp.add(reg)
        if reg in in_to_outs:
            for n in in_to_outs[reg]:
                inner(n, depth + 1)
        explored.add(reg)
        out.append(reg)

    for reg in all_regs:
        if reg in explored:
            continue
        inner(reg, 0)
    return list(reversed(out))


@bench
def part_1(input: Input):
    registers = Registers()
    ops = list(parse(input))
    ordered = dfs_topo_sort(ops)
    out_reg_to_ops: DefaultDict[Reg, list[Op]] = DefaultDict(list)
    for op in ops:
        out_reg_to_ops[op.out].append(op)

    for reg in ordered:
        for op in out_reg_to_ops[reg]:
            op.fn(registers, op.inputs, op.out)
    return registers[Reg("a")]


@bench
def part_2(input: Input):
    registers = Registers()
    ops = list(parse(input))
    ordered = dfs_topo_sort(ops)
    out_reg_to_ops: DefaultDict[Reg, list[Op]] = DefaultDict(list)
    for op in ops:
        out_reg_to_ops[op.out].append(op)

    for reg in ordered:
        for op in out_reg_to_ops[reg]:
            op.fn(registers, op.inputs, op.out)
    final_a = registers[Reg("a")]
    registers = Registers()
    registers[Reg("b")] = final_a
    for reg in ordered:
        if reg == Reg("b"):
            continue
        for op in out_reg_to_ops[reg]:
            op.fn(registers, op.inputs, op.out)
    return registers[Reg("a")]


if __name__ == "__main__":
    part_1(input_file)
    part_2(input_file)
