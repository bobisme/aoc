#!/usr/bin/env python

from dataclasses import dataclass, field
from enum import Enum
from typing import LiteralString, NamedTuple
import time

Input = list[str] | list[LiteralString]

CONTROL_1: Input = (
    """\
inc a
jio a, +2
tpl a
inc a
""".splitlines()
)


class Inst(Enum):
    Unk = -1
    Hlf = 0
    Tpl = 1
    Inc = 2
    Jmp = 3
    Jie = 4
    Jio = 5

    @staticmethod
    def from_str(s: str) -> "Inst":
        match s:
            case "hlf":
                return Inst.Hlf
            case "tpl":
                return Inst.Tpl
            case "inc":
                return Inst.Inc
            case "jmp":
                return Inst.Jmp
            case "jie":
                return Inst.Jie
            case "jio":
                return Inst.Jio
        assert not "unreachable"
        return Inst.Unk


class Reg(Enum):
    Unk = -1
    A = 0
    B = 1

    @staticmethod
    def from_str(s: str) -> "Reg":
        if s == "a":
            return Reg.A
        elif s == "b":
            return Reg.B
        assert not f"unreachable {s}"
        return Reg.Unk


type Registers = dict[Reg, int]


class Op(NamedTuple):
    inst: Inst
    arg1: Reg | int
    arg2: int | None = None

    @staticmethod
    def from_str(s: str) -> "Op":
        inst, args = s.split(" ", maxsplit=1)
        inst = Inst.from_str(inst)
        args = args.split(", ", maxsplit=1)
        if inst == Inst.Jmp:
            return Op(inst, int(args[0]))
        if inst in (Inst.Jie, Inst.Jio):
            return Op(inst, Reg.from_str(args[0]), int(args[1]))
        return Op(inst, Reg.from_str(args[0]))


@dataclass(slots=True)
class Machine:
    rom: list[Op]
    ptr: int = 0
    regs: Registers = field(default_factory=lambda: {Reg.A: 0, Reg.B: 0})

    def exec(self, op: Op):
        match op.inst:
            case Inst.Hlf:
                assert isinstance(op.arg1, Reg)
                self.regs[op.arg1] = self.regs[op.arg1] // 2
                self.ptr += 1
            case Inst.Tpl:
                assert isinstance(op.arg1, Reg)
                self.regs[op.arg1] = self.regs[op.arg1] * 3
                self.ptr += 1
            case Inst.Inc:
                assert isinstance(op.arg1, Reg)
                self.regs[op.arg1] = self.regs[op.arg1] + 1
                self.ptr += 1
            case Inst.Jmp:
                assert isinstance(op.arg1, int)
                self.ptr += op.arg1
            case Inst.Jie:
                assert isinstance(op.arg1, Reg)
                assert isinstance(op.arg2, int)
                if self.regs[op.arg1] % 2 == 0:
                    self.ptr += op.arg2
                else:
                    self.ptr += 1
            case Inst.Jio:
                assert isinstance(op.arg1, Reg)
                assert isinstance(op.arg2, int)
                if self.regs[op.arg1] == 1:
                    self.ptr += op.arg2
                else:
                    self.ptr += 1

    def run(self):
        while self.ptr < len(self.rom):
            self.exec(self.rom[self.ptr])


def part_1(input: Input):
    ops = list(Op.from_str(line) for line in input)
    machine = Machine(ops)
    machine.run()
    return machine.regs[Reg.B]


def part_2(input: Input):
    ops = list(Op.from_str(line) for line in input)
    machine = Machine(ops)
    machine.regs[Reg.A] = 1
    machine.run()
    return machine.regs[Reg.B]


def _test():
    def assert_eq(a, b):
        assert a == b, f"{a} != {b}"

    assert_eq(part_1(CONTROL_1), 0)
    # assert_eq(part_2(CONTROL_1), 0)


def run(fn, year=2015, day=23, part=0):
    start = time.perf_counter_ns()
    res = fn()
    elapsed_ns = time.perf_counter_ns() - start
    print(f"{year}\t{day}\t{part}\t{res}\t{elapsed_ns}")
    return res


if __name__ == "__main__":
    with open("2015-23.input") as f:
        input_file = [line.rstrip("\n") for line in f.readlines()]
    _test()
    run(lambda: part_1(input_file), part=1)
    run(lambda: part_2(input_file), part=2)
