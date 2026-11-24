#!/usr/bin/env python

import time
from dataclasses import dataclass, field
from typing import Literal, LiteralString

Input = list[str] | list[LiteralString]

CONTROL_1: Input = """\
cpy 41 a
inc a
inc a
dec a
jnz a 2
dec a
""".splitlines()


@dataclass
class Instruction:
    op: Literal["cpy", "inc", "dec", "jnz"]
    a: str | int
    b: str | int | None = None

    @staticmethod
    def from_str(inst: str) -> "Instruction":
        if inst.startswith("cpy"):
            _, a, b = inst.split(" ")
            try:
                a = int(a)
            except ValueError:
                pass
            try:
                b = int(b)
            except ValueError:
                pass
            return Instruction("cpy", a, b)
        if inst.startswith("inc"):
            _, a = inst.split(" ")
            try:
                a = int(a)
            except ValueError:
                pass
            return Instruction("inc", a)
        if inst.startswith("dec"):
            _, a = inst.split(" ")
            try:
                a = int(a)
            except ValueError:
                pass
            return Instruction("dec", a)
        else:  # jnz
            _, a, b = inst.split(" ")
            try:
                a = int(a)
            except ValueError:
                pass
            try:
                b = int(b)
            except ValueError:
                pass
            return Instruction("jnz", a, b)


@dataclass
class Computer:
    instructions: list[Instruction] = field(default_factory=list)
    a: int = 0
    b: int = 0
    c: int = 0
    d: int = 0
    idx: int = 0

    def exec(self) -> bool:
        if self.idx >= len(self.instructions):
            return False
        inst = self.instructions[self.idx]
        match inst.op:
            case "cpy":
                a_val = getattr(self, inst.a) if isinstance(inst.a, str) else inst.a
                assert isinstance(inst.b, str)
                setattr(self, inst.b, a_val)
                self.idx += 1
            case "inc":
                assert isinstance(inst.a, str)
                setattr(self, inst.a, getattr(self, inst.a) + 1)
                self.idx += 1
            case "dec":
                assert isinstance(inst.a, str)
                setattr(self, inst.a, getattr(self, inst.a) - 1)
                self.idx += 1
            case "jnz":
                a_val = getattr(self, inst.a) if isinstance(inst.a, str) else inst.a
                if a_val != 0:
                    assert isinstance(inst.b, int)
                    self.idx += inst.b
                else:
                    self.idx += 1
        return True

    def run(self):
        while self.exec():
            continue


def part_1(input: Input):
    instructions = [Instruction.from_str(line) for line in input]
    computer = Computer(instructions=instructions)
    computer.run()
    return computer.a


def part_2(input: Input):
    instructions = [Instruction.from_str(line) for line in input]
    computer = Computer(instructions=instructions, c=1)
    computer.run()
    return computer.a


def _test():
    def assert_eq(a, b):
        assert a == b, f"{a} != {b}"

    assert_eq(part_1(CONTROL_1), 42)
    # assert_eq(part_2(CONTROL_1), 0)


def run(fn, year=2016, day=12, part=0):
    start = time.perf_counter_ns()
    res = fn()
    elapsed_ns = time.perf_counter_ns() - start
    print(f"{year}\t{day}\t{part}\t{res}\t{elapsed_ns}")
    return res


if __name__ == "__main__":
    with open("2016-12.input") as f:
        input_file = [line.rstrip("\n") for line in f]
    _test()
    run(lambda: part_1(input_file), part=1)
    run(lambda: part_2(input_file), part=2)
