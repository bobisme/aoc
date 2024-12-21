#!/usr/bin/env python

import math
import os
import re
import sys
from typing import Iterable


CONTROL_1 = """\
Register A: 729
Register B: 0
Register C: 0

Program: 0,1,5,4,3,0
""".splitlines()

CONTROL_2 = """\
Register A: 2024
Register B: 0
Register C: 0

Program: 0,3,5,4,3,0
""".splitlines()

with open("2024-17.input") as f:
    input_file = [line.strip() for line in f.readlines()]

Registers = dict[str, int]


def dbg(*args, **kwargs):
    if os.getenv("DEBUG") == "1":
        print(*args, **kwargs)


def parse(input: Iterable[str]) -> tuple[Registers, list[int]]:
    program = False
    registers = {}
    reg_pat = re.compile(r"Register (\w+): (\d+)")

    for line in input:
        if program:
            prog = [int(x) for x in line.lstrip("Program: ").split(",")]
            return registers, prog
        if not line:
            program = True
            continue
        reg, strval = reg_pat.findall(line)[0]
        registers[reg] = int(strval)
    return registers, []


class Halt(Exception):
    pass


class Computer:
    a: int = 0
    b: int = 0
    c: int = 0
    program: list[int]
    program_bit_len = 0
    pointer = 0
    output: list[int]

    def __init__(self, registers: Registers, program: list[int]) -> None:
        self.a = registers.get("A", 0)
        self.b = registers.get("B", 0)
        self.c = registers.get("C", 0)

        self.program = program
        self.output = []

    def __repr__(self) -> str:
        return f"A:{self.a} B:{self.b} C:{self.c} | {self.program}"

    def read_ops(self, p: int) -> tuple[int, int]:
        if p >= len(self.program) or (p + 1) >= len(self.program):
            raise Halt()
        return self.program[p], self.program[p + 1]

    def handle(self, opcode: int, operand: int):
        match opcode:
            case 0:
                self.adv(operand)
            case 1:
                self.bxl(operand)
            case 2:
                self.bst(operand)
            case 3:
                self.jnz(operand)
            case 4:
                self.bxc(operand)
            case 5:
                self.out(operand)
            case 6:
                self.bdv(operand)
            case 7:
                self.cdv(operand)
            case _:
                raise ValueError(f"Invalid instruction: {opcode}")

    def combo(self, operand: int) -> int:
        match operand:
            case 0 | 1 | 2 | 3:
                return operand
            case 4:
                return self.a
            case 5:
                return self.b
            case 6:
                return self.c
            case _:
                raise ValueError(f"Invalid combo operator: {operand}")

    def adv(self, operand: int):
        dbg("adv", operand)
        self.a = math.trunc(self.a / (2 ** self.combo(operand)))
        self.pointer += 2

    def bxl(self, operand: int):
        dbg("bxl", operand)
        self.b = self.b ^ operand
        self.pointer += 2

    def bst(self, operand: int):
        dbg("bst", operand)
        self.b = self.combo(operand) % 8
        self.pointer += 2

    def jnz(self, operand: int):
        dbg("jnz", operand)
        if self.a == 0:
            self.pointer += 2
            return
        self.pointer = operand

    def bxc(self, _: int):
        dbg("bxc")
        self.b = self.b ^ self.c
        self.pointer += 2

    def out(self, operand: int):
        dbg("out", operand)
        self.output.append(self.combo(operand) % 8)
        self.pointer += 2

    def bdv(self, operand: int):
        dbg("bdv", operand)
        self.b = math.trunc(self.a / (2 ** self.combo(operand)))
        self.pointer += 2

    def cdv(self, operand: int):
        dbg("cdv", operand)
        self.c = math.trunc(self.a / (2 ** self.combo(operand)))
        self.pointer += 2

    def execute(self) -> list[int]:
        while True:
            try:
                opcode, operand = self.read_ops(self.pointer)
                self.handle(opcode, operand)
            except Halt:
                return self.output


def part_1(input: Iterable[str]):
    def test_1():
        comp = Computer({"C": 9}, [2, 6])
        opcode, operand = comp.read_ops(0)
        assert opcode == 2, repr(comp)
        assert operand == 6, repr(comp)
        comp.handle(opcode, operand)
        assert comp.b == 1, repr(comp)
        print("test 1 passed")

    def test_2():
        comp = Computer({"A": 10}, [5, 0, 5, 1, 5, 4])
        out = comp.execute()
        assert out == [0, 1, 2], f"{repr(comp)} {out}"
        print("test 2 passed")

    def test_3():
        comp = Computer({"A": 2024}, [0, 1, 5, 4, 3, 0])
        out = comp.execute()
        assert comp.a == 0, f"{repr(comp)} {out}"
        assert out == [4, 2, 5, 6, 7, 7, 7, 7, 3, 1, 0], f"{repr(comp)} {out}"
        print("test 3 passed")

    def test_4():
        comp = Computer({"B": 29}, [1, 7])
        out = comp.execute()
        assert comp.b == 26, f"{repr(comp)} {out}"
        print("test 4 passed")

    def test_5():
        comp = Computer({"B": 2024, "C": 43690}, [4, 0])
        out = comp.execute()
        assert comp.b == 44354, f"{repr(comp)} {out}"
        print("test 5 passed")

    # test_1()
    # test_2()
    # test_3()
    # test_4()
    # test_5()

    registers, program = parse(input)
    computer = Computer(registers, program)
    print(",".join(str(x) for x in computer.execute()))


def search(
    registers: Registers, program: list[int], idx: int, lo=0, hi=100_000_000_000_000
) -> tuple[int, int, int]:
    target = program[idx]
    while lo < hi:
        mid_a = (lo + hi) // 2
        computer = Computer(registers, program)
        computer.a = mid_a
        mid_output = computer.execute()
        try:
            mid_val = mid_output[idx]
        except IndexError:
            mid_val = -1
        if len(mid_output) > len(program):
            mid_val = sys.maxsize
        if mid_val < target:
            lo = mid_a + 1
        elif mid_val > target:
            hi = mid_a
        else:
            return mid_a, lo, hi
    return -1, -1, -1


def program_to_octal(program: list[int]) -> int:
    return int("".join(str(x) for x in reversed(program)), 8)


def search_oct(registers: Registers, program: list[int], lo=0, hi=sys.maxsize) -> int:
    oct_prg = program_to_octal(program)
    mid_a = -1
    while lo < hi:
        mid_a = (lo + hi) // 2
        computer = Computer(registers, program)
        computer.a = mid_a
        output = computer.execute()
        if len(output) < len(program):
            mid_val = -1
        elif len(output) > len(program):
            mid_val = sys.maxsize
        else:
            mid_val = program_to_octal(output)
        print(mid_a, mid_val)
        if mid_val < oct_prg:
            lo = mid_a + 1
        elif mid_val > oct_prg:
            hi = mid_a
        else:
            return mid_a
    raise Exception("search failed")


def find_lowest(registers: Registers, program: list[int], hi_a: int) -> int:
    out_a = hi_a
    oct_prg = program_to_octal(program)
    for a in range(hi_a, -1, -1):
        computer = Computer(registers, program)
        computer.a = a
        output = computer.execute()
        oct_out = program_to_octal(output)
        print("looking for lowest", hi_a, oct_out, oct_prg)
        if oct_out < oct_prg:
            break
        out_a = a
    return out_a


def part_2(input):
    # def test_6():
    #     registers, program = parse(CONTROL_2)
    #     comp = Computer(registers, program)
    #     comp.a = 117440
    #     out = comp.execute()
    #     assert out == [0, 3, 5, 4, 3, 0], f"{repr(comp)} {out}"
    #     print("test 6 passed")
    #
    # test_6()
    #
    registers, program = parse(input)
    # mid_a = search_oct(registers, program)
    # computer = Computer(registers, program)
    # computer.a = mid_a
    # output = computer.execute()
    # print(mid_a, program, output)
    #
    # low_a = find_lowest(registers, program, mid_a)
    # print("low a", low_a)
    for sig in range(0, 8):
        a = sig << (3 * (len(program) - 2))
        computer = Computer(registers, program)
        computer.a = a
        output = computer.execute()
        print(output)


if __name__ == "__main__":
    part_1(input_file)
    part_2(CONTROL_1)
