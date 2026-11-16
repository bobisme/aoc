#!/usr/bin/env python

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal, LiteralString

Input = list[str] | list[LiteralString]

CONTROL_1: Input = """\
value 5 goes to bot 2
bot 2 gives low to bot 1 and high to bot 0
value 3 goes to bot 1
bot 1 gives low to output 1 and high to bot 0
bot 0 gives low to output 2 and high to output 0
value 2 goes to bot 2
""".splitlines()

PART_1_OUT = None


@dataclass(slots=True)
class Inst:
    low_type: str
    low_idx: int
    high_type: str
    high_idx: int


@dataclass
class Bot:
    id: int
    instructions: Inst
    chips: list[int] = field(default_factory=list)

    def give(self, chip: int) -> None:
        self.chips.append(chip)
        if len(self.chips) == 2:
            self.chips.sort()
            if self.chips[0] == 17 and self.chips[1] == 61:
                global PART_1_OUT
                PART_1_OUT = self.id

    def act(self, bots: dict[int, "Bot"], outputs: dict[int, int]):
        if len(self.chips) < 2:
            return
        if self.instructions.low_type == "bot":
            bots[self.instructions.low_idx].give(self.chips[0])
            bots[self.instructions.low_idx].act(bots, outputs)
        else:
            outputs[self.instructions.low_idx] = self.chips[0]
        if self.instructions.high_type == "bot":
            bots[self.instructions.high_idx].give(self.chips[1])
            bots[self.instructions.high_idx].act(bots, outputs)
        else:
            outputs[self.instructions.high_idx] = self.chips[1]
        self.chips = []


def run_sim(input: Input):
    bots: dict[int, Bot] = {}
    outputs = {}
    distributions = []
    for line in input:
        if line.startswith("value"):
            val, bot_id = line.split(" goes to bot ")
            distributions.append((int(val.replace("value ", "")), int(bot_id)))
        else:
            bot_id, rest = line.split(" gives low to ", 1)
            low, high = rest.split(" and high to ", 1)
            bot_id = int(bot_id.replace("bot ", ""))
            low_dst, low_idx = low.split(" ")
            high_dst, high_idx = high.split(" ")
            low_idx, high_idx = int(low_idx), int(high_idx)
            bots[bot_id] = Bot(
                bot_id, instructions=Inst(low_dst, low_idx, high_dst, high_idx)
            )
    for val, bot_idx in distributions:
        bots[bot_idx].give(val)
        bots[bot_idx].act(bots, outputs)
    return outputs


def part_1(input: Input):
    run_sim(input)
    return PART_1_OUT


def part_2(input: Input):
    outputs = run_sim(input)
    return outputs[0] * outputs[1] * outputs[2]


def run(fn, year=2016, day=10, part=0):
    start = time.perf_counter_ns()
    res = fn()
    elapsed_ns = time.perf_counter_ns() - start
    print(f"{year}\t{day}\t{part}\t{res}\t{elapsed_ns}")
    return res


if __name__ == "__main__":
    with open("2016-10.input") as f:
        input_file = [line.rstrip("\n") for line in f]
    run(lambda: part_1(input_file), part=1)
    run(lambda: part_2(input_file), part=2)
