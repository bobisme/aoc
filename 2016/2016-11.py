#!/usr/bin/env python

import heapq
import re
import time
from collections.abc import Generator
from dataclasses import dataclass
from enum import Enum
from itertools import chain, combinations
from typing import LiteralString, Self

Input = list[str] | list[LiteralString]

CONTROL_1: Input = """\
The first floor contains a hydrogen-compatible microchip and a lithium-compatible microchip.
The second floor contains a hydrogen generator.
The third floor contains a lithium generator.
The fourth floor contains nothing relevant.
""".splitlines()

# Rules:
# - If a chip is near an RTG and is not connected to its own, it's fried
# - Elevator can only carry 2 things
# - Elevator only works if carrying at least 1 RTG or chip
# - Elevator always stops on each floor -- items will irradiate each other
# - You and elevator start on first floor
# Goal:
# - Get everything to 4th floor intact


class Dir(Enum):
    UP = 1
    DOWN = 2


@dataclass(slots=True)
class RTG:
    element: str

    def __hash__(self) -> int:
        return hash(self.element)


@dataclass(slots=True)
class Microchip:
    element: str

    def __hash__(self) -> int:
        return hash(self.element)


@dataclass(slots=True)
class Move:
    dir: Dir
    items: tuple[
        RTG | Microchip | None,
        RTG | Microchip | None,
    ]


@dataclass
class State:
    chips: list[set[str]]
    generators: list[set[str]]
    elevator: int = 1

    def __hash__(self) -> int:
        return hash(
            (
                self.elevator,
                tuple(tuple(sorted(floor)) for floor in self.chips),
                tuple(tuple(sorted(floor)) for floor in self.generators),
            )
        )

    @staticmethod
    def from_input(input: Input) -> "State":
        chip_pattern = re.compile(r"(\w+)-compatible microchip")
        generator_pattern = re.compile(r"(\w+) generator")

        chips: list[set[str]] = []
        generators: list[set[str]] = []

        for line in input:
            found_chips = {x for x in chip_pattern.findall(line)}
            found_generators = {x for x in generator_pattern.findall(line)}
            chips.append(found_chips)
            generators.append(found_generators)

        return State(chips, generators)

    def print(self) -> None:
        elements = sorted(chain(*self.chips))
        symbols = [x[:2].capitalize() for x in elements]
        for floor in range(4, 0, -1):
            print(f"F{floor}", end=" ")
            if self.elevator == floor:
                print("E ", end=" ")
            else:
                print(". ", end=" ")
            for ei, element in enumerate(elements):
                if element in self.generators[floor - 1]:
                    print(f"{symbols[ei]}G", end=" ")
                else:
                    print(".  ", end=" ")
                if element in self.chips[floor - 1]:
                    print(f"{symbols[ei]}M", end=" ")
                else:
                    print(".  ", end=" ")
            print()
        print()

    def clone(self) -> "State":
        return State(
            chips=[floor.copy() for floor in self.chips],
            generators=[floor.copy() for floor in self.generators],
            elevator=self.elevator,
        )

    def is_stable(self) -> bool:
        for floor in range(len(self.chips)):
            generators = self.generators[floor]
            chips = self.chips[floor]

            if not generators:
                continue

            if any(chip not in generators for chip in chips):
                return False
        return True

    def with_move(self, move: Move) -> "State":
        new = self.clone()
        source = new.elevator - 1
        dest = source + (1 if move.dir == Dir.UP else -1)

        for item in move.items:
            if item is None:
                continue

            floors = new.chips if isinstance(item, Microchip) else new.generators

            floors[source].remove(item.element)
            floors[dest].add(item.element)

        new.elevator = dest + 1
        return new

    def is_lower_empty(self, floor: int) -> bool:
        return all(
            len(self.chips[i]) == 0 and len(self.generators[i]) == 0
            for i in range(floor - 1)
        )

    def moves(self) -> Generator[Move]:
        floor_idx = self.elevator - 1
        items = {Microchip(x) for x in self.chips[floor_idx]} | {
            RTG(x) for x in self.generators[floor_idx]
        }
        cargo = chain(
            ((item, None) for item in items),
            combinations(items, 2),
        )
        directions = []
        if self.elevator < 4:
            directions.append(Dir.UP)
        if self.elevator > 1 and not self.is_lower_empty(self.elevator):
            directions.append(Dir.DOWN)

        for pair in cargo:
            for dir in directions:
                yield Move(dir, pair)

    def valid_moves(self) -> Generator[Move]:
        for move in self.moves():
            new = self.with_move(move)
            if new.is_stable():
                yield move

    def next_states(self) -> Generator["State"]:
        for move in self.moves():
            new = self.with_move(move)
            if new.is_stable():
                yield new

    def floors_remaining(self) -> int:
        out = len(self.chips) - 1
        for i in range(len(self.chips)):
            if len(self.chips[i]) == 0 and len(self.generators[i]) == 0:
                out -= 1
            else:
                break
        return out


@dataclass(slots=True)
class Item:
    state: State
    steps: int
    prev: Self | None = None

    def __hash__(self) -> int:
        return hash(self.state)

    def __lt__(self, other: "Item"):
        return self.steps < other.steps


def bfs(initial_state: State) -> Item:
    q = [Item(initial_state, 0)]
    visited: set[State] = set()
    while len(q) > 0:
        item = heapq.heappop(q)
        if item.state.floors_remaining() == 0:
            return item
        for s in item.state.next_states():
            if s in visited:
                continue
            visited.add(s)
            heapq.heappush(q, Item(s, item.steps + 1, prev=item))
    raise ValueError("crap")


def part_1(input: Input):
    initial_state = State.from_input(input)
    # State.from_input(input).print()
    final = bfs(initial_state)
    # path = [final]
    # item = final
    # while item.prev:
    #     path.append(item.prev)
    #     item = item.prev
    # for item in reversed(path):
    #     item.state.print()
    return final.steps


def part_2(input: Input):
    initial_state = State.from_input(input)
    initial_state.chips[0].add("elerium")
    initial_state.chips[0].add("dilithium")
    initial_state.generators[0].add("elerium")
    initial_state.generators[0].add("dilithium")
    # State.from_input(input).print()
    final = bfs(initial_state)
    return final.steps


def _test():
    def assert_eq(a, b):
        assert a == b, f"{a} != {b}"

    assert_eq(part_1(CONTROL_1), 11)
    # assert_eq(part_2(CONTROL_1), 0)


def run(fn, year=2016, day=11, part=0):
    start = time.perf_counter_ns()
    res = fn()
    elapsed_ns = time.perf_counter_ns() - start
    print(f"{year}\t{day}\t{part}\t{res}\t{elapsed_ns}")
    return res


if __name__ == "__main__":
    with open("2016-11.input") as f:
        input_file = [line.rstrip("\n") for line in f]
    _test()
    run(lambda: part_1(input_file), part=1)
    run(lambda: part_2(input_file), part=2)
