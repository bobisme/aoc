#!/usr/bin/env python

from collections import namedtuple
from dataclasses import dataclass
from functools import reduce
import heapq
import math
from typing import LiteralString
import time

Input = list[str] | list[LiteralString]

CONTROL_1: Input = (
    """\
""".splitlines()
)


@dataclass
class Item:
    name: str
    cost: int
    damage: int = 0
    armor: int = 0


WEAPONS = (
    Item(name="Dagger", cost=8, damage=4),
    Item(name="Shortsword", cost=10, damage=5),
    Item(name="Warhammer", cost=25, damage=6),
    Item(name="Longsword", cost=40, damage=7),
    Item(name="Greataxe", cost=74, damage=8),
)

ARMOR = (
    Item(name="Leather", cost=13, armor=1),
    Item(name="Chainmail", cost=31, armor=2),
    Item(name="Splintmail", cost=53, armor=3),
    Item(name="Bandedmail", cost=75, armor=4),
    Item(name="Steel armor", cost=102, armor=5),
)

RINGS = (
    Item(name="Damage +1", cost=25, damage=1),
    Item(name="Damage +2", cost=50, damage=2),
    Item(name="Damage +3", cost=100, damage=3),
    Item(name="Defense +1", cost=20, armor=1),
    Item(name="Defense +2", cost=40, armor=2),
    Item(name="Defense +3", cost=80, armor=3),
)

ITEMS = WEAPONS + ARMOR + RINGS


@dataclass
class Boss:
    hp: int
    damage: int
    armor: int


def parse(input: Input) -> Boss:
    return Boss(
        hp=int(input[0][-3:]),
        damage=int(input[1][-1:]),
        armor=int(input[2][-1:]),
    )


def wins(boss: Boss, player_damage: int, player_armor: int) -> bool:
    return math.ceil(boss.hp / max(player_damage - boss.armor, 1)) <= math.ceil(
        100 / max(boss.damage - player_armor, 1)
    )


@dataclass(slots=True)
class QNode:
    idx: int
    damage: int
    armor: int
    cost: int
    prev_idxs: tuple[int, ...]

    def __lt__(self, other) -> bool:
        return self.cost < other.cost


def part_1_search(input: Input) -> int:
    boss = parse(input)
    print(boss)
    best_cost = 10**10
    best: tuple[int, tuple[int, ...]] | None = None
    item_pool: list[Item] = list(ARMOR)
    for item in RINGS:
        item_pool.extend([item] * 2)
    end_idx = len(item_pool) - 1
    for widx, weapon in enumerate(WEAPONS):
        print(f"weapon {weapon}")
        q = [QNode(0, weapon.damage, 0, weapon.cost, tuple())]
        while q:
            n = heapq.heappop(q)
            if n.cost >= best_cost:
                continue
            item = item_pool[n.idx]
            if n.idx == end_idx:
                if wins(boss, n.damage, n.armor):
                    print(f"wins with cost {n.cost}")
                    if n.cost < best_cost:
                        best_cost = n.cost
                        best = (widx, n.prev_idxs)
                if wins(boss, n.damage + item.damage, n.armor + item.armor):
                    print(f"wins with cost {n.cost + item.cost}")
                    if n.cost + item.cost < best_cost:
                        best_cost = n.cost + item.cost
                        best = (widx, n.prev_idxs + (n.idx,))
            else:
                heapq.heappush(
                    q,
                    QNode(
                        n.idx + 1,
                        n.damage + item.damage,
                        n.armor + item.armor,
                        n.cost + item.cost,
                        n.prev_idxs + (n.idx,),
                    ),
                )
                heapq.heappush(
                    q, QNode(n.idx + 1, n.damage, n.armor, n.cost, n.prev_idxs)
                )
    print(f"best = {best}")
    assert best_cost != 85
    return best_cost


def part_2(input: Input):
    boss = parse(input)
    return 0


def _test():
    def assert_eq(a, b):
        assert a == b, f"{a} != {b}"

    boss = Boss(hp=12, damage=7, armor=2)
    assert_eq(wins(boss, player_damage=5, player_armor=5), True)
    # assert_eq(part_1(CONTROL_1), 0)
    # assert_eq(part_2(CONTROL_1), 0)


def run(fn, year=2015, day=21, part=0):
    start = time.perf_counter_ns()
    res = fn()
    elapsed_ns = time.perf_counter_ns() - start
    print(f"{year}\t{day}\t{part}\t{res}\t{elapsed_ns}")
    return res


if __name__ == "__main__":
    with open("2015-21.input") as f:
        input_file = [line.rstrip("\n") for line in f.readlines()]
    _test()
    run(lambda: part_1_search(input_file), part=1)
    # run(lambda: part_2(input_file), part=2)
