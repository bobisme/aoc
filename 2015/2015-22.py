#!/usr/bin/env python

from enum import Enum
import heapq
from dataclasses import dataclass, field
from typing import LiteralString, NamedTuple
import time

Input = list[str] | list[LiteralString]


@dataclass(slots=True)
class Player:
    health: int
    damage: int = 0
    armor: int = 0
    mana: int = 0

    def copy(self) -> "Player":
        return Player(self.health, self.damage, self.armor, self.mana)


@dataclass(slots=True)
class Boss:
    health: int
    damage: int = 0

    def copy(self) -> "Boss":
        return Boss(self.health, self.damage)


class Effects(NamedTuple):
    shield: int = 0
    poison: int = 0
    recharge: int = 0

    def apply(self, boss: Boss, player: Player) -> tuple[Boss, Player, "Effects"]:
        boss = boss.copy()
        player = player.copy()
        if self.shield > 0:
            player.armor = 7
        else:
            player.armor = 0
        if self.poison > 0:
            boss.health -= 3
        if self.recharge > 0:
            player.mana += 101
        return (
            boss,
            player,
            Effects(
                max(self.shield - 1, 0),
                max(self.poison - 1, 0),
                max(self.recharge - 1, 0),
            ),
        )


class EffectAlreadyApplied(Exception):
    pass


class Spell(Enum):
    """
    Magic Missile costs 53 mana. It instantly does 4 damage.
    Drain costs 73 mana. It instantly does 2 damage and heals you for 2 hit points.
    Shield costs 113 mana. It starts an effect that lasts for 6 turns. While it is active, your armor is increased by 7.
    Poison costs 173 mana. It starts an effect that lasts for 6 turns. At the start of each turn while it is active, it deals the boss 3 damage.
    Recharge costs 229 mana. It starts an effect that lasts for 5 turns. At the start of each turn while it is active, it gives you 101 new mana.
    """

    MagicMissile = 1
    Drain = 2
    Shield = 3
    Poison = 4
    Recharge = 5

    def cost(self) -> int:
        match self:
            case self.MagicMissile:
                return 53
            case self.Drain:
                return 73
            case self.Shield:
                return 113
            case self.Poison:
                return 173
            case self.Recharge:
                return 229

    def cast(
        self, boss: Boss, player: Player, effects: Effects
    ) -> tuple[Boss, Player, Effects]:
        boss = boss.copy()
        player = player.copy()
        player.mana -= self.cost()
        match self:
            case self.MagicMissile:
                boss.health -= 4
            case self.Drain:
                boss.health -= 2
                player.health += 2
            case self.Shield:
                if effects.shield > 0:
                    raise EffectAlreadyApplied()
                effects = Effects(6, effects.poison, effects.recharge)
            case self.Poison:
                if effects.poison > 0:
                    raise EffectAlreadyApplied()
                effects = Effects(effects.shield, 6, effects.recharge)
            case self.Recharge:
                if effects.recharge > 0:
                    raise EffectAlreadyApplied()
                effects = Effects(effects.shield, effects.poison, 5)
        return boss, player, effects


INSTANTS = (Spell.MagicMissile, Spell.Drain)
EFFECTS = (Spell.Shield, Spell.Poison, Spell.Recharge)
SPELLS = INSTANTS + EFFECTS


def parse(input: Input) -> Boss:
    return Boss(
        health=int(input[0].split(": ", maxsplit=1)[1]),
        damage=int(input[1].split(": ", maxsplit=1)[1]),
    )


@dataclass(slots=True)
class QNode:
    boss: Boss
    player: Player
    effects: Effects = field(default_factory=Effects)
    total_mana: int = 0
    turn: int = 0
    log: list[Spell] = field(default_factory=list)

    def __lt__(self, other: "QNode") -> bool:
        # return self.player.mana < other.player.mana
        return self.heuristic() < other.heuristic()

    def heuristic(self):
        return self.total_mana


def search(boss: Boss, player: Player, hard_mode=False) -> int:
    q = [QNode(boss, player)]
    best_mana = 10**10

    i = 0
    while q:
        i += 1
        n = heapq.heappop(q)
        if n.total_mana > best_mana:
            continue
        is_player_turn = n.turn % 2 == 0

        boss, player = n.boss, n.player.copy()

        # check if someone is dead
        if boss.health <= 0:
            if n.total_mana < best_mana:
                best_mana = n.total_mana
            continue

        if hard_mode and is_player_turn:
            player.health -= 1

        if player.health <= 0:
            continue

        boss, player, next_effects = n.effects.apply(boss, player)

        # check if boss is dead again
        if boss.health <= 0:
            if n.total_mana < best_mana:
                best_mana = n.total_mana
            continue

        # choose spells
        if is_player_turn:
            was_cast = False
            for spell in SPELLS:
                if spell.cost() > player.mana:
                    continue
                try:
                    b_next, p_next, e_next = spell.cast(boss, player, next_effects)
                except EffectAlreadyApplied:
                    continue
                was_cast = True
                heapq.heappush(
                    q,
                    QNode(
                        b_next,
                        p_next,
                        e_next,
                        total_mana=n.total_mana + spell.cost(),
                        turn=n.turn + 1,
                        log=n.log + [spell],
                    ),
                )
            if not was_cast:  # LOSE!
                continue
        else:  # boss turn
            b_damage = max(boss.damage - player.armor, 1)
            player.health -= b_damage
            heapq.heappush(
                q,
                QNode(
                    boss,
                    player,
                    next_effects,
                    total_mana=n.total_mana,
                    turn=n.turn + 1,
                    log=n.log,
                ),
            )
    return best_mana


def part_1(input: Input):
    boss = parse(input)
    best = search(boss, Player(health=50, mana=500))
    assert best == 953, best
    return best


def part_2(input: Input):
    boss = parse(input)
    best = search(boss, Player(health=50, mana=500), hard_mode=True)
    assert 900 < best < 1295, best
    return best


def _test():
    def assert_eq(a, b):
        assert a == b, f"{a} != {b}"

    assert_eq(search(Boss(health=13, damage=8), Player(health=10, mana=250)), 226)
    assert_eq(search(Boss(health=14, damage=8), Player(health=10, mana=250)), 641)

    # assert_eq(part_1(CONTROL_1), 0)
    # assert_eq(part_2(CONTROL_1), 0)


def run(fn, year=2015, day=22, part=0):
    start = time.perf_counter_ns()
    res = fn()
    elapsed_ns = time.perf_counter_ns() - start
    print(f"{year}\t{day}\t{part}\t{res}\t{elapsed_ns}")
    return res


if __name__ == "__main__":
    with open("2015-22.input") as f:
        input_file = [line.rstrip("\n") for line in f.readlines()]
    _test()
    run(lambda: part_1(input_file), part=1)
    run(lambda: part_2(input_file), part=2)
