#!/usr/bin/env python

from typing import NamedTuple
import re

CONTROL_1 = """\
Card 1: 41 48 83 86 17 | 83 86  6 31 17  9 48 53
Card 2: 13 32 20 16 61 | 61 30 68 82 17 32 24 19
Card 3:  1 21 53 59 44 | 69 82 63 72 16 21 14  1
Card 4: 41 92 73 84 69 | 59 84 76 51 58  5 54 83
Card 5: 87 83 26 28 32 | 88 30 70 12 93 22 82 36
Card 6: 31 18 13 56 72 | 74 77 10 23 35 67 36 11
""".splitlines()

with open("2023-4.input") as f:
    input = [line.strip() for line in f.readlines()]


class Card(
    NamedTuple("Card", [("id", int), ("winners", set[int]), ("numbers", set[int])])
):
    @staticmethod
    def parse(line: str):
        id = re.findall(r"(\d+):", line)[0]
        id = int(id)
        _, rest = line.split(":", 1)
        w, n = rest.split("|", 1)
        winners = {int(x) for x in re.findall(r"\d+", w)}
        numbers = {int(x) for x in re.findall(r"\d+", n)}
        return Card(id, winners, numbers)

    def matches(self) -> set:
        return self.winners & self.numbers

    def match_count(self) -> int:
        return len(self.matches())

    def score(self) -> int:
        m = self.match_count()
        if m == 0:
            return 0
        return 2 ** (m - 1)


def main(input):
    print("sum", sum(Card.parse(line).score() for line in input))


def main2(input):
    card_map: dict[int, Card] = dict()
    score_map: dict[int, int] = dict()
    count_map: dict[int, int] = dict()
    won: list[int] = []
    q: list[int] = []
    for line in input:
        card = Card.parse(line)
        card_map[card.id] = card
        score_map[card.id] = card.match_count()
        count_map[card.id] = 1
        q.append(card.id)

    for id, _ in card_map.items():
        card = card_map[id]
        m = card.match_count()
        card_count = count_map[id]
        if m == 0:
            continue
        new_ids = list(range(card.id + 1, card.id + m + 1))
        print(new_ids)
        for id in new_ids:
            count_map[id] += card_count
        print(count_map)
    print("sum = ", sum(v for (k, v) in count_map.items()))


if __name__ == "__main__":
    main2(input)
