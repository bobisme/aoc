#!/usr/bin/env python
from dataclasses import dataclass
from string import digits
from collections import defaultdict
from pprint import pp

CONTROL_1 = """\
32T3K 765
T55J5 684
KK677 28
KTJJT 220
QQQJA 483
""".splitlines()

CARD_RANKS = [str(x) for x in range(2, 10)] + list("TJQKA")
CARD_RANK_MAP = dict((c, i) for (i, c) in enumerate(CARD_RANKS))

JOKER_RANKS = ["J"] + [str(x) for x in range(2, 10)] + list("TQKA")
JOKER_RANK_MAP = dict((c, i) for (i, c) in enumerate(JOKER_RANKS))
pp(JOKER_RANK_MAP)


def card_counts(cards: str, rankmap=CARD_RANK_MAP) -> list[tuple[int, str]]:
    counts = defaultdict(int)
    for card in cards:
        counts[card] += 1
    return list(
        reversed(
            sorted(
                ((count, card) for (card, count) in counts.items()),
                key=lambda x: (x[0], rankmap[x[1]]),
            )
        )
    )


def hand_score(cards, rankmap=CARD_RANK_MAP):
    return tuple(rankmap[c] for c in cards)


def rank_key(cards, counts, rankmap=CARD_RANK_MAP):
    sort_list = (
        tuple((x[0]) for x in counts),
        *hand_score(cards, rankmap=rankmap),
    )
    return sort_list


def joker_rank_key(cards):
    counts = card_counts(cards, rankmap=JOKER_RANK_MAP)
    possible_hands = [counts]
    jokers = 0
    for count in counts:
        if count[1] == "J":
            jokers = count[0]
    if jokers == 0:
        return (
            tuple((x[0]) for x in counts),
            hand_score(cards, rankmap=JOKER_RANK_MAP),
        )
    # no_wild_counts, wild_counts = counts
    # counts = max((no_wild_counts, wild_counts))
    # counts = self.counts_with_jokers()
    # non_wild_counts = (
    #     tuple((x[0]) for x in counts),
    #     hand_score(cards, rankmap=JOKER_RANK_MAP),
    # )
    for i, card_count in enumerate(counts):
        with_wildcards = counts.copy()
        count, card = with_wildcards[i]
        count = count if card != "J" else 0
        with_wildcards[i] = (count + jokers, card)
        with_wildcards = [x for x in with_wildcards if x[1] != "J"]
        with_wildcards.sort(key=lambda x: (x[0], JOKER_RANK_MAP[x[1]]))
        with_wildcards.reverse()
        possible_hands.append(with_wildcards)
    # pp(possible_hands)
    # exit(0)
    counts = max(possible_hands)
    return (
        tuple((x[0]) for x in counts),
        hand_score(cards, rankmap=JOKER_RANK_MAP),
    )


@dataclass
class Hand:
    cards: str
    bid: int

    def __lt__(self, other):
        return joker_rank_key(self.cards) < joker_rank_key(other.cards)


with open("2023-7.input") as f:
    input = [line.strip() for line in f.readlines()]


def main(input):
    hands = []
    for line in input:
        cards, bid = line.split(" ", 1)
        hands.append(Hand(cards=cards, bid=int(bid)))
    # pp(hands)
    # counts = [(hand.counts(), int(hand.bid)) for hand in hands]
    # pp(counts)
    ordered = list(sorted(hands))
    pp([(hand.cards, joker_rank_key(hand.cards)) for hand in ordered])
    winnings = [(i + 1) * hand.bid for i, hand in enumerate(ordered)]
    # pp(winnings)
    pp(sum(winnings))
    # assert winnings == 6440


if __name__ == "__main__":
    main(input)
