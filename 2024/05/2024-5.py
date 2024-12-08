#!/usr/bin/env python

from functools import cmp_to_key
from pprint import pp

CONTROL_1 = """\
47|53
97|13
97|61
97|47
75|29
61|13
75|53
29|13
97|29
53|29
61|53
97|53
61|29
47|13
75|47
97|75
47|61
75|61
47|29
75|13
53|13

75,47,61,53,29
97,61,53,29,13
75,29,13
75,97,47,61,53
61,13,29
97,13,75,29,47
""".splitlines()

with open("2024-5.input") as f:
    input_file = [line.strip() for line in f.readlines()]


def is_pair_ordered(order: list[list[int]], a: int, b: int) -> bool:
    if a in order[b]:
        return False
    return True


def is_ordered(order: list[list[int]], update: list[int]) -> bool:
    if len(update) <= 1:
        return True
    a = update[0]
    for b in update[1:]:
        if not is_pair_ordered(order, a, b):
            return False
    return is_ordered(order, update[1:])


def middle(update: list[int]) -> int:
    return update[len(update) // 2]


def part_1(input):
    order = [[] for _ in range(100)]
    updates = []
    mode = 0
    for line in input:
        if mode == 0:
            if not line:
                mode = 1
                continue
            a, b = line.split("|")
            order[int(a)].append(int(b))
        else:
            updates.append([int(x) for x in line.split(",")])

    out = 0
    for update in updates:
        if is_ordered(order, update):
            out += middle(update)
    print(out)


def part_2(input):
    order = [[] for _ in range(100)]
    updates = []
    mode = 0
    for line in input:
        if mode == 0:
            if not line:
                mode = 1
                continue
            a, b = line.split("|")
            order[int(a)].append(int(b))
        else:
            updates.append([int(x) for x in line.split(",")])

    def cmp(a: int, b: int) -> int:
        if b in order[a]:
            return 1
        if a in order[b]:
            return -1
        return 0

    out = 0
    for update in updates:
        if not is_ordered(order, update):
            srt = sorted(update, key=cmp_to_key(cmp))
            out += middle(srt)
    print(out)


if __name__ == "__main__":
    part_1(input_file)
    part_2(input_file)
