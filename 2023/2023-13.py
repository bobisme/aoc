#!/usr/bin/env python

import numpy as np
from typing import Deque


CONTROL_1 = """\
#.##..##.
..#.##.#.
##......#
##......#
..#.##.#.
..##..##.
#.#.##.#.

#...##..#
#....#..#
..##..###
#####.##.
#####.##.
..##..###
#....#..#
""".splitlines()

CONTROL_2 = """\
..##..##.
..#.##.#.
##......#
##......#
..#.##.#.
..##..##.
#.#.##.#.

#...##..#
#....#..#
..##..###
#####.##.
#####.##.
..##..###
#....#..#
""".splitlines()

with open("2023-13.input") as f:
    input = [line.strip() for line in f.readlines()]


def parse_patterns(input):
    start = 0
    for i in range(len(input)):
        if input[i] == "":
            yield [list(line) for line in input[start:i]]
            start = i + 1
    yield [list(line) for line in input[start:]]


def levenshtein(a, b):
    distances = np.zeros((len(a) + 1, len(b) + 1))
    for t1 in range(len(a) + 1):
        distances[t1][0] = t1

    for t2 in range(len(b) + 1):
        distances[0][t2] = t2

    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            if a[i - 1] == b[j - 1]:
                cost = 0
            else:
                cost = 1

            distances[i][j] = min(
                distances[i - 1][j] + 1,  # Deletion
                distances[i][j - 1] + 1,  # Insertion
                distances[i - 1][j - 1] + cost,  # Substitution
            )
    return distances[len(a)][len(b)]


def check_mirror(pattern: np.array, previously_fixed: bool) -> (int, bool):
    # for line in pattern:
    #     print("".join(line))
    d = list()
    mirror_after = 0
    smudge_fixed = False
    for j in range(len(pattern)):
        if j != 0:
            p_slice = pattern[j : j + len(d)]
            d_slice = np.array(list(reversed(d))[: len(p_slice)])
            L = 0
            for i in range(len(p_slice)):
                L += levenshtein(p_slice[i], d_slice[i])
            if L == 1.0:
                mirror_after = j
                smudge_fixed = True
                break
        d.append(pattern[j])
    return mirror_after, smudge_fixed


def value_of_pattern(pattern):
    rows = np.array(pattern)
    cols = np.transpose(rows)
    # print(rows)
    hm, vm = 0, 0
    hm, fixed = check_mirror(rows, False)
    if fixed is False:
        vm, fixed = check_mirror(cols, fixed)

    return vm + 100 * hm


def main(input):
    patterns = list(parse_patterns(input))
    x = sum(value_of_pattern(pattern) for pattern in patterns)
    print(f"{x=}")


if __name__ == "__main__":
    main(input)
