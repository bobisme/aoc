#!/usr/bin/env python
import re
from collections import defaultdict

CONTROL_1 = """\
3   4
4   3
2   5
1   3
3   9
3   3
""".splitlines()

with open("2024-1.input") as f:
    input_file = [line.strip() for line in f.readlines()]


def part_1(input):
    left, right = [], []
    for line in input:
        a, b = re.split(r"\s+", line)
        # print(l, r)
        left.append(int(a))
        right.append(int(b))
    left.sort()
    right.sort()
    z = list(zip(left, right))
    # print(z)
    d = list(abs(a - b) for (a, b) in z)
    # print(d)
    s = sum(d)
    print(s)


def part_2(input):
    counts = defaultdict(int)
    left, right = [], []
    for line in input:
        a, b = re.split(r"\s+", line)
        left.append(int(a))
        right.append(int(b))
    for x in right:
        counts[x] += 1
    s = sum(x * counts[x] for x in left)
    print(s)


if __name__ == "__main__":
    part_2(input_file)
