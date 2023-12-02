#!/usr/bin/env python

import re

CONTROL = """\
1abc2
pqr3stu8vwx
a1b2c3d4e5f
treb7uchet
""".split()
CONTROL_2 = """\
two1nine
eightwothree
abcone2threexyz
xtwone3four
4nineeightseven2
zoneight234
7pqrstsixteen
""".split()

map = {
    "0": 0,
    "1": 1,
    "2": 2,
    "3": 3,
    "4": 4,
    "5": 5,
    "6": 6,
    "7": 7,
    "8": 8,
    "9": 9,
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
}


def to_int(val):
    return map[val]


with open("2023-1.input") as f:
    input = [line.strip() for line in f.readlines()]


def get_digits(val):
    found = [int(x) if x else 0 for x in re.findall(r"\d", val)]
    return found if found else [0]


def get_matches(val):
    matches = []
    for k, v in map.items():
        i = val.find(k)
        if i >= 0:
            matches.append((i, v))
        i = val.rfind(k)
        if i >= 0:
            matches.append((i, v))
    matches.sort()
    return matches


def digit_calibration(line):
    matches = list(get_digits(line))
    return matches[0] * 10 + matches[-1]


def calibration(line):
    matches = get_matches(line)
    return matches[0][1] * 10 + matches[-1][1]


test_digit_cal = digit_calibration("ddgjgcrssevensix37twooneightgt")
assert test_digit_cal == 37, test_digit_cal
test_cal = calibration("ddgjgcrssevensix37twooneightgt")
assert test_cal == 78, test_cal

for line in CONTROL:
    print(line, digit_calibration(line), calibration(line))


def sums(lines):
    print("digit sum", sum(digit_calibration(line) for line in lines))
    print("sum", sum(calibration(line) for line in lines))


sums(CONTROL)
sums(CONTROL_2)
sums(input)
