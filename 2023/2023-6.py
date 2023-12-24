#!/usr/bin/env python

from dataclasses import dataclass
from functools import reduce
import math
import re


@dataclass
class Race:
    time: int
    record_distance: int


CONTROL_1 = """\
Time:      7  15   30
Distance:  9  40  200
""".splitlines()

with open("2023-6.input") as f:
    input = [line.strip() for line in f.readlines()]


# held = x
# remaining = time - held
# speed = held
# velocity = speed / distance
def quadratic(a: int, b: int, c: int) -> tuple[float, float]:
    s = (b**2 - 4 * a * c) ** 0.5
    plus = (-b + s) / (2 * a)
    minus = (-b - s) / (2 * a)
    return plus, minus


def main(input):
    print(input)
    times = [int(x) for x in re.findall(r"\d+", input[0])]
    distances = [int(x) for x in re.findall(r"\d+", input[1])]
    races = [
        Race(time=time, record_distance=distances[i]) for i, time in enumerate(times)
    ]
    legal = []
    for race in races:
        low, high = quadratic(-1, race.time, -race.record_distance)
        low, high = math.ceil(low + 1e-6), math.ceil(high)
        r = range(math.ceil(low), math.ceil(high))
        legal.append(len(r))
        # print(low, high)
    print(legal)
    print(reduce(lambda acc, x: acc * x, legal, 1))
    final_time = [int(x) for x in re.findall(r"\d+", input[0].replace(" ", ""))][0]
    final_dist = [int(x) for x in re.findall(r"\d+", input[1].replace(" ", ""))][0]
    race = Race(final_time, final_dist)
    print(race)
    low, high = quadratic(-1, final_time, -final_dist)
    low, high = math.ceil(low + 1e-6), math.ceil(high)
    r = range(math.ceil(low), math.ceil(high))
    print(len(r))


if __name__ == "__main__":
    main(input)
