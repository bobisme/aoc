#!/usr/bin/env python

import itertools
import re
from typing import LiteralString, NamedTuple
import time


Input = list[str] | list[LiteralString]

CONTROL_1: Input = (
    """\
Comet can fly 14 km/s for 10 seconds, but then must rest for 127 seconds.
Dancer can fly 16 km/s for 11 seconds, but then must rest for 162 seconds.
""".splitlines()
)


class Stats(NamedTuple):
    speed: int
    flight_time: int
    rest_time: int


def parse(input: Input) -> dict[str, Stats]:
    pattern = re.compile(
        r"(\w+) can fly (\d+) km/s for (\d+) seconds, but then must rest for (\d+) seconds."
    )
    data = {}
    for line in input:
        matches = next(pattern.finditer(line))
        assert matches is not None
        reindeer, speed, flight_time, rest_time = matches.groups()
        data[reindeer] = Stats(int(speed), int(flight_time), int(rest_time))
    return data


def simulate(stats: Stats, end_t: int) -> float:
    dist = 0.0
    t = 0
    for mode in itertools.cycle(("flying", "resting")):
        rem_t = end_t - t
        if rem_t <= 0:
            return dist
        if mode == "flying":
            d_t = min(stats.flight_time, rem_t)
            dist += d_t * stats.speed
            t += d_t
        else:
            t += min(stats.rest_time, rem_t)
    assert False


def part_1(input: Input, t: int):
    stats = parse(input)
    return int(max(simulate(s, t) for s in stats.values()))


def part_2(input: Input, max_t: int):
    stats = parse(input)
    scores = {reindeer: 0 for reindeer in stats.keys()}
    for t in range(1, max_t + 1):
        lead, _ = max(
            ((rd, simulate(s, t)) for (rd, s) in stats.items()), key=lambda x: x[1]
        )
        scores[lead] += 1
    return max(scores.values())


def _test():
    def assert_eq(a, b):
        assert a == b, f"{a} != {b}"

    assert_eq(part_1(CONTROL_1, 1000), 1120)


def run(fn, year=2015, day=14, part=0):
    start = time.perf_counter_ns()
    res = fn()
    elapsed_ns = time.perf_counter_ns() - start
    print(f"{year}\t{day}\t{part}\t{res}\t{elapsed_ns}")
    return res


if __name__ == "__main__":
    with open("2015-14.input") as f:
        input_file = [line.rstrip("\n") for line in f.readlines()]
    _test()
    run(lambda: part_1(input_file, 2503), part=1)
    run(lambda: part_2(input_file, 2503), part=2)
