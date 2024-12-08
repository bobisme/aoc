#!/usr/bin/env python

from collections.abc import Generator, Iterable


CONTROL_1 = """\
7 6 4 2 1
1 2 7 8 9
9 7 6 2 1
1 3 2 4 5
8 6 4 4 1
1 3 6 7 9
""".splitlines()

with open("2024-2.input") as f:
    input_file = [line.strip() for line in f.readlines()]


def is_safe(report: list[int], accept=0) -> bool:
    last = report[0]
    mode = 0
    errors = 0
    for x in report[1:]:
        if not (1 <= abs(x - last) <= 3):
            errors += 1
        if x > last:
            if mode == 0:
                mode = 1
            elif mode == 1:
                pass
            else:
                errors += 1
        elif x < last:
            if mode == 0:
                mode = -1
            elif mode == -1:
                pass
            else:
                errors += 1
        else:
            errors += 1
        last = x
    return errors <= accept


def part_1(input):
    safe = 0
    for line in input:
        report = [int(x) for x in line.split(" ")]
        if is_safe(report):
            safe += 1
    print(safe)


def without(iter: Iterable[int], index: int) -> Generator[int, None, None]:
    return (x for (i, x) in enumerate(iter) if i != index)


def part_2(input):
    safe = 0
    for line in input:
        report = [int(x) for x in line.split(" ")]
        for i in range(len(report)):
            if is_safe(list(without(report, i))):
                safe += 1
                break
    print(safe)


if __name__ == "__main__":
    part_2(input_file)
