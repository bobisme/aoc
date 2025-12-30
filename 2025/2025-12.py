#!/usr/bin/env python

from dataclasses import dataclass, field
from functools import reduce
from itertools import chain
from threading import local
from typing import Callable, Generator, Iterator, LiteralString
import math
import time

Input = list[str] | list[LiteralString]

CONTROL_1: Input = (
    """\
0:
###
##.
##.

1:
###
##.
.##

2:
.##
###
##.

3:
##.
###
##.

4:
###
#..
###

5:
###
.#.
###

4x4: 0 0 0 0 2 0
12x5: 1 0 1 0 2 2
12x5: 1 0 1 0 3 2
""".splitlines()
)

with open("2025-12.input") as f:
    input_file = [line.rstrip("\n") for line in f.readlines()]

Op = Callable[["Present"], "Present"]


@dataclass
class Present:
    shape: list[list[int]]

    def __repr__(self) -> str:
        return "\n".join(
            "".join("#" if x == 1 else "." for x in row) for row in self.shape
        )

    def __hash__(self) -> int:
        return hash(repr(self))

    @staticmethod
    def from_input(input: Input) -> "Present":
        data = [[int(x == "#") for x in row] for row in input]
        return Present(shape=data)

    def apply(self, ops: list[Op]) -> "Present":
        return reduce(lambda p, op: op(p), ops, self)


def noop(src: Present) -> Present:
    return Present([row.copy() for row in src.shape.copy()])


def rot_cw(src: Present) -> Present:
    data = [[0 for _ in row] for row in src.shape]
    for src_i, dst_j in ((0, 2), (1, 1), (2, 0)):
        for src_j, dst_i in ((0, 0), (1, 1), (2, 2)):
            data[dst_i][dst_j] = src.shape[src_i][src_j]
    return Present(data)


def rot_ccw(src: Present) -> Present:
    data = [[0 for _ in row] for row in src.shape]
    for src_i, dst_j in ((0, 0), (1, 1), (2, 2)):
        for src_j, dst_i in ((0, 2), (1, 1), (2, 0)):
            data[dst_i][dst_j] = src.shape[src_i][src_j]
    return Present(data)


def flip_v(src: Present) -> Present:
    data = [row.copy() for row in src.shape.copy()]
    data[0], data[2] = data[2], data[0]
    return Present(data)


def flip_h(src: Present) -> Present:
    data = [row.copy() for row in src.shape.copy()]
    for row_i in range(3):
        data[row_i][0], data[row_i][2] = data[row_i][2], data[row_i][0]
    return Present(data)


@dataclass
class PresentGroup:
    elements = (
        (noop,),
        (rot_cw,),
        (rot_cw, rot_cw),
        (rot_ccw,),
        (flip_v,),
        (flip_h,),
        (rot_ccw, flip_v),
        (rot_ccw, flip_h),
    )

    def act(self, element_idx: int, present: Present) -> Present:
        return reduce(lambda p, op: op(p), self.elements[element_idx], present)


class PresentOrbit:
    elements: set[Present]

    def __init__(self, group: PresentGroup, index: int, init: Present):
        self.index = index
        self.init = init
        self.elements = {group.act(i, init) for i in range(len(group.elements))}

    def __repr__(self) -> str:
        return f"{self.index}:\n{repr(self.init)}"

    def __hash__(self) -> int:
        return hash(self.index)

    def __iter__(self) -> Iterator[Present]:
        return self.elements.__iter__()

    def __len__(self) -> int:
        return len(self.elements)

    def min_area(self) -> int:
        return sum(x for row in self.init.shape for x in row)


class PresentSet:
    def __init__(self, group: PresentGroup, init_presents: list[Present]) -> None:
        self.present_orbits = {
            i: PresentOrbit(group, i, p) for (i, p) in enumerate(init_presents)
        }

    def __getitem__(self, idx: int) -> PresentOrbit:
        return self.present_orbits[idx]


EQUIV = [
    ((rot_ccw, rot_ccw), (flip_v, flip_h)),
    ((rot_cw, rot_cw), (rot_ccw, rot_ccw)),
    ((rot_cw, rot_ccw), (noop,)),
    ((rot_ccw, rot_cw), (noop,)),
    ((flip_v, flip_v), (noop,)),
    ((flip_h, flip_h), (noop,)),
]

EQUIV_MAP = {}
for left, right in EQUIV:
    EQUIV_MAP[left] = right
    EQUIV_MAP[right] = left


@dataclass(slots=True)
class Region:
    size: tuple[int, ...]
    qtys: tuple[int, ...]
    area: int = field(init=False)

    def __post_init__(self):
        self.area = self.size[0] * self.size[1]

    @property
    def w(self) -> int:
        return self.size[0]

    @property
    def h(self) -> int:
        return self.size[1]


@dataclass
class Place:
    region: Region

    def __post_init__(self):
        self.area = [[0 for _ in range(self.region.w)] for _ in range(self.region.h)]

    def __repr__(self) -> str:
        return "\n".join(
            "".join("#" if x == 1 else "." for x in row) for row in self.area
        )

    def copy(self) -> "Place":
        other = Place(self.region)
        other.area = [row.copy() for row in self.area]
        return other

    def can_insert(self, present: Present, offset: tuple[int, int]) -> bool:
        x, y = offset
        return all(
            self.area[i + y][j + x] == 0
            for i, row in enumerate(present.shape)
            for j, val in enumerate(row)
            if val == 1
        )

    def insert(self, present: Present, offset: tuple[int, int]):
        x, y = offset
        for i, row in enumerate(present.shape):
            for j, val in enumerate(row):
                if val == 1:
                    self.area[i + y][j + x] = val


def parse(input: Input) -> tuple[list[Present], list[Region]]:
    presents = []
    regions = []
    for i in range(6):
        presents.append(Present.from_input(input[i * 5 + 1 : i * 5 + 4]))
    for line in input[30:]:
        size, idxs = line.split(": ")
        regions.append(
            Region(
                tuple(map(int, size.split("x", maxsplit=1))),
                tuple(map(int, idxs.split(" "))),
            )
        )
    return presents, regions


@dataclass
class QElem:
    place: Place
    # unplaced presents
    orbits: list[PresentOrbit]
    offset: tuple[int, int]


@dataclass
class Record:
    orbits: list[PresentOrbit]
    offset: tuple[int, int]


iters = local()
iters.it = 0


def part_1(input: Input):
    presents, regions = parse(input)
    group = PresentGroup()
    presents = PresentSet(group, presents)

    def try_place(place: Place, orbits: list[PresentOrbit]) -> bool:
        # print()
        # print(place)
        if len(orbits) == 0:
            print("GOT IT")
            # print(place)
            return True
        for x in range(place.region.w - 2):
            for y in range(place.region.h - 2):
                for p in orbits[0]:
                    iters.it += 1
                    # HACK
                    if iters.it > 10_000_000:
                        print("too many iterations")
                        return False
                    if place.can_insert(p, (x, y)):
                        next_place = place.copy()
                        next_place.insert(p, (x, y))
                        if try_place(next_place, orbits[1:]):
                            return True
        return False

    def try_fill(region: Region) -> bool:
        iters.it = 0
        place = Place(region)
        orbits = list(
            chain(*([presents[i]] * count for (i, count) in enumerate(region.qtys)))
        )
        min_area = sum(o.min_area() for o in orbits)
        unpacked_area = sum(9 for _ in orbits)
        # print(f"{min_area=} {region.area()=} {unpacked_area=}")

        # WTF, is this the solution? just filter
        if min_area > region.area:
            return False
        if unpacked_area > region.area * 1.5:
            return False
        return True
        return try_place(place, orbits)

    return sum(try_fill(r) for r in regions)


def part_1_real(input: Input):
    "The real answer."

    def quick_parse(input: Input) -> Generator[tuple[int, int]]:
        for line in input[30:]:
            size, idxs = line.split(": ", maxsplit=1)
            area = math.prod(map(int, size.split("x", maxsplit=1)))
            qtys = sum(map(int, idxs.split(" ")))
            yield area, qtys

    return sum(1 for (area, qtys) in quick_parse(input) if (9 * qtys) <= area)


def run(fn, year=2025, day=12, part=0):
    start = time.perf_counter_ns()
    res = fn()
    elapsed_ns = time.perf_counter_ns() - start
    print(f"{year}\t{day}\t{part}\t{res}\t{elapsed_ns}")


if __name__ == "__main__":
    run(lambda: part_1_real(input_file), part=1)
