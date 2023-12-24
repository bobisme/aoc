#!/usr/bin/env python

from collections.abc import Iterator
from typing import Optional, Self
from dataclasses import dataclass


CONTROL_1 = """\
-L|F7
7S-7|
L|7||
-L-J|
L|-JF
""".splitlines()
CONTROL_2 = """\
7-F7-
.FJ|7
SJLL7
|F--J
LJ.LJ
""".splitlines()
CONTROL_3 = """\
...........
.S-------7.
.|F-----7|.
.||.....||.
.||.....||.
.|L-7.F-J|.
.|..|.|..|.
.L--J.L--J.
...........
""".splitlines()
CONTROL_4 = """\
.F----7F7F7F7F-7....
.|F--7||||||||FJ....
.||.FJ||||||||L7....
FJL7L7LJLJ||LJ.L-7..
L--J.L7...LJS7F-7L7.
....F-J..F7FJ|L7L7L7
....L7.F7||L7|.L7L7|
.....|FJLJ|FJ|F7|.LJ
....FJL-7.||.||||...
....L---J.LJ.LJLJ...
""".splitlines()
CONTROL_5 = """\
FF7FSF7F7F7F7F7F---7
L|LJ||||||||||||F--J
FL-7LJLJ||||||LJL-77
F--JF--7||LJLJ7F7FJ-
L---JF-JLJ.||-FJLJJ7
|F|F-JF---7F7-L7L|7|
|FFJF7L7F-JF7|JL---7
7-L-JL7||F7|L7F-7F7|
L.L7LFJ|||||FJL7||LJ
L7JLJL-JLJLJL--JLJ.L
""".splitlines()

with open("2023-10.input") as f:
    input = [line.strip() for line in f.readlines()]


@dataclass
class Coords:
    i: int
    j: int

    # def __eq__(self, other: Self) -> bool:
    #     return self.i == other.i and self.j == other.j
    def __hash__(self) -> int:
        return hash((self.i, self.j))


def start_coords(input) -> Coords:
    for j, line in enumerate(input):
        for i, c in enumerate(line):
            if c == "S":
                return Coords(i, j)
    raise Exception("fart")


def neighbors(
    ii: int, ij: int, rows: int, cols: int
) -> tuple[Optional[Coords], Optional[Coords], Optional[Coords], Optional[Coords]]:
    top, right, bottom, left = None, None, None, None
    if ij - 1 >= 0:
        top = Coords(ii, ij - 1)
    if ii + 1 < cols:
        right = Coords(ii + 1, ij)
    if ij + 1 < rows:
        bottom = Coords(ii, ij + 1)
    if ii - 1 >= 0:
        left = Coords(ii - 1, ij)
    return (top, right, bottom, left)


def top_is_ok(p: str, current_p: str) -> bool:
    return current_p in ("S", "|", "J", "L") and p in ("|", "F", "7")


def right_is_ok(p: str, current_p: str) -> bool:
    return current_p in ("S", "-", "F", "L") and p in ("-", "J", "7")


def bottom_is_ok(p: str, current_p: str) -> bool:
    return current_p in ("S", "|", "F", "7") and p in ("|", "J", "L")


def left_is_ok(p: str, current_p: str) -> bool:
    return current_p in ("S", "-", "J", "7") and p in ("-", "F", "L")


def is_at_edge(c: Coords, cols: int, rows: int) -> bool:
    return c.i == 0 or c.j == 0 or c.i == cols - 1 or c.j == rows - 1


def main(input):
    for line in input:
        print(line)

    def part(coords: Coords):
        return input[coords.j][coords.i]

    rows = len(input)
    cols = len(input[0])
    start = start_coords(input)

    def explore_df(
        current: Coords, q: list[Coords], branch: list[Coords]
    ) -> tuple[list[Coords], bool]:
        current_p = part(current)
        if current_p == "S" and len(branch) > 0:
            return branch, True
        if current in visited:
            return branch, False
        n = neighbors(current.i, current.j, rows, cols)
        top, right, bottom, left = n
        if top and (
            current_p == "S" or current_p in ("|", "J", "L") and top != branch[-1]
        ):
            p = part(top)
            if p in ("|", "F", "7"):
                branch, end = explore_df(top, q, branch + [current])
                if end:
                    return branch, True

        if right and (
            current_p == "S" or current_p in ("-", "F", "L") and right != branch[-1]
        ):
            p = part(right)
            if p in ("-", "J", "7"):
                branch, end = explore_df(right, q, branch + [current])
                if end:
                    return branch, True
        if bottom and (
            current_p == "S" or current_p in ("|", "F", "7") and bottom != branch[-1]
        ):
            p = part(bottom)
            if p in ("|", "J", "L"):
                branch, end = explore_df(bottom, q, branch + [current])
                if end:
                    return branch, True
        if left and (
            current_p == "S" or current_p in ("-", "J", "7") and left != branch[-1]
        ):
            p = part(left)
            if p in ("S", "-", "F", "L"):
                branch, end = explore_df(left, q, branch + [current])
                if end:
                    return branch, True

    def explore_bf(q: list[Coords], visited: set[Coords], branch_len: int) -> int:
        current = q.pop()
        current_p = part(current)
        visited.add(current)
        # if current in branch:
        #     return branch, False
        n = neighbors(current.i, current.j, rows, cols)
        top, right, bottom, left = n
        if top and top not in visited:
            p = part(top)
            if top_is_ok(p, current_p):
                q.append(top)
        if right and right not in visited:
            p = part(right)
            if right_is_ok(p, current_p):
                q.append(right)
        if bottom and bottom not in visited:
            p = part(bottom)
            if bottom_is_ok(p, current_p):
                q.append(bottom)
        if left and left not in visited:
            p = part(left)
            if left_is_ok(p, current_p):
                q.append(left)
        return False

    # path = explore_df(current, q, [])
    # print(path)
    # print("path len", len(path))
    # print("mid", len(path) // 2)
    q: list[Coords] = []
    q.append(start)
    visited = set()
    branch_len = -1
    while len(q) > 0:
        print(q)
        if explore_bf(q, visited, branch_len):
            print("BREAK")
            break
        branch_len += 1
    print("len", branch_len)
    print("mid", (branch_len) // 2)


def main2(input):
    input = [list(line) for line in input]
    for line in input:
        print(line)
    print("-" * 60)

    def part(coords: Coords):
        return input[coords.j][coords.i]

    rows = len(input)
    cols = len(input[0])
    start = start_coords(input)

    def explore_bf(q: list[Coords], visited: set[Coords], branch_len: int) -> int:
        current = q.pop()
        current_p = part(current)
        # input[current.j][current.i] = str(len(visited))
        if current_p == "S" and len(visited) > 0:
            return True
        visited.add(current)
        n = neighbors(current.i, current.j, rows, cols)
        top, right, bottom, left = n
        if top and top not in visited:
            p = part(top)
            if top_is_ok(p, current_p):
                # print("top", current_p, "->", p)
                q.append(top)
        if right and right not in visited:
            p = part(right)
            if right_is_ok(p, current_p):
                # print("right", current_p, "->", p)
                q.append(right)
        if bottom and bottom not in visited:
            p = part(bottom)
            if bottom_is_ok(p, current_p):
                # print("bottom", current_p, "->", p)
                q.append(bottom)
        if left and left not in visited:
            p = part(left)
            if left_is_ok(p, current_p):
                # print("left", current_p, "->", p)
                q.append(left)
        return False

    def start_shape(coords: Coords) -> str:
        top, right, bottom, left = neighbors(coords.i, coords.j, rows, cols)
        ok = []
        if top and top_is_ok(part(top), "S"):
            ok.append("top")
        if right and right_is_ok(part(right), "S"):
            ok.append("right")
        if bottom and bottom_is_ok(part(bottom), "S"):
            ok.append("bottom")
        if left and left_is_ok(part(left), "S"):
            ok.append("left")
        a, b = ok[0], ok[1]
        if (a, b) == ("top", "right"):
            return "L"
        if (a, b) == ("top", "bottom"):
            return "|"
        if (a, b) == ("top", "left"):
            return "J"
        if (a, b) == ("right", "bottom"):
            return "F"
        if (a, b) == ("right", "left"):
            return "-"
        if (a, b) == ("bottom", "left"):
            return "7"
        raise Exception("anus")

    q: list[Coords] = []
    q.append(start)
    visited = set()
    branch_len = -1
    while len(q) > 0:
        if explore_bf(q, visited, branch_len):
            print("BREAK")
            break
        branch_len += 1
    print("len", branch_len)
    print("mid", (branch_len) // 2)
    # for coords in visited:
    #     input[coords.j][coords.i] = "X"
    for i in range(0, cols):
        for j in range(0, rows):
            coords = Coords(i, j)
            if coords not in visited:
                input[coords.j][coords.i] = "."

    internal_points = 0
    for i in range(0, cols):
        for j in range(0, rows):
            coords = Coords(i, j)
            if part(coords) != ".":
                continue
            if is_at_edge(coords, cols, rows):
                input[j][i] = "O"
                continue
            r_cross = 0
            open = None
            for i2 in range(i + 1, cols):
                coords = Coords(i2, j)
                p = part(coords)
                # if p == '|':
                #     r_cross += 1
                # if p in ("|", "7", "J"):
                #     open = False
                #     r_cross += 1
                if open == "F":
                    if p in ("L", "F", "J", "O", "I", "."):
                        open = None
                        r_cross += 1
                    elif p == "7":
                        open = None
                elif open == "L":
                    if p in ("L", "F", "7", "O", "I", "."):
                        open = None
                        r_cross += 1
                    elif p == "J":
                        open = None
                else:
                    if p in ("|", "J", "7"):
                        r_cross += 1
                    elif p in ("L", "F"):
                        open = p
                    elif p == "S":
                        open = start_shape(coords)
            # input[j][i] = str(intersections)
            if r_cross % 2 == 0:
                input[j][i] = "O"
            else:
                input[j][i] = "I"
                internal_points += 1
    for i in range(0, cols):
        for j in range(0, rows):
            coords = Coords(i, j)
            if part(coords) not in ("O", "I"):
                input[j][i] = "."

    for line in input:
        print("".join(line))
    print("internal points", internal_points)


if __name__ == "__main__":
    main2(input)
