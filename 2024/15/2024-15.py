#!/usr/bin/env python

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal, TypeGuard

CONTROL_1 = """\
##########
#..O..O.O#
#......O.#
#.OO..O.O#
#..O@..O.#
#O#..O...#
#O..O..O.#
#.OO.O.OO#
#....O...#
##########

<vv>^<v^>v>^vv^v>v<>v^v<v<^vv<<<^><<><>>v<vvv<>^v^>^<<<><<v<<<v^vv^v>^
vvv<<^>^v^^><<>>><>^<<><^vv^^<>vvv<>><^^v>^>vv<>v<<<<v<^v>^<^^>>>^<v<v
><>vv>v^v^<>><>>>><^^>vv>v<^^^>>v^v^<^^>v^^>v^<^v>v<>>v^v^<v>v^^<^^vv<
<<v<^>>^^^^>>>v^<>vvv^><v<<<>^^^vv^<vvv>^>v<^^^^v<>^>vvvv><>>v^<<^^^^^
^><^><>>><>^^<<^^v>>><^<v>^<vv>>v>>>^v><>^v><<<<v>>v<v<v>vvv>^<><<>^><
^>><>^v<><^vvv<^^<><v<<<<<><^v<<<><<<^^<v<^^^><^>>^<v^><<<^>>^v<v^v<v^
>^>>^v>vv>^<<^v<>><<><<v<<v><>v<^vv<<<>^^v^>^^>>><<^v>>v^v><^^>>^<>vv^
<><^^>^^^<><vvvvv^v<v<<>^v<v>v<<^><<><<><<<^^<<<^<<>><<><^^^>^^<>^>v<>
^^>vv<^v^v<vv>^<><v<^v>^^^>>>^^vvv^>vvv<>>>^<^>>>>>^<<^v>^vvv<>^<><<v>
v^^>>><<^^<>>^v^<v^vv<>v^<<>^<^v^v><^<<<><<^<v><v<>vv>>v><v^<vv<>v^<<^
""".splitlines()

with open("2024-15.input") as f:
    input_file = [line.strip() for line in f.readlines()]


@dataclass
class Pos:
    i: int
    j: int

    def __hash__(self) -> int:
        return hash((self.i, self.j))

    def __repr__(self) -> str:
        return f"({self.i},{self.j})"

    def __add__(self, other: "Pos") -> "Pos":
        return Pos(self.i + other.i, self.j + other.j)


class Grid:
    bot: Pos

    def __init__(self, input: Iterable[str]) -> None:
        self.grid = [list(line) for line in input]
        self.bot = self.find_bot()

    def __repr__(self) -> str:
        return "\n".join("".join(row) for row in self.grid)

    @property
    def height(self):
        return len(self.grid)

    @property
    def width(self):
        return len(self.grid[0])

    def find_bot(self) -> Pos:
        for i in range(1, self.width - 1):
            for j in range(1, self.height - 1):
                if self.grid[i][j] == "@":
                    return Pos(i, j)
        raise ValueError("Couldn't find bot")

    def move(self, pos: Pos, vector: Pos) -> bool:
        item = self.grid[pos.i][pos.j]
        next_pos = Pos(pos.i + vector.i, pos.j + vector.j)
        next_item = self.grid[next_pos.i][next_pos.j]

        if next_item == "O":
            if not self.move(next_pos, vector):
                return False
        next_item = self.grid[next_pos.i][next_pos.j]
        if item == "#" or item == ".":
            return False
        if next_item == "#":
            return False
        if next_item == ".":
            self.grid[next_pos.i][next_pos.j] = item
            self.grid[pos.i][pos.j] = "."
            if item == "@":
                self.bot = next_pos
            return True
        raise Exception("Unreachable")

    def move_left(self, pos: Pos) -> bool:
        return self.move(pos, Pos(0, -1))

    def move_right(self, pos: Pos) -> bool:
        return self.move(pos, Pos(0, 1))

    def move_up(self, pos: Pos) -> bool:
        return self.move(pos, Pos(-1, 0))

    def move_down(self, pos: Pos) -> bool:
        return self.move(pos, Pos(1, 0))


def parse(input: Iterable[str]) -> tuple[Grid, str]:
    grid = []
    moves = []

    mode = "grid"
    for line in input:
        if line != "":
            if mode == "grid":
                grid.append(line)
            else:
                moves.append(line)
        else:
            mode = "moves"
    return Grid(grid), "".join(moves)


def move_bot(grid: Grid, dir: str):
    if dir == "^":
        grid.move_up(grid.bot)
    elif dir == ">":
        grid.move_right(grid.bot)
    elif dir == "v":
        grid.move_down(grid.bot)
    elif dir == "<":
        grid.move_left(grid.bot)


def part_1(input):
    grid, moves = parse(input)
    for dir in moves:
        move_bot(grid, dir)
    out = 0
    for i in range(1, grid.height - 1):
        for j in range(1, grid.width - 1):
            if grid.grid[i][j] == "O":
                out += 100 * i + j
    print(out)


LR = Literal["<", ">"]
UD = Literal["^", "v"]
Dir = Literal[LR, UD]


class Grid2:
    bot: Pos

    def __init__(self, input: Iterable[str]) -> None:
        single_grid = [list(line) for line in input]
        single_width = len(single_grid[0])
        height = len(single_grid)
        self.grid = [["." for _ in range(2 * single_width)] for _ in range(height)]
        for i in range(height):
            for j in range(single_width):
                item = single_grid[i][j]
                if item == ".":
                    continue
                if item == "#":
                    self.grid[i][j * 2] = "#"
                    self.grid[i][j * 2 + 1] = "#"
                elif item == "O":
                    self.grid[i][j * 2] = "["
                    self.grid[i][j * 2 + 1] = "]"
                elif item == "@":
                    self.grid[i][j * 2] = "@"
                    self.grid[i][j * 2 + 1] = "."

        self.bot = self.find_bot()

    def __repr__(self) -> str:
        return "\n".join("".join(row) for row in self.grid)

    @property
    def height(self):
        return len(self.grid)

    @property
    def width(self):
        return len(self.grid[0])

    def find_bot(self) -> Pos:
        for i in range(1, self.width - 1):
            for j in range(1, self.height - 1):
                if self.grid[i][j] == "@":
                    return Pos(i, j)
        raise ValueError("Couldn't find bot")

    def can_move(self, pos: Pos, dir: Dir) -> bool:
        if dir == "^":
            vec = Pos(-1, 0)
        elif dir == ">":
            vec = Pos(0, 1)
        elif dir == "v":
            vec = Pos(1, 0)
        elif dir == "<":
            vec = Pos(0, -1)

        item = self.grid[pos.i][pos.j]
        if item == "#":
            return False
        if item == ".":
            return True
        if item == "[":
            return all(
                (
                    self.can_move(pos + vec, dir),
                    self.can_move(pos + Pos(0, 1) + vec, dir),
                )
            )
        elif item == "]":
            return all(
                (
                    self.can_move(pos + vec, dir),
                    self.can_move(pos + Pos(0, -1) + vec, dir),
                )
            )
        elif item == "@":
            return self.can_move(pos + vec, dir)
        raise Exception("Unreachable")

    def move_sideways(self, pos: Pos, dir: Literal[">", "<"]) -> bool:
        if dir == ">":
            next_pos = Pos(pos.i, pos.j + 1)
        elif dir == "<":
            next_pos = Pos(pos.i, pos.j - 1)

        item = self.grid[pos.i][pos.j]
        next_item = self.grid[next_pos.i][next_pos.j]

        if next_item == "[" or next_item == "]":
            if not self.move(next_pos, dir):
                return False
        next_item = self.grid[next_pos.i][next_pos.j]
        if item == "#" or item == ".":
            return False
        if next_item == "#":
            return False
        if next_item == ".":
            self.grid[next_pos.i][next_pos.j] = item
            self.grid[pos.i][pos.j] = "."
            if item == "@":
                self.bot = next_pos
            return True
        raise Exception("Unreachable")

    def move_vertically(self, pos: Pos, dir: Literal["^", "v"]) -> bool:
        if dir == "^":
            next_pos = Pos(pos.i - 1, pos.j)
        elif dir == "v":
            next_pos = Pos(pos.i + 1, pos.j)

        if not self.can_move(pos, dir):
            return False

        item = self.grid[pos.i][pos.j]
        next_item = self.grid[next_pos.i][next_pos.j]

        if item in ("#", "."):
            return False
        if next_item == "#":
            return False
        elif next_item == "[":
            self.move_vertically(next_pos, dir)
            self.move_vertically(next_pos + Pos(0, 1), dir)
        elif next_item == "]":
            self.move_vertically(next_pos, dir)
            self.move_vertically(next_pos + Pos(0, -1), dir)
        next_item = self.grid[next_pos.i][next_pos.j]
        if next_item == ".":
            if item == "@":
                self.grid[next_pos.i][next_pos.j] = item
                self.grid[pos.i][pos.j] = "."
                self.bot = next_pos
                return True
            elif item == "[" and self.grid[next_pos.i][next_pos.j + 1] == ".":
                self.grid[next_pos.i][next_pos.j] = item
                self.grid[next_pos.i][next_pos.j + 1] = "]"
                self.grid[pos.i][pos.j] = "."
                self.grid[pos.i][pos.j + 1] = "."
                return True
            elif item == "]" and self.grid[next_pos.i][next_pos.j - 1] == ".":
                self.grid[next_pos.i][next_pos.j] = item
                self.grid[next_pos.i][next_pos.j - 1] = "["
                self.grid[pos.i][pos.j] = "."
                self.grid[pos.i][pos.j - 1] = "."
                return True
            return False
        raise Exception("Unreachable")

    def move(self, pos: Pos, dir: Literal["^", ">", "v", "<"]) -> bool:
        if dir in ("<", ">"):
            return self.move_sideways(pos, dir)
        if dir in ("^", "v"):
            return self.move_vertically(pos, dir)


def parse_2(input: Iterable[str]) -> tuple[Grid2, list[Dir]]:
    grid = []
    moves = []

    def check_dirs(data: list[str]) -> TypeGuard[list[Dir]]:
        return all(x in ("^", ">", "v", "<") for x in data)

    mode = "grid"
    for line in input:
        if line != "":
            if mode == "grid":
                grid.append(line)
            else:
                moves.append(line)
        else:
            mode = "moves"
    dirs = list("".join(moves))
    if check_dirs(dirs):
        return Grid2(grid), dirs
    raise ValueError("not a dir")


def part_2(input):
    grid, moves = parse_2(input)
    for dir in moves:
        grid.move(grid.bot, dir)
    out = 0
    for i in range(1, grid.height - 1):
        for j in range(1, grid.width - 1):
            if grid.grid[i][j] == "[":
                out += 100 * i + j
    print(out)


if __name__ == "__main__":
    part_1(input_file)
    part_2(input_file)
