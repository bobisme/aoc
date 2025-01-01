#!/usr/bin/env python

# NOTE to readers:
# I needed so much help with this, I don't consider that I solved this one.
# Maybe it was being sick for the last week, but I could not wrap my head around it.

from collections import deque
from functools import cache
from itertools import permutations, product
import sys
from typing import Deque, Iterable, Literal, Protocol, Sequence

CONTROL_1 = """\
029A
980A
179A
456A
379A
""".splitlines()

with open("2024-21.input") as f:
    input_file = [line.strip() for line in f.readlines()]

Dir = Literal["^", ">", "v", "<"]
DIRS: tuple[Dir, ...] = ("^", ">", "v", "<")
Input = Literal[Dir, "A"]
INPUTS: tuple[Input, ...] = ("^", ">", "v", "<", "A")


class Panic(ValueError):
    pass


def opposite(dir: Dir) -> Dir:
    if dir == "^":
        return "v"
    if dir == ">":
        return "<"
    if dir == "v":
        return "^"
    if dir == "<":
        return ">"


class Pad(Protocol):
    x: int
    y: int
    moves: list[Dir]

    def __init__(self) -> None:
        self.moves = []

    def is_pos_ok(self, pos: tuple[int, int]) -> bool: ...
    def activate(self): ...

    def check(self):
        if not self.is_pos_ok((self.x, self.y)):
            raise Panic()

    def input(self, input: str):
        for i in input:
            if i in DIRS:
                self.move(i)
            elif i == "A":
                self.activate()

    def move(self, dir: Dir, track=True):
        if dir == "^":
            self.y -= 1
        if dir == ">":
            self.x += 1
        if dir == "v":
            self.y += 1
        if dir == "<":
            self.x -= 1
        if track:
            self.moves.append(dir)
        self.check()

    def undo(self):
        if not self.moves:
            return
        popped = self.moves.pop()
        self.move(opposite(popped), track=False)


class NumPad(Pad):
    out: list[str]

    def __init__(self):
        super().__init__()
        self.x = 2
        self.y = 3
        self.out = []

    def __repr__(self):
        def get_char(x, y):
            if (x, y) == (self.x, self.y):
                return "\033[7m{}\033[0m"
            return "{}"

        pad = [
            "+---+---+---+",
            f"| {get_char(0,0).format('7')} | {get_char(1,0).format('8')} | {get_char(2,0).format('9')} |",
            "+---+---+---+",
            f"| {get_char(0,1).format('4')} | {get_char(1,1).format('5')} | {get_char(2,1).format('6')} |",
            "+---+---+---+",
            f"| {get_char(0,2).format('1')} | {get_char(1,2).format('2')} | {get_char(2,2).format('3')} |",
            "+---+---+---+",
            f"    | {get_char(1,3).format('0')} | {get_char(2,3).format('A')} |",
            "    +---+---+",
        ]
        return "\n".join(pad)

    def is_pos_ok(self, pos: tuple[int, int]) -> bool:
        if (
            pos[0] < 0
            or pos[1] < 0
            or pos[0] > 2
            or pos[1] > 3
            or (pos[0], pos[1]) == (0, 3)
        ):
            return False
        return True

    def activate(self):
        pos = (self.x, self.y)
        match pos:
            case (0, 0):
                self.out.append("7")
            case (1, 0):
                self.out.append("8")
            case (2, 0):
                self.out.append("9")
            case (0, 1):
                self.out.append("4")
            case (1, 1):
                self.out.append("5")
            case (2, 1):
                self.out.append("6")
            case (0, 2):
                self.out.append("1")
            case (1, 2):
                self.out.append("2")
            case (2, 2):
                self.out.append("3")
            case (1, 3):
                self.out.append("0")
            case (2, 3):
                self.out.append("A")
            case _:
                raise Panic()


class DirPad(Pad):
    x: int
    y: int
    moves: list[Dir]
    out: list[str]
    controller: Pad

    def __init__(self, controller: Pad) -> None:
        super().__init__()
        self.x = 2
        self.y = 0
        self.controller = controller

    def __repr__(self):
        def get_char(x, y):
            if (x, y) == (self.x, self.y):
                return "\033[7m{}\033[0m"
            return "{}"

        pad = [
            "    +---+---+",
            f"    | {get_char(1,0).format('^')} | {get_char(2,0).format('A')} |",
            "+---+---+---+",
            f"| {get_char(0,1).format('<')} | {get_char(1,1).format('v')} | {get_char(2,1).format('>')} |",
            "+---+---+---+",
        ]
        return "\n".join(pad)

    def is_pos_ok(self, pos: tuple[int, int]) -> bool:
        if (
            pos[0] < 0
            or pos[1] < 0
            or pos[0] > 2
            or pos[1] > 1
            or (pos[0], pos[1]) == (0, 0)
        ):
            return False
        return True

    def activate(self):
        pos = (self.x, self.y)
        match pos:
            case (1, 0):
                self.controller.input("^")
            case (2, 0):
                self.controller.input("A")
            case (0, 1):
                self.controller.input("<")
            case (1, 1):
                self.controller.input("v")
            case (2, 1):
                self.controller.input(">")
            case _:
                raise Panic()


def search(numpad: NumPad, dir1: DirPad, dir2: DirPad, target: str) -> list[Input]:
    q: Deque[list[Input]] = Deque([[]])
    while q:
        input_list = q.popleft()
        # print(f"{len(input_list)=}")
        for next_input in INPUTS:
            numpad = NumPad()
            dir1 = DirPad(numpad)
            dir2 = DirPad(dir1)
            next_inputs = input_list + [next_input]
            try:
                dir2.input("".join(next_inputs))
                outstr = "".join(numpad.out)
                if outstr:
                    print("out", f"{outstr=}")
                if "".join(numpad.out) == target:
                    return next_inputs
                if outstr and not target.startswith(outstr):
                    continue
                q.append(next_inputs)
            except Panic:
                continue
    return []


numpad_positions = {
    "A": (3, 2),
    "0": (3, 1),
    "1": (2, 0),
    "2": (2, 1),
    "3": (2, 2),
    "4": (1, 0),
    "5": (1, 1),
    "6": (1, 2),
    "7": (0, 0),
    "8": (0, 1),
    "9": (0, 2),
}

dirpad_positions = {
    "A": (0, 2),
    "^": (0, 1),
    "<": (1, 0),
    "v": (1, 1),
    ">": (1, 2),
}


def are_moves_valid(
    input: Iterable[str], positions: dict[str, tuple[int, int]]
) -> bool:
    pos = positions["A"]
    for c in input:
        match c:
            case "A":
                continue
            case "^":
                pos = (pos[0] - 1, pos[1])
            case "v":
                pos = (pos[0] + 1, pos[1])
            case "<":
                pos = (pos[0], pos[1] - 1)
            case ">":
                pos = (pos[0], pos[1] + 1)
            case _:
                raise ValueError("NO")
        if pos not in positions.values():
            return False
    return True


def all_paths(input: Iterable[str], positions: dict[str, tuple[int, int]]):
    curr_key = "A"
    parts = []
    for target_key in input:
        curr_pos = positions[curr_key]
        target_pos = positions[target_key]
        di = target_pos[0] - curr_pos[0]
        dj = target_pos[1] - curr_pos[1]
        curr_key = target_key
        path = ""
        if di > 0:
            path += "v" * di
        if di < 0:
            path += "^" * -di
        if dj > 0:
            path += ">" * dj
        if dj < 0:
            path += "<" * -dj
        parts.append(list("".join(x + ("A",)) for x in set(permutations(path))))
    results = ["".join(x) for x in list(product(*parts))]
    return [x for x in results if are_moves_valid(x, positions)]


Sequences = dict[tuple[str, str], list[str]]


def sequences_for_keypad(positions: dict[str, tuple[int, int]]) -> Sequences:
    sequences: Sequences = {}
    max_row = max(i for (i, _) in positions.values())
    max_col = max(j for (_, j) in positions.values())
    pos_map = {pos: key for (key, pos) in positions.items()}
    for a in positions.keys():
        for b in positions.keys():
            if a == b:
                sequences[(a, b)] = ["A"]
                continue
            possibilities = []
            q = deque([(positions[a], "")])  # [(position, path)]
            optimal = sys.maxsize
            while q:
                (i, j), moves = q.popleft()
                nexts = (
                    (i - 1, j, "^"),
                    (i + 1, j, "v"),
                    (i, j - 1, "<"),
                    (i, j + 1, ">"),
                )
                for next_i, next_j, next_move in nexts:
                    if next_i < 0 or next_j < 0 or next_i > max_row or next_j > max_col:
                        continue
                    key = pos_map.get((next_i, next_j))
                    if key is None:
                        continue
                    if key == b:
                        if optimal < len(moves) + 1:
                            break
                        optimal = len(moves) + 1
                        possibilities.append(moves + next_move + "A")
                    else:
                        q.append(((next_i, next_j), moves + next_move))
                else:
                    continue
                break
            sequences[(a, b)] = possibilities
    return sequences


num_sequences = sequences_for_keypad(numpad_positions)
dir_sequences = sequences_for_keypad(dirpad_positions)
dir_lengths = {key: len(val[0]) for key, val in dir_sequences.items()}


def bfs_paths(input: str, sequences: Sequences):
    # print(input)
    options = [sequences[(a, b)] for a, b in zip("A" + input, input)]
    return ["".join(x) for x in product(*options)]


@cache
def dist_between_keys(a: str, b: str, depth=2) -> int:
    if depth == 1:
        return dir_lengths[(a, b)]
    min_len = sys.maxsize
    for seq in dir_sequences[(a, b)]:
        length = 0
        for a2, b2 in zip("A" + seq, seq):
            length += dist_between_keys(a2, b2, depth=depth - 1)
        min_len = min(min_len, length)
    return min_len


def solve(input: Sequence[str], layers=2):
    out = 0
    for line in input:
        print("solving for input", line)
        paths = bfs_paths(line, num_sequences)
        next_paths = paths
        for _ in range(layers):
            current_paths = []
            for path in next_paths:
                current_paths.extend(bfs_paths(path, dir_sequences))
            next_paths = current_paths

        min_len = min(len(p) for p in next_paths)
        out += min_len * int(line.rstrip("A"))
    return out


def solve_faster(input: Sequence[str], layers=2):
    out = 0
    for line in input:
        inputs = bfs_paths(line, num_sequences)
        min_len = sys.maxsize
        for seq in inputs:
            length = 0
            for a, b in zip("A" + seq, seq):
                length += dist_between_keys(a, b, depth=layers)
            min_len = min(min_len, length)
        print(line, min_len)
        out += min_len * int(line.rstrip("A"))
    return out


def part_1(input: Sequence[str]):
    print(solve(input, layers=2))


def part_2(input):
    print(solve_faster(input, layers=25))


if __name__ == "__main__":
    part_1(input_file)
    part_2(input_file)
