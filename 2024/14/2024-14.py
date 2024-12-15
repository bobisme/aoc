#!/usr/bin/env python

from dataclasses import dataclass
import re
import time

CONTROL_1 = """\
p=0,4 v=3,-3
p=6,3 v=-1,-3
p=10,3 v=-1,2
p=2,0 v=2,-1
p=0,0 v=1,3
p=3,0 v=-2,-2
p=7,6 v=-1,-3
p=3,0 v=-1,-2
p=9,3 v=2,3
p=7,3 v=-1,2
p=2,4 v=2,-3
p=9,5 v=-3,-3
""".splitlines()

with open("2024-14.input") as f:
    input_file = [line.strip() for line in f.readlines()]


@dataclass
class Bot:
    pos: tuple[int, int]
    vel: tuple[int, int]

    def move(self):
        self.pos = (self.pos[0] + self.vel[0], self.pos[1] + self.vel[1])


def parse(input) -> list[Bot]:
    pattern = re.compile(r"p=(\d+),(\d+) v=(-?\d+),(-?\d+)")
    out = []
    for line in input:
        px, py, vx, vy = pattern.findall(line)[0]
        out.append(Bot((int(px), int(py)), (int(vx), int(vy))))
    return out


def visualize(bots: list[Bot], width=101, height=103):
    vis = [[0 for _ in range(width)] for _ in range(height)]
    for bot in bots:
        vis[bot.pos[1] % height][bot.pos[0] % width] += 1
    return vis


def part_1(input, width=101, height=103):
    bots = parse(input)
    for _ in range(100):
        for bot in bots:
            bot.move()
    vis = visualize(bots, width, height)
    # for line in vis:
    #     print("".join("." if x == 0 else str(x) for x in line))
    total = 1
    for i_start in (0, height // 2 + 1):
        for j_start in (0, width // 2 + 1):
            quadrant = 0
            for i in range(i_start, i_start + height // 2):
                for j in range(j_start, j_start + width // 2):
                    quadrant += vis[i][j]
            total *= quadrant
    print(total)


def search_for_straight_line(vis: list[list[int]], len_=10) -> bool:
    width = len(vis[0])
    height = len(vis)
    for i in range(height):
        for j in range(0, width - len_):
            if sum(vis[i][j : j + len_]) >= len_:
                return True
    return False


def part_2(input, width=101, height=103):
    bots = parse(input)
    for i in range(1_000_000):
        for bot in bots:
            bot.move()
        vis = visualize(bots, width, height)
        if search_for_straight_line(vis):
            for line in vis:
                print("".join("." if x == 0 else str(x) for x in line))
            print(i + 1)
        # time.sleep(0.02)


if __name__ == "__main__":
    # part_1(CONTROL_1, width=11, height=7)
    part_1(input_file, width=101, height=103)
    part_2(input_file)
