#!/usr/bin/env python
import re
from collections import defaultdict
from typing import Generator, Any, NamedTuple, List, Literal

from colored import Fore, Style


class Cubes(
    NamedTuple("Cubes", [("count", int), ("color", Literal["red", "green", "blue"])])
):
    def __repr__(self):
        return f"{self.count} {getattr(Fore, self.color)}{self.color}{Style.reset}"


class Hand(NamedTuple("Hand", [("cubes", List[Cubes])])):
    def __repr__(self):
        return repr(self.cubes)

    def power(self) -> int:
        x = 1
        for cubes in self.cubes:
            x *= cubes.count
        return x


class GameStats(
    NamedTuple("GameStats", [("max_green", int), ("max_blue", int), ("max_red", int)])
):
    def as_hand(self):
        return Hand(
            cubes=[
                Cubes(self.max_red, "red"),
                Cubes(self.max_green, "green"),
                Cubes(self.max_blue, "blue"),
            ]
        )

    def __repr__(self):
        return f"Totals: {repr(self.as_hand())}"


class Game(NamedTuple("Game", [("id", int), ("hands", List[Hand])])):
    def __repr__(self):
        return f"Game {self.id}: {self.hands}"

    def stats(self) -> GameStats:
        counts = defaultdict(lambda: 0)
        for hand in self.hands:
            for cubes in hand.cubes:
                counts[cubes.color] = max(counts[cubes.color], cubes.count)
        return GameStats(
            max_green=counts.get("green", 0),
            max_red=counts.get("red", 0),
            max_blue=counts.get("blue", 0),
        )


COLOR_PATTERN = re.compile(r"(\d+) (red|green|blue)")


def parse_colors(cube_text: str) -> Generator[Cubes, Any, Any]:
    colors = re.findall(COLOR_PATTERN, cube_text)
    for color in colors:
        yield Cubes(int(color[0]), color[1])


def parse_hands(record_text: str) -> Generator[Hand, Any, Any]:
    cubes = record_text.split("; ")
    for cube_text in cubes:
        yield Hand(cubes=list(parse_colors(cube_text)))


def parse_game(line: str) -> Game:
    game, record_text = line.split(": ")
    _, id = game.split(" ", 1)
    return Game(id=int(id), hands=list(parse_hands(record_text)))


CONTROL_1 = """\
Game 1: 3 blue, 4 red; 1 red, 2 green, 6 blue; 2 green
Game 2: 1 blue, 2 green; 3 green, 4 blue, 1 red; 1 green, 1 blue
Game 3: 8 green, 6 blue, 20 red; 5 blue, 4 red, 13 green; 5 green, 1 red
Game 4: 1 green, 3 red, 6 blue; 3 green, 6 red; 3 green, 15 blue, 14 red
Game 5: 6 red, 1 blue, 3 green; 2 blue, 1 red, 2 green
""".splitlines()


def possible_games(
    games, r: int = 0, g: int = 0, b: int = 0
) -> Generator[Game, Any, Any]:
    for game in games:
        stats = game.stats()
        if stats.max_blue > b or stats.max_green > g or stats.max_red > r:
            pass
        else:
            yield game


def main(input):
    games = [parse_game(line) for line in input]
    table = dict((game.id, (game, game.stats())) for game in games)
    possible = list(possible_games(games, r=12, g=13, b=14))
    possible_ids = {game.id for game in possible}
    for id, (game, stats) in table.items():
        if id in possible_ids:
            print(f"{Style.bold}Game {id}{Style.reset}")
        else:
            print(f"{Style.dim}Game {id}{Style.reset}")
        for hand in game.hands:
            print("  ", hand)
        print("  ", stats)
        print("  Power:", stats.as_hand().power())
    print("=" * 60)
    print([game.id for game in possible])
    print("part 1 sum", sum(game.id for game in possible))
    print("part 2 sum", sum(game.stats().as_hand().power() for game in games))


if __name__ == "__main__":
    with open("2023-2.input") as f:
        input = [line.strip() for line in f.readlines()]
    main(input)
