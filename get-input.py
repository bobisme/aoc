#!/usr/bin/env python3
import argparse
import datetime
import errno
import os
import sys
from textwrap import dedent

import requests

SESSION_TOKEN = os.getenv("AOC_TOKEN")


def get_input(year, day):
    url = f"https://adventofcode.com/{year}/day/{day}/input"
    headers = {"Cookie": f"session={SESSION_TOKEN}"}
    r = requests.get(url, headers=headers)

    if r.status_code != 200:
        sys.exit(f"/api/alerts response: {r.status_code}: {r.reason} \n{r.content}")

    return r.text


def init_script(year, day):
    filename = f"{year}-{day:02d}.py"
    try:
        with open(filename, "x") as f:
            f.write(
                dedent(
                    f"""\
                    #!/usr/bin/env python

                    from typing import LiteralString
                    import timeit

                    Input = list[str] | list[LiteralString]

                    CONTROL_1: Input = \"""\\
                    \""".splitlines()

                    with open("{year}-{day:02d}.input") as f:
                        input_file = [line.strip() for line in f.readlines()]


                    def part_1(input: Input):
                        for line in input:
                            print(line)
                        return 0


                    def part_2(input: Input):
                        for line in input:
                            print(line)
                        return 0


                    def _test():
                        def assert_eq(a, b):
                            assert a == b, f"{{a}} != {{b}}"

                        assert_eq(part_1(CONTROL_1), 0)
                        # assert_eq(part_2(CONTROL_1), 0)


                    def _bench(fn, count=100):
                        return timeit.timeit(fn, number=count) / count * 1_000


                    if __name__ == "__main__":
                        _test()
                        print("tests: PASS")
                        print("-" * 40)
                        print("part_1:", part_1(CONTROL_1))
                        print("part_2:", part_2(input_file))
                        print("-" * 40)
                        print("part_1 bench: {{:.1f}}ms".format(_bench(lambda: part_1(input_file), count=1)))
                        print("part_2 bench: {{:.1f}}ms".format(_bench(lambda: part_2(input_file), count=1)))
                    """
                )
            )
    except OSError as e:
        if e.errno == errno.EEXIST:
            print(filename, "already exists, skipping")
        else:
            raise
    os.chmod(filename, 0o744)


def main():
    date = datetime.date.today()
    parser = argparse.ArgumentParser(
        prog="get-input.py",
        description="Fetch Advent of Code Input",
        epilog="Must supply browser session id in AOC_TOKEN env var.",
    )
    parser.add_argument("-y", "--year", default=date.year)
    parser.add_argument("-d", "--day", default=date.day)
    args = parser.parse_args()
    year = args.year
    day = args.day
    input = get_input(year, day)
    with open(f"{year}-{day:02d}.input", mode="w") as f:
        f.write(input)
    init_script(year, day)
    print(f"input file = {year}-{day:02d}.input")
    print(f"script = {year}-{day:02d}.py")


if __name__ == "__main__":
    main()
