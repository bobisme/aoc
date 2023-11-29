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
    filename = f"{year}-{day}.py"
    try:
        with open(filename, "x") as f:
            f.write(
                dedent(
                    f"""\
                    #!/usr/bin/env python

                    with open("{year}-{day}.input") as f:
                        input = [line.strip() for line in f.readlines()]

                    print(input)
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
    with open(f"{year}-{day}.input", mode="w") as f:
        f.write(input)
    init_script(year, day)
    print(f"input file = {year}-{day}.input")
    print(f"script = {year}-{day}.py")


if __name__ == "__main__":
    main()
