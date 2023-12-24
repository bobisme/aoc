#!/usr/bin/env python

CONTROL_1 = """\
""".splitlines()

with open("2023-24.input") as f:
    input_file = [line.strip() for line in f.readlines()]


def main(input):
    for line in input:
        print(line)

if __name__ == "__main__":
    main(input_file)
