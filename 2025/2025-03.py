#!/usr/bin/env python

CONTROL_1 = """\
987654321111111
811111111111119
234234234234278
818181911112111
""".splitlines()

with open("2025-03.input") as f:
    input_file = [line.strip() for line in f.readlines()]


def largest_char(s: str) -> tuple[int, str]:
    return max(enumerate(s), key=lambda x: x[1])


def part_1(input):

    out = 0
    for line in input:
        i, a = largest_char(line[:-1])
        _, b = largest_char(line[i + 1 :])
        out += int(a + b)
    return out


def part_2(input):
    # slow, overgeneralized
    def search(s: str, start=0, parent=0, curr_max=0, depth=0) -> int:
        if depth == 12:
            return max(parent, curr_max)

        level = 12 - depth - 1
        zeroes = 10**level
        for i in range(start, len(s) - (level)):
            d = ord(s[i]) - ord("0")
            next_ = parent + (d * zeroes)
            if (next_ // zeroes) < (curr_max // zeroes):
                continue
            res = search(
                s, start=i + 1, parent=next_, curr_max=curr_max, depth=depth + 1
            )
            if res > curr_max:
                curr_max = res
        return curr_max

    # just do what I did for part 1
    def scan_search(s: str, parent=0, depth=0) -> int:
        if depth == 12:
            return parent

        level = 12 - depth - 1
        zeroes = 10**level
        (pos, c) = largest_char(s[: len(s) - level])
        d = ord(c) - ord("0")
        return scan_search(
            s[pos + 1 :], parent=(parent + (d * zeroes)), depth=depth + 1
        )

    return sum(scan_search(line) for line in input)


def _test():
    def assert_eq(a, b):
        assert a == b, f"{a} != {b}"

    assert_eq(part_1(CONTROL_1), 357)
    assert_eq(part_2(CONTROL_1), 3121910778619)
    x = [
        "2411323321122342222312224225222113222113323212322221243612222112223322233231224121422335412222222422"
    ]
    assert_eq(part_2(x), 654222222422)
    print("tests: PASS")


if __name__ == "__main__":
    _test()
    print(part_1(input_file))
    print(part_2(input_file))
