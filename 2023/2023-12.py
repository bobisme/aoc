#!/usr/bin/env python
from collections import deque
import enum
import re
from typing import Any, Generator, NamedTuple, Optional
from itertools import cycle, combinations, islice, permutations, product, zip_longest
from functools import cache

CONTROL_1 = """\
???.### 1,1,3
.??..??...?##. 1,1,3
?#?#?#?#?#?#?#? 1,3,1,6
????.#...#... 4,1,1
????.######..#####. 1,6,5
?###???????? 3,2,1
""".splitlines()

with open("2023-12.input") as f:
    input = [line.strip() for line in f.readlines()]


def matches_groups(row, expected: list[int]):
    found = re.findall("#+", row)
    counts = [len(x) for x in found]
    return counts == expected


assert matches_groups("##.###.#", [2, 3, 1])


def q_indexes(row):
    for i, c in enumerate(row):
        if c == "?":
            yield i


def n_springs(row):
    return sum(1 for c in row if c == "#")


Row = NamedTuple("Row", [("pattern", str), ("groups", tuple[int, ...]), ("slen", int)])


def parse_row(row):
    pattern, groups = row.split(" ", 1)
    groups = tuple(int(x) for x in groups.split(","))
    return Row(pattern, groups, len(pattern))


def parse_x_row(row, x=5):
    pattern, groups = row.split(" ", 1)
    groups = [int(x) for x in groups.split(",")]
    return Row("?".join([pattern] * x), tuple(list(groups) * x), slen=len(pattern))


def get_chunks(xs, n):
    n = max(1, n)
    return (xs[i : i + n] for i in range(0, len(xs), n))


def check_chunk(pattern, gen) -> bool:
    if len(gen) > len(pattern):
        return False
    chunk_len = len(gen)
    for i in range(chunk_len):
        c = pattern[i]
        if c == "?":
            continue
        if c != gen[i]:
            return False
    return True


def check(row: Row, test: list[tuple[str, int]]) -> bool:
    len_gen = sum(x[1] for x in test)
    if len_gen > len(row.pattern):
        # print("size doesnt match")
        return False
    pattern = row.pattern
    r = min(len(pattern), len_gen)
    for i in range(r):
        c = pattern[i]
        if c == "?":
            continue
        generated = "".join((x[0] * x[1] for x in test))
        if c != generated[i]:
            # print(f"{c} != {generated[i]}")
            return False
    return True


def check_possible(pattern: str, test: list[tuple[str, int]]) -> bool:
    len_gen = sum(x[1] for x in test)
    if len_gen > len(pattern):
        # print("size doesnt match")
        return False
    pattern = pattern
    r = min(len(pattern), len_gen)
    for i in range(r):
        c = pattern[i]
        if c == "?":
            continue
        generated = "".join((x[0] * x[1] for x in test))
        if c != generated[i]:
            # print(f"{c} != {generated[i]}")
            return False
    return True


# def sliding_window(iterable, n):
#     # sliding_window('ABCDEFG', 4) --> ABCD BCDE CDEF DEFG
#     it = iter(iterable)
#     window = deque(islice(it, n - 1), maxlen=n)
#     for x in it:
#         window.append(x)
#         yield tuple(window)
#
#
# def check_sep_springs(
#     pattern_chunk: str, pieces: tuple[int, Optional[int], str]
# ) -> bool:
#     a, b, c = pieces
#     generated =
#     assert len(generated) == len(pattern_chunk)
#     r = len(generated)
#     for i in range(r):
#         c = pattern_chunk[i]
#         if c == "?":
#             continue
#         if c != generated[i]:
#             # print(f"{c} != {generated[i]}")
#             return False
#     return True
#
#
# def check(row: Row, seps: list[int], springs: list[int]) -> bool:
#     CHUNK_LEN = 3
#     section_len = sum(seps) + sum(springs)
#     if section_len > len(row.pattern):
#         return False
#     # seps, springs
#     chars = cycle((".", "#"))
#     inter = list(sliding_window(zip(range(len(seps)), chars), 3))
#     for i_window in inter:
#         ()
#     # for i in range(0, len(inter) - CHUNK_LEN, CHUNK_LEN):
#     #     pre = inter[:i]
#     #     post = inter[i:CHUNK_LEN]
#     return check_sep_springs(row.pattern[:section_len], s1, g, s2)


def possible_patterns(
    row: Row,
    pat_section: str,
    separators: list[int],
    prefix: list[tuple[str, int]],
    sep_i: int,
    r: range,
) -> list[str]:
    sep_end_i = len(separators) - 1
    pats = []
    for s in r:
        test = prefix + [(".", s)]
        if check_possible(pat_section, test):
            if sep_i < len(row.groups):
                test = test + [("#", row.groups[sep_i])]
            if sep_i >= sep_end_i:
                if test != "":
                    pats.append(test)
                break
            separators_remaining = r.stop - s
            next_separator_basis = separators[sep_i + 1]
            pats.extend(
                possible_patterns(
                    row,
                    pat_section,
                    separators,
                    test,
                    sep_i + 1,
                    range(next_separator_basis, separators_remaining),
                )
            )
        else:
            break
    return pats


@cache
def all_possible(pattern: str) -> list[str]:
    qs = [i for i, c in enumerate(pattern) if c == "?"]

    def inner(p: list[str], qi: int) -> Generator[str, Any, Any]:
        p1 = p.copy()
        p2 = p.copy()
        p1[qs[qi]] = "."
        p2[qs[qi]] = "#"
        if qi == len(qs) - 1:
            yield "".join(p1)
            yield "".join(p2)
        else:
            yield from inner(p1, qi + 1)
            yield from inner(p2, qi + 1)
        # for generated in product(".#", repeat=len(pattern)):
        #     if not check_chunk(pattern, "".join(generated)):
        #         continue
        #     yield "".join(generated)

    out = list(inner(list(pattern), 0))
    for o in out:
        print(o)
    return out


def arrangements_for_row(row: Row) -> int:
    print(f"{row=}")
    qs = list(q_indexes(row.pattern))
    # print(f"{qs=}")
    n_visible_springs = n_springs(row.pattern)
    # print(f"{n_visible_springs=}")
    needed_springs = sum(row.groups) - n_visible_springs
    # print(f"{needed_springs=}")
    needed_dots = len(qs) - needed_springs
    # print(f"{needed_dots=}")
    # char_pool = "." * needed_dots + "#" * needed_springs
    # print(f"{char_pool=}")
    # pset = set()
    separators = [1 for _ in range(len(row.groups) + 1)]
    separators[0] = 0
    separators[-1] = 0
    sum_groups = sum(row.groups)
    seps = separators.copy()
    valid_count = 0
    max_dots = len(row.pattern) - sum_groups
    sep_end_i = len(separators) - 1

    def ccc(prefix: list[tuple[str, int]], sep_i: int, r: range):
        n = 0
        for s in r:
            test = prefix + [(".", s)]
            if check(row, test):
                if sep_i < len(row.groups):
                    test = test + [("#", row.groups[sep_i])]
                if sep_i >= sep_end_i:
                    if test != "":
                        n += 1
                    break
                separators_remaining = r.stop - s
                next_separator_basis = separators[sep_i + 1]
                n += ccc(
                    test,
                    sep_i + 1,
                    range(next_separator_basis, separators_remaining),
                )
            else:
                break
        return n

    n = ccc([], 0, range(max_dots + 1))
    print(f"{n=}")
    return n


def arrangements_for_row2(row: Row, joins: int = 1) -> int:
    print(f"{row=}")
    qs = list(q_indexes(row.pattern))
    # print(f"{qs=}")
    n_visible_springs = n_springs(row.pattern)
    # print(f"{n_visible_springs=}")
    needed_springs = sum(row.groups) - n_visible_springs
    # print(f"{needed_springs=}")
    needed_dots = len(qs) - needed_springs
    # print(f"{needed_dots=}")
    # char_pool = "." * needed_dots + "#" * needed_springs
    # print(f"{char_pool=}")
    # pset = set()
    separators = [1 for _ in range(len(row.groups) + 1)]
    separators[0] = 0
    separators[-1] = 0
    sum_groups = sum(row.groups)
    seps = separators.copy()
    valid_count = 0
    max_dots = len(row.pattern) - sum_groups
    sep_end_i = len(separators) - 1
    # pats = possible_patterns(
    #     row, row.pattern[:20], separators, [], 0, range(max_dots + 1)
    # )
    # chunks = [
    #     "".join(x)
    #     for x in zip_longest(*([iter(row.pattern)] * (row.slen + 1)), fillvalue="")
    # ]
    groups = row.groups * joins

    def ccc(prefix: str, groups: list[int], chunks: list[str], chunk_i: int):
        n = 0
        chunk = chunks[chunk_i]
        possible = list(all_possible(chunk))
        for p in possible:
            pattern = prefix + p
            if chunk_i == len(chunks) - 1:
                if matches_groups(pattern, groups):
                    n += 1
            else:
                n += ccc(pattern, groups, chunks, chunk_i + 1)
        return n

    chunks = [row.pattern] + ["?", row.pattern] * (joins - 1)
    n = ccc("", groups, chunks, 0)
    print(f"{n=}")
    return n


@cache
def gimme(pattern: str, pat_start: int, remaining_groups: tuple[int, ...]) -> list[str]:
    # print(f"{pattern[pat_start:]=} {pat_start=} {remaining_groups=}")
    # print(f"{pattern[pat_start:]=} {remaining_groups=}")
    remaining_separators = len(pattern[pat_start:]) - sum(remaining_groups)
    if pat_start == len(pattern) and len(remaining_groups) == 0:
        return []
    elif pat_start > len(pattern):
        return []
    if pat_start >= len(pattern):
        return []
    start = 0
    if not remaining_groups:
        start = remaining_separators
    out = []
    out_n = 0
    for n_separators in range(start, remaining_separators + 1):
        if not remaining_groups and n_separators == 0:
            continue
        generated = f"{'.' * n_separators}"
        if remaining_groups:
            generated += f"{'#' * remaining_groups[0]}"
        if len(remaining_groups) > 1:
            generated += "."

        if not check_chunk(pattern[pat_start:], generated):
            continue
        # print(f"{generated=}")
        next_pat_start = pat_start + len(generated)
        if next_pat_start == len(pattern) and len(remaining_groups[1:]) == 0:
            out.append(generated)
            out_n += 1
            break
        # if pat_start >= len(pattern) and len(remaining_groups) == 0:
        #     n += 1
        #     break
        # else:
        for sub in gimme(pattern, next_pat_start, remaining_groups[1:]):
            out.append(generated + sub)
            out_n += 1
    print(out_n)
    return out


@cache
def gimme_n(pattern: str, pat_start: int, remaining_groups: tuple[int, ...]) -> int:
    # print(f"{pattern[pat_start:]=} {pat_start=} {remaining_groups=}")
    # print(f"{pattern[pat_start:]=} {remaining_groups=}")
    remaining_separators = len(pattern[pat_start:]) - sum(remaining_groups)
    if pat_start == len(pattern) and len(remaining_groups) == 0:
        return 0
    elif pat_start > len(pattern):
        return 0
    if pat_start >= len(pattern):
        return 0
    start = 0
    if not remaining_groups:
        start = remaining_separators
    out = []
    out_n = 0
    for n_separators in range(start, remaining_separators + 1):
        if not remaining_groups and n_separators == 0:
            continue
        generated = f"{'.' * n_separators}"
        if remaining_groups:
            generated += f"{'#' * remaining_groups[0]}"
        if len(remaining_groups) > 1:
            generated += "."

        if not check_chunk(pattern[pat_start:], generated):
            continue
        # print(f"{generated=}")
        next_pat_start = pat_start + len(generated)
        if next_pat_start == len(pattern) and len(remaining_groups[1:]) == 0:
            out_n += 1
            break
        # if pat_start >= len(pattern) and len(remaining_groups) == 0:
        #     n += 1
        #     break
        # else:
        out_n += gimme_n(pattern, next_pat_start, remaining_groups[1:])
    return out_n


def main(input):
    for row in input:
        print(row)
    rows = [parse_x_row(x, 5) for x in input]
    total_total = 0
    for r in rows:
        # print("-" * 60)
        # print(r.pattern, ",".join((str(x) for x in r.groups)))
        total = gimme_n(r.pattern, 0, r.groups)
        # print(x)
        total_total += total
        # print(f"{total=}")
    print(f"{total_total=}")
    # total = sum(1 for r in rows[4:5] for s in gimme(r.pattern, 0, r.groups))


if __name__ == "__main__":
    main(input)
