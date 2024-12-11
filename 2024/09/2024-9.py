#!/usr/bin/env python

from collections.abc import Generator
from collections import deque
from typing import Deque, NamedTuple, Optional


CONTROL_1 = """\
2333133121414131402
""".splitlines()

with open("2024-9.input") as f:
    input_file = [line.strip() for line in f.readlines()]


def build_dense(map: list[int]) -> list[Optional[int]]:
    dense: list[Optional[int]] = [None for _ in range(sum(map))]
    dense_i = 0
    id = 0
    for i in range(len(map)):
        if i % 2 == 0:
            for _ in range(map[i]):
                dense[dense_i] = id
                dense_i += 1
            id += 1
        else:
            for _ in range(map[i]):
                dense_i += 1
    return dense


def free_spaces(dense: list[Optional[int]]) -> Generator[int, None, None]:
    for i in range(len(dense)):
        if dense[i] is None:
            yield i


def end_files(dense: list[Optional[int]]) -> Generator[int, None, None]:
    for i in range(len(dense) - 1, 0, -1):
        if dense[i] is not None:
            yield i


def checksum(dense: list[Optional[int]]) -> int:
    out = 0
    for i, x in enumerate(dense):
        if x is None:
            continue
        out += i * x

    return out


def part_1(input):
    map = [int(x) for x in input[0]]
    dense = build_dense(map)
    for free, filled in zip(free_spaces(dense), end_files(dense)):
        if free >= filled:
            break
        dense[free] = dense[filled]
        dense[filled] = None
    print(checksum(dense))


Item = NamedTuple("Item", (("id", Optional[int]), ("size", int)))


def build_sparse(map: list[int]) -> Deque[Item]:
    d = deque()
    id = 0
    for i in range(len(map)):
        if i % 2 == 0:
            d.append(Item(id, map[i]))
            id += 1
        else:
            if map[i] == 0:
                continue
            d.append(Item(None, map[i]))
    return d


def sparse_to_dense(sparse: Deque[Item]) -> list[Optional[int]]:
    out = []
    for x in sparse:
        out.extend([x.id] * x.size)
    return out


def part_2(input):
    map = [int(x) for x in input[0]]
    sparse = build_sparse(map)
    left_i, right_i = (0, len(sparse) - 1)
    while left_i < right_i:
        while sparse[left_i].id is not None and sparse[left_i].size > 0:
            left_i += 1
        while sparse[right_i].id is None:
            right_i -= 1
        file = sparse[right_i]
        # search for free space
        slotted = False
        for j in range(left_i, right_i):
            if sparse[j].id is None and sparse[j].size >= file.size:
                sparse[right_i] = Item(None, file.size)
                new_size = sparse[j].size - file.size
                if new_size == 0:
                    del sparse[j]
                else:
                    sparse[j] = Item(None, new_size)
                sparse.insert(j, file)
                slotted = True
                break
        if not slotted:
            right_i -= 1
    dense = sparse_to_dense(sparse)
    # print("".join("." if x is None else str(x) for x in dense))
    print(checksum(dense))


if __name__ == "__main__":
    part_1(input_file)
    part_2(input_file)
