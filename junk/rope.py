from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import cast


@dataclass(slots=True)
class RopeNode[Str: str | bytes | bytearray]:
    s: Str | None = None
    left: "RopeNode[Str] | None" = None
    right: "RopeNode[Str] | None" = None
    left_len: int = 0
    len: int = field(init=False)

    def __post_init__(self):
        if self.s is not None:
            self.len = len(self.s)
        else:
            assert self.left is not None and self.right is not None
            self.left_len = len(self.left)
            self.len = self.left.len + self.right.len

    def __len__(self) -> int:
        return self.len

    def __bytes__(self) -> bytes:
        if self.s is not None:
            if isinstance(self.s, str):
                return bytes(self.s, "ascii")
            if isinstance(self.s, bytes):
                return self.s
            if isinstance(self.s, bytearray):
                return bytes(self.s)
        assert self.left is not None and self.right is not None
        return bytes(self.left) + bytes(self.right)

    def __str__(self) -> str:
        if self.s is not None:
            if isinstance(self.s, str):
                return self.s
            if isinstance(self.s, bytes):
                return str(self.s, "ascii")
            if isinstance(self.s, bytearray):
                return str(self.s, "ascii")
        assert self.left is not None and self.right is not None
        return str(self.left) + str(self.right)

    def __iter__(self) -> Iterator[Str]:
        if self.s is not None:
            if len(self.s) == 0:
                return
            if isinstance(self.s, str):
                yield from cast(Iterator[Str], self.s)
            else:
                for i in range(len(self.s)):
                    yield cast(Str, self.s[i : i + 1])
        else:
            assert self.left is not None and self.right is not None
            yield from self.left
            yield from self.right

    def __getitem__(self, key: int | slice) -> Str:
        if isinstance(key, slice):
            if self.s is not None:
                return cast(Str, self.s[key])
            assert self.left is not None and self.right is not None
            if key.start < self.left_len:
                if key.stop <= self.left_len:
                    return self.left[key]
                else:
                    left_slice = self.left[key.start :]
                    right_slice = self.right[: key.stop - self.left_len]
                    if isinstance(left_slice, str):
                        return cast(Str, left_slice + cast(str, right_slice))
                    if isinstance(left_slice, bytes):
                        return cast(Str, left_slice + cast(bytes, right_slice))
                    else:
                        return cast(Str, left_slice + cast(bytearray, right_slice))
            return self.right[key.start - self.left_len : key.stop - self.left_len]
        else:
            if self.s is not None:
                return self[key : key + 1]
            assert self.left is not None and self.right is not None
            if key < self.left_len:
                return self.left[key]
            return self.right[key - self.left_len]

    def __eq__(self, other: object, /) -> bool:
        assert isinstance(other, RopeNode)
        if len(self) != len(other):
            return False
        for a, b in zip(self, other):
            if a != b:
                return False
        return True

    def __add__(self, other: "RopeNode") -> "RopeNode":
        return RopeNode.concat(self, other)

    @staticmethod
    def concat(r1: "RopeNode", r2: "RopeNode") -> "RopeNode":
        return RopeNode(left=r1, right=r2)

    def split(self, idx: int) -> tuple["RopeNode", "RopeNode"]:
        """
        Given s = "abcdefg", idx = 3: ("abc", "defg")
        """
        if self.s is not None:
            left = self.s[:idx]
            right = self.s[idx:]
            return RopeNode(left), RopeNode(right)

        assert self.left is not None and self.right is not None
        if idx < self.left_len:
            (l1, l2) = self.left.split(idx)
            return l1, RopeNode.concat(l2, self.right)

        (r1, r2) = self.right.split(idx - self.left_len)
        return RopeNode.concat(self.left, r1), r2

    def insert(self, idx: int, s: str) -> "RopeNode":
        insert = rope(s)
        left, right = self.split(idx)
        return RopeNode.concat(RopeNode.concat(left, insert), right)

    def replace(self, start: int, end: int, s: Str) -> "RopeNode":
        insert = rope(s)
        left, mid = self.split(start)
        _, right = mid.split(end - start)
        return RopeNode.concat(RopeNode.concat(left, insert), right)


def rope(s: str | bytes | bytearray) -> RopeNode:
    return RopeNode(s)


def __test_rope():
    for s in (
        cast(str, "hello there"),
        cast(bytes, b"hello there"),
        bytearray(b"hello there"),
    ):
        r = rope(s)
        assert str(r) == "hello there"
        r2 = r.insert(3, "ahoy")
        assert str(r2) == "helahoylo there"
        r3 = r.insert(0, "ahoy ")
        assert str(r3) == "ahoy hello there"
        r4 = r.replace(4, 5, " no")
        assert str(r4) == "hell no there"
        r5 = r.replace(6, 11, "dude")
        assert str(r5) == "hello dude", str(r5)
        if isinstance(s[3:5], str):
            assert r[3:5] == "lo"
        else:
            assert str(r[3:5], "ascii") == "lo", r[3:5]
    r = rope("abcdefgh")
    left, right = RopeNode.split(r, 3)
    assert str(left) == "abc"
    assert str(right) == "defgh"
    r2 = RopeNode.concat(*r.split(3))
    print(r2, repr(r2), r2[1:5])
    assert str(r2[1:5]) == "bcde"
    new_root = RopeNode(left_len=len(left), left=left, right=right)
    assert str(new_root) == "abcdefgh"
    r3 = new_root.replace(2, 5, "!!!")
    assert str(r3) == "ab!!!fgh"


if __name__ == "__main__":
    __test_rope()
