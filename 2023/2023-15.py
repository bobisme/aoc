#!/usr/bin/env python

# If the operation character is a dash (-), go to the relevant box and remove
# the lens with the given label if it is present in the box.
#
# If the operation character is an equals sign (=), it will be followed by a
# number indicating the focal length of the lens that needs to go into the
# relevant box; be sure to use the label maker to mark the lens with the label
# given in the beginning of the step so you can find it later. There are two
# possible situations:
#
# - If there is already a lens in the box with the same label, replace the old
#   lens with the new lens: remove the old lens and put the new lens in its
#   place, not moving any other lenses in the box.
# - If there is not already a lens in the box with the same label, add the lens
#   to the box immediately behind any lenses already in the box. Don't move any
#   of the other lenses when you do this. If there aren't any lenses in the box,
#   the new lens goes all the way to the front of the box.

from dataclasses import dataclass
from enum import Enum
import re
from typing import DefaultDict, Optional, OrderedDict, Self

PATTERN = re.compile(r"([a-z]+)([=-])(\d+)?")

CONTROL_1 = """\
rn=1,cm-,qp=3,cm=2,qp-,pc=4,ot=9,ab=5,pc-,pc=6,ot=7
""".strip().split(",")

with open("2023-15.input") as f:
    input = f.read().strip().split(",")


def lame_hash(input: str) -> int:
    out = 0
    for c in input:
        out += ord(c)
        out = out * 17
        out &= 0xFF
    return out


assert lame_hash("HASH") == 52


class Op(Enum):
    REMOVE = "-"
    PLACE = "="


@dataclass
class Lens:
    label: str
    power: int

    def __repr__(self):
        return f"[{self.label} {self.power}]"

    @property
    def hash(self) -> int:
        return lame_hash(self.label)


@dataclass
class Step:
    label: str
    op: Op
    lens: Optional[int]

    def __repr__(self):
        op = "place" if self.op == Op.PLACE else "remove"
        return f"{op}: {self.label} {self.lens if self.lens else ''}"

    @staticmethod
    def parse(seq: str):
        a, b, c = re.findall(PATTERN, seq)[0]
        op = Op.REMOVE if b == "-" else Op.PLACE
        return Step(a, op, int(c) if c else None)

    def get_lens(self) -> Lens:
        assert self.lens is not None
        return Lens(self.label, self.lens)


def main(input):
    print(len(input))
    total = sum(lame_hash(seq) for seq in input)
    print(f"{total=}")
    boxes = DefaultDict(list)
    steps = [Step.parse(seq) for seq in input]
    for step in steps:
        print(step)
        if step.op == Op.PLACE:
            lens = step.get_lens()
            box = boxes[lens.hash]
            already_there_i = None
            for i, l in enumerate(box):
                if l.label == lens.label:
                    already_there_i = i
                    break
            if already_there_i is None:
                box.append(lens)
            else:
                box[already_there_i] = lens
        else:
            box = boxes[lame_hash(step.label)]
            i = next((i for (i, x) in enumerate(box) if x.label == step.label), None)
            if i is not None:
                del box[i]
    total_power = 0
    for box_i, box in boxes.items():
        print(f"BOX {box_i}: {box}")
        for lens_i, lens in enumerate(box):
            total_power += (1 + box_i) * (1 + lens_i) * lens.power
    print(f"{total_power=}")


if __name__ == "__main__":
    main(input)
