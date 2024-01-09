#!/usr/bin/env python
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Callable, Deque, NamedTuple, Optional, Self
import enum
from pprint import pp, pformat
from functools import reduce
from itertools import zip_longest
import re
import datetime

from z3 import Option

CONTROL_1 = """\
px{a<2006:qkq,m>2090:A,rfg}
pv{a>1716:R,A}
lnx{m>1548:A,A}
rfg{s<537:gd,x>2440:R,A}
qs{s>3448:A,lnx}
qkq{x<1416:A,crn}
crn{x>2662:A,R}
in{s<1351:px,qqz}
qqz{s>2770:qs,m<1801:hdj,R}
gd{a>3333:R,R}
hdj{m>838:A,pv}

{x=787,m=2655,a=1222,s=2876}
{x=1679,m=44,a=2067,s=496}
{x=2036,m=264,a=79,s=2244}
{x=2461,m=1339,a=466,s=291}
{x=2127,m=1623,a=2188,s=1013}
""".splitlines()

with open("2023-19.input") as f:
    input_file = [line.strip() for line in f.readlines()]


class Op(str, enum.ReprEnum):
    ACCEPT = "A"
    REJECT = "R"
    SEND = "S"


Part = NamedTuple("Part", [("x", int), ("m", int), ("a", int), ("s", int)])


def part_total(part: Part) -> int:
    return sum(part)


class CmpOp(NamedTuple("CmpOp", [("key", str), ("cmp", str), ("val", int)])):
    def interval(self) -> "Interval":
        if self.cmp == "<":
            return Interval(1, self.val)
        return Interval(self.val + 1, 4001)


@dataclass
class Rule:
    text: str
    match: Callable[[Part], bool]
    op: Op
    cmp_op: Optional[CmpOp] = None
    dst: Optional[str] = None

    # cmp: Callable[[Part], Op]
    def __repr__(self):
        return self.text


@dataclass
class Workflow:
    id: str
    rules: list[Rule]

    def handle(self, part: Part) -> tuple[Op, Optional[str]]:
        for rule in self.rules:
            matches = rule.match(part)
            # print(f"@ rule {rule.text}: {matches}")
            if matches:
                return rule.op, rule.dst
        raise Exception("NO MATCH")


def parse_part(line: str) -> Part:
    x, m, a, s = line.strip("{}").split(",", 4)
    return Part(int(x[2:]), int(m[2:]), int(a[2:]), int(s[2:]))


RULE_PATTERN = re.compile(r"([xmas])([<>])(\d+):(\w+)")


def match_fn(key: str, val: int, cmp: str) -> Callable[[Part], bool]:
    if cmp == "<":
        cmp_method = "__lt__"
    else:
        cmp_method = "__gt__"

    def inner(part: Part) -> bool:
        attr = getattr(part, key)
        method = getattr(attr, cmp_method)
        return method(val)

    return inner


def parse_rule(text: str) -> Rule:
    if ":" in text:
        key, cmp, val, dst = re.findall(RULE_PATTERN, text)[0]
        if dst == "A":
            op = Op.ACCEPT
            dst = None
        elif dst == "R":
            op = Op.REJECT
            dst = None
        else:
            op = Op.SEND
            dst = dst
        return Rule(
            text=text,
            match=match_fn(key, int(val), cmp),
            op=op,
            cmp_op=CmpOp(key, cmp, int(val)),
            dst=dst,
        )
    if text == "A":
        op = Op.ACCEPT
        dst = None
    elif text == "R":
        op = Op.REJECT
        dst = None
    else:
        op = Op.SEND
        dst = text
    return Rule(text=text, match=lambda _: True, op=op, dst=dst)


def parse_workflow(line: str) -> Workflow:
    id, rest = line.split("{", 1)
    rules = rest.rstrip("}").split(",")
    rules = [parse_rule(x) for x in rules]
    return Workflow(id, rules)


def parse(input) -> tuple[dict[str, Workflow], list[Part]]:
    workflows = {}
    parts = []
    last_line = 0
    for i, line in enumerate(input):
        if not line:
            last_line = i
            break
        wf = parse_workflow(line)
        workflows[wf.id] = wf
    for line in input[last_line + 1 :]:
        parts.append(parse_part(line))
    return workflows, parts


def shall_accept(workflows: dict[str, Workflow], part: Part) -> bool:
    wf = workflows["in"]
    while True:
        # print(f"checking wf {wf.id}")
        op, dst = wf.handle(part)
        if op == Op.ACCEPT:
            return True
        elif op == Op.REJECT:
            return False
        assert op == Op.SEND
        assert dst is not None
        wf = workflows[dst]


def part_1(input):
    workflows, parts = parse(input)
    accepted = [part for part in parts if shall_accept(workflows, part)]
    print(f"{sum(part_total(part) for part in accepted)=}")


class Interval(NamedTuple("Interval", [("start", int), ("end", int)])):
    def __repr__(self) -> str:
        return f"I({self.start}..{self.end})"

    def __len__(self):
        if self.end <= self.start:
            return 0
        return self.end - self.start

    def __sub__(self, other) -> tuple["Interval", "Interval"]:
        # Case when there is no overlap
        if self.end <= other.start or other.end <= self.start:
            return (self, Interval(0, 0))
        # Case when 'other' completely covers 'self'
        if other.start <= self.start and other.end >= self.end:
            return (Interval(0, 0), Interval(0, 0))
        # Case when 'other' overlaps the start of 'self'
        if other.start <= self.start:
            return (Interval(other.end, self.end), Interval(0, 0))
        # Case when 'other' overlaps the end of 'self'
        if other.end >= self.end:
            return (Interval(self.start, other.start), Interval(0, 0))
        # Case when 'other' is completely inside 'self'
        return (Interval(self.start, other.start), Interval(other.end, self.end))

    def __and__(self, other: Self) -> "Interval":
        new_start = max(self.start, other.start)
        new_end = min(self.end, other.end)
        if new_start <= new_end:
            return Interval(new_start, new_end)
        return Interval(0, 0)

    def __or__(self, other: Self) -> tuple["Interval", "Interval"]:
        if self.end >= other.start and other.end >= self.start:
            new_start = min(self.start, other.start)
            new_end = max(self.end, other.end)
            return Interval(new_start, new_end), Interval(0, 0)
        return self, other


# def intersect_intervals(base: Interval, others: list[Interval]) -> list[Interval]:
#     if not others:
#         return []
#     out = [base & other for other in others]
#     return [x for x in out if len(x) > 0]


def union_intervals(intervals: list[Interval]) -> list[Interval]:
    if not intervals:
        return []
    sorted_intervals = sorted(intervals, key=lambda x: x.start)
    merged_intervals = [sorted_intervals[0]]
    for current in sorted_intervals[1:]:
        last_merged = merged_intervals[-1]
        if current.start <= last_merged.end:
            merged_intervals[-1] = Interval(
                min(last_merged.start, current.start), max(last_merged.end, current.end)
            )
        else:
            merged_intervals.append(current)
    # return [x for x in merged_intervals if len(x) > 0]
    return merged_intervals


# def difference_intervals(intervals: list[Interval]) -> list[Interval]:
#     if not intervals:
#         return []
#     difference_set = [intervals[0]]
#     for current in intervals[1:]:
#         new_difference_set = []
#         for interval in difference_set:
#             new_difference_set.extend(interval - current)
#         difference_set = new_difference_set
#     return [x for x in difference_set if len(x) > 0]

Split = NamedTuple("Split", [("matched", "PartStats"), ("unmatched", "PartStats")])


@dataclass
class PartStats:
    x: Interval
    m: Interval
    a: Interval
    s: Interval

    def __hash__(self) -> int:
        return hash((self.x, self.m, self.a, self.s))

    def __lt__(self, other) -> bool:
        return (self.x, self.m, self.a, self.s) < (other.x, other.m, other.a, other.s)

    def split(self, cmp_op: CmpOp) -> Split:
        unmatched, matched = self.copy(), self.copy()
        key = cmp_op.key
        interval = cmp_op.interval()
        self_interval = getattr(self, key)
        x, y = self_interval - interval
        # NOTE: let's assume there will never be a hole
        assert y == Interval(0, 0), y
        setattr(unmatched, key, x)
        setattr(matched, key, self_interval & interval)
        return Split(unmatched=unmatched, matched=matched)

    def copy(self) -> "PartStats":
        return PartStats(x=self.x, m=self.m, a=self.a, s=self.s)

    # def __sub__(self: Self, other: Self) -> tuple[Self, Optional[Self]]:
    #     x = self.x - other.x
    #     m = self.m - other.m
    #     a = self.a - other.a
    #     s = self.s - other.s

    # x = reduce(lambda acc, x: acc + difference_intervals([x] + other.x), self.x, [])
    # m = reduce(lambda acc, m: acc + difference_intervals([m] + other.m), self.m, [])
    # a = reduce(lambda acc, a: acc + difference_intervals([a] + other.a), self.a, [])
    # s = reduce(lambda acc, s: acc + difference_intervals([s] + other.s), self.s, [])
    # return PartStats(x=x, m=m, a=a, s=s)

    #
    # def __and__(self, other):
    #     x = reduce(lambda acc, x: acc + intersect_intervals(x, other.x), self.x, [])
    #     m = reduce(lambda acc, m: acc + intersect_intervals(m, other.m), self.m, [])
    #     a = reduce(lambda acc, a: acc + intersect_intervals(a, other.a), self.a, [])
    #     s = reduce(lambda acc, s: acc + intersect_intervals(s, other.s), self.s, [])
    #     return PartStats(x, m, a, s)
    #
    # def __or__(self, other):
    #     return PartStats(
    #         union_intervals(self.x + other.x),
    #         union_intervals(self.m + other.m),
    #         union_intervals(self.a + other.a),
    #         union_intervals(self.s + other.s),
    #     )

    @classmethod
    def zero(cls) -> "PartStats":
        return cls(Interval(0, 0), Interval(0, 0), Interval(0, 0), Interval(0, 0))

    @classmethod
    def one(cls) -> "PartStats":
        return cls(Interval(0, 1), Interval(0, 1), Interval(0, 1), Interval(0, 1))

    @classmethod
    def full(cls) -> "PartStats":
        return cls(
            Interval(1, 4001),
            Interval(1, 4001),
            Interval(1, 4001),
            Interval(1, 4001),
        )

    def is_zero(self) -> bool:
        return self == PartStats.zero()

    def combinations(self) -> int:
        return len(self.x) * len(self.m) * len(self.a) * len(self.s)


@dataclass
class StatsGroup:
    accepted: list[PartStats] = field(default_factory=list)
    rejected: list[PartStats] = field(default_factory=list)
    remaining: list[PartStats] = field(default_factory=list)
    dst: Optional[str] = None
    sending: Optional[PartStats] = None

    @classmethod
    def default(cls) -> "StatsGroup":
        return StatsGroup([], [], [PartStats.full()])


@dataclass
class RuleIntervals:
    accepted: Optional[PartStats] = None
    rejected: Optional[PartStats] = None
    remaining: Optional[PartStats] = None
    dst: Optional[str] = None
    sending: Optional[PartStats] = None


def intervals_for_rule(
    rule: Rule,
    remaining: Optional[PartStats] = None,
) -> RuleIntervals:
    # print(f"evaluating rule {rule}")
    if remaining is None:
        remaining = PartStats.full()
    if rule.cmp_op is None:
        if rule.op == Op.ACCEPT:
            return RuleIntervals(accepted=remaining)
        if rule.op == Op.REJECT:
            return RuleIntervals(rejected=remaining)
        if rule.op == Op.SEND:
            assert rule.dst
            return RuleIntervals(dst=rule.dst, sending=remaining)
    assert rule.cmp_op is not None
    split = remaining.split(rule.cmp_op)
    if rule.op == Op.ACCEPT:
        return RuleIntervals(accepted=split.matched, remaining=split.unmatched)
    elif rule.op == Op.REJECT:
        return RuleIntervals(rejected=split.matched, remaining=split.unmatched)
    assert rule.dst is not None
    return RuleIntervals(
        remaining=split.unmatched,
        dst=rule.dst,
        sending=split.matched,
    )


def handle_intervals_for_workflow(
    workflows: dict[str, Workflow],
    wf_key: str,
    remaining: PartStats,
) -> StatsGroup:
    # print(f"WORKFLOW: {wf_key} handling {remaining}")
    accepted = []
    rejected = []
    q = deque()
    q.append((wf_key, 0, remaining))
    while q:
        (wf_key, rule_i, rem) = q.popleft()
        wf = workflows[wf_key]
        rule = wf.rules[rule_i]
        intr = intervals_for_rule(rule, rem)
        if intr.accepted:
            accepted.append(intr.accepted)
        if intr.rejected:
            rejected.append(intr.rejected)
        if intr.dst:
            assert intr.sending is not None
            q.append((intr.dst, 0, intr.sending))
        if intr.remaining:
            q.append((wf_key, rule_i + 1, intr.remaining))
    out = StatsGroup(
        accepted=[x for x in accepted if x and x != PartStats.zero()],
        rejected=[x for x in rejected if x and x != PartStats.zero()],
        remaining=[],
    )
    return out


def draw_line(width, thick=False, double=False, dots=False, end="\n", pre="", post=""):
    char = "─"
    if thick:
        char = "━"
    if double:
        char = "═"
    if dots:
        char = "┈"
    if thick and dots:
        char = "┉"
    print(pre, char * width, post, sep="", end=end)


@contextmanager
def test(name: str):
    print(" ", name, end="... ")
    try:
        yield None
        print("PASS")
        draw_line(58, pre=" ", dots=True)
    except Exception:
        print("FAIL")
        raise


def run_tests():
    draw_line(58, double=True, pre="╒", post="╕")
    print("│", "RUNNING TESTS".ljust(56), "│")
    draw_line(58, pre="└", post="┘")

    workflows, _ = parse(CONTROL_1)

    with test("subtract intervals"):
        sub = Interval(1, 4001) - Interval(1, 2001)
        assert sub == (Interval(2001, 4001), Interval(0, 0)), sub
        sub = Interval(1, 4001) - Interval(1001, 2001)
        assert sub == (Interval(1, 1001), Interval(2001, 4001)), sub

    with test("cmp_op.interval()"):
        cmp_op = CmpOp(key="a", cmp="<", val=1000)
        assert cmp_op.interval() == Interval(1, 1000), cmp_op.interval()
        cmp_op = CmpOp(key="a", cmp=">", val=1000)
        assert cmp_op.interval() == Interval(1001, 4001), cmp_op.interval()

    with test("split PartStats"):
        ps = PartStats(
            x=Interval(1, 4001),
            m=Interval(1, 4001),
            a=Interval(1, 4001),
            s=Interval(1, 4001),
        )
        cmp_op = CmpOp(key="a", cmp="<", val=1000)
        split = ps.split(cmp_op)
        assert split.unmatched == PartStats(
            x=Interval(1, 4001),
            m=Interval(1, 4001),
            a=Interval(1000, 4001),
            s=Interval(1, 4001),
        ), split.unmatched
        assert split.matched == PartStats(
            x=Interval(1, 4001),
            m=Interval(1, 4001),
            a=Interval(1, 1000),
            s=Interval(1, 4001),
        ), split.matched

    with test("split partial PartStats"):
        ps = PartStats(
            x=Interval(200, 3001),
            m=Interval(300, 2001),
            a=Interval(350, 2501),
            s=Interval(400, 3501),
        )
        cmp_op = CmpOp(key="a", cmp="<", val=1000)
        split = ps.split(cmp_op)
        assert split.unmatched == PartStats(
            x=Interval(200, 3001),
            m=Interval(300, 2001),
            a=Interval(1000, 2501),
            s=Interval(400, 3501),
        ), split.unmatched
        assert split.matched == PartStats(
            x=Interval(200, 3001),
            m=Interval(300, 2001),
            a=Interval(350, 1000),
            s=Interval(400, 3501),
        ), split.matched

    with test("rule: compare and send"):
        px = workflows["px"]
        rule: Rule = px.rules[0]
        assert rule.op == Op.SEND, rule
        assert rule.dst == "qkq", rule.dst
        assert rule.cmp_op == CmpOp(key="a", cmp="<", val=2006), rule.cmp_op
        assert rule.match(Part(x=1, m=2, a=1000, s=4))
        assert not rule.match(Part(x=1, m=2, a=5000, s=4))

    with test("rule: compare and accept"):
        px = workflows["px"]
        rule: Rule = px.rules[1]
        assert rule.op == Op.ACCEPT, rule
        assert rule.dst is None, rule.dst
        assert rule.cmp_op == CmpOp(key="m", cmp=">", val=2090), rule.cmp_op
        assert not rule.match(Part(x=1, m=2090, a=1000, s=4))
        assert rule.match(Part(x=1, m=2091, a=5000, s=4))

    with test("rule: send"):
        px = workflows["px"]
        rule: Rule = px.rules[2]
        assert rule.op == Op.SEND, rule
        assert rule.dst == "rfg", rule.dst
        assert rule.cmp_op is None, rule.cmp_op
        assert rule.match(Part(0, 0, 0, 0)), rule.match

    with test("intervals for rule: sending"):
        px = workflows["px"]
        rule: Rule = px.rules[0]
        intr = intervals_for_rule(rule)
        assert intr.accepted is None, intr.accepted
        assert intr.rejected is None, intr.rejected
        assert intr.remaining == PartStats(
            x=Interval(1, 4001),
            m=Interval(1, 4001),
            a=Interval(2006, 4001),
            s=Interval(1, 4001),
        ), intr.remaining
        assert intr.dst == "qkq", intr.dst
        assert intr.sending == PartStats(
            x=Interval(1, 4001),
            m=Interval(1, 4001),
            a=Interval(1, 2006),
            s=Interval(1, 4001),
        ), intr.sending

    with test("intervals for rule: accepting"):
        px = workflows["px"]
        rule: Rule = px.rules[1]
        intr = intervals_for_rule(rule)
        assert intr.accepted == PartStats(
            x=Interval(1, 4001),
            m=Interval(2091, 4001),
            a=Interval(1, 4001),
            s=Interval(1, 4001),
        ), intr.accepted
        assert intr.rejected is None, intr.rejected
        assert intr.remaining == PartStats(
            x=Interval(1, 4001),
            m=Interval(1, 2091),
            a=Interval(1, 4001),
            s=Interval(1, 4001),
        ), intr.remaining
        assert intr.dst is None, intr.dst
        assert intr.sending is None, intr.sending

    with test("intervals for rule: rejecting"):
        rule = workflows["pv"].rules[0]
        intr = intervals_for_rule(rule)
        assert intr.accepted is None, intr.accepted
        assert intr.rejected == PartStats(
            x=Interval(1, 4001),
            m=Interval(1, 4001),
            a=Interval(1717, 4001),
            s=Interval(1, 4001),
        ), intr.rejected
        assert intr.remaining == PartStats(
            x=Interval(1, 4001),
            m=Interval(1, 4001),
            a=Interval(1, 1717),
            s=Interval(1, 4001),
        ), intr.remaining
        assert intr.dst is None, intr.dst
        assert intr.sending is None, intr.sending

    with test("intervals for rule: sending again"):
        # qqz │ s>2770:qs │ m<1801:hdj │ R
        # qs  │ s>3448:A  │ lnx
        # lnx │ m>1548:A  │ A
        # hdj │ m>838:A   │ pv
        # pv  │ a>1716:R  │ A
        rule = workflows["qqz"].rules[1]
        rem = PartStats.full()
        rem.m = Interval(1, 3001)
        intr = intervals_for_rule(rule, rem)
        assert intr.accepted is None, intr.accepted
        assert intr.rejected is None, intr.rejected
        assert intr.remaining == PartStats(
            x=Interval(1, 4001),
            m=Interval(1801, 3001),
            a=Interval(1, 4001),
            s=Interval(1, 4001),
        ), intr.remaining
        assert intr.dst == "hdj", intr.dst
        assert intr.sending == PartStats(
            x=Interval(1, 4001),
            m=Interval(1, 1801),
            a=Interval(1, 4001),
            s=Interval(1, 4001),
        ), intr.sending

    with test("handle intervals for terminal workflow"):
        remaining = PartStats(
            x=Interval(1, 4001),
            m=Interval(1, 4001),
            a=Interval(200, 3000),
            s=Interval(1, 4001),
        )
        group = handle_intervals_for_workflow(workflows, "pv", remaining)
        assert group.remaining == [], group.remaining
        assert group.accepted == [
            PartStats(
                x=Interval(1, 4001),
                m=Interval(1, 4001),
                a=Interval(200, 1717),
                s=Interval(1, 4001),
            )
        ], group.accepted
        assert group.rejected == [
            PartStats(
                x=Interval(1, 4001),
                m=Interval(1, 4001),
                a=Interval(1717, 3000),
                s=Interval(1, 4001),
            )
        ], group.rejected

    with test("handle intervals for workflow"):
        remaining = PartStats(
            x=Interval(1, 4001),
            m=Interval(1, 4001),
            a=Interval(200, 3000),
            s=Interval(1, 4001),
        )
        group = handle_intervals_for_workflow(workflows, "hdj", remaining)
        assert group.remaining == [], group.remaining

        group.accepted.sort()
        assert group.accepted == [
            PartStats(
                x=Interval(start=1, end=4001),
                m=Interval(start=1, end=839),
                a=Interval(start=200, end=1717),
                s=Interval(start=1, end=4001),
            ),
            PartStats(
                x=Interval(start=1, end=4001),
                m=Interval(start=839, end=4001),
                a=Interval(start=200, end=3000),
                s=Interval(start=1, end=4001),
            ),
        ], group.accepted
        assert group.rejected == [
            PartStats(
                x=Interval(start=1, end=4001),
                m=Interval(start=1, end=839),
                a=Interval(start=1717, end=3000),
                s=Interval(start=1, end=4001),
            ),
        ], group.rejected

    with test("handle intervals for workflow"):
        # qqz │ s>2770:qs │ m<1801:hdj │ R
        # qs  │ s>3448:A  │ lnx
        # lnx │ m>1548:A  │ A
        # hdj │ m>838:A   │ pv
        # pv  │ a>1716:R  │ A
        group = handle_intervals_for_workflow(workflows, "qqz", PartStats.full())
        assert group.remaining == [], group.remaining

        # group.accepted.sort()
        accepted = group.accepted
        assert len(accepted) == 5, len(accepted)
        expected_accepted = [
            PartStats(
                x=Interval(start=1, end=4001),
                m=Interval(start=1, end=4001),
                a=Interval(start=1, end=4001),
                s=Interval(start=3449, end=4001),
            ),
            PartStats(
                x=Interval(start=1, end=4001),
                m=Interval(start=1549, end=4001),
                a=Interval(start=1, end=4001),
                s=Interval(start=2771, end=3449),
            ),
            PartStats(
                x=Interval(start=1, end=4001),
                m=Interval(start=1, end=1549),
                a=Interval(start=1, end=4001),
                s=Interval(start=2771, end=3449),
            ),
            PartStats(
                x=Interval(start=1, end=4001),
                m=Interval(start=839, end=1801),
                a=Interval(start=1, end=4001),
                s=Interval(start=1, end=2771),
            ),
            PartStats(
                x=Interval(start=1, end=4001),
                m=Interval(start=1, end=839),
                a=Interval(start=1717, end=4001),
                s=Interval(start=1, end=2771),
            ),
        ]
        for i, exp in enumerate(expected_accepted):
            assert exp in accepted, f"index {i}, {exp}\nnot in\n{pformat(accepted)}"

    draw_line(60)
    print("ALL TESTS PASSED")
    draw_line(60, thick=True)


CONTROL_TARGET = 167_409_079_868_000


def part_2(input):
    print("━" * len(input[0]))
    for line in input:
        print(line)
    print("━" * len(input[-1]))
    # run_tests()
    workflows, _ = parse(input)
    # rule = workflows["in"].rules[0]
    # pp(rule)
    accepted_combinations = 0
    wf_keys = list(workflows.keys())
    for wf_key in ("in",):
        res = handle_intervals_for_workflow(workflows, wf_key, PartStats.full())
        comb = sum(x.combinations() for x in res.accepted)
        assert comb > 0, res
        accepted_combinations += comb
    print(f"{accepted_combinations=}")
    print("off by", accepted_combinations - CONTROL_TARGET)
    # accepted = [part for part in parts if shall_accept(workflows, part)]
    # for part in parts[:]:
    #     print(f"{shall_accept(workflows, part)=}")
    # print(f"{sum(part_total(part) for part in accepted)=}")


if __name__ == "__main__":
    start = datetime.datetime.now()
    part_2(input_file)
    print(f"elapsed: {datetime.datetime.now() - start}")
