#!/usr/bin/env python
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Callable, NamedTuple, Optional, Self
import enum
from pprint import pp
from functools import reduce
import re

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


CmpOp = NamedTuple("CmpOp", [("key", str), ("cmp", str), ("val", int)])


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
    # for line in input:
    #     print(line)
    workflows, parts = parse(input)
    # pp(workflows)
    # pp(parts)
    accepted = [part for part in parts if shall_accept(workflows, part)]
    # for part in parts[:]:
    #     print(f"{shall_accept(workflows, part)=}")
    print(f"{sum(part_total(part) for part in accepted)=}")


class Interval(NamedTuple("Interval", [("start", int), ("end", int)])):
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


@dataclass
class PartStats:
    x: Interval
    m: Interval
    a: Interval
    s: Interval

    def split(self, other: Self, key: str) -> tuple[Self, Self]:
        a, b = self, self
        x, y = getattr(self, key) - getattr(other, key)
        setattr(a, key, x)
        setattr(b, key, y)
        return a, b

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


def intervals_for_rule(
    rule: Rule,
    accepted: Optional[PartStats] = None,
    rejected: Optional[PartStats] = None,
    remaining: Optional[PartStats] = None,
) -> StatsGroup:
    if accepted is None:
        accepted = PartStats.zero()
    if rejected is None:
        rejected = PartStats.zero()
    if remaining is None:
        remaining = PartStats.full()
    # remaining = PartStats.full() - accepted - rejected
    if rule.cmp_op is None:
        if rule.op == Op.ACCEPT:
            return StatsGroup([accepted, remaining], [rejected], [])
        if rule.op == Op.REJECT:
            return StatsGroup([accepted], [rejected, remaining], [])
        if rule.op == Op.SEND:
            assert rule.dst
            return StatsGroup(
                [accepted], [rejected], [], dst=rule.dst, sending=remaining
            )
    assert rule.cmp_op is not None

    cmp_op = rule.cmp_op
    rem_stat = getattr(remaining, cmp_op.key)
    acc_stat = getattr(accepted, cmp_op.key)
    rej_stat = getattr(rejected, cmp_op.key)
    cmp_interval = Interval(0, 0)
    if cmp_op.cmp == "<":
        cmp_interval = Interval(1, cmp_op.val)
    if cmp_op.cmp == ">":
        cmp_interval = Interval(cmp_op.val + 1, 4001)
    passing_intervals = [cmp_interval, rem_stat]
    # for rem in rem_stat:
    #     diffed.extend(difference_intervals([rem] + passing_intervals))
    # print(f"{diffed=}")

    if rule.op == Op.ACCEPT:
        R = remaining
        X = PartStats.zero()
        setattr(X, cmp_op.key, [cmp_interval])
        Y = PartStats.zero()
        setattr(Y, cmp_op.key, list(Interval(1, 4001) - cmp_interval))
        Z = R - Y
        A2 = accepted | Z
        R2 = R - X
        return StatsGroup(A2, rej, R2)
    # if rule.op == Op.REJECT:
    #     R = remaining
    #     X = PartStats.zero()
    #     setattr(X, cmp_op.key, [cmp_interval])
    #     Y = PartStats.zero()
    #     setattr(Y, cmp_op.key, list(Interval(1, 4001) - cmp_interval))
    #     Z = R - Y
    #     A2 = rejected | Z
    #     R2 = R - X
    #     return StatsGroup(acc, A2, R2)
    # # SEND
    # R = remaining.copy()
    # X = PartStats.zero()
    # setattr(X, cmp_op.key, [cmp_interval])
    # Y = PartStats.zero()
    # setattr(Y, cmp_op.key, list(Interval(1, 4001) - cmp_interval))
    # Z = R - Y
    # R2 = R - X
    # # TODO: this is wrong, but I'm writing it down
    # assert rule.dst
    # # setattr(accepted, cmp_op.key, union_intervals(inter + acc_stat))
    # return StatsGroup(acc, rej, R2, dst=rule.dst, sending=Z)


def handle_intervals_for_workflow(
    workflows: dict[str, Workflow],
    wf_key: str,
    accepted: PartStats,
    rejected: PartStats,
    remaining: PartStats,
) -> StatsGroup:
    wf = workflows[wf_key]
    group = StatsGroup(accepted, rejected, remaining)
    for rule in wf.rules:
        group = intervals_for_rule(
            rule, group.accepted, group.rejected, group.remaining
        )
        # print(acc, rej, rem, snd)
        if group.dst and group.sending:
            dst, snd_rem = group.dst, group.sending
            group = handle_intervals_for_workflow(
                workflows, dst, group.accepted, group.rejected, snd_rem
            )
    return group


def count_combinations(stats: PartStats) -> int:
    x = sum(len(i) for i in stats.x)
    m = sum(len(i) for i in stats.m)
    a = sum(len(i) for i in stats.a)
    s = sum(len(i) for i in stats.s)
    return x * m * a * s


def combinations_for_workflow(
    workflows: dict[str, Workflow],
    wf_key: str,
    accepted: PartStats,
    rejected: PartStats,
    remaining: PartStats,
) -> int:
    wf = workflows[wf_key]
    acc, rej, rem = accepted, rejected, remaining
    n_accepted: int = 0
    for rule in wf.rules:
        group = intervals_for_rule(rule, acc, rej, rem)
        n_accepted += count_combinations(acc)
        # print(acc, rej, rem, snd)
        if group.dst and group.sending:
            dst, snd_rem = group.dst, group.sending
            n_accepted = combinations_for_workflow(workflows, dst, acc, rej, snd_rem)
    return n_accepted

    # return handle_intervals_for_rule(workflows, rule, acc, rej, snd_rem)
    # acc, rej, rem, snd = intervals_for_rule(rule, accepted, rejected, remaining)


def draw_line(width, thick=False):
    char = "─" if not thick else "━"
    print(char * width)


@contextmanager
def test(name: str):
    print(name, end="... ")
    try:
        yield None
        print("PASS")
    except Exception as e:
        print("FAIL")
        raise


def tests():
    draw_line(60, thick=True)
    print("RUNNING TESTS")
    draw_line(60)

    workflows, _ = parse(CONTROL_1)

    with test("split PartStats"):
        ps = PartStats(
            x=Interval(1, 4001),
            m=Interval(1, 4001),
            a=Interval(1, 4001),
            s=Interval(1, 4001),
        )
        rule = Rule(
            "fake",
            match=lambda x: True,
            op=Op.ACCEPT,
            cmp_op=CmpOp(key="a", cmp="<", val=1000),
        )

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
        group = intervals_for_rule(rule)
        assert group.accepted == PartStats.zero(), group.accepted
        assert group.rejected == PartStats.zero(), group.rejected
        assert group.remaining == PartStats(
            x=Interval(1, 4001),
            m=Interval(1, 4001),
            a=Interval(2006, 4001),
            s=Interval(1, 4001),
        ), group.remaining
        assert group.dst == "qkq", group.dst
        assert group.sending == PartStats(
            x=Interval(1, 4001),
            m=Interval(1, 4001),
            a=Interval(1, 2006),
            s=Interval(1, 4001),
        ), group.sending

    with test("intervals for rule: accepting"):
        px = workflows["px"]
        rule: Rule = px.rules[1]
        group = intervals_for_rule(rule)
        assert group.accepted == PartStats(
            x=Interval(1, 4001),
            m=Interval(2091, 4001),
            a=Interval(1, 4001),
            s=Interval(1, 4001),
        ), group.accepted
        assert group.rejected == PartStats.zero(), group.rejected
        assert group.remaining == PartStats(
            x=Interval(1, 4001),
            m=Interval(1, 2091),
            a=Interval(1, 4001),
            s=Interval(1, 4001),
        ), group.remaining
        assert group.dst is None, group.dst
        assert group.sending is None, group.sending

    with test("intervals for rule: rejecting"):
        rule = workflows["pv"].rules[0]
        group = intervals_for_rule(rule)
        assert group.accepted == PartStats.zero(), group.accepted
        assert group.rejected == PartStats(
            x=Interval(1, 4001),
            m=Interval(1, 4001),
            a=Interval(1717, 4001),
            s=Interval(1, 4001),
        ), group.rejected
        assert group.remaining == PartStats(
            x=Interval(1, 4001),
            m=Interval(1, 4001),
            a=Interval(1, 1717),
            s=Interval(1, 4001),
        ), group.remaining
        assert group.dst is None, group.dst
        assert group.sending is None, group.sending

    with test("handle intervals for rule"):
        remaining = PartStats(
            x=Interval(1, 4001),
            m=Interval(1, 4001),
            a=Interval(200, 3000),
            s=Interval(1, 4001),
        )
        group = handle_intervals_for_workflow(
            workflows, "hdj", PartStats.zero(), PartStats.zero(), remaining
        )
        assert group.remaining == PartStats.zero(), group.remaining
        assert group.accepted == PartStats(
            x=Interval(1, 4001),
            m=Interval(1, 839),
            a=Interval(200, 1717),
            s=Interval(1, 4001),
        ), group.accepted
        assert group.rejected == PartStats(
            x=Interval(1, 4001),
            m=Interval(1, 4001),
            a=Interval(200, 3000),
            s=Interval(1, 4001),
        ), group.rejected

    draw_line(60)
    print("ALL TESTS PASSED")
    draw_line(60, thick=True)


def part_2(input):
    print("━" * len(input[0]))
    for line in input:
        print(line)
    print("━" * len(input[-1]))
    tests()
    workflows, _ = parse(input)
    # rule = workflows["in"].rules[0]
    # pp(rule)
    res = handle_intervals_for_workflow(
        workflows, "pv", PartStats.zero(), PartStats.zero(), PartStats.full()
    )
    # print(f"{acc=}\n{rej=}\n{rem=}")
    x = sum(len(i) for i in res.accepted.x)
    m = sum(len(i) for i in res.accepted.m)
    a = sum(len(i) for i in res.accepted.a)
    s = sum(len(i) for i in res.accepted.s)
    # combinations = x * m * a * s
    combinations = (1 + x) * (1 + m) * (1 + a) * (1 + s)
    print(f"{x=} {m=} {a=} {s=} {combinations=}")
    # accepted = [part for part in parts if shall_accept(workflows, part)]
    # for part in parts[:]:
    #     print(f"{shall_accept(workflows, part)=}")
    # print(f"{sum(part_total(part) for part in accepted)=}")


if __name__ == "__main__":
    part_2(CONTROL_1)
