import pytest
import importlib.util
import sys


def module_from_file(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


challenge = module_from_file("challenge", "2023-19.py")
from challenge import (
    Rule,
    Interval,
    intersect_intervals,
    union_intervals,
    difference_intervals,
    intervals_for_rule,
    handle_intervals_for_workflow,
    CONTROL_1,
    PartStats,
    parse,
    combinations_for_workflow,
    count_combinations,
)


def test_simple_workflow():
    workflows, _ = parse(CONTROL_1)
    (
        acc,
        _,
        _,
    ) = handle_intervals_for_workflow(
        workflows, "pv", PartStats.zero(), PartStats.zero(), PartStats.full()
    )
    combinations = count_combinations(acc)
    assert combinations == 1716 * 4000 * 4000 * 4000


def test_sending_workflow():
    workflows, _ = parse(CONTROL_1)
    (
        acc,
        rej,
        rem,
    ) = handle_intervals_for_workflow(
        workflows, "hdj", PartStats.zero(), PartStats.zero(), PartStats.full()
    )
    accepted = (4000 - 838) * 4000 * 4000 * 4000
    remaining = 4000 * 4000 * 4000 * 4000
    next_rem = remaining - accepted
    rejected = (4000 - 1716) * (4000 - 838) * 4000 * 4000
    accepted = next_rem - rejected + accepted
    assert count_combinations(rem) == 0
    assert count_combinations(acc) == accepted
    assert count_combinations(rej) == rejected


def test_full_control():
    workflows, _ = parse(CONTROL_1)
    (
        acc,
        rej,
        rem,
    ) = handle_intervals_for_workflow(
        workflows, "in", PartStats.zero(), PartStats.zero(), PartStats.full()
    )

    assert count_combinations(rem) == 0
    combinations = count_combinations(acc)
    expected = 167409079868000
    diff = expected - combinations
    assert diff == 0


def test_intervals_for_rule_simple():
    workflows, _ = parse(CONTROL_1)
    rule = workflows["pv"].rules[0]  # a>1716:R
    (
        acc,
        rej,
        rem,
        snd,
    ) = intervals_for_rule(rule, PartStats.zero(), PartStats.zero(), PartStats.full())
    assert count_combinations(acc) == 0
    assert sum(len(x) for x in rej.x) == 4000
    assert sum(len(m) for m in rej.m) == 4000
    assert sum(len(a) for a in rej.a) == (4000 - 1716)
    assert sum(len(s) for s in rej.s) == 4000

    assert sum(len(x) for x in rem.x) == 4000
    assert sum(len(m) for m in rem.m) == 4000
    assert sum(len(a) for a in rem.a) == 1716
    assert sum(len(s) for s in rem.s) == 4000

    assert count_combinations(rej) == (4000 - 1716) * 4000 * 4000 * 4000
    assert snd is None
    workflows, _ = parse(CONTROL_1)


def test_intervals_for_rule_cmp_accept():
    workflows, _ = parse(CONTROL_1)
    rule = workflows["hdj"].rules[0]  # m>838:A
    (
        acc,
        rej,
        rem,
        snd,
    ) = intervals_for_rule(rule, PartStats.zero(), PartStats.zero(), PartStats.full())

    assert (acc.x, acc.m, acc.a, acc.s) == (
        [Interval(1, 4001)],
        [Interval(839, 4001)],
        [Interval(1, 4001)],
        [Interval(1, 4001)],
    )
    assert (rem.x, rem.m, rem.a, rem.s) == (
        [Interval(1, 4001)],
        [Interval(1, 839)],
        [Interval(1, 4001)],
        [Interval(1, 4001)],
    )
    assert (rej.x, rej.m, rej.a, rej.s) == ([], [], [], [])
    # dst, snd_rem = snd
    # workflows, _ = parse(CONTROL_1)


def test_intervals_for_rule_remaining_send():
    workflows, _ = parse(CONTROL_1)
    rule = workflows["hdj"].rules[1]  # pv
    (
        acc,
        rej,
        rem,
        snd,
    ) = intervals_for_rule(rule, PartStats.zero(), PartStats.zero(), PartStats.full())
    assert count_combinations(acc) == 0
    assert count_combinations(rej) == 0
    assert count_combinations(rem) == 0
    dst, snd_rem = snd
    assert dst == "pv"
    assert count_combinations(snd_rem) == 4000 * 4000 * 4000 * 4000
    workflows, _ = parse(CONTROL_1)


def test_intervals_for_rule_remaining_accept():
    workflows, _ = parse(CONTROL_1)
    rule = workflows["rfg"].rules[2]  # A
    (
        acc,
        rej,
        rem,
        _,
    ) = intervals_for_rule(rule, PartStats.zero(), PartStats.zero(), PartStats.full())
    assert count_combinations(acc) == 4000 * 4000 * 4000 * 4000
    assert count_combinations(rej) == 0
    assert count_combinations(rem) == 0


def test_intervals_for_rule_immediate_accepted():
    workflows, _ = parse(CONTROL_1)
    rule = workflows["rfg"].rules[2]  # A
    (
        acc,
        rej,
        rem,
        _,
    ) = intervals_for_rule(
        rule,
        PartStats(x=[Interval(10, 20)], m=[], a=[], s=[]),
        PartStats.zero(),
        PartStats(x=[Interval(20, 30)], m=[], a=[], s=[Interval(1, 100)]),
    )
    assert acc.x == [Interval(10, 30)]
    assert acc.m == []
    assert acc.a == []
    assert acc.s == [Interval(1, 100)]
    assert (rem.x, rem.m, rem.a, rem.s) == ([], [], [], [])
    assert (rej.x, rej.m, rej.a, rej.s) == ([], [], [], [])


def test_intervals_for_rule_immediate_reject():
    workflows, _ = parse(CONTROL_1)
    rule = workflows["gd"].rules[1]  # R
    (
        acc,
        rej,
        rem,
        _,
    ) = intervals_for_rule(
        rule,
        PartStats(x=[Interval(10, 20)], m=[], a=[], s=[]),
        PartStats(x=[], m=[Interval(69, 420)], a=[], s=[]),
        PartStats(x=[Interval(20, 30)], m=[], a=[], s=[Interval(1, 100)]),
    )
    assert (acc.x, acc.m, acc.a, acc.s) == ([Interval(10, 20)], [], [], [])
    assert (rej.x, rej.m, rej.a, rej.s) == (
        [Interval(20, 30)],
        [Interval(69, 420)],
        [],
        [Interval(1, 100)],
    )
    assert (rem.x, rem.m, rem.a, rem.s) == ([], [], [], [])


def test_simple_workflow_comb():
    workflows, _ = parse(CONTROL_1)
    combinations = combinations_for_workflow(
        workflows, "pv", PartStats.zero(), PartStats.zero(), PartStats.full()
    )
    assert combinations == 1716 * 4000 * 4000 * 4000


def test_sending_workflow_comb():
    workflows, _ = parse(CONTROL_1)
    combinations = combinations_for_workflow(
        workflows, "hdj", PartStats.zero(), PartStats.zero(), PartStats.full()
    )
    assert combinations != 0


def test_interval_length():
    interval = Interval(3, 5)
    assert len(interval) == 2


def test_interval_subtraction():
    interval1 = Interval(1, 5)
    interval2 = Interval(3, 7)
    result = interval1 - interval2
    assert result == (Interval(1, 3), Interval(0, 0))


def test_sub_no_overlap():
    interval1 = Interval(1, 3)
    interval2 = Interval(4, 6)
    assert interval1 - interval2 == (interval1, Interval(0, 0))


def test_sub_partial_overlap_start():
    interval1 = Interval(1, 5)
    interval2 = Interval(0, 3)
    assert interval1 - interval2 == (Interval(3, 5), Interval(0, 0))


def test_sub_partial_overlap_end():
    interval1 = Interval(1, 5)
    interval2 = Interval(3, 6)
    assert interval1 - interval2 == (Interval(1, 3), Interval(0, 0))


def test_sub_complete_overlap():
    interval1 = Interval(1, 5)
    interval2 = Interval(1, 5)
    assert interval1 - interval2 == (Interval(0, 0), Interval(0, 0))


def test_sub_inner_overlap():
    interval1 = Interval(1, 5)
    interval2 = Interval(2, 4)
    assert interval1 - interval2 == (Interval(1, 2), Interval(4, 5))


def test_sub_outer_overlap():
    interval1 = Interval(2, 4)
    interval2 = Interval(1, 5)
    assert interval1 - interval2 == (Interval(0, 0), Interval(0, 0))


def test_interval_intersection():
    interval1 = Interval(1, 5)
    interval2 = Interval(3, 7)
    assert (interval1 & interval2) == Interval(3, 5)


def test_interval_union():
    interval1 = Interval(1, 5)
    interval2 = Interval(3, 7)
    assert (interval1 | interval2) == (Interval(1, 7), Interval(0, 0))


def test_intersect_intervals():
    base = Interval(1, 5)
    others = [Interval(3, 7), Interval(8, 10)]
    assert intersect_intervals(base, others) == [Interval(3, 5)]


def test_intersect_intervals_no_intersection():
    base = Interval(1, 100)
    others = [Interval(200, 300), Interval(400, 500)]
    assert intersect_intervals(base, others) == []


def test_union_intervals():
    intervals = [Interval(1, 3), Interval(2, 5), Interval(6, 8)]
    assert union_intervals(intervals) == [Interval(1, 5), Interval(6, 8)]


def test_difference_intervals():
    intervals = [Interval(1, 5), Interval(2, 4), Interval(3, 6)]
    assert difference_intervals(intervals) == [Interval(1, 2)]


def test_PartStats_sub():
    a = PartStats(x=[Interval(1, 101), Interval(120, 200)], m=[], a=[], s=[])
    b = PartStats(x=[Interval(50, 151)], m=[], a=[], s=[])
    expected = PartStats(x=[Interval(1, 50), Interval(151, 200)])
    assert a - b == expected


def test_PartStats_and():
    a = PartStats(x=[Interval(1, 101), Interval(120, 200)], m=[], a=[], s=[])
    b = PartStats(x=[Interval(50, 151), Interval(190, 300)], m=[], a=[], s=[])
    expected = PartStats(x=[Interval(50, 101), Interval(120, 151), Interval(190, 200)])
    assert a & b == expected

    R = PartStats(
        x=[Interval(start=1, end=4001)],
        m=[Interval(start=1, end=4001)],
        a=[Interval(start=1, end=4001)],
        s=[Interval(start=1, end=4001)],
    )
    X = PartStats(x=[], m=[Interval(start=839, end=4001)], a=[], s=[])
    Y = R & X
    assert Y == PartStats(m=[Interval(start=839, end=4001)])


# Add more tests as necessary for edge cases and error handling
