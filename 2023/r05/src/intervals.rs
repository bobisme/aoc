use core::cmp;
use core::ops::Range;
use std::collections::HashSet;

use itertools::Itertools;
use tracing::info;

pub const fn kmin(a: i64, b: i64) -> i64 {
    if a < b {
        return a;
    }
    b
}
pub const fn kmax(a: i64, b: i64) -> i64 {
    if a > b {
        return a;
    }
    b
}

pub const fn kclone_range(r: &Range<i64>) -> Range<i64> {
    r.start..r.end
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Interval {
    Scalar(i64),
    Range(Range<i64>),
    Empty,
}

impl Ord for Interval {
    fn cmp(&self, other: &Self) -> cmp::Ordering {
        match (self, other) {
            (Interval::Scalar(a), Interval::Scalar(b)) => a.cmp(b),
            (Interval::Scalar(a), Interval::Range(b)) => a.cmp(&b.start),
            (Interval::Range(a), Interval::Scalar(b)) => a.start.cmp(b),
            (Interval::Range(a), Interval::Range(b)) => a.start.cmp(&b.start),
            (Interval::Empty, Interval::Empty) => cmp::Ordering::Equal,
            (Interval::Empty, _) => cmp::Ordering::Less,
            (_, Interval::Empty) => cmp::Ordering::Greater,
        }
    }
}

impl PartialOrd for Interval {
    fn partial_cmp(&self, other: &Self) -> Option<cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Interval {
    pub const fn start(&self) -> i64 {
        match self {
            Interval::Scalar(x) => *x,
            Interval::Range(r) => r.start,
            Interval::Empty => 0,
        }
    }

    pub const fn end(&self) -> i64 {
        match self {
            Interval::Scalar(x) => *x + 1,
            Interval::Range(r) => r.end,
            Interval::Empty => 0,
        }
    }

    pub const fn len(&self) -> usize {
        match self {
            Interval::Scalar(_) => 1,
            Interval::Range(r) => {
                if r.end <= r.start {
                    return 0;
                }
                (r.end - r.start).unsigned_abs() as usize
            }
            Interval::Empty => 0,
        }
    }

    #[must_use]
    pub const fn is_empty(&self) -> bool {
        match self {
            Interval::Scalar(_) => false,
            Interval::Range(r) => (r.end - r.start) <= 0,
            Interval::Empty => true,
        }
    }

    pub const fn contains(&self, x: i64) -> bool {
        match self {
            Interval::Scalar(s) => x == *s,
            Interval::Range(r) => r.start <= x && x < r.end,
            Interval::Empty => false,
        }
    }

    pub const fn intersects(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Scalar(a), Self::Scalar(b)) => *a == *b,
            (Self::Scalar(s), Self::Range(r)) | (Self::Range(r), Self::Scalar(s)) => {
                r.start <= *s || *s < r.end
            }
            (Self::Range(a), Self::Range(b)) => a.start <= b.end && b.start <= a.end,
            _ => false,
        }
    }

    pub const fn kclone(&self) -> Self {
        match self {
            Interval::Scalar(x) => Interval::Scalar(*x),
            Interval::Range(r) => Interval::Range(kclone_range(r)),
            Interval::Empty => Interval::Empty,
        }
    }

    const fn _bitand(&self, rhs: &Self) -> Self {
        match (self, rhs) {
            (Interval::Empty, _) | (_, Interval::Empty) => Interval::Empty,
            (Interval::Scalar(a), Interval::Scalar(b)) => {
                if *a == *b {
                    return Interval::Scalar(*a);
                }
                Interval::Empty
            }
            (Interval::Scalar(s), Interval::Range(r))
            | (Interval::Range(r), Interval::Scalar(s)) => {
                if r.start <= *s && *s < r.end {
                    Interval::Scalar(*s)
                } else {
                    Interval::Empty
                }
            }
            (Interval::Range(a), Interval::Range(b)) => {
                let start = kmax(a.start, b.start);
                let end = kmin(a.end, b.end);
                if start < end {
                    Interval::Range(start..end)
                } else {
                    Interval::Empty
                }
            }
        }
    }

    const fn _bitxor(&self, rhs: &Self) -> (Interval, Interval) {
        match (self, rhs) {
            (Interval::Scalar(a), Interval::Scalar(b)) => {
                if *a == *b {
                    return (Interval::Empty, Interval::Empty);
                }
                (self.kclone(), Interval::Empty)
            }
            (Interval::Scalar(a), Interval::Range(_)) => {
                if rhs.contains(*a) {
                    return (Interval::Empty, Interval::Empty);
                }
                (self.kclone(), Interval::Empty)
            }
            (Interval::Range(a), Interval::Scalar(b)) => {
                if !self.contains(*b) {
                    return (self.kclone(), Interval::Empty);
                }
                let upper = Interval::Range((*b + 1)..a.end);
                let lower = Interval::Range(a.start..*b);
                if lower.is_empty() {
                    return (upper, Interval::Empty);
                }
                (upper, lower)
            }
            (Interval::Range(a), Interval::Range(b)) => {
                // Case where ranges do not overlap
                if a.end <= b.start || b.end <= a.start {
                    return (Interval::Range(a.start..a.end), Interval::Empty);
                }

                // Overlapping cases
                match (a.start < b.start, a.end > b.end) {
                    (true, true) => {
                        // 'a' completely covers 'b', resulting in two disjoint ranges
                        (
                            Interval::Range(a.start..b.start),
                            Interval::Range(b.end..a.end),
                        )
                    }
                    (true, false) => {
                        // Overlap at the end of 'a'
                        (Interval::Range(a.start..b.start), Interval::Empty)
                    }
                    (false, true) => {
                        // Overlap at the beginning of 'a'
                        (Interval::Range(b.end..a.end), Interval::Empty)
                    }
                    (false, false) => {
                        // 'b' completely covers 'a', resulting in emptiness
                        (Interval::Empty, Interval::Empty)
                    }
                }
            }
            (x, Interval::Empty) => (x.kclone(), Interval::Empty),
            (Interval::Empty, _) => (Interval::Empty, Interval::Empty),
        }
    }
}

impl From<Range<i64>> for Interval {
    fn from(value: Range<i64>) -> Self {
        Self::Range(value)
    }
}

impl From<i64> for Interval {
    fn from(value: i64) -> Self {
        Self::Scalar(value)
    }
}

/// Intersection of intervals.
impl std::ops::BitAnd for &Interval {
    type Output = Interval;

    fn bitand(self, rhs: Self) -> Self::Output {
        self._bitand(rhs)
    }
}

impl std::ops::BitAnd for Interval {
    type Output = Interval;

    fn bitand(self, rhs: Self) -> Self::Output {
        std::ops::BitAnd::bitand(&self, &rhs)
    }
}

const _: () = {
    let and = Interval::Range(40..61)._bitand(&Interval::Range(20..51));
    match and {
        Interval::Range(r) => assert!(r.start == 40 && r.end == 51),
        _ => panic!(),
    }
};

/// Difference
impl std::ops::BitXor for &Interval {
    type Output = (Interval, Interval);

    fn bitxor(self, rhs: Self) -> Self::Output {
        self._bitxor(rhs)
    }
}

impl std::ops::BitXor for Interval {
    type Output = (Interval, Interval);
    fn bitxor(self, rhs: Self) -> Self::Output {
        std::ops::BitXor::bitxor(&self, &rhs)
    }
}

/// Union
impl std::ops::BitOr for &Interval {
    type Output = (Interval, Interval);

    fn bitor(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            (Interval::Scalar(a), Interval::Scalar(b)) => {
                if a == b {
                    return ((self.clone()), Interval::Empty);
                }
                ((self.clone()), rhs.clone())
            }
            (Interval::Scalar(a), Interval::Range(b))
            | (Interval::Range(b), Interval::Scalar(a)) => {
                if b.contains(a) {
                    return ((rhs.clone()), Interval::Empty);
                }
                ((self.clone()), (rhs.clone()))
            }
            (Interval::Range(a), Interval::Range(b)) => {
                let x = self & rhs;
                if x.is_empty() {
                    return ((self.clone()), (rhs.clone()));
                };
                let i = Interval::from(cmp::min(a.start, b.start)..cmp::max(a.end, b.end));
                ((i), Interval::Empty)
            }
            (x, Interval::Empty) | (Interval::Empty, x) => (x.clone(), Interval::Empty),
        }
    }
}

impl std::ops::BitOr for Interval {
    type Output = (Interval, Interval);

    fn bitor(self, rhs: Self) -> Self::Output {
        &self | &rhs
    }
}

impl std::ops::Add for Interval {
    type Output = Interval;

    fn add(self, other: Interval) -> Self::Output {
        &self + &other
    }
}

impl std::ops::Add for &Interval {
    type Output = Interval;

    fn add(self, other: &Interval) -> Self::Output {
        match (self, other) {
            (Interval::Scalar(a), Interval::Scalar(b)) => Interval::Scalar(a + b),
            (Interval::Scalar(_), Interval::Range(_)) => other + self,
            (Interval::Range(a), Interval::Scalar(b)) => {
                Interval::Range((a.start + b)..(a.end + b))
            }
            (Interval::Range(a), Interval::Range(b)) => {
                Interval::from((a.start + b.start)..(a.end + b.end))
            }
            (x, Interval::Empty) => x.clone(),
            (Interval::Empty, _) => Interval::Empty,
        }
    }
}

impl std::ops::Sub for Interval {
    type Output = Self;

    fn sub(self, other: Interval) -> Self::Output {
        match (self, other) {
            (Self::Scalar(a), Self::Scalar(b)) => Self::Scalar(a - b),
            (Self::Scalar(_), Self::Range(_)) => todo!(),
            (Self::Range(a), Self::Scalar(b)) => Self::from((a.start - b)..(a.end - b)),
            (Self::Range(a), Self::Range(b)) => Self::from((a.start - b.start)..(a.end - b.end)),
            (x, Interval::Empty) => x.clone(),
            (Interval::Empty, _) => Interval::Empty,
        }
    }
}

pub fn union_intervals(intervals: &[Interval]) -> Vec<Interval> {
    let mut result = Vec::new();

    for interval in intervals {
        let mut added = false;

        // Merge with existing intervals in the result if possible
        for existing in result.iter_mut() {
            let (union, is_empty) = &*existing | interval;
            if !matches!(is_empty, Interval::Empty) {
                continue;
            }

            *existing = union;
            added = true;
            break;
        }

        // If the interval did not merge with any existing intervals, add it to the result
        if !added {
            result.push(interval.clone());
        }
    }

    result
}

pub fn difference_intervals(intervals: &[Interval]) -> Vec<Interval> {
    if intervals.is_empty() {
        return Vec::new();
    }

    let mut result = vec![intervals[0].clone()];

    for interval in &intervals[1..] {
        let mut new_result = Vec::new();
        for existing in result.iter() {
            let (diff1, diff2) = existing ^ interval;
            if !matches!(diff1, Interval::Empty) {
                new_result.push(diff1);
            }
            if !matches!(diff2, Interval::Empty) {
                new_result.push(diff2);
            }
        }
        result = new_result;
    }

    result
        .into_iter()
        .filter(|x| !(matches!(x, Interval::Empty) || x.is_empty()))
        .collect()
}

pub fn lowest(intervals: &[Interval]) -> Option<i64> {
    intervals.iter().min().map(|x| x.start())
}

#[cfg(test)]
mod test {
    use super::*;
    use assert2::assert;

    #[test]
    fn test_union_intervals_with_empty_intervals() {
        let intervals = vec![Interval::Empty, Interval::Empty];
        let result = union_intervals(&intervals);
        assert!(result == vec![Interval::Empty]);
    }

    #[test]
    fn test_union_intervals_with_scalar_intervals() {
        let intervals = vec![Interval::Scalar(1), Interval::Scalar(2)];
        let result = union_intervals(&intervals);
        assert!(result == vec![Interval::Scalar(1), Interval::Scalar(2)]);
    }

    #[test]
    fn test_union_intervals_with_overlapping_ranges() {
        let intervals = vec![Interval::Range(1..3), Interval::Range(2..4)];
        let result = union_intervals(&intervals);
        assert!(result == vec![Interval::Range(1..4)]);
    }

    #[test]
    fn test_union_intervals_with_disjoint_ranges() {
        let intervals = vec![Interval::Range(1..3), Interval::Range(4..6)];
        let result = union_intervals(&intervals);
        assert!(result == vec![Interval::Range(1..3), Interval::Range(4..6)]);
    }

    #[test]
    fn test_union_intervals_with_mixed_types() {
        let intervals = vec![Interval::Scalar(1), Interval::Range(3..5), Interval::Empty];
        let result = union_intervals(&intervals);
        assert!(result == vec![Interval::Scalar(1), Interval::Range(3..5)]);
    }

    #[test]
    fn test_difference_scalar_with_scalar() {
        let a = Interval::Scalar(1);
        let b = Interval::Scalar(1);
        let (diff1, diff2) = a._bitxor(&b);
        assert!(diff1 == Interval::Empty);
        assert!(diff2 == Interval::Empty);
    }

    #[test]
    fn test_difference_scalar_with_range() {
        let a = Interval::Scalar(1);
        let b = Interval::Range(1..3);
        let (diff1, diff2) = a._bitxor(&b);
        assert!(diff1 == Interval::Empty);
        assert!(diff2 == Interval::Empty);
    }

    #[test]
    fn test_difference_range_with_scalar() {
        let a = Interval::Range(1..4);
        let b = Interval::Scalar(2);
        let (diff1, diff2) = a._bitxor(&b);
        assert!(diff1 == Interval::Range(3..4));
        assert!(diff2 == Interval::Range(1..2));
    }

    #[test]
    fn test_difference_range_with_range() {
        let a = Interval::Range(1..5);
        let b = Interval::Range(2..4);
        let (diff1, diff2) = a ^ b;
        assert!(diff1 == Interval::Range(1..2));
        assert!(diff2 == Interval::Range(4..5));
    }

    #[test]
    fn test_difference_with_empty() {
        let a = Interval::Range(1..3);
        let b = Interval::Empty;
        let (diff1, diff2) = a._bitxor(&b);
        assert!(diff1 == Interval::Range(1..3));
        assert!(diff2 == Interval::Empty);
    }

    #[test]
    fn intersect_empty_with_any() {
        assert!(Interval::Empty & Interval::Scalar(5) == Interval::Empty);
        assert!(Interval::Range(1..3) & Interval::Empty == Interval::Empty);
    }

    #[test]
    fn intersect_scalars() {
        assert_eq!(
            Interval::Scalar(5) & Interval::Scalar(5),
            Interval::Scalar(5)
        );
        assert!(Interval::Scalar(5) & Interval::Scalar(6) == Interval::Empty);
    }

    #[test]
    fn intersect_scalar_with_range() {
        assert_eq!(
            Interval::Scalar(2) & Interval::Range(1..3),
            Interval::Scalar(2)
        );
        assert!(Interval::Scalar(4) & Interval::Range(1..3) == Interval::Empty);
    }

    #[test]
    fn intersect_ranges() {
        assert_eq!(
            Interval::Range(1..5) & Interval::Range(3..7),
            Interval::Range(3..5)
        );
        assert_eq!(
            Interval::Range(1..3) & Interval::Range(4..6),
            Interval::Empty
        );
    }

    #[test]
    fn add_scalar_to_scalar() {
        let a = Interval::Scalar(3);
        let b = Interval::Scalar(4);
        assert!(&a + &b == Interval::Scalar(7));
    }

    #[test]
    fn add_scalar_to_range() {
        let scalar = Interval::Scalar(2);
        let range = Interval::Range(3..6);
        assert!(&scalar + &range == Interval::Range(5..8));
    }

    #[test]
    fn add_range_to_scalar() {
        let range = Interval::Range(1..4);
        let scalar = Interval::Scalar(3);
        assert!(&range + &scalar == Interval::Range(4..7));
    }

    #[test]
    fn add_range_to_range() {
        let range1 = Interval::Range(1..3);
        let range2 = Interval::Range(4..6);
        assert!(&range1 + &range2 == Interval::Range(5..9));
    }

    #[test]
    fn add_empty_to_any() {
        let empty = Interval::Empty;
        let scalar = Interval::Scalar(3);
        let range = Interval::Range(1..4);
        assert!(&empty + &scalar == Interval::Empty);
        assert!(&range + &empty == range);
    }

    #[test]
    fn add_any_to_empty() {
        let empty = Interval::Empty;
        let scalar = Interval::Scalar(3);
        let range = Interval::Range(1..4);
        assert!(&scalar + &empty == scalar);
        assert!(&empty + &range == Interval::Empty);
    }

    #[test]
    fn test_difference_intervals_basic() {
        let intervals = vec![Interval::Range(1..5), Interval::Range(3..7)];
        let result = difference_intervals(&intervals);
        assert!(result == vec![Interval::Range(1..3)]);
    }

    #[test]
    fn test_difference_intervals_long() {
        let intervals = vec![
            Interval::Range(1..5),
            Interval::Range(3..7),
            Interval::Empty,
            Interval::Scalar(2),
        ];
        let result = difference_intervals(&intervals);
        assert!(result == vec![Interval::Range(1..2)]);
    }
}
