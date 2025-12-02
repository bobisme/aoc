use std::{ops::RangeInclusive, sync::LazyLock};

static INPUT: LazyLock<String> =
    std::sync::LazyLock::new(|| std::fs::read_to_string("2025-2.input").unwrap());

fn parse(input: &str) -> Vec<RangeInclusive<u64>> {
    let ranges = input.strip_suffix("\n").unwrap().split(",");
    ranges
        .map(|r| {
            let (start, end) = r.split_once("-").unwrap();
            let start = start.parse::<u64>().unwrap();
            let end = end.parse::<u64>().unwrap();
            start..=end
        })
        .collect()
}

fn part_1(input: &str) -> u64 {
    let ranges = parse(input);
    let mut invalid_count = 0;
    let mut buf = String::new();
    for range in ranges {
        for i in range {
            let s = i.to_string();
            let n = s.len();
            if !n.is_multiple_of(2) {
                continue;
            }
            if s[0..(n / 2)] == s[(n / 2)..] {
                invalid_count += i;
            }
        }
    }
    invalid_count
}

fn part_2(input: &str) -> u64 {
    fn pat_fills(s: &str, pat: &str) -> bool {
        !(pat.len()..s.len())
            .step_by(pat.len())
            .any(|start| &s[start..(start + pat.len())] != pat)
    }

    fn has_repeats(s: &str) -> bool {
        (1..(s.len() / 2 + 1))
            .filter(|pat_len| s.len().is_multiple_of(*pat_len))
            .any(|pat_len| pat_fills(s, &s[..pat_len]))
    }

    let ranges = parse(input);
    let mut invalid_count = 0;
    for range in ranges {
        for i in range {
            let s = i.to_string();
            let n = s.len();
            if has_repeats(&s) {
                invalid_count += i;
            }
        }
    }
    invalid_count
}

fn main() {
    println!("{}", part_1(&INPUT));
    println!("{}", part_2(&INPUT));
}

#[cfg(test)]
mod test {
    use super::*;

    const CONTROL_1: &'static str = r#"11-22,95-115,998-1012,1188511880-1188511890,222220-222224,1698522-1698528,446443-446449,38593856-38593862,565653-565659,824824821-824824827,2121212118-2121212124
"#;

    #[test]
    fn test() {
        assert_eq!(part_1(&CONTROL_1), 1227775554);
        assert_eq!(part_2(&CONTROL_1), 4174379265);
    }
}
