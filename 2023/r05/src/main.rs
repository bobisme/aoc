use std::collections::{HashMap, HashSet};
use std::ops::Range;
use std::rc::Rc;
use std::str::FromStr;
use std::{cmp, i64};

use color_eyre::eyre::{bail, ensure, eyre};
use color_eyre::Result;
use itertools::Itertools;
use tracing::{debug, info};

mod intervals;
use intervals::Interval;

use crate::intervals::{difference_intervals, lowest, union_intervals};

const INPUT: &str = include_str!("../../2023-5.input");

const CONTROL_1: &str = r#"seeds: 79 14 55 13

seed-to-soil map:
50 98 2
52 50 48

soil-to-fertilizer map:
0 15 37
37 52 2
39 0 15

fertilizer-to-water map:
49 53 8
0 11 42
42 0 7
57 7 4

water-to-light map:
88 18 7
18 25 70

light-to-temperature map:
45 77 23
81 45 19
68 64 13

temperature-to-humidity map:
0 69 1
1 0 69

humidity-to-location map:
60 56 37
56 93 4"#;

#[derive(Clone, Copy, Default, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Stage {
    #[default]
    Unknown,
    Input,
    Seed,
    Soil,
    Fertilizer,
    Water,
    Light,
    Temperature,
    Humidity,
    Location,
}
const STAGE_ORDER: [Stage; 8] = [
    Stage::Seed,
    Stage::Soil,
    Stage::Fertilizer,
    Stage::Water,
    Stage::Light,
    Stage::Temperature,
    Stage::Humidity,
    Stage::Location,
];

impl Stage {
    pub const fn next(&self) -> Option<Self> {
        match self {
            Self::Input => Some(Self::Seed),
            Self::Seed => Some(Self::Soil),
            Self::Soil => Some(Self::Fertilizer),
            Self::Fertilizer => Some(Self::Water),
            Self::Water => Some(Self::Light),
            Self::Light => Some(Self::Temperature),
            Self::Temperature => Some(Self::Humidity),
            Self::Humidity => Some(Self::Location),
            Self::Location => None,
            Self::Unknown => None,
        }
    }
}

impl FromStr for Stage {
    type Err = color_eyre::Report;

    fn from_str(s: &str) -> std::prelude::v1::Result<Self, Self::Err> {
        match s {
            "seed" => Ok(Self::Seed),
            "soil" => Ok(Self::Soil),
            "fertilizer" => Ok(Self::Fertilizer),
            "water" => Ok(Self::Water),
            "light" => Ok(Self::Light),
            "temperature" => Ok(Self::Temperature),
            "humidity" => Ok(Self::Humidity),
            "location" => Ok(Self::Location),
            _ => Err(eyre!("FUCK: {}", s)),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct StageMap {
    pub stage: Stage,
    pub interval: Interval,
    pub offset: i64,
}

impl StageMap {
    pub fn new(stage: Stage, interval: Interval, offset: i64) -> Self {
        Self {
            stage,
            interval,
            offset,
        }
    }

    pub fn map(&self, x: i64) -> i64 {
        x + self.offset
    }

    pub fn map_range(&self, r: Range<i64>) -> Range<i64> {
        let start = r.start + self.offset;
        let end = r.end + self.offset;
        start..end
    }

    pub fn input_range(&self) -> Range<i64> {
        match &self.interval {
            Interval::Range(r) => r.clone(),
            _ => todo!(),
        }
    }

    pub fn output_range(&self) -> Range<i64> {
        self.map_range(self.input_range())
    }
}

#[derive(Debug, Clone)]
struct Ranges {
    src: Vec<Interval>,
    dst: Vec<Interval>,
}

impl Ranges {
    fn new(src: Vec<Interval>, dst: Vec<Interval>) -> Ranges {
        Ranges { src, dst }
    }

    fn to_stage_map(&self, stage: Stage, i: usize) -> StageMap {
        let to = self.dst[i].start();
        let fro = self.src[i].start();
        StageMap {
            stage,
            interval: self.src[i].clone(),
            offset: to - fro,
        }
    }
}

#[derive(Debug, PartialEq, Eq, Hash, Clone)]
struct MapLine {
    src: Interval,
    dst: Interval,
    offset: i64,
}

fn parse_ranges_line(line: &str) -> Result<MapLine> {
    let parts: Vec<_> = line
        .split_whitespace()
        .filter_map(|s| s.parse::<i64>().ok())
        .collect();
    ensure!(parts.len() == 3);
    let (dst, src, n) = (parts[0], parts[1], parts[2]);
    Ok(MapLine {
        dst: Interval::from(dst..dst + n),
        src: Interval::from(src..src + n),
        offset: dst - src,
    })
}

fn parse_range_name(line: &str) -> Result<(Stage, Stage)> {
    let parts = line.replace(" map:", "");
    let parts: Vec<&str> = parts.splitn(2, "-to-").collect();
    ensure!(parts.len() == 2);
    Ok((parts[0].parse()?, parts[1].parse()?))
}

fn parse_ranges_in_map(input: &[&str]) -> Result<(Stage, Ranges)> {
    let (from_, _) = parse_range_name(input[0])?;

    let mut srcs = Vec::new();
    let mut dsts = Vec::new();

    for line in &input[1..] {
        if line.is_empty() {
            break;
        }

        let map_line = parse_ranges_line(line)?;
        srcs.push(map_line.src);
        dsts.push(map_line.dst);
    }

    Ok((from_, Ranges::new(srcs, dsts)))
}

fn parse2(input: Vec<&str>) -> Result<(Vec<Interval>, HashMap<Stage, Ranges>)> {
    if input.is_empty() || !input[0].starts_with("seeds: ") {
        bail!("Invalid input format");
    }

    let seeds_str = input[0].replace("seeds: ", "");
    let seeds: Vec<_> = seeds_str
        .split_whitespace()
        .filter_map(|s| s.parse::<i64>().ok())
        .collect();

    let mut seeds_ranges = Vec::new();
    for chunk in seeds.chunks_exact(2) {
        seeds_ranges.push(Interval::Range(chunk[0]..chunk[0] + chunk[1]));
    }

    let mut ranges = HashMap::new();
    let mut i = 2;
    while i < input.len() {
        let (stage, r) = parse_ranges_in_map(&input[i..])?;
        ranges.insert(stage, r.clone());
        // ranges.insert(stage, r);
        i += r.src.len() + 2;
    }

    Ok((seeds_ranges, ranges))
}

/// Node of the interval tree.
#[derive(Default, Debug)]
pub struct IntNode {
    pub center: i64,
    pub range: Range<i64>,
    pub left: Option<Rc<IntNode>>,
    pub right: Option<Rc<IntNode>>,
    pub intervals_by_start: Vec<StageMap>,
    pub intervals_by_end: Vec<StageMap>,
}

impl IntNode {
    pub fn new(center: i64) -> Self {
        Self {
            center,
            ..Default::default()
        }
    }
    pub fn add_center_interval(&mut self, r: StageMap) {
        self.intervals_by_start.push(r.clone());
        self.intervals_by_end.push(r.clone());
        self.intervals_by_start
            .sort_by(|a, b| a.input_range().start.cmp(&b.input_range().start));
        self.intervals_by_end
            .sort_by(|a, b| a.input_range().end.cmp(&b.input_range().end));
        self.intervals_by_end.reverse();
    }

    pub fn center_range(&self) -> Range<i64> {
        if self.intervals_by_start.is_empty() {
            return Default::default();
        }
        let start = self.intervals_by_start[0].input_range().start;
        let end = self.intervals_by_end[0].input_range().end;
        start..end
    }

    pub fn around_point(&self, p: i64) -> Vec<StageMap> {
        if self.center_range().contains(&p) {
            let mut v: Vec<_> = self
                .intervals_by_start
                .iter()
                .take_while(|x| x.input_range().start <= p)
                .cloned()
                .collect();
            v.extend(
                self.intervals_by_end
                    .iter()
                    .take_while(|x| p < x.input_range().end)
                    .cloned(),
            );
            return v;
        }
        if p == self.center {
            return self.intervals_by_start.clone();
        }
        Default::default()
    }
}

pub fn center_of_range(r: &Range<i64>) -> i64 {
    (r.end - 1 - r.start) / 2 + r.start
}

fn subscriber() -> impl tracing::subscriber::Subscriber {
    use tracing_subscriber::{fmt::format::FmtSpan, prelude::*};
    tracing_subscriber::registry()
        .with(tracing_subscriber::EnvFilter::from_default_env())
        .with(
            tracing_subscriber::fmt::layer()
                .compact()
                .with_target(false)
                .without_time()
                .with_span_events(FmtSpan::CLOSE),
        )
}

pub fn full_range<'a>(ranges: impl Iterator<Item = &'a Interval>) -> Range<i64> {
    let mut start_min = i64::MAX;
    let mut end_max = 0;
    for r in ranges {
        start_min = cmp::min(r.start(), start_min);
        end_max = cmp::max(r.end(), end_max);
    }
    start_min..end_max
}

pub fn make_node(maps: Vec<StageMap>) -> (IntNode, Vec<StageMap>, Vec<StageMap>) {
    let full = full_range(maps.iter().map(|sm| &sm.interval));
    let center = center_of_range(&full);
    let mut node = IntNode::new(center);
    let mut lower = Vec::new();
    let mut higher = Vec::new();
    for stage_map in maps.iter() {
        if stage_map.input_range().contains(&node.center) {
            node.add_center_interval(stage_map.clone());
        } else if stage_map.input_range().end <= node.center {
            lower.push(stage_map.clone());
        } else {
            higher.push(stage_map.clone());
        }
    }
    dbg!((&node, &lower, &higher));
    (node, lower, higher)
}

pub fn make_tree(maps: Vec<StageMap>) -> IntNode {
    let (mut node, lower, higher) = make_node(maps);
    if !lower.is_empty() {
        node.left = Some(Rc::new(make_tree(lower)));
    }
    if !higher.is_empty() {
        node.right = Some(Rc::new(make_tree(higher)));
    }
    node
}
fn handwritten(seed_ranges: Vec<Interval>, seed: Vec<&StageMap>) {
    let a = &seed_ranges[1];
    let c = &seed[0].interval;
    let d = &seed[1].interval;
    let x = (a & c) + Interval::from(seed[0].offset);
    let y = (a & d) + Interval::from(seed[1].offset);
    let w = a ^ c;
    let v = (&w.0 ^ d, &w.1 ^ d);
    // info!(?a, ?c, ?d, ?x, ?y, ?w, ?v, "setup");
    let parts = [x, y, w.0, w.1, v.0 .0, v.0 .1, v.1 .0, v.1 .1];
    let mut unioned = HashSet::new();
    unioned.insert(Interval::Empty);
    for part in parts.iter() {
        let mut new_union = HashSet::new();
        for upart in unioned.iter() {
            let (a, b) = part | upart;
            new_union.insert(a);
            new_union.insert(b);
        }
        unioned = new_union;
    }
    let unioned: Vec<_> = unioned.iter().filter(|x| !x.is_empty()).collect();
}

fn do_stage(stage_maps: &[&StageMap], input_intervals: &[Interval]) -> Vec<Interval> {
    // this needs to change so each iter is differenced/removes mapped sections
    let mut parts_to_union = Vec::new();
    for input_interval in input_intervals.iter() {
        let mut parts_to_difference = vec![input_interval.clone()];
        for map in stage_maps.iter() {
            let a = (input_interval & &map.interval);
            let b = &a + &Interval::Scalar(map.offset);
            if !a.is_empty() {
                parts_to_difference.push(a);
            }
            if !b.is_empty() {
                parts_to_union.push(b.clone());
            }
        }
        // debug!(x = ?parts_to_difference, "to diff");
        parts_to_union.extend(difference_intervals(&parts_to_difference));
    }
    union_intervals(&parts_to_union)
        .into_iter()
        .sorted()
        .collect()
}

fn main() -> Result<()> {
    color_eyre::install().unwrap();
    tracing::subscriber::set_global_default(subscriber()).unwrap();
    let input: Vec<_> = INPUT.lines().collect();
    let (seed_ranges, range_maps) = parse2(input)?;
    // println!("{:?}", seed_ranges);
    // println!("{:?}", range_maps);
    let mut stage_maps: Vec<_> = range_maps
        .iter()
        .flat_map(|(stage, ranges)| {
            ranges
                .src
                .iter()
                .enumerate()
                .map(|(i, _)| ranges.to_stage_map(*stage, i))
        })
        .collect();
    stage_maps.extend(seed_ranges.iter().map(|x| StageMap {
        stage: Stage::Location,
        interval: x.clone(),
        offset: 0,
    }));
    // let root = make_tree(stage_maps);
    // dbg!(root.around_point(52));
    let grouped_stage_maps = stage_maps.iter().group_by(|x| x.stage);
    let stage_maps_by_stage: HashMap<_, _> = grouped_stage_maps
        .into_iter()
        .map(|(k, v)| (k, v.collect_vec()))
        .collect();
    let mut inputs = seed_ranges;
    for stage in STAGE_ORDER[..STAGE_ORDER.len() - 1].iter() {
        // info!(?stage, ?inputs, "in");
        let maps = &stage_maps_by_stage[&stage];
        // info!(?stage, ?maps, "maps");
        let next_inputs = do_stage(maps, &inputs);
        // if stage == &Stage::Humidity {
        //     info!(?next_inputs, "out");
        // }
        inputs = next_inputs;
    }
    print!("LOWEST = {}", lowest(&inputs).unwrap());
    Ok(())
}

#[cfg(test)]
mod test {
    use core::time;
    use std::{thread, time::Duration};

    use crate::intervals::lowest;

    use super::*;
    use assert2::assert;

    #[test]
    fn it_works() {
        assert!(
            parse_ranges_line("50 98 2").unwrap()
                == MapLine {
                    src: Interval::Range(98..100),
                    dst: Interval::Range(50..52),
                    offset: -48,
                }
        )
    }

    #[test]
    fn hold_hand() {
        // let input: Vec<_> = CONTROL_1.lines().collect();
        // let (seed_ranges, range_maps) = parse2(input).unwrap();
        // // println!("{:?}", seed_ranges);
        // // println!("{:?}", range_maps);
        // let mut stage_maps: Vec<_> = range_maps
        //     .iter()
        //     .flat_map(|(stage, ranges)| {
        //         ranges
        //             .src
        //             .iter()
        //             .enumerate()
        //             .map(|(i, _)| ranges.to_stage_map(*stage, i))
        //     })
        //     .collect();
        // stage_maps.extend(seed_ranges.iter().map(|x| StageMap {
        //     stage: Stage::Location,
        //     interval: x.clone(),
        //     offset: 0,
        // }));
        use Stage::*;
        let stage_maps = [
            StageMap {
                stage: Seed,
                interval: Interval::Range(98..100),
                offset: -48,
            },
            StageMap {
                stage: Seed,
                interval: Interval::Range(50..98),
                offset: 2,
            },
            StageMap {
                stage: Soil,
                interval: Interval::Range(15..52),
                offset: -15,
            },
            StageMap {
                stage: Soil,
                interval: Interval::Range(52..54),
                offset: -15,
            },
            StageMap {
                stage: Soil,
                interval: Interval::Range(0..15),
                offset: 39,
            },
            StageMap {
                stage: Light,
                interval: Interval::Range(77..100),
                offset: -32,
            },
            StageMap {
                stage: Light,
                interval: Interval::Range(45..64),
                offset: 36,
            },
            StageMap {
                stage: Light,
                interval: Interval::Range(64..77),
                offset: 4,
            },
            StageMap {
                stage: Water,
                interval: Interval::Range(18..25),
                offset: 70,
            },
            StageMap {
                stage: Water,
                interval: Interval::Range(25..95),
                offset: -7,
            },
            StageMap {
                stage: Temperature,
                interval: Interval::Range(69..70),
                offset: -69,
            },
            StageMap {
                stage: Temperature,
                interval: Interval::Range(0..69),
                offset: 1,
            },
            StageMap {
                stage: Humidity,
                interval: Interval::Range(56..93),
                offset: 4,
            },
            StageMap {
                stage: Humidity,
                interval: Interval::Range(93..97),
                offset: -37,
            },
            StageMap {
                stage: Fertilizer,
                interval: Interval::Range(53..61),
                offset: -4,
            },
            StageMap {
                stage: Fertilizer,
                interval: Interval::Range(11..53),
                offset: -11,
            },
            StageMap {
                stage: Fertilizer,
                interval: Interval::Range(0..7),
                offset: 42,
            },
            StageMap {
                stage: Fertilizer,
                interval: Interval::Range(7..11),
                offset: 50,
            },
            StageMap {
                stage: Location,
                interval: Interval::Range(79..93),
                offset: 0,
            },
            StageMap {
                stage: Location,
                interval: Interval::Range(55..68),
                offset: 0,
            },
        ];
        let grouped_stage_maps = stage_maps.iter().group_by(|x| x.stage);
        let stage_maps_by_stage: HashMap<_, _> = grouped_stage_maps
            .into_iter()
            .map(|(k, v)| (k, v.collect_vec()))
            .collect();
        let input_range = Interval::Range(79..93);
        let maps = [
            // &StageMap {
            //     stage: Seed,
            //     interval: Interval::Range(98..100),
            //     offset: -48,
            // },
            &StageMap {
                stage: Seed,
                interval: Interval::Range(50..98),
                offset: 2,
            },
        ];
        let done = do_stage(&maps[..], &[input_range]);
        thread::sleep(Duration::from_millis(200));
        assert!(done == vec![Interval::Range(81..95)]);
        dbg!(done);
    }
}
