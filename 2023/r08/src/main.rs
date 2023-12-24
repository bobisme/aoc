use std::{
    cmp,
    collections::{HashMap, HashSet},
    fs::read_to_string,
    io::Write,
};

use regex::Regex;
const INPUT_FILE: &str = include_str!("../2023-8.input");

const CONTROL_1: &str = r#"RL

AAA = (BBB, CCC)
BBB = (DDD, EEE)
CCC = (ZZZ, GGG)
DDD = (DDD, DDD)
EEE = (EEE, EEE)
GGG = (GGG, GGG)
ZZZ = (ZZZ, ZZZ)"#;

const CONTROL_2: &str = r#"LLR

AAA = (BBB, BBB)
BBB = (AAA, ZZZ)
ZZZ = (ZZZ, ZZZ)"#;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Dir {
    Left,
    Right,
}

impl From<Dir> for &str {
    fn from(value: Dir) -> Self {
        match value {
            Dir::Left => "L",
            Dir::Right => "R",
        }
    }
}

impl From<Dir> for char {
    fn from(value: Dir) -> Self {
        match value {
            Dir::Left => 'L',
            Dir::Right => 'R',
        }
    }
}

impl From<char> for Dir {
    fn from(c: char) -> Self {
        match c {
            'L' => Self::Left,
            'R' => Self::Right,
            _ => unreachable!(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Node {
    label: String,
    left: String,
    right: String,
    z: u8,
}

impl Node {
    pub fn is_a(&self) -> bool {
        self.label.chars().nth(2) == Some('A')
    }
}

impl std::fmt::Display for Node {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{label} -L-> {left:?} -R-> {right:?}",
            label = self.label,
            left = self.left,
            right = self.right
        )
    }
}

impl std::str::FromStr for Node {
    type Err = Box<dyn std::error::Error>;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let re = Regex::new(r"([A-Z]{3})\s+=\s+\(([A-Z]{3}), ([A-Z]{3})\)").unwrap();
        let caps = re.captures(s).unwrap();

        let label = caps[1].to_string();
        let left = caps[2].to_string();
        let right = caps[3].to_string();
        let z = if label.chars().nth(2).unwrap() == 'Z' {
            1
        } else {
            0
        };

        Ok(Self {
            label,
            left,
            right,
            z,
        })
    }
}

fn parse(input: &[String]) -> (impl Iterator<Item = Dir> + Clone + '_, Vec<Node>) {
    let directions = input[0].chars().map(Dir::from);
    let nodes = input
        .iter()
        .skip(2)
        .map(|l| l.parse::<Node>())
        .collect::<Result<_, _>>()
        .unwrap();
    // let nodes_map: HashMap<String, Node> = nodes.map(|node| (node.label.clone(), node)).collect();

    (directions, nodes)
}
pub fn greatest_common_divisor(mut n: u64, mut m: u64) -> u64 {
    assert!(n != 0 && m != 0);
    while m != 0 {
        if m < n {
            std::mem::swap(&mut m, &mut n);
        }
        m %= n;
    }
    n
}
fn least_common_multiple(x: u64, y: u64) -> u64 {
    (x * y) / greatest_common_divisor(x, y)
}

fn part_2() {
    // let input = read_lines("../2023-8.input");
    let input: Vec<_> = INPUT_FILE.lines().map(String::from).collect();
    let (directions, nodes) = parse(&input);
    let mut directions = directions.enumerate().cycle();
    let map: HashMap<&str, &Node> = HashMap::from_iter(nodes.iter().map(|n| (n.label.as_str(), n)));
    let roots: Vec<&Node> = nodes.iter().filter(|n| n.is_a()).collect();
    // let root = *map.get("AAA").unwrap();
    let mut steps = [0; 6];
    for (root_i, root) in roots.iter().enumerate() {
        let mut node = root;
        // let mut step_count = 0;
        let mut visited = HashSet::new();
        // let bar = indicatif::ProgressBar::new(10_000_000_000);
        while node.z != 1 {
            println!("visiting {node}");
            let (di, direction) = directions.next().unwrap();
            println!("going {direction:?}");
            // if visited.contains(&(&node.label, di)) {
            //     panic!("ERROR: CYCLE!");
            // }
            visited.insert((&node.label, di));
            // bar.inc(1);
            steps[root_i] += 1;
            // println!("{step_count} {0} {direction:?}", node.label);
            let node_label = match direction {
                Dir::Left => &node.left,
                Dir::Right => &node.right,
            };

            node = map.get(node_label.as_str()).unwrap();
        }
    }
    println!("steps count = {steps:?}");
    println!("calculating LCM");
    let l = steps
        .iter()
        .cloned()
        .take(6)
        .reduce(least_common_multiple)
        .unwrap();
    println!("lcm = {l}");
    // bar.finish();
}

fn part_1() {
    // let input = read_lines("../2023-8.input");
    let input: Vec<_> = INPUT_FILE.lines().map(String::from).collect();
    let (directions, nodes) = parse(&input);
    let mut directions = directions.enumerate().cycle();
    let map: HashMap<&str, &Node> = HashMap::from_iter(nodes.iter().map(|n| (n.label.as_str(), n)));
    let root = *map.get("AAA").unwrap();
    let mut node = root;
    let mut step_count = 0;
    let mut visited = HashSet::new();
    let bar = indicatif::ProgressBar::new(10_000_000_000);
    while node.label != "ZZZ" {
        println!("visiting {node}");
        let (di, direction) = directions.next().unwrap();
        println!("going {direction:?}");
        if visited.contains(&(&node.label, di)) {
            dbg!((&node.label, direction, step_count));
            panic!("ERROR: CYCLE!");
        }
        visited.insert((&node.label, di));
        // bar.inc(1);
        step_count += 1;
        // println!("{step_count} {0} {direction:?}", node.label);
        let node_label = match direction {
            Dir::Left => &node.left,
            Dir::Right => &node.right,
        };

        node = map.get(node_label.as_str()).unwrap();
    }
    // bar.finish();

    println!("step count = {}", step_count);
}

fn main() {
    part_2()
}

#[cfg(test)]
mod test {
    use super::*;
    use assert2::assert;

    #[test]
    fn it_works() {
        let input: Vec<_> = INPUT_FILE.lines().map(String::from).collect();
        let (mut directions, nodes) = parse(&input);
        let d: Vec<&str> = directions.map(Dir::into).collect();

        let expected = "LLRLRRLLRLRRLLRLRRLRRRLRLRLRRRLLRLRRRLRLRRRLRLRLLLRRLRLRLLRLRRLRRRLRRRLLRRLRLRRRLRRLRRRLRLLRRLRRRLRRRLRRLRLRRLLLRLRLLRRRLRRLLRLRLRRLLRLRRLLRLRRLRRLLRRRLRLRLRRRLLRRRLRRLRRRLRRRLRLRRRLRRLLLRRRLRLLLRRRLRLLRLLRRRLLRRLRRRLRRRLRLLRLRLRRRLLRRLRRRLRRLRLLRRRLRRLRRRLRRRLRRRLRLRRRLRRRLRLRRRR";
        assert!((&d).join("") == expected);
    }
}
