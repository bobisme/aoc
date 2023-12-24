use std::{
    cmp,
    collections::{HashMap, HashSet, VecDeque},
    time::Instant,
};

use rayon::prelude::*;

const INPUT_FILE: &str = include_str!("../../2023-23.input");

const CONTROL_1: &str = r#"#.#####################
#.......#########...###
#######.#########.#.###
###.....#.>.>.###.#.###
###v#####.#v#.###.#.###
###.>...#.#.#.....#...#
###v###.#.#.#########.#
###...#.#.#.......#...#
#####.#.#.#######.#.###
#.....#.#.#.......#...#
#.#####.#.#.#########v#
#.#...#...#...###...>.#
#.#.#v#######v###.###v#
#...#.>.#...>.>.#.###.#
#####v#.#.###v#.#.###.#
#.....#...#...#.#.#...#
#.#########.###.#.#.###
#...###...#...#...#.###
###.###.#.###v#####v###
#...#...#.#.>.>.#.>.###
#.###.###.#.###.#.#v###
#.....###...###...#...#
#####################.#
"#;

#[derive(Debug, PartialEq, Eq, Hash, Clone, Copy)]
pub struct Pos {
    i: i32,
    j: i32,
}

impl Pos {
    pub const fn new(i: i32, j: i32) -> Self {
        Self { i, j }
    }
    pub const fn new_usize(i: usize, j: usize) -> Self {
        Self {
            i: i as i32,
            j: j as i32,
        }
    }
}

pub struct Field {
    input: Vec<String>,
    rows: usize,
    cols: usize,
}

impl Field {
    fn new(input: Vec<String>) -> Field {
        let rows = input.len();
        let cols = input[0].len();
        Field { input, rows, cols }
    }

    fn get(&self, pos: Pos) -> char {
        self.input[pos.j as usize]
            .chars()
            .nth(pos.i as usize)
            .unwrap()
    }

    const fn in_bounds(&self, pos: Pos) -> bool {
        pos.i >= 0 && pos.i < (self.cols as i32) && pos.j >= 0 && pos.j < (self.rows as i32)
    }

    pub fn print(&self, marks: impl Iterator<Item = Pos>, char: char) {
        println!("\x1b[31;34m{}\x1b[0m", "-".repeat(self.cols));
        let mut field: Vec<Vec<String>> = self
            .input
            .iter()
            .map(|row| {
                row.chars()
                    .map(|c| format!("\x1b[31;30m{}\x1b[0m", c))
                    .collect()
            })
            .collect();
        for mark in marks {
            if !self.in_bounds(mark) {
                continue;
            }
            field[mark.j as usize][mark.i as usize] = format!("O");
        }

        for row in field {
            println!("{}", row.join(""));
        }
    }
}

fn pleft(pos: Pos) -> Pos {
    Pos {
        i: pos.i - 1,
        j: pos.j,
    }
}

fn pright(pos: Pos) -> Pos {
    Pos {
        i: pos.i + 1,
        j: pos.j,
    }
}

fn pup(pos: Pos) -> Pos {
    Pos {
        i: pos.i,
        j: pos.j - 1,
    }
}

fn pdown(pos: Pos) -> Pos {
    Pos {
        i: pos.i,
        j: pos.j + 1,
    }
}

fn neighbors2(field: &Field, pos: Pos) -> Vec<Pos> {
    let up = pup(pos);
    let down = pdown(pos);
    let left = pleft(pos);
    let right = pright(pos);
    let mut neighbors = Vec::new();

    if field.in_bounds(up) {
        neighbors.push(up);
    }
    if field.in_bounds(left) {
        neighbors.push(left);
    }
    if field.in_bounds(right) {
        neighbors.push(right);
    }
    if field.in_bounds(down) {
        neighbors.push(down);
    }

    neighbors
        .into_iter()
        .filter(|&p| field.get(p) != '#')
        .collect()
}

fn find_longest(field: &Field) -> i32 {
    let start = Pos::new(1, 0);
    let mut distances = vec![vec![None; field.cols]; field.rows];
    let mut longest = 0;
    fn inner(
        field: &Field,
        pos: Pos,
        explored: &mut HashSet<Pos>,
        path: &HashSet<Pos>,
        distances: &mut Vec<Vec<Option<i32>>>,
        longest: &mut i32,
    ) -> i32 {
        if explored.contains(&pos) || path.contains(&pos) {
            return 0;
        }
        // if let Some(&cached_dist) = distances[pos.j as usize][pos.i as usize].as_ref() {
        //     return cached_dist;
        // }
        // let end = Pos::new_usize(field.cols - 2, field.rows - 1);
        // if d > 0 {
        //     return Some(d);
        // }
        let mut new_path = path.clone();
        new_path.insert(pos);
        explored.insert(pos);
        let ns = neighbors2(field, pos);

        // terminal
        if pos == Pos::new_usize(field.cols - 2, field.rows - 1) {
            // field.print(new_path.iter().cloned(), 'O');
            let len = (new_path.len() - 1) as i32;
            println!("^^ path len = {}, longest = {longest}", len);
            if len > *longest {
                println!("^^ new longest!");
                *longest = len;
            }
            // return Some(0);
        }

        let max_dist = ns
            .into_iter()
            .filter(|n| !path.contains(n))
            .map(|n| inner(field, n, explored, &new_path, distances, longest))
            .max()
            .unwrap_or(-1);
        explored.remove(&pos);
        let out = max_dist + 1;
        distances[pos.j as usize][pos.i as usize] = Some(out);
        out
    }
    let mut explored = HashSet::new();
    let out = inner(
        field,
        start,
        &mut explored,
        &HashSet::new(),
        &mut distances,
        &mut longest,
    );
    for row in distances.iter() {
        let r = row
            .iter()
            .map(|x| format!("{:>4}", x.unwrap_or(0)))
            .collect::<Vec<_>>()
            .join(" ");
        println!("{r:?}");
    }
    longest
}

// fn find_longest(field: &Field) -> i32 {
//     let start = Pos::new(1, 0);
//     let mut cache = vec![vec![None; field.cols]; field.rows];
//     let mut stack: VecDeque<(Pos, i32)> = VecDeque::new();
//     stack.push_back((start, 0));
//
//     while let Some((pos, dist)) = stack.pop_front() {
//         if pos.i < 0
//             || pos.j < 0
//             || pos.i >= field.cols as i32
//             || pos.j >= field.rows as i32
//             || field.input[pos.j as usize].chars().nth(pos.i as usize) == Some('#')
//         {
//             continue;
//         }
//
//         if let Some(cached) = cache[pos.j as usize][pos.i as usize] {
//             if cached >= dist {
//                 continue;
//             }
//         }
//
//         cache[pos.j as usize][pos.i as usize] = Some(dist);
//
//         let neighbors = neighbors2(field, pos);
//         for neighbor in neighbors {
//             stack.push_back((neighbor, dist + 1));
//         }
//     }
//
//     cache
//         .into_iter()
//         .flatten()
//         .filter_map(|x| x)
//         .max()
//         .unwrap_or(0)
// }

fn part_1(input: Vec<String>) {
    let field = Field::new(input);
    let dist = find_longest(&field);
    // field.print(Some(path), 'O');
    println!("Distance: {}", dist);
}

fn main() {
    let start = Instant::now();
    part_1(INPUT_FILE.lines().map(|x| x.to_string()).collect());
    let elapsed = start.elapsed();
    println!("elapsed: {elapsed:?}");
}
