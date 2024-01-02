use std::{
    cmp,
    collections::{BinaryHeap, HashMap, HashSet},
    io::Write,
};

use glam::IVec2;
use itertools::Itertools;

const INPUT_FILE: &str = include_str!("../../2023-17.input");
const CONTROL_1: &str = r#"2413432311323
3215453535623
3255245654254
3446585845452
4546657867536
1438598798454
4457876987766
3637877979653
4654967986887
4564679986453
1224686865563
2546548887735
4322674655533
"#;
const CONTROL_2: &str = r#"111111111111
999999999991
999999999991
999999999991
999999999991
"#;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Dir {
    Up,
    Down,
    Left,
    Right,
    Invalid,
}

impl std::fmt::Display for Dir {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        use std::fmt::Write;
        match self {
            Self::Up => f.write_char('^'),
            Self::Down => f.write_char('v'),
            Self::Left => f.write_char('<'),
            Self::Right => f.write_char('>'),
            Self::Invalid => f.write_char('X'),
        }
    }
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BinDir {
    Vertical,
    Horizontal,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct QItem {
    pos: IVec2,
    dir: BinDir,
    total_heat: u32,
}

impl QItem {
    pub fn key(&self) -> QItemKey {
        QItemKey(self.pos, self.dir)
    }
}

impl PartialOrd for QItem {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for QItem {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // reverse because it's a max-heap and we need min-heap
        other.total_heat.cmp(&self.total_heat)
    }
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct QItemKey(pub IVec2, pub BinDir);

#[derive(Debug, Default, Clone)]
pub struct Q {
    q_set: HashMap<QItemKey, i32>,
    q: BinaryHeap<QItem>,
}

impl Q {
    pub fn contains(&self, item: &QItem) -> bool {
        self.q_set.get(&item.key()).is_some_and(|&x| x > 0)
    }

    pub fn pop(&mut self) -> Option<QItem> {
        let item = self.q.pop()?;
        if let Some(x) = self.q_set.get_mut(&item.key()) {
            *x -= 1;
        }
        Some(item)
    }

    pub fn push(&mut self, item: QItem) {
        self.q_set
            .entry(item.key())
            .and_modify(|x| *x += 1)
            .or_insert(1);
        self.q.push(item)
    }
}

pub struct Field {
    input: Vec<Vec<u32>>,
    pub rows: usize,
    pub cols: usize,
}
type AdjList = HashMap<(IVec2, BinDir), HashSet<(IVec2, BinDir, u32)>>;
impl Field {
    pub fn new(input: Vec<Vec<u32>>) -> Self {
        let rows = input.len();
        let cols = input[0].len();
        Self { input, rows, cols }
    }

    pub fn in_bounds(&self, pos: IVec2) -> bool {
        0 <= pos.x && pos.x < self.cols as i32 && 0 <= pos.y && pos.y < self.rows as i32
    }

    pub fn get(&self, pos: IVec2) -> u32 {
        assert!(self.in_bounds(pos));
        self.input[pos.y as usize][pos.x as usize]
    }

    fn next_nodes_in_dir(
        &self,
        pos: IVec2,
        dir: Dir,
        min_dist: i32,
        max_dist: i32,
    ) -> Vec<(IVec2, BinDir, u32)> {
        let mut heat = 0;
        let mut nodes = Vec::with_capacity((max_dist - min_dist) as usize);
        for i in min_dist..=max_dist {
            let p = match dir {
                Dir::Up => IVec2::new(pos.x, pos.y - i),
                Dir::Down => IVec2::new(pos.x, pos.y + i),
                Dir::Left => IVec2::new(pos.x - i, pos.y),
                Dir::Right => IVec2::new(pos.x + i, pos.y),
                Dir::Invalid => unreachable!(),
            };
            if !self.in_bounds(p) {
                continue;
            }
            let bin_dir = match dir {
                Dir::Up | Dir::Down => BinDir::Vertical,
                _ => BinDir::Horizontal,
            };
            heat += self.get(p);
            nodes.push((p, bin_dir, heat));
        }
        if nodes.is_empty() {
            return nodes;
        }
        let init_heat: u32 = interpolate_points(pos, nodes[0].0)
            .into_iter()
            .skip(1)
            .map(|p| self.get(p))
            .sum();
        for n in nodes.iter_mut() {
            n.2 += init_heat;
        }
        nodes
    }
    pub fn next_nodes(
        &self,
        in_dir: BinDir,
        pos: IVec2,
        min_dist: i32,
        max_dist: i32,
    ) -> Vec<(IVec2, BinDir, u32)> {
        let mut nodes = Vec::with_capacity((max_dist - min_dist) as usize * 4);
        match in_dir {
            BinDir::Vertical => {
                nodes.append(&mut self.next_nodes_in_dir(pos, Dir::Left, min_dist, max_dist));
                nodes.append(&mut self.next_nodes_in_dir(pos, Dir::Right, min_dist, max_dist));
            }
            BinDir::Horizontal => {
                nodes.append(&mut self.next_nodes_in_dir(pos, Dir::Up, min_dist, max_dist));
                nodes.append(&mut self.next_nodes_in_dir(pos, Dir::Down, min_dist, max_dist));
            }
        }
        nodes
    }

    pub fn build_adj_list(&self, min_dist: i32, max_dist: i32) -> AdjList {
        let mut adj_list: AdjList = HashMap::new();
        for j in 0..self.rows {
            for i in 0..self.cols {
                let pos = IVec2::new(i as i32, j as i32);
                for in_dir in [BinDir::Vertical, BinDir::Horizontal] {
                    for n in self.next_nodes(in_dir, pos, min_dist, max_dist) {
                        adj_list
                            .entry((pos, in_dir))
                            .and_modify(|x| {
                                x.insert(n);
                            })
                            .or_insert(HashSet::from([n]));
                    }
                }
            }
        }
        adj_list
    }

    pub fn dijkstra(
        &self,
        min_dist: i32,
        max_dist: i32,
    ) -> HashMap<(IVec2, BinDir), (IVec2, BinDir)> {
        let start_pos = IVec2::new(0, 0);
        let adj_list = self.build_adj_list(min_dist, max_dist);
        let mut q = Q::default();
        for &(pos, dir) in adj_list.keys() {
            if pos == start_pos {
                continue;
            }
            q.push(QItem {
                pos,
                dir,
                total_heat: u32::MAX,
            });
        }
        let mut heat = HashMap::new();
        q.push(QItem {
            pos: start_pos,
            dir: BinDir::Vertical,
            total_heat: 0,
        });
        q.push(QItem {
            pos: start_pos,
            dir: BinDir::Horizontal,
            total_heat: 0,
        });
        heat.insert((start_pos, BinDir::Horizontal), 0);
        heat.insert((start_pos, BinDir::Vertical), 0);
        println!("POPULATED QUEUE");
        let mut prev = HashMap::new();
        while let Some(item) = q.pop() {
            let neighbors = adj_list.get(&(item.pos, item.dir));
            if neighbors.is_none() {
                continue;
            }
            let neighbors = neighbors.unwrap();
            for &(n_pos, n_dir, n_heat) in neighbors {
                // dbg!(&item);
                let new_heat = heat.get(&(item.pos, item.dir)).unwrap_or(&u32::MAX) + n_heat;
                if new_heat < heat.get(&(n_pos, n_dir)).copied().unwrap_or(u32::MAX) {
                    heat.insert((n_pos, n_dir), new_heat);
                    prev.insert((n_pos, n_dir), (item.pos, item.dir));
                    q.push(QItem {
                        pos: n_pos,
                        dir: n_dir,
                        total_heat: new_heat,
                    });
                }
            }
        }
        prev
    }
}

fn retrace(
    prev: &HashMap<(IVec2, BinDir), (IVec2, BinDir)>,
    end: IVec2,
    end_dir: BinDir,
) -> Vec<IVec2> {
    let mut out = Vec::new();
    out.push(end);
    let mut n = prev.get(&(end, end_dir));
    while let Some(node) = n {
        out.push(node.0);
        n = prev.get(node)
    }
    out.reverse();
    out
}

/// Fill in any gaps between a and b with interpolated matrix positions.
fn interpolate_points(a: IVec2, b: IVec2) -> Vec<IVec2> {
    let mut out = Vec::new();
    let dx = (b.x - a.x).abs();
    let dy = (b.y - a.y).abs();
    if dx == 0 && dy == 0 {
        return out;
    }

    let sx = if a.x < b.x { 1 } else { -1 };
    let sy = if a.y < b.y { 1 } else { -1 };

    let mut err = dx - dy;
    let mut x = a.x;
    let mut y = a.y;

    while x != b.x || y != b.y {
        out.push(IVec2::new(x, y));
        let e2 = 2 * err;
        if e2 > -dy {
            err -= dy;
            x += sx;
        }
        if e2 < dx {
            err += dx;
            y += sy;
        }
    }
    out
}

/// Fill in any gaps between each position of the path.
fn interpolated_path(path: &[IVec2]) -> Vec<IVec2> {
    let mut out = Vec::with_capacity(path.len());
    if path.is_empty() {
        return out;
    }
    for i in 0..path.len() - 1 {
        let points = interpolate_points(path[i], path[i + 1]);
        out.extend(points);
    }
    out.push(path[path.len() - 1]);
    out
}

fn draw_line(width: usize) {
    const CHAR: &[u8] = b"\xE2\x94\x81";
    let mut out = std::io::stdout();
    out.write_all(b"\x1b[34m").unwrap();
    for _ in 0..width {
        out.write_all(CHAR).unwrap();
    }
    out.write_all(b"\x1b[0m\n").unwrap();
}

fn detect_dir(a: IVec2, b: IVec2) -> Dir {
    match (b.x - a.x, b.y - a.y) {
        (x, _) if x > 0 => Dir::Right,
        (x, _) if x < 0 => Dir::Left,
        (_, y) if y > 0 => Dir::Down,
        (_, y) if y < 0 => Dir::Up,
        _ => Dir::Invalid,
    }
}

fn vis_path(field: &Field, path: &[IVec2]) {
    draw_line(field.cols);
    let mut out: Vec<Vec<String>> = field
        .input
        .iter()
        .map(|row| row.iter().map(|x| format!("\x1b[30m{x}\x1b[0m")).collect())
        .collect();
    let interpolated = interpolated_path(path);
    for i in 1..interpolated.len() {
        let p = interpolated[i - 1];
        let n = interpolated[i];
        let dir = detect_dir(p, n);
        out[n.y as usize][n.x as usize] = dir.to_string();
    }
    for row in out {
        for col in row {
            print!("{col}");
        }
        println!();
    }
    draw_line(field.cols);
}

fn path_heat(field: &Field, path: &[IVec2]) -> u32 {
    let mut heat = 0;
    let interpolated = interpolated_path(path);
    for pos in interpolated[1..].iter() {
        heat += field.get(*pos);
    }
    heat
}

fn main() {
    // let min_dist = 1;
    // let max_dist = 3;
    let begin = std::time::Instant::now();
    let input = INPUT_FILE
        .lines()
        .map(|line| line.chars().map(|c| c.to_digit(10).unwrap()).collect())
        .collect();
    let field = Field::new(input);
    let prev = field.dijkstra(4, 10);
    let start = IVec2::new(0, 0);
    let end = IVec2::new(field.cols as i32 - 1, field.rows as i32 - 1);
    let mut min_path = Vec::new();
    let mut min_heat = u32::MAX;
    for d in [BinDir::Horizontal, BinDir::Vertical] {
        let path = retrace(&prev, end, d);
        if path[0] != start {
            continue;
        }
        let heat = path_heat(&field, &path);
        if heat < min_heat {
            min_heat = heat;
            min_path = path;
        }
    }
    // vis_path(&field, &min_path);
    // dbg!(&min_path);
    println!("min path heat = {min_heat}");
    println!("time: {:?}", begin.elapsed());
}

#[cfg(test)]
mod test {
    use super::*;
    use assert2::assert;

    #[test]
    fn it_works() {
        let min_dist = 4;
        let max_dist = 10;
        let input = CONTROL_1
            .lines()
            .map(|line| line.chars().map(|c| c.to_digit(10).unwrap()).collect())
            .collect();
        let field = Field::new(input);
        let nodes = field.next_nodes_in_dir(IVec2::new(0, 0), Dir::Down, min_dist, max_dist);
        let expected = Vec::from([
            (IVec2::new(0, 4), BinDir::Vertical, 13, 4),
            (IVec2::new(0, 5), BinDir::Vertical, 14, 5),
            (IVec2::new(0, 6), BinDir::Vertical, 18, 6),
            (IVec2::new(0, 7), BinDir::Vertical, 21, 7),
            (IVec2::new(0, 8), BinDir::Vertical, 25, 8),
            (IVec2::new(0, 9), BinDir::Vertical, 29, 9),
            (IVec2::new(0, 10), BinDir::Vertical, 30, 10),
        ]);
        assert!(nodes == expected);
    }
}
