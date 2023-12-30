use std::{
    cmp,
    collections::{BinaryHeap, HashMap, HashSet, VecDeque},
    ops::Deref,
};

use itertools::Itertools;
use parking_lot::Mutex;
use rayon::prelude::*;

const INPUT_FILE: &str = include_str!("../../2023-25.input");
const CONTROL_1: &str = r#"jqt: rhn xhk nvd
rsh: frs pzl lsr
xhk: hfx
cmg: qnr nvd lhk bvb
rhn: xhk bvb hfx
bvb: xhk hfx
pzl: lsr hfx nvd
qnr: nvd
ntq: jqt hfx bvb xhk
nvd: lhk
lsr: lhk
rzs: qnr cmg lsr rsh
frs: qnr lhk lsr
"#;

fn parse_line(line: &str) -> (String, HashSet<String>) {
    let mut split = line.splitn(2, ": ");
    let left = split.next().unwrap();
    let right = split.next().unwrap().split(' ');
    (left.to_string(), right.map(|x| x.to_string()).collect())
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct QItem {
    val: u32,
    key: String,
    v: u32,
}

impl PartialOrd for QItem {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for QItem {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.val.cmp(&other.val)
    }
}

#[derive(Debug, Default)]
struct Q {
    q: BinaryHeap<QItem>,
    set: HashSet<String>,
    versions: HashMap<String, u32>,
}

impl Q {
    fn push(&mut self, key: &str, val: u32, ver: u32) {
        let item = QItem {
            val,
            key: key.to_string(),
            v: ver,
        };
        self.set.insert(item.key.clone());
        self.q.push(item);
    }
    fn pop(&mut self) -> Option<QItem> {
        let mut popped;
        loop {
            popped = self.q.pop()?;
            let expected = *self.versions.get(&popped.key).unwrap_or(&0);
            if popped.v == expected {
                break;
            }
        }
        self.set.remove(&popped.key);
        Some(popped)
    }
    fn contains(&self, key: &String) -> bool {
        self.set.contains(key)
    }
    fn update(&mut self, key: &str, val: u32) {
        let v = *self
            .versions
            .entry(key.to_string())
            .and_modify(|counter| *counter += 1)
            .or_insert(1);
        self.push(key, val, v);
    }
}

#[repr(transparent)]
#[derive(Default, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct Vertex(u32);

impl std::fmt::Debug for Vertex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "│{}│", self.0)
    }
}

#[derive(Default, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct Edge(Vertex, Vertex);

impl std::fmt::Debug for Edge {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "║{}─{}║", self.0 .0, self.1 .0)
    }
}

impl Edge {
    fn new(u: Vertex, v: Vertex) -> Self {
        if u < v {
            Self(u, v)
        } else {
            Self(v, u)
        }
    }
}

#[derive(Debug, Default)]
struct Intern {
    last_id: u32,
    intern_map: HashMap<String, Vertex>,
}

impl Intern {
    fn fetch_next_id(&mut self) -> u32 {
        let next = self.last_id + 1;
        self.last_id = next;
        next
    }

    fn intern(&mut self, str: &str) -> Vertex {
        let v = self.intern_map.entry(str.to_string()).or_insert_with(|| {
            let next = self.last_id + 1;
            self.last_id = next;
            Vertex(next)
        });
        *v
    }
}

#[derive(Debug, Default, Clone)]
struct Graph {
    adj_list: HashMap<Vertex, Vec<Vertex>>,
    merge_map: HashMap<Vertex, (Vertex, Vertex)>,
}

impl Graph {
    fn add_edge(&mut self, u: Vertex, v: Vertex) {
        self.adj_list.entry(u).or_default().push(v);
        self.adj_list.entry(v).or_default().push(u);
    }

    #[must_use]
    fn len(&self) -> usize {
        self.adj_list.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn shortest_path(&self, src: Vertex, dst: Vertex) -> Vec<Vertex> {
        let mut q = VecDeque::new();
        let mut explored = HashSet::new();
        let mut prev = HashMap::new();
        q.push_back(src);
        explored.insert(src);
        while let Some(v) = q.pop_front() {
            if v == dst {
                break;
            }
            for n in self.adj_list[&v].iter() {
                if explored.contains(n) {
                    continue;
                }
                explored.insert(*n);
                prev.insert(*n, v);
                q.push_back(*n);
            }
        }
        let mut path = Vec::new();
        let mut curr = dst;
        while let Some(&p) = prev.get(&curr) {
            path.push(p);
            curr = p;
            if p == src {
                break;
            }
        }
        path.reverse();
        path
    }

    fn most_connected_vert(&self, excl: Option<Vertex>) -> Vertex {
        *self
            .adj_list
            .iter()
            .filter(|(&k, _)| excl.is_none() || excl.unwrap() != k)
            .max_by(|a, b| a.1.len().cmp(&b.1.len()))
            .unwrap()
            .0
    }

    fn most_connected_to(&self, v: Vertex) -> Vertex {
        let adj = &self.adj_list[&v];
        let mut counts = HashMap::new();
        for x in adj.iter() {
            counts.entry(x).and_modify(|x| *x += 1).or_insert(1);
        }
        *counts.into_iter().max().unwrap().0
    }

    fn merge(&mut self, u: Vertex, v: Vertex, intern: &mut Intern) -> Vertex {
        let mut neighbors = self.adj_list[&u].clone();
        if let Some(v_adj) = self.adj_list.get(&v) {
            neighbors.append(&mut v_adj.clone());
        }
        neighbors.retain(|&x| x != u && x != v);

        let new_vert = Vertex(intern.fetch_next_id());
        self.merge_map.insert(new_vert, (u, v));
        self.adj_list.remove(&u);
        self.adj_list.remove(&v);

        for n in neighbors.iter() {
            let adj = self.adj_list.get_mut(n).unwrap();
            adj.retain(|&x| x != u && x != v);
            adj.push(new_vert);
        }
        self.adj_list.insert(new_vert, neighbors);
        new_vert
    }

    fn merged_size(&self, v: Vertex) -> usize {
        if !self.merge_map.contains_key(&v) {
            return 1;
        }
        let merged = self.merge_map[&v];
        self.merged_size(merged.0) + self.merged_size(merged.1)
    }
}

fn shoer_wagner_pass(graph: &mut Graph, intern: &mut Intern, init: Vertex) -> usize {
    let mut vert = init;
    while graph.len() > 2 {
        let tightest = graph.most_connected_to(vert);
        vert = graph.merge(vert, tightest, intern);
    }
    let key = graph.adj_list.keys().next().unwrap();
    graph.adj_list[key].len()
}

fn main() {
    let input = CONTROL_1.lines();

    let mut graph = Graph::default();
    let mut intern = Mutex::new(Intern::default());
    let adj_list: HashMap<String, HashSet<String>> = input.map(parse_line).collect();
    let start = std::time::Instant::now();
    for (vert, adj) in adj_list {
        for other in adj {
            let mut intern = intern.lock();
            let u = intern.intern(&vert);
            let v = intern.intern(&other);
            graph.add_edge(u, v);
        }
    }
    let in_path_count = Mutex::new(HashMap::<Edge, u32>::new());
    let vert_count = graph.adj_list.keys().len();
    // let progress_total = vert_count * (vert_count - 1);
    // let bar = indicatif::ProgressBar::new(progress_total as u64);
    // graph
    //     .adj_list
    //     .keys()
    //     .combinations(2)
    //     .par_bridge()
    //     .for_each(|comb| {
    //         bar.inc(1);
    //         let path = graph.shortest_path(*comb[0], *comb[1]);
    //         let mut in_path_count = in_path_count.lock();
    //         for edge in path.iter().cloned().tuple_windows::<(Vertex, Vertex)>() {
    //             in_path_count
    //                 .entry(Edge::new(edge.0, edge.1))
    //                 .and_modify(|counter| *counter += 1)
    //                 .or_insert(1);
    //         }
    //     });
    // let mut vert_traverse_counts = in_path_count
    //     .lock()
    //     .iter()
    //     .map(|(k, v)| (*k, *v))
    //     .collect_vec();
    // vert_traverse_counts.sort_by(|a, b| a.1.cmp(&b.1));
    // vert_traverse_counts.reverse();
    // let top = vert_traverse_counts.iter().take(3).collect_vec();
    // println!("{:?}", top);
    // bar.finish();
    let bar = indicatif::ProgressBar::new(graph.adj_list.len() as u64);
    let minies = graph
        .adj_list
        .keys()
        .par_bridge()
        .map(|init| {
            let mut g = graph.clone();
            let key = graph.adj_list.keys().next().unwrap();
            let mut intern = intern.lock();
            shoer_wagner_pass(&mut g, &mut intern, *key)
        })
        .collect::<Vec<_>>();
    dbg!(minies);
    // println!("local min cut = {} edges", local_min_cut);
    // for key in G.adj_list.keys() {
    //     println!("group size: {}", G.merged_size(*key));
    // }
    println!("{:?}", start.elapsed());
}

#[cfg(test)]
mod test {
    use super::*;
    use assert2::assert;

    #[test]
    fn it_works() {
        let input = CONTROL_1.lines();
        let mut graph = Graph::default();
        let mut intern = Intern::default();
        let adj_list: HashMap<String, HashSet<String>> = input.map(parse_line).collect();
        for (vert, adj) in adj_list {
            for other in adj {
                let u = intern.intern(&vert);
                let v = intern.intern(&other);
                graph.add_edge(u, v);
            }
        }

        let mut graph_iter = graph.adj_list.iter().take(2);
        let (&u, _) = graph_iter.next().unwrap();
        let (&v, _) = graph_iter.next().unwrap();

        let new = graph.merge(u, v, &mut intern);
        assert!(!graph.adj_list.contains_key(&u));
        assert!(!graph.adj_list.contains_key(&v));
        assert!(graph.adj_list.contains_key(&new));
        assert!(graph.merge_map[&new] == (u, v));
    }
}
