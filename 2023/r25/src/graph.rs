use std::{
    cmp,
    collections::{BTreeMap, HashMap},
};

use rand::{rngs::ThreadRng, seq::IteratorRandom};

#[repr(transparent)]
#[derive(Default, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Vertex(pub u32);

impl std::fmt::Debug for Vertex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "│{}│", self.0)
    }
}

#[derive(Default, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Edge(pub Vertex, pub Vertex);

impl std::fmt::Debug for Edge {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "║{}─{}║", self.0 .0, self.1 .0)
    }
}

impl Edge {
    pub fn new(u: Vertex, v: Vertex) -> Self {
        if u < v {
            Self(u, v)
        } else {
            Self(v, u)
        }
    }

    /// Returns the other vertex or None if not contained.
    pub fn contains(&self, v: Vertex) -> Option<Vertex> {
        match (self.0 == v, self.1 == v) {
            (true, false) => Some(self.1),
            (false, true) => Some(self.0),
            _ => None,
        }
    }
}

#[derive(Debug, Default)]
pub struct Intern {
    last_id: u32,
    pub intern_map: HashMap<String, Vertex>,
    pub rev_map: HashMap<Vertex, String>,
}

impl Intern {
    pub fn fetch_next_id(&mut self) -> u32 {
        let next = self.last_id + 1;
        self.last_id = next;
        next
    }

    pub fn intern(&mut self, str: &str) -> Vertex {
        let v = self.intern_map.entry(str.to_string()).or_insert_with(|| {
            let next = self.last_id + 1;
            self.last_id = next;
            Vertex(next)
        });
        self.rev_map.insert(*v, str.to_string());
        *v
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Cut {
    pub s: Vertex,
    pub t: Vertex,
    pub weight: u32,
}

impl PartialOrd for Cut {
    fn partial_cmp(&self, other: &Self) -> Option<cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Cut {
    fn cmp(&self, other: &Self) -> cmp::Ordering {
        self.weight.cmp(&other.weight)
    }
}

#[derive(Debug, Default, Clone)]
pub struct DisjointSet {
    parents: HashMap<Vertex, Vertex>,
    rank: HashMap<Vertex, usize>,
}

impl DisjointSet {
    fn find(&mut self, v: Vertex) -> Option<Vertex> {
        let p = self.parents.get(&v)?;
        if *p != v {
            let found = self.find(*p)?;
            self.parents.insert(v, found);
        }
        self.parents.get(&v).copied()
    }

    fn union(&mut self, u: Vertex, v: Vertex) {
        let root_u = self.find(u).unwrap();
        let root_v = self.find(v).unwrap();

        if root_u == root_v {
            return;
        }
        match (self.rank[&root_u], self.rank[&root_v]) {
            (a, b) if a < b => {
                self.parents.insert(root_u, root_v);
            }
            (a, b) if a > b => {
                self.parents.insert(root_v, root_u);
            }
            _ => {
                self.parents.insert(root_v, root_u);
                if let Some(x) = self.rank.get_mut(&root_u) {
                    *x += 1;
                }
            }
        }
    }
    fn count_merged(&mut self, vertex: Vertex) -> usize {
        let Some(root) = self.find(vertex) else {
            return 0;
        };
        let verts: Vec<_> = self.parents.keys().cloned().collect();
        verts
            .iter()
            .filter_map(|x| self.find(*x))
            .filter(|&x| x == root)
            .count()
    }
}

#[derive(Debug, Default, Clone)]
pub struct Graph {
    /// map vertex -> list { (vertex, weight) }
    pub adj_list: BTreeMap<Vertex, Vec<Vertex>>,
    pub disjoint_set: DisjointSet,
}

impl Graph {
    pub fn add_edge(&mut self, u: Vertex, v: Vertex) {
        self.adj_list.entry(u).or_default().push(v);
        self.adj_list.entry(v).or_default().push(u);
        self.disjoint_set.parents.insert(u, u);
        self.disjoint_set.parents.insert(v, v);
        self.disjoint_set.rank.insert(u, 0);
        self.disjoint_set.rank.insert(v, 0);
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.adj_list.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn contract(&mut self, u: Vertex, v: Vertex) {
        self.disjoint_set.union(u, v);
        for (&vert, list) in self.adj_list.iter_mut() {
            let found_vert = self.disjoint_set.find(vert).unwrap();
            *list = list
                .iter()
                .filter_map(|&x| {
                    if self.disjoint_set.find(x).unwrap() == found_vert {
                        None
                    } else {
                        self.disjoint_set.find(x)
                    }
                })
                .collect()
        }
    }
}

pub fn rand_edge(graph: &Graph, rng: &mut ThreadRng) -> Edge {
    let u = graph
        .adj_list
        .iter()
        .filter(|(_, list)| !list.is_empty())
        .map(|(k, _)| k)
        .choose(rng)
        .cloned()
        .unwrap();
    let v = graph.adj_list[&u].iter().choose(rng).cloned().unwrap();
    Edge::new(u, v)
}

pub fn karger(graph: &mut Graph, rng: &mut ThreadRng) -> (usize, (usize, usize)) {
    let n_verts = graph.adj_list.len();
    for _ in (0..).take(n_verts - 2) {
        let edge = rand_edge(graph, rng);
        graph.contract(edge.0, edge.1);
    }
    let (cut_weight, (s, t)) = graph
        .adj_list
        .iter()
        .max_by(|(_, a_list), (_, b_list)| a_list.len().cmp(&b_list.len()))
        .map(|(k, list)| (list.len(), (k, list[0])))
        .unwrap();
    let s_size = graph.disjoint_set.count_merged(*s);
    let t_size = graph.disjoint_set.count_merged(t);
    (cut_weight, (s_size, t_size))
}

#[cfg(test)]
mod test {
    #![allow(unused)]
    use std::collections::BTreeSet;

    use crate::input::{parse_line, CONTROL_1};

    use super::*;
    use assert2::assert;

    fn setup() -> (BTreeMap<Vertex, BTreeSet<Vertex>>, Intern) {
        let input = CONTROL_1.lines();
        let mut intern = Intern::default();
        let adj_list: BTreeMap<_, _> = input.map(|x| parse_line(x, &mut intern)).collect();
        (adj_list, intern)
    }

    #[test]
    fn test_something() {
        let (mut adj_list, intern) = setup();
        let mut graph = Graph::default();
        for (k, v) in adj_list.iter() {
            for x in v.iter() {
                graph.add_edge(*k, *x);
            }
        }
        assert!(graph.adj_list[&Vertex(1)].contains(&Vertex(2)));
        assert!(
            graph.adj_list[&Vertex(3)]
                .iter()
                .filter(|x| **x == Vertex(1))
                .count()
                == 1
        );
        assert!(
            graph.adj_list[&Vertex(3)]
                .iter()
                .filter(|x| **x == Vertex(2))
                .count()
                == 1
        );
        graph.contract(Vertex(1), Vertex(2));
        assert!(!graph.adj_list[&Vertex(1)].contains(&Vertex(2)));
        assert!(
            graph.adj_list[&Vertex(3)]
                .iter()
                .filter(|x| **x == Vertex(1))
                .count()
                == 2
        );
        assert!(
            graph.adj_list[&Vertex(3)]
                .iter()
                .filter(|x| **x == Vertex(2))
                .count()
                == 0
        );
        assert!(graph.disjoint_set.count_merged(Vertex(1)) == 2);
    }
    #[test]
    fn run_karger() {
        let (mut adj_list, intern) = setup();
        let mut graph = Graph::default();
        for (k, v) in adj_list.iter() {
            for x in v.iter() {
                graph.add_edge(*k, *x);
            }
        }
        karger(&mut graph, &mut rand::thread_rng());
        // dbg!(&graph);
        // assert!(false);
    }
}
