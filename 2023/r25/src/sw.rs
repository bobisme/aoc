use std::{
    cmp,
    collections::{BTreeMap, BTreeSet},
};

use itertools::Itertools;
use parking_lot::Mutex;
use rand::seq::IteratorRandom;

use crate::graph::{Cut, Edge, Intern, Vertex};

pub fn weight_to_set(set: &BTreeSet<Vertex>, edges: &BTreeMap<Edge, u32>) -> BTreeMap<Vertex, u32> {
    let mut weights = BTreeMap::new();
    for (edge, &weight) in edges.iter() {
        let other = match (set.contains(&edge.0), set.contains(&edge.1)) {
            (true, false) => edge.1,
            (false, true) => edge.0,
            (true, true) => {
                // eprintln!("both in set");
                continue;
            }
            (false, false) => {
                // eprintln!("not connected to set");
                continue;
            }
        };
        weights
            .entry(other)
            .and_modify(|x| *x += weight)
            .or_insert(weight);
    }
    weights
}

pub fn stoer_wagner_pass(
    vertices: &BTreeSet<Vertex>,
    edges: &BTreeMap<Edge, u32>,
    init: Vertex,
) -> Cut {
    let mut a = BTreeSet::new();
    a.insert(init);
    let mut last = init;
    let mut next_last = init;
    let mut weight = u32::MAX;
    for _ in 2..=vertices.len() {
        // dbg!(&a);
        let weights = weight_to_set(&a, edges);
        // dbg!(&weights);
        let next_vert = vertices
            .iter()
            .filter(|v| !a.contains(v))
            .max_by_key(|x| *weights.get(x).unwrap_or(&0));
        next_last = last;
        last = *next_vert.unwrap();
        let s_t_weight = *edges.get(&Edge::new(next_last, last)).unwrap_or(&0);
        // dbg!((next_last, last, s_t_weight));
        weight = weights[&last] + s_t_weight;
        a.insert(last);
    }
    let s = next_last;
    let t = last;
    // dbg!((s, t, weight));
    Cut { s, t, weight }
}

fn merge(
    s: Vertex,
    t: Vertex,
    merged: Vertex,
    vertices: &mut BTreeSet<Vertex>,
    edges: &mut BTreeMap<Edge, u32>,
) {
    // println!("merging {s:?} and {t:?} into {merged:?}");
    vertices.remove(&s);
    vertices.remove(&t);
    vertices.insert(merged);
    let s_neighbors = edges
        .keys()
        .filter_map(|e| match (e.0 == s, e.1 == s) {
            (true, false) => Some(e.1),
            (false, true) => Some(e.0),
            _ => None,
        })
        .collect_vec();
    let t_neighbors = edges
        .keys()
        .filter_map(|e| match (e.0 == t, e.1 == t) {
            (true, false) => Some(e.1),
            (false, true) => Some(e.0),
            _ => None,
        })
        .collect_vec();
    let mut s_t_edges: BTreeMap<Vertex, u32> = BTreeMap::new();
    for n in s_neighbors {
        let edge = Edge::new(s, n);
        let w = edges[&edge];
        edges.remove(&edge);
        s_t_edges.entry(n).and_modify(|x| *x += w).or_insert(w);
    }
    for n in t_neighbors {
        if n == s {
            continue;
        }
        let edge = Edge::new(t, n);
        let w = edges[&edge];
        edges.remove(&edge);
        s_t_edges.entry(n).and_modify(|x| *x += w).or_insert(w);
    }
    for (u, weight) in s_t_edges {
        edges.insert(Edge::new(u, merged), weight);
    }
    // dbg!(&edges);
}

#[allow(unused)]
#[derive(Debug, Clone, Copy)]
pub struct MergeOp {
    s: Vertex,
    t: Vertex,
    size: usize,
}

pub fn calc_group_size(
    _edges: &BTreeMap<Edge, u32>,
    vert: Vertex,
    _cut_vert: Vertex,
    merges: &BTreeMap<Vertex, MergeOp>,
) -> usize {
    merges.get(&vert).map(|x| x.size).unwrap_or(1)
}

pub fn calc_group_sizes(
    edges: &BTreeMap<Edge, u32>,
    cut: &Cut,
    merges: &BTreeMap<Vertex, MergeOp>,
) -> (usize, usize) {
    let s_size = calc_group_size(edges, cut.s, cut.t, merges);
    let t_size = calc_group_size(edges, cut.t, cut.s, merges);
    (s_size, t_size)
}

pub fn collect_vertices(adj_list: &BTreeMap<Vertex, BTreeSet<Vertex>>) -> BTreeSet<Vertex> {
    adj_list
        .iter()
        .flat_map(|(k, v)| std::iter::once(k).chain(v).cloned())
        .collect()
}

pub fn stoer_wagner(
    adj_list: &BTreeMap<Vertex, BTreeSet<Vertex>>,
    intern: &Mutex<Intern>,
    init: Vertex,
) -> (Cut, (usize, usize), BTreeMap<Vertex, MergeOp>) {
    let mut min_cut = Cut {
        s: Vertex(0),
        t: Vertex(0),
        weight: u32::MAX,
    };
    let mut group_sizes = (0, 0);
    let mut vertices = collect_vertices(adj_list);
    let init_size = vertices.len();
    let mut edges: BTreeMap<Edge, u32> = adj_list
        .iter()
        .flat_map(|(v, ns)| ns.iter().map(|n| (Edge::new(*v, *n), 1)))
        .collect();
    let mut merges: BTreeMap<Vertex, MergeOp> = BTreeMap::new();
    let bar = indicatif::ProgressBar::new(init_size as u64);
    while vertices.len() > 1 {
        bar.inc(1);
        let cut = stoer_wagner_pass(&vertices, &edges, init);
        // dbg!(vertices.len());
        // dbg!(&cut);
        if cut < min_cut {
            min_cut = cut;
            let sizes = calc_group_sizes(&edges, &cut, &merges);
            let a = cmp::max(sizes.0, sizes.1);
            let b = init_size - a;
            group_sizes = (a, b);
            if cut.weight == 3 {
                // dbg!(&merges);
                // dbg!(&edges);
                break;
            }
        }
        let merged = Vertex(intern.lock().fetch_next_id());
        let s_size = merges.get(&cut.s).map(|x| x.size).unwrap_or(1);
        let t_size = merges.get(&cut.t).map(|x| x.size).unwrap_or(1);
        merges.insert(
            merged,
            MergeOp {
                s: cut.s,
                t: cut.t,
                size: s_size + t_size,
            },
        );
        merge(cut.s, cut.t, merged, &mut vertices, &mut edges);
    }
    bar.finish();
    (min_cut, group_sizes, merges)
}

pub fn karger_stein(
    adj_list: &BTreeMap<Vertex, BTreeSet<Vertex>>,
    intern: &Mutex<Intern>,
) -> (Cut, (usize, usize)) {
    let mut rng = rand::thread_rng();
    let mut last_cut = Cut {
        s: Vertex(0),
        t: Vertex(0),
        weight: u32::MAX,
    };
    let mut vertices = collect_vertices(adj_list);
    let _init_size = vertices.len();
    let mut edges: BTreeMap<Edge, u32> = adj_list
        .iter()
        .flat_map(|(v, ns)| ns.iter().map(|n| (Edge::new(*v, *n), 1)))
        .collect();
    let mut merges: BTreeMap<Vertex, MergeOp> = BTreeMap::new();
    while vertices.len() > 2 {
        let (edge, &weight) = edges.iter().choose(&mut rng).unwrap();
        let cut = Cut {
            s: edge.0,
            t: edge.1,
            weight,
        };
        last_cut = cut;
        let merged = Vertex(intern.lock().fetch_next_id());
        let s_size = merges.get(&cut.s).map(|x| x.size).unwrap_or(1);
        let t_size = merges.get(&cut.t).map(|x| x.size).unwrap_or(1);
        merges.insert(
            merged,
            MergeOp {
                s: cut.s,
                t: cut.t,
                size: s_size + t_size,
            },
        );
        merge(edge.0, edge.1, merged, &mut vertices, &mut edges);
    }
    dbg!(&merges);
    dbg!(last_cut);
    let s_size = merges.get(&last_cut.s).map(|x| x.size).unwrap_or(1);
    let t_size = merges.get(&last_cut.t).map(|x| x.size).unwrap_or(1);
    (last_cut, (s_size, t_size))
}

#[cfg(test)]
mod test {
    #![allow(unused)]
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
    fn test_weights_to_set() {
        let (mut adj_list, mut intern) = setup();
        let rzs = intern.intern("rzs");
        let frs = intern.intern("frs");
        let qnr = intern.intern("qnr");
        let cmg = intern.intern("cmg");
        let lsr = intern.intern("lsr");
        let rsh = intern.intern("rsh");
        let lhk = intern.intern("lhk");
        let intern = Mutex::new(intern);

        let mut edges: BTreeMap<Edge, u32> = adj_list
            .iter()
            .flat_map(|(v, ns)| ns.iter().map(|n| (Edge::new(*v, *n), 1)))
            .collect();
        let mut set = BTreeSet::new();
        set.insert(rzs);
        set.insert(frs);
        let weights = weight_to_set(&set, &edges);
        let expected = BTreeMap::from([(qnr, 2), (lsr, 2), (cmg, 1), (rsh, 2), (lhk, 1)]);
        assert!(weights == expected);
    }

    #[test]
    fn test_weights_to_set_uses_weights() {
        let (mut adj_list, mut intern) = setup();
        let rzs = intern.intern("rzs");
        let frs = intern.intern("frs");
        let qnr = intern.intern("qnr");
        let cmg = intern.intern("cmg");
        let lsr = intern.intern("lsr");
        let rsh = intern.intern("rsh");
        let lhk = intern.intern("lhk");
        let intern = Mutex::new(intern);

        let mut edges: BTreeMap<Edge, u32> = adj_list
            .iter()
            .flat_map(|(v, ns)| ns.iter().map(|n| (Edge::new(*v, *n), 2)))
            .collect();
        let mut set = BTreeSet::new();
        set.insert(rzs);
        set.insert(frs);
        let weights = weight_to_set(&set, &edges);
        let expected = BTreeMap::from([(qnr, 4), (lsr, 4), (cmg, 2), (rsh, 4), (lhk, 2)]);
        assert!(weights == expected);
    }

    #[test]
    fn test_stoer_wagner_pass() {
        let (mut adj_list, mut intern) = setup();
        let rzs = intern.intern("rzs");
        let frs = intern.intern("frs");
        let qnr = intern.intern("qnr");
        let cmg = intern.intern("cmg");
        let lsr = intern.intern("lsr");
        let rsh = intern.intern("rsh");
        let lhk = intern.intern("lhk");
        let intern = Mutex::new(intern);
        let mut edges: BTreeMap<Edge, u32> = adj_list
            .iter()
            .flat_map(|(v, ns)| ns.iter().map(|n| (Edge::new(*v, *n), 1)))
            .collect();
        let mut vertices: BTreeSet<_> = adj_list.keys().cloned().collect();
        let cut = stoer_wagner_pass(&vertices, &edges, rzs);
        let mut intern = intern.lock();
        assert!(
            cut == Cut {
                s: intern.intern("rsh"),
                t: intern.intern("frs"),
                weight: 4
            }
        );
    }

    #[test]
    fn test_merge() {
        let (mut adj_list, mut intern) = setup();
        let nvd = intern.intern("nvd");
        let bvb = intern.intern("bvb");
        let intern = Mutex::new(intern);
        let mut edges: BTreeMap<Edge, u32> = adj_list
            .iter()
            .flat_map(|(v, ns)| ns.iter().map(|n| (Edge::new(*v, *n), 1)))
            .collect();
        let mut vertices: BTreeSet<_> = adj_list.keys().cloned().collect();
        let new_v = Vertex(16);
        merge(nvd, bvb, new_v, &mut vertices, &mut edges);
        assert!(!vertices.contains(&nvd));
        assert!(!vertices.contains(&bvb));
        assert!(vertices.contains(&new_v));
        assert!(!edges.keys().any(|x| x.0 == nvd || x.1 == nvd));
        assert!(!edges.keys().any(|x| x.0 == bvb || x.1 == bvb));
        assert!(edges.keys().any(|x| x.0 == new_v || x.1 == new_v));
    }

    #[test]
    fn test_merge_chain() {
        let (mut adj_list, mut intern) = setup();
        let nvd = intern.intern("nvd");
        let bvb = intern.intern("bvb");
        let intern = Mutex::new(intern);
        let mut edges: BTreeMap<Edge, u32> = adj_list
            .iter()
            .flat_map(|(v, ns)| ns.iter().map(|n| (Edge::new(*v, *n), 1)))
            .collect();
        let mut vertices = collect_vertices(&adj_list);
        assert!(vertices.contains(&Vertex(12)));
        merge(Vertex(5), Vertex(6), Vertex(16), &mut vertices, &mut edges);
        merge(Vertex(1), Vertex(2), Vertex(17), &mut vertices, &mut edges);
        merge(Vertex(3), Vertex(14), Vertex(18), &mut vertices, &mut edges);
        merge(
            Vertex(18),
            Vertex(13),
            Vertex(19),
            &mut vertices,
            &mut edges,
        );
        merge(
            Vertex(19),
            Vertex(17),
            Vertex(20),
            &mut vertices,
            &mut edges,
        );
        dbg!(&edges);
        let count_4 = edges.keys().filter_map(|e| e.contains(Vertex(4))).count();
        assert!(count_4 == 5);
        assert!(edges[&Edge::new(Vertex(4), Vertex(20))] == 1);
        let set: BTreeSet<_> = vertices
            .iter()
            .filter(|&&x| x != Vertex(4))
            .copied()
            .collect();
        dbg!(&set);
        let weights = weight_to_set(&set, &edges);
        assert!(weights[&Vertex(4)] == 5);
    }

    #[test]
    fn test_stoer_wagner() {
        let (mut adj_list, mut intern) = setup();
        let rzs = intern.intern("rzs");
        let frs = intern.intern("frs");
        let qnr = intern.intern("qnr");
        let cmg = intern.intern("cmg");
        let lsr = intern.intern("lsr");
        let rsh = intern.intern("rsh");
        let lhk = intern.intern("lhk");
        let intern = Mutex::new(intern);
        let (cut, group_sizes, _) = stoer_wagner(&adj_list, &intern, cmg);
        let mut intern = intern.lock();
        assert!(group_sizes == (6, 9));
        assert!(
            cut == Cut {
                s: Vertex(4),
                t: Vertex(23),
                weight: 4
            }
        );
    }
}
