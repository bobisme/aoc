pub mod graph;
pub mod input;
pub mod queue;
pub mod sw;

use std::collections::{BTreeMap, BTreeSet, HashSet};

use graph::{Graph, Intern, Vertex};
use input::parse_line;
use parking_lot::Mutex;
use rayon::prelude::*;

use crate::{
    graph::{karger, Edge},
    input::INPUT_FILE,
};

pub fn print_graph(graph: &Graph, intern: &Intern) {
    println!("strict graph {{\n\tlayout=neato");
    let mut printed = HashSet::new();
    for (v, adj) in graph.adj_list.iter() {
        println!("{0} [label=\"{0}|{1}\"]", v.0, &intern.rev_map[v]);
        let v_name = v.0.to_string();
        for a in adj.iter() {
            let edge = Edge::new(*v, Vertex(a.0));
            if printed.contains(&edge) {
                continue;
            }
            println!("\t{} -- {}", v_name, a.0);
            printed.insert(edge);
        }
    }
    println!("}}");
}

fn main() {
    let input = INPUT_FILE.lines();

    let mut graph = Graph::default();
    let intern = Mutex::new(Intern::default());
    let adj_list: BTreeMap<Vertex, BTreeSet<Vertex>> =
        input.map(|x| parse_line(x, &mut intern.lock())).collect();
    let start = std::time::Instant::now();
    for (k, v) in adj_list.iter() {
        for x in v.iter() {
            graph.add_edge(*k, *x);
        }
    }
    let mut min_cut = usize::MAX;
    let mut min_groups = (0, 0);
    while min_cut > 1 {
        println!("trying a batch of graphs");
        let (cut_weight, (s, t)) = (0..24)
            .into_par_iter()
            .map(|_| graph.clone())
            .map(|mut g| karger(&mut g, &mut rand::thread_rng()))
            .min_by_key(|(c, _)| *c)
            .unwrap();
        if cut_weight < min_cut {
            min_cut = cut_weight;
            min_groups = (s, t);
        }
        println!("best so far: cut weight={min_cut} groups={min_groups:?}");
    }
    println!("{:?}", start.elapsed());
}
