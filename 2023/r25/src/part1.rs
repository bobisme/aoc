pub mod graph;
pub mod input;
pub mod queue;
pub mod sw;

use std::{
    cmp,
    collections::{BTreeMap, BTreeSet, BinaryHeap, HashMap, HashSet, VecDeque},
};

use graph::{Graph, Intern, Vertex};
use input::parse_line;
use itertools::Itertools;
use parking_lot::Mutex;
use rand::prelude::*;
use rayon::prelude::*;

use crate::{
    graph::{karger, Edge},
    input::{CONTROL_1, INPUT_FILE},
    sw::{karger_stein, stoer_wagner},
};

// fn print_graph(graph: &Graph, intern: &Intern) {
//     println!("strict graph {{\n\tlayout=neato");
//     let mut printed = HashSet::new();
//     for (v, adj) in graph.adj_list.iter() {
//         println!("{0} [label=\"{0}|{1}\"]", v.0, &intern.rev_map[v]);
//         let v_name = v.0.to_string();
//         for a in adj.iter() {
//             let edge = Edge::new(*v, a.0);
//             if printed.contains(&edge) {
//                 continue;
//             }
//             println!("\t{} -- {}", v_name, a.0 .0);
//             printed.insert(edge);
//         }
//     }
//     println!("}}");
// }
fn main() {
    let input = INPUT_FILE.lines();

    let mut graph = Graph::default();
    let intern = Mutex::new(Intern::default());
    let adj_list: BTreeMap<Vertex, BTreeSet<Vertex>> =
        input.map(|x| parse_line(x, &mut intern.lock())).collect();
    let start = std::time::Instant::now();
    // for (u, adj) in adj_list {
    //     for v in adj {
    //         graph.add_edge(u, v, 1);
    //     }
    // }
    // print_graph(&graph, &intern.lock());
    // let progress_total = vert_count * (vert_count - 1);
    // let bar = indicatif::ProgressBar::new(progress_total as u64);
    // let in_path_count = Mutex::new(HashMap::<Edge, u32>::new());
    // let vert_count = graph.adj_list.keys().len();
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
    // let bar = indicatif::ProgressBar::new(graph.adj_list.len() as u64);
    // let minies = graph
    //     .adj_list
    //     .keys()
    //     // .par_bridge()
    //     .map(|init| {
    //         let mut g = graph.clone();
    //         let cut = stoer_wagner(&mut g, &intern, *init);
    //         bar.inc(1);
    //         cut
    //     })
    //     .collect::<Vec<_>>();
    // bar.finish();
    // let init = adj_list.keys().choose(&mut rand::thread_rng()).unwrap();
    // let mut g = graph.clone();
    // // let (cut, group_sizes, _) = stoer_wagner(&adj_list, &intern, *init);
    // let (mut cut, mut groups) = karger_stein(&adj_list, &intern);
    // while cut.weight > 3 || (groups.0 == 1 || groups.1 == 1) {
    //     (cut, groups) = karger_stein(&adj_list, &intern);
    // }
    // dbg!(cut);
    // dbg!(groups);
    // println!("groups sizes: {}, {}", group_sizes.0, group_sizes.1);
    // println!("multiplied = {}", group_sizes.0 * group_sizes.1);
    // println!("groups {:?}, {:?}", g.unroll(cut.s), g.unroll(cut.t));
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
        // let mut g = graph.clone();
        // let (cut_weight, (s, t)) = karger(&mut g, &mut rand::thread_rng());
        if cut_weight < min_cut {
            min_cut = cut_weight;
            min_groups = (s, t);
        }
        println!("best so far: cut weight={min_cut} groups={min_groups:?}");
    }
    println!("{:?}", start.elapsed());
}
