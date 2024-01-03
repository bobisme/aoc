use std::collections::BTreeSet;

use crate::graph::{Intern, Vertex};

pub const INPUT_FILE: &str = include_str!("../../2023-25.input");
pub const CONTROL_1: &str = r#"jqt: rhn xhk nvd
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

pub fn parse_line(line: &str, intern: &mut Intern) -> (Vertex, BTreeSet<Vertex>) {
    let mut split = line.splitn(2, ": ");
    let left = split.next().unwrap();
    let right = split.next().unwrap().split(' ');
    (
        intern.intern(left),
        right.map(|x| intern.intern(x)).collect(),
    )
}
