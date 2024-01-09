use indicatif::ProgressBar;
use std::{
    cmp,
    collections::{HashMap, HashSet, VecDeque},
    rc::Rc,
    u64,
};

#[allow(dead_code)]
const INPUT_FILE: &str = include_str!("../../2023-20.input");
#[allow(dead_code)]
const CONTROL_1: &str = r#"broadcaster -> a, b, c
%a -> b
%b -> c
%c -> inv
&inv -> a"#;
#[allow(dead_code)]
const CONTROL_2: &str = r#"broadcaster -> a
%a -> inv, con
&inv -> b
%b -> con
&con -> output"#;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Pulse {
    High,
    Low,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum State {
    Pass,
    /// Flip-flop modules (prefix %) are either on or off; they are initially off. If a
    /// flip-flop module receives a high pulse, it is ignored and nothing happens. However, if a
    /// flip-flop module receives a low pulse, it flips between on and off. If it was off, it turns
    /// on and sends a high pulse. If it was on, it turns off and sends a low pulse.
    FlipFlop(bool),
    /// Conjunction modules (prefix &) remember the type of the most recent pulse received from
    /// each of their connected input modules; they initially default to remembering a low pulse
    /// for each input. When a pulse is received, the conjunction module first updates its memory
    /// for that input. Then, if it remembers high pulses for all inputs, it sends a low pulse;
    /// otherwise, it sends a high pulse.
    Conj {
        in_mem: HashMap<String, Pulse>,
    },
}

impl State {
    fn pulse(&mut self, input: &str, p: Pulse) -> Option<Pulse> {
        match self {
            State::Pass => Some(p),
            State::FlipFlop(s) => {
                if let Pulse::High = p {
                    return None;
                }
                *s = !*s;
                match s {
                    true => Some(Pulse::High),
                    _ => Some(Pulse::Low),
                }
            }
            State::Conj { in_mem } => {
                // dbg!(&input);
                *in_mem.get_mut(input).unwrap() = p;
                if in_mem.iter().all(|(k, v)| *v == Pulse::High) {
                    Some(Pulse::Low)
                } else {
                    Some(Pulse::High)
                }
            }
        }
    }

    fn reset(&mut self) {
        match self {
            State::Pass => {}
            State::FlipFlop(is_on) => *is_on = false,
            State::Conj { in_mem } => {
                in_mem.iter_mut().for_each(|(k, v)| *v = Pulse::Low);
            }
        }
    }
}

#[derive(Default, Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct Stats {
    highs: i32,
    lows: i32,
}

impl std::ops::Add for Stats {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self {
            highs: self.highs + rhs.highs,
            lows: self.lows + rhs.lows,
        }
    }
}
impl std::iter::Sum for Stats {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Stats::default(), |acc, x| acc + x)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct Mod {
    id: String,
    state: State,
    outs: Vec<String>,
}

#[derive(Debug, Clone)]
struct Modules(HashMap<String, Mod>);

impl Modules {
    fn reset(&mut self) {
        for (_, module) in self.iter_mut() {
            module.state.reset();
        }
    }

    fn press_button(&mut self, dst: &str) -> bool {
        // self.reset();
        let mut q = VecDeque::from(vec![(
            Some(Pulse::Low),
            String::from("button"),
            String::from("broadcaster"),
        )]);
        while let Some((Some(pulse), from, module_name)) = q.pop_front() {
            // dbg!((pulse, &module_name));
            if pulse == Pulse::Low && module_name == dst {
                // dbg!(&module_name);
                return true;
            }
            let m = self.get_mut(&module_name);
            if m.is_none() {
                continue;
            }
            let module = m.unwrap();
            let pulse = module.state.pulse(&from, pulse);
            if pulse.is_some() {
                for out in module.outs.iter() {
                    // println!("{module_name} -{pulse:?}-> {out}");
                    q.push_back((pulse, module_name.clone(), out.clone()));
                }
            }
        }
        false
    }
}

impl std::ops::Deref for Modules {
    type Target = HashMap<String, Mod>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl std::ops::DerefMut for Modules {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

fn parse_line(line: &str) -> Mod {
    let (name, outs) = line.split_once(" -> ").unwrap();
    let outs = outs.split(", ").map(|x| x.to_string()).collect();
    let state = match name.chars().next().unwrap() {
        '%' => State::FlipFlop(false),
        '&' => State::Conj {
            in_mem: HashMap::new(),
        },
        _ => State::Pass,
    };
    Mod {
        id: name.trim_start_matches(['%', '&']).to_string(),
        state,
        outs,
    }
}

fn parse(input: impl Iterator<Item = &'static str>) -> Modules {
    let mut out = HashMap::new();
    let mods = input.map(parse_line);
    for module in mods {
        out.insert(module.id.clone(), module);
    }
    out.insert(
        "output".to_string(),
        Mod {
            id: "output".to_string(),
            state: State::Pass,
            outs: vec![],
        },
    );
    let conjuctions: HashSet<String> = out
        .iter_mut()
        .filter(|(_, v)| matches!(v.state, State::Conj { in_mem: _ }))
        .map(|(k, _)| k.clone())
        .collect();
    let inputs: Vec<_> = out
        .iter()
        .map(|(mod_name, mod_)| {
            (
                mod_name.clone(),
                mod_.outs
                    .iter()
                    .filter(|x| conjuctions.contains(*x))
                    .cloned()
                    .collect::<Vec<_>>(),
            )
        })
        .collect();
    for (input, outs) in inputs {
        for o in outs {
            let mod_ = out.get_mut(&o).unwrap();
            if let State::Conj { in_mem } = &mut mod_.state {
                in_mem.insert(input.clone(), Pulse::Low);
            }
        }
    }
    Modules(out)
}

fn get_inputs(mods: &Modules, target: &str) -> Vec<String> {
    mods.iter()
        .filter_map(|(mod_name, mod_)| {
            let to_target = mod_.outs.iter().any(|x| *x == target);
            if to_target {
                return Some(mod_name.clone());
            }
            None
        })
        .collect()
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct Node {
    id: String,
    deps: HashSet<Rc<Node>>,
}

impl std::hash::Hash for Node {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.id.hash(state);
    }
}

impl std::fmt::Display for Node {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "[{}]<-{}",
            self.id,
            self.deps
                .iter()
                .map(|d| d.id.clone())
                .collect::<Vec<_>>()
                .join(",")
        )
    }
}

fn build_graph(
    modules: &Modules,
    dep_map: &mut HashMap<String, HashSet<String>>,
    visited: &mut HashSet<String>,
    id: &str,
) {
    if visited.contains(id) {
        return;
    }
    visited.insert(id.to_owned());
    let inputs = HashSet::from_iter(get_inputs(modules, id));
    for input in inputs.iter() {
        if visited.contains(input) {
            continue;
        }
        build_graph(modules, dep_map, visited, input);
    }
    dep_map.insert(id.to_string(), inputs);
}

fn toposort(
    dep_map: &HashMap<String, HashSet<String>>,
    visited: &mut HashSet<String>,
    id: &str,
    list: &mut Vec<String>,
) {
    visited.insert(id.to_string());
    let deps = dep_map.get(id).unwrap();
    for dep in deps {
        if visited.contains(dep) {
            continue;
        }
        toposort(dep_map, visited, dep, list);
    }
    list.push(id.to_string());
}

fn gcd(a: u64, b: u64) -> u64 {
    let mut a = a;
    let mut b = b;
    while b != 0 {
        let temp = b;
        b = a % b;
        a = temp;
    }
    a
}
fn lcm(a: u64, b: u64) -> u64 {
    (a * b) / gcd(a, b)
}

fn calc_pushes_to_low_out(
    modules: &Modules,
    dep_map: &HashMap<String, HashSet<String>>,
    visited: &mut HashSet<String>,
    cache: &mut HashMap<String, u64>,
    id: &str,
) -> u64 {
    visited.insert(id.to_string());
    if cache.contains_key(id) {
        return *cache.get(id).unwrap();
    }
    let mod_ = modules.get(id).unwrap();
    let deps = dep_map.get(id).unwrap();
    let out = if deps.is_empty() {
        match &mod_.state {
            State::Pass => 1,
            State::FlipFlop(_) => 2,
            _ => unreachable!(),
        }
    } else {
        let dep_counts = deps.iter().filter_map(|d| {
            if visited.contains(d) {
                return None;
            }
            Some(calc_pushes_to_low_out(modules, dep_map, visited, cache, d))
        });
        match &mod_.state {
            State::Pass => 1,
            State::FlipFlop(_) => 2 * dep_counts.into_iter().fold(u64::MAX, cmp::min),
            State::Conj { in_mem: _ } => dep_counts.fold(1, lcm),
        }
    };
    cache.insert(id.to_string(), out);
    out
}
fn calc_pushes_to_high_out(
    modules: &Modules,
    dep_map: &HashMap<String, HashSet<String>>,
    visited: &mut HashSet<String>,
    cache: &mut HashMap<String, u64>,
    id: &str,
) -> u64 {
    visited.insert(id.to_string());
    if cache.contains_key(id) {
        return *cache.get(id).unwrap();
    }
    let mod_ = modules.get(id).unwrap();
    let deps = dep_map.get(id).unwrap();
    if deps.is_empty() {
        match &mod_.state {
            State::Pass => 0,
            State::FlipFlop(_) => 1,
            _ => unreachable!(),
        }
    } else {
        let dep_counts = deps.iter().filter_map(|d| {
            if visited.contains(d) {
                return None;
            }
            Some(calc_pushes_to_high_out(modules, dep_map, visited, cache, d))
        });
        match &mod_.state {
            State::Pass => 1,
            State::FlipFlop(_) => 2 * dep_counts.into_iter().fold(u64::MAX, cmp::min),
            State::Conj { in_mem } => dep_counts.into_iter().fold(u64::MAX, cmp::min),
        }
    }
}
fn calc_pushes_to_low_in(
    modules: &Modules,
    dep_map: &HashMap<String, HashSet<String>>,
    id: &str,
) -> u64 {
    let mut cache = HashMap::new();
    let mut visited = HashSet::new();
    let dep_counts = dep_map
        .get(id)
        .unwrap()
        .iter()
        .map(|d| calc_pushes_to_low_out(modules, dep_map, &mut visited, &mut cache, d));
    let mod_ = modules.get(id).unwrap();
    match &mod_.state {
        State::Pass => dep_counts.into_iter().fold(1, cmp::min),
        State::FlipFlop(_) => dep_counts.into_iter().fold(1, cmp::min),
        State::Conj { in_mem: _ } => dep_map
            .get(id)
            .unwrap()
            .iter()
            .map(|d| calc_pushes_to_high_out(modules, dep_map, &mut visited, &mut cache, d))
            .fold(1, lcm),
    }
}

fn all_the_presses(modules: &mut Modules, target: &str) -> u64 {
    (1..)
        .take_while(|_| !modules.press_button(target))
        .last()
        .unwrap_or(1)
}

fn main() {
    let start = std::time::Instant::now();
    let mut modules = parse(INPUT_FILE.lines());
    modules.insert(
        "rx".to_string(),
        Mod {
            id: "rx".to_string(),
            state: State::Pass,
            outs: vec![],
        },
    );
    let mut visited = HashSet::new();
    let mut dep_map = HashMap::new();
    build_graph(&modules, &mut dep_map, &mut visited, "rx");
    drop(visited);
    dbg!(&dep_map.get("rs"));
    // return;
    let mut sorted = Vec::new();
    let mut visited = HashSet::new();
    toposort(&dep_map, &mut visited, "rx", &mut sorted);
    let name = "bp";
    let inputs = get_inputs(&modules, name);
    println!("[{name}]<-{{{}}}", inputs.join(", "));
    // dbg!(all_the_presses(&mut modules, name));
    let mut results = HashMap::new();
    for target in sorted.iter().take(sorted.len() - 1) {
        let measured = all_the_presses(&mut modules, target) + 1;
        modules.reset();
        results.insert(target.to_string(), measured);
        println!("run {target} = {measured:?}");
    }
    let rs_deps = get_inputs(&modules, "rs");
    let rs = rs_deps
        .iter()
        .filter_map(|x| results.get(x))
        .inspect(|x| {
            dbg!(x);
        })
        .cloned()
        .fold(1, lcm);
    dbg!(rs);
    println!("time: {:?}", start.elapsed());
}
