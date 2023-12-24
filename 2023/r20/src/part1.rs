use std::collections::{HashMap, HashSet, VecDeque};

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
        if let Self::Conj { in_mem } = self {
            in_mem.iter_mut().for_each(|(k, v)| *v = Pulse::Low);
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

    fn press_button(&mut self) -> Stats {
        // self.reset();
        let mut q = VecDeque::from(vec![(
            Some(Pulse::Low),
            String::from("button"),
            String::from("broadcaster"),
        )]);
        let mut highs = 0;
        let mut lows = 0;
        while let Some((Some(pulse), from, module_name)) = q.pop_front() {
            match pulse {
                Pulse::High => highs += 1,
                Pulse::Low => lows += 1,
                _ => {}
            }
            let m = self.get_mut(&module_name);
            if m.is_none() {
                continue;
            }
            let module = m.unwrap();
            let pulse = module.state.pulse(&from, pulse);
            if pulse.is_some() {
                for out in module.outs.iter() {
                    println!("{module_name} -{pulse:?}-> {out}");
                    q.push_back((pulse, module_name.clone(), out.clone()));
                }
            }
        }
        Stats { highs, lows }
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

fn main() {
    let mut modules = parse(INPUT_FILE.lines());
    dbg!(&modules);
    let stats: Stats = (0..1_000).map(|_| modules.press_button()).sum();
    // let (high, low) = ;
    dbg!(stats);
    let val = stats.highs * stats.lows;
    println!("end value = {val}");
    // modules.reset();
    // let broadcaster = modules.get_mut("broadcaster").unwrap();
    // dbg!(broadcaster.state.pulse(Pulse::High));
    // let con = modules.get_mut("con").unwrap();
    // dbg!(con.state.pulse(Pulse::High));
}
