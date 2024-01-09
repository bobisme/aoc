#![allow(unused)]
use std::collections::{BTreeSet, BinaryHeap, HashMap};

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
    set: BTreeSet<String>,
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
