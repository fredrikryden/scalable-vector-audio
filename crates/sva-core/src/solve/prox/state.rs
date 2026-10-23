// Concern: a composition's varied numbers as one vector: node defaults as centres, line fields as deviations or their own | Non-concern: pricing, stepping | IO: (Model, varies) -> State -> scopes

use std::collections::BTreeMap;

use crate::cli_error::CliError;
use crate::model::Model;
use crate::solve::Vary;

#[derive(Clone, Debug)]
pub struct Var {
    pub vary: Vary,
    pub lo: f64,
    pub hi: f64,
    pub log: bool,
    /// About a JND of what the number moves, in its coordinate.
    pub res: f64,
    pub unit: &'static str,
}

impl Var {
    pub fn of(v: &Vary) -> Result<Var, CliError> {
        let res = v.resolution().ok_or_else(|| {
            CliError::usage(
                "solve.prox_needs_step",
                format!(
                    "`--vary {}` gives no `/<step>`; under `--method prox` a step is the number's \
                     resolution, about a JND of what it moves, as `/1.22,log` or `/0.5`",
                    v.name
                ),
            )
        })?;
        let (lo, hi) = v.bounds();
        Ok(Var {
            vary: v.clone(),
            lo,
            hi,
            log: v.log,
            res,
            unit: v.rendered_unit(),
        })
    }

    pub fn matches(&self, path: &str, name: &str) -> bool {
        self.vary.names(path, name)
    }

    pub fn bits(&self) -> f64 {
        self.vary.bits()
    }

    pub fn value(&self, coord: f64) -> f64 {
        self.vary.value(coord)
    }

    /// Zero inside linear bounds is the number's absence.
    pub fn zero_free(&self) -> bool {
        !self.log && self.lo <= 0.0 && self.hi >= 0.0
    }

    pub fn text(&self, value: f64) -> String {
        let v = format!("{:.6e}", value)
            .parse::<f64>()
            .expect("a number prints back");
        let s = format!("{v}");
        let s = match s.as_str() {
            "-0" => "0".to_string(),
            _ => s,
        };
        format!("{s}{}", self.unit)
    }
}

/// Refuses a `--vary` that names no node default and no score field `model` plays.
pub fn reaches(model: &Model, vars: &[Var]) -> Result<(), CliError> {
    let score = model.score.clone().unwrap_or_default();
    let reached = |v: &Var| {
        model
            .nodes
            .iter()
            .any(|(p, n)| n.defaults.iter().any(|(d, _)| v.matches(p, d)))
            || model
                .lines
                .iter()
                .any(|l| l.fields.iter().any(|(f, _)| v.matches(&score, f)))
    };
    match vars.iter().find(|v| !reached(v)) {
        Some(v) => Err(CliError::usage(
            "solve.reaches_nothing",
            format!(
                "`--vary {}` names no node default and no score field of the composition \
                 `@{}` plays",
                v.vary.name, model.root
            ),
        )),
        None => Ok(()),
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum Role {
    Centre {
        node: usize,
        name: String,
    },
    Dev {
        line: usize,
        name: String,
    },
    Own {
        line: usize,
        name: String,
        root: bool,
    },
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Slot {
    Fused { centre: usize, dev: usize },
    Own(usize),
    Fixed(f64),
}

#[derive(Clone)]
pub struct State {
    pub vars: Vec<Var>,
    pub x: Vec<f64>,
    pub roles: Vec<Role>,
    pub var_of: Vec<usize>,
    pub nodes: Vec<String>,
    pub model: Model,
    pub centre: BTreeMap<(usize, String), usize>,
    pub dev: BTreeMap<(usize, String), usize>,
    pub own: BTreeMap<(usize, String), usize>,
    pub root: Vec<Option<usize>>,
}

impl State {
    pub fn of(model: &Model, vars: &[Var]) -> Result<State, CliError> {
        let var_for = |path: &str, name: &str| vars.iter().position(|v| v.matches(path, name));
        let score = model.score.clone().unwrap_or_default();
        let nodes: Vec<String> = model.nodes.keys().cloned().collect();
        let mut s = State {
            vars: vars.to_vec(),
            x: Vec::new(),
            roles: Vec::new(),
            var_of: Vec::new(),
            nodes: nodes.clone(),
            model: model.clone(),
            centre: BTreeMap::new(),
            dev: BTreeMap::new(),
            own: BTreeMap::new(),
            root: vec![None; model.lines.len()],
        };
        for (n, path) in nodes.iter().enumerate() {
            for (name, value) in &model.nodes[path].defaults {
                if let Some(v) = var_for(path, name) {
                    let c = vars[v].vary.coord(*value);
                    s.push(
                        c,
                        v,
                        Role::Centre {
                            node: n,
                            name: name.clone(),
                        },
                    );
                    s.centre.insert((n, name.clone()), s.x.len() - 1);
                }
            }
        }
        for (i, line) in model.lines.iter().enumerate() {
            let node = nodes
                .iter()
                .position(|p| *p == line.node)
                .expect("a line's node");
            let defaults = &model.nodes[&line.node].defaults;
            for (name, default) in defaults {
                let Some(v) = var_for(&line.node, name) else {
                    if let Some(v) = var_for(&score, name) {
                        let written = line.fields.iter().find(|(f, _)| f == name);
                        let value = written.map_or(*default, |(_, x)| *x);
                        s.push(
                            vars[v].vary.coord(value),
                            v,
                            Role::Own {
                                line: i,
                                name: name.clone(),
                                root: false,
                            },
                        );
                        s.own.insert((i, name.clone()), s.x.len() - 1);
                    }
                    continue;
                };
                if model.score.is_none() {
                    continue;
                }
                let c = s.x[s.centre[&(node, name.clone())]];
                let d = line
                    .fields
                    .iter()
                    .find(|(f, _)| f == name)
                    .map_or(0.0, |(_, value)| vars[v].vary.coord(*value) - c);
                s.push(
                    d,
                    v,
                    Role::Dev {
                        line: i,
                        name: name.clone(),
                    },
                );
                s.dev.insert((i, name.clone()), s.x.len() - 1);
            }
            for (name, value) in &line.fields {
                if defaults.iter().any(|(d, _)| d == name) {
                    continue;
                }
                let Some(v) = var_for(&score, name) else {
                    continue;
                };
                let root = !vars[v].log && vars[v].lo == 0.0 && s.root[i].is_none();
                s.push(
                    vars[v].vary.coord(*value),
                    v,
                    Role::Own {
                        line: i,
                        name: name.clone(),
                        root,
                    },
                );
                s.own.insert((i, name.clone()), s.x.len() - 1);
                if root {
                    s.root[i] = Some(s.x.len() - 1);
                }
            }
        }
        Ok(s)
    }

    fn push(&mut self, x: f64, var: usize, role: Role) {
        self.x.push(x);
        self.var_of.push(var);
        self.roles.push(role);
    }

    pub fn node_index(&self, path: &str) -> usize {
        self.nodes.iter().position(|p| p == path).expect("a node")
    }

    pub fn line_node(&self, line: usize) -> usize {
        self.node_index(&self.model.lines[line].node)
    }

    pub fn alive(&self, line: usize) -> bool {
        self.root[line].is_none_or(|r| self.x[r] > 0.0)
    }

    /// The names a line's node reads, each with where its value comes from.
    pub fn slots(&self, line: usize) -> Vec<(String, Slot)> {
        let l = &self.model.lines[line];
        let n = self.line_node(line);
        let mut out = Vec::new();
        for (name, value) in &self.model.nodes[&l.node].defaults {
            let slot = match (
                self.centre.get(&(n, name.clone())),
                self.dev.get(&(line, name.clone())),
            ) {
                (Some(c), Some(d)) => Slot::Fused {
                    centre: *c,
                    dev: *d,
                },
                (Some(c), None) if self.model.score.is_none() => Slot::Own(*c),
                _ if self.own.contains_key(&(line, name.clone())) => {
                    Slot::Own(self.own[&(line, name.clone())])
                }
                _ => Slot::Fixed(
                    l.fields
                        .iter()
                        .find(|(f, _)| f == name)
                        .map_or(*value, |(_, v)| *v),
                ),
            };
            out.push((name.clone(), slot));
        }
        for (name, value) in &l.fields {
            if out.iter().any(|(n, _)| n == name) {
                continue;
            }
            let slot = match self.own.get(&(line, name.clone())) {
                Some(o) => Slot::Own(*o),
                None => Slot::Fixed(*value),
            };
            out.push((name.clone(), slot));
        }
        out
    }

    /// The root node's names and their centres.
    pub fn root_slots(&self) -> Vec<(String, Slot)> {
        let n = self.node_index(&self.model.root);
        self.model.nodes[&self.model.root]
            .defaults
            .iter()
            .map(|(name, value)| {
                let slot = match self.centre.get(&(n, name.clone())) {
                    Some(c) => Slot::Own(*c),
                    None => Slot::Fixed(*value),
                };
                (name.clone(), slot)
            })
            .collect()
    }

    pub fn coord_of(&self, slot: Slot) -> Option<f64> {
        match slot {
            Slot::Fused { centre, dev } => Some(self.x[centre] + self.x[dev]),
            Slot::Own(o) => Some(self.x[o]),
            Slot::Fixed(_) => None,
        }
    }

    pub fn var_of_slot(&self, slot: Slot) -> Option<&Var> {
        match slot {
            Slot::Fused { centre, .. } => Some(&self.vars[self.var_of[centre]]),
            Slot::Own(o) => Some(&self.vars[self.var_of[o]]),
            Slot::Fixed(_) => None,
        }
    }

    /// The bounds coordinate `i` is held in.
    pub fn bounds(&self, i: usize) -> (f64, f64) {
        let var = &self.vars[self.var_of[i]];
        (var.lo, var.hi)
    }

    /// The value a slot holds, clamped into its bounds.
    pub fn value(&self, slot: Slot) -> f64 {
        match slot {
            Slot::Fixed(v) => v,
            _ => {
                let var = self.var_of_slot(slot).expect("a varied slot");
                var.value(
                    self.coord_of(slot)
                        .expect("a coordinate")
                        .clamp(var.lo, var.hi),
                )
            }
        }
    }
}
