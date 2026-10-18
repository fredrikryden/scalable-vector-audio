// Concern: checks each score line against the node it plays | Non-concern: reading a score (score.rs), lowering its degrees and bars (reserved.rs) | IO: (&Graph[, beside]) -> Refusals

use crate::graph::{Graph, resolve_ref_path};
use crate::parameters::free_parameters;
use crate::refusal::Refusal;
use crate::score::{Event, Score, refused};

impl Graph {
    /// Every score line here against each node it plays, here or in `beside`.
    pub fn check_scores(&self, beside: Option<&Graph>) -> Result<(), Vec<Refusal>> {
        let refusals: Vec<Refusal> = self
            .paths()
            .filter_map(|path| Some((path, self.score(path)?)))
            .flat_map(|(path, score)| self.checked(path, score, beside))
            .collect();
        match refusals.is_empty() {
            true => Ok(()),
            false => Err(refusals),
        }
    }

    /// Each line's first event per node it plays against that node: a line's events bind
    /// the same keys, so one refusal per line and node says all there is.
    fn checked(&self, path: &str, score: &Score, beside: Option<&Graph>) -> Vec<Refusal> {
        let text = self.text(path).expect("every node holds its text");
        let mut refusals = Vec::new();
        let mut seen: Vec<(usize, &str)> = Vec::new();
        for event in &score.events {
            let key = (event.line, event.node.as_str());
            if seen.last().is_some_and(|(line, _)| *line != event.line) {
                seen.clear();
            }
            if seen.contains(&key) {
                continue;
            }
            seen.push(key);
            let Some((key, message)) = self.unread(path, event, beside) else {
                continue;
            };
            let line = &text[event.span.start..event.span.end];
            let column = line.find(&format!("\"{key}\"")).map_or(1, |at| at + 1);
            let diag = refused(
                event.span.start + column - 1,
                (event.line, column),
                &message,
            );
            refusals.push(Refusal::new(path, diag.span, diag.code, diag.message));
        }
        refusals
    }

    /// The first key this line binds that its node does not read, or the first parameter the
    /// node reads that neither the line nor a default of the node binds.
    fn unread(
        &self,
        path: &str,
        event: &Event,
        beside: Option<&Graph>,
    ) -> Option<(String, String)> {
        let target = resolve_ref_path(path, &event.node)?;
        let defined = self
            .defined(&target)
            .or_else(|| beside.and_then(|b| b.defined(&target)))?;
        let is_node = |name: &str| {
            resolve_ref_path(&target, name)
                .is_some_and(|n| self.defines(&n) || beside.is_some_and(|b| b.defines(&n)))
        };
        let mut reads: Vec<String> = defined.defaults.iter().map(|(k, _)| k.clone()).collect();
        reads.extend(free_parameters(
            &defined.body,
            &defined.defaults,
            &[],
            is_node,
        ));
        let mut given: Vec<(&str, &str)> = event
            .args
            .iter()
            .map(|(k, _)| (k.as_str(), k.as_str()))
            .collect();
        if event.deg.is_some() {
            given.push(("f0", "deg"));
        }
        let reads_list = reads.join("`, `");
        if let Some((name, written)) = given
            .iter()
            .find(|(name, _)| !reads.iter().any(|r| r == name))
        {
            let set = match name == written {
                true => format!("`{name}`"),
                false => format!("`{written}`, which sets `{name}`,"),
            };
            return Some((
                written.to_string(),
                format!(
                    "{set} is no parameter `@{}` reads; it reads `{reads_list}`",
                    event.node
                ),
            ));
        }
        let names: Vec<&str> = given.iter().map(|(name, _)| *name).collect();
        let unbound = free_parameters(&defined.body, &defined.defaults, &names, is_node);
        let first = unbound.first()?;
        Some((
            "node".to_string(),
            format!(
                "`@{}` reads `{first}`, which neither this line nor a default line of it binds",
                event.node
            ),
        ))
    }
}
