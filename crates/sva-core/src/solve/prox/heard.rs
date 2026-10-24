// Concern: what a priced fit tells its caller as it runs: steps, the activity and work, stages listed, a refusal's activity | Non-concern: the JSON (fit.rs) | IO: (events, a clock) -> lines heard

use std::cell::RefCell;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use super::eval::Residual;
use super::{Progress, Solution};
use crate::cli_error::CliError;

/// A cost bound: each activity line is written and handed to the caller.
pub const DEFAULT_HEARTBEAT_SECS: f64 = 1.0;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Action {
    Seeding,
    ReadingReference,
    RenderingNoise,
    Descending,
    Debiasing,
    RankingProposals,
    TryingProposal,
    EvaluatingAtCap,
    Measuring,
}

impl Action {
    pub fn name(self) -> &'static str {
        match self {
            Action::Seeding => "seeding",
            Action::ReadingReference => "reading_reference",
            Action::RenderingNoise => "rendering_noise",
            Action::Descending => "descending",
            Action::Debiasing => "debiasing",
            Action::RankingProposals => "ranking_proposals",
            Action::TryingProposal => "trying_proposal",
            Action::EvaluatingAtCap => "evaluating_at_cap",
            Action::Measuring => "measuring",
        }
    }

    pub fn unit(self) -> &'static str {
        match self {
            Action::RenderingNoise => "render",
            Action::Descending | Action::Debiasing => "step",
            Action::RankingProposals => "proposal",
            Action::TryingProposal => "trial",
            Action::Measuring => "draw",
            Action::Seeding | Action::ReadingReference | Action::EvaluatingAtCap => "",
        }
    }
}

pub struct Act {
    action: Action,
    of: Option<usize>,
    done: usize,
    edit: Option<String>,
    line: Option<String>,
}

impl Act {
    pub fn new(action: Action) -> Act {
        Act {
            action,
            of: None,
            done: 0,
            edit: None,
            line: None,
        }
    }

    pub fn of(self, n: usize) -> Act {
        Act {
            of: Some(n),
            ..self
        }
    }

    pub fn from(self, done: usize) -> Act {
        Act { done, ..self }
    }

    pub fn edit(self, label: &str, line: String) -> Act {
        Act {
            edit: Some(label.to_string()),
            line: Some(line),
            ..self
        }
    }
}

struct Frame {
    act: Act,
    done: usize,
}

#[derive(Clone, Debug)]
pub struct Doing {
    pub action: Action,
    pub stage: Option<String>,
    pub edit: Option<String>,
    pub count: Option<(usize, usize)>,
    pub part: Option<(usize, usize)>,
    pub comparisons: u64,
    /// Comparisons a second over the last window at least the heartbeat long; none before one.
    pub per_second: Option<f64>,
    pub seconds: f64,
}

pub enum Heard<'a> {
    Step(&'a str, &'a Progress<'a>),
    Doing(&'a Doing),
    Listed {
        label: &'a str,
        solution: &'a Solution,
        at_cap: Option<&'a Residual>,
        length: Option<f64>,
        files: &'a [(String, String)],
        least: bool,
    },
    /// None where the refusal came before any activity.
    Failed {
        doing: Option<&'a Doing>,
        code: &'static str,
    },
}

struct At {
    stage: Option<String>,
    part: Option<(usize, usize)>,
    frames: Vec<Frame>,
    began: f64,
    last: f64,
    window: (f64, u64),
    per_second: Option<f64>,
    /// The innermost activity an error left since the last one began.
    failed: Option<Doing>,
}

pub type Report<'a> = &'a dyn Fn(&Heard);
/// Seconds.
pub type Clock<'a> = &'a dyn Fn() -> f64;

pub struct Hearing<'a> {
    report: Report<'a>,
    clock: Clock<'a>,
    every: f64,
    pub work: Arc<AtomicU64>,
    at: RefCell<At>,
}

impl<'a> Hearing<'a> {
    pub fn new(report: Report<'a>, clock: Clock<'a>, every: f64) -> Hearing<'a> {
        let began = clock();
        Hearing {
            report,
            clock,
            every,
            work: Arc::default(),
            at: RefCell::new(At {
                stage: None,
                part: None,
                frames: Vec::new(),
                began,
                last: began,
                window: (began, 0),
                per_second: None,
                failed: None,
            }),
        }
    }

    pub fn heard(&self, heard: &Heard) {
        (self.report)(heard);
    }

    pub fn staged(&self, label: &str, part: (usize, usize)) {
        let mut at = self.at.borrow_mut();
        at.stage = Some(label.to_string());
        at.part = Some(part);
    }

    pub fn step(&self, p: &Progress) {
        let label = {
            let at = self.at.borrow();
            at.frames
                .iter()
                .rev()
                .find_map(|f| f.act.line.clone())
                .or_else(|| at.stage.clone())
                .unwrap_or_default()
        };
        self.heard(&Heard::Step(&label, p));
    }

    fn doing(&self, now: f64) -> Option<Doing> {
        let mut at = self.at.borrow_mut();
        let comparisons = self.work.load(Ordering::Relaxed);
        let (then, before) = at.window;
        if now > then && now - then >= self.every {
            at.per_second = Some((comparisons - before) as f64 / (now - then));
            at.window = (now, comparisons);
        }
        let top = at.frames.last()?;
        Some(Doing {
            action: top.act.action,
            stage: at.stage.clone(),
            edit: at.frames.iter().rev().find_map(|f| f.act.edit.clone()),
            count: top.act.of.map(|n| (top.done, n)),
            part: at.part,
            comparisons,
            per_second: at.per_second,
            seconds: now - at.began,
        })
    }

    fn told(&self, now: f64) {
        if let Some(d) = self.doing(now) {
            self.at.borrow_mut().last = now;
            self.heard(&Heard::Doing(&d));
        }
    }

    fn begin(&self, act: Act) {
        {
            let mut at = self.at.borrow_mut();
            at.failed = None;
            let done = act.done;
            at.frames.push(Frame { act, done });
        }
        self.told((self.clock)());
    }

    fn end(&self, failed: bool) {
        let now = (self.clock)();
        if failed && self.at.borrow().failed.is_none() {
            let d = self.doing(now);
            self.at.borrow_mut().failed = d;
        }
        self.at.borrow_mut().frames.pop();
        if !failed {
            self.told(now);
        }
    }

    /// Heard once `every` has passed since the last line, so between units, never inside one.
    pub fn tick(&self, done: usize) {
        if let Some(top) = self.at.borrow_mut().frames.last_mut() {
            top.done = done;
        }
        let now = (self.clock)();
        if now - self.at.borrow().last >= self.every {
            self.told(now);
        }
    }

    pub async fn during<T>(
        &self,
        act: Act,
        run: impl Future<Output = Result<T, CliError>>,
    ) -> Result<T, CliError> {
        self.begin(act);
        let out = run.await;
        self.end(out.is_err());
        out
    }

    pub fn within<T>(
        &self,
        act: Act,
        run: impl FnOnce() -> Result<T, CliError>,
    ) -> Result<T, CliError> {
        self.begin(act);
        let out = run();
        self.end(out.is_err());
        out
    }

    pub fn failed(&self, error: &CliError) {
        let doing = self.at.borrow_mut().failed.take();
        self.heard(&Heard::Failed {
            doing: doing.as_ref(),
            code: error.code(),
        });
    }
}
