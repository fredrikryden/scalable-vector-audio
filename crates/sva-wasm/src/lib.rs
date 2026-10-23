// Concern: the JS surface — a composition a page fills node by node, renders, streams, persists | Non-concern: the pipeline, the store's medium | IO: (path, text) -> samples, blocks, counters

//! A page holds no directory: nodes arrive one at a time, so a composition is BUILT rather
//! than read. A refusal crosses as a thrown `Error`: `name` is the CLI's error code,
//! `refusal` the envelope it would print.

mod analyze;
mod cost;
mod describe;
mod opfs;
mod seed;
mod stats;
mod threads;

pub use analyze::analyze;
pub use describe::{outline, readings};
pub use opfs::DirectoryHandle;
pub use seed::seed;
pub use stats::{subscribe, unsubscribe};
pub use threads::start_threads;

use std::borrow::Cow;
use std::cell::{Cell, RefCell};
use std::num::NonZeroUsize;
use std::pin::Pin;
use std::rc::Rc;
use std::sync::Arc;

use sva_core::{
    Asked, CliError, Diagnostic, Job, Out, Printed, Rendered, Report, Representation, SAMPLE_LIMIT,
    counters_json, error_envelope, execute_over, query_data, stats_json, stream_stats_json,
    work_json,
};
use sva_engine::{
    Abandon, BootPhase, CacheStats, DEFAULT_STORE_BYTES, EngineError, Handle, Never, Placed,
    Session, Store, StoreBoot, Tier,
};
use wasm_bindgen::prelude::wasm_bindgen;
use wasm_bindgen::{JsCast, JsValue};

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_name = Error)]
    type JsErr;

    #[wasm_bindgen(constructor, js_class = "Error")]
    fn new(message: &str) -> JsErr;

    #[wasm_bindgen(method, setter)]
    fn set_name(this: &JsErr, name: &str);

    #[wasm_bindgen(method, setter)]
    fn set_refusal(this: &JsErr, refusal: JsValue);

    #[wasm_bindgen(js_namespace = JSON, catch)]
    fn parse(text: &str) -> Result<JsValue, JsValue>;

    #[wasm_bindgen(js_namespace = console)]
    fn warn(message: &str);

    type MessageChannel;

    #[wasm_bindgen(constructor)]
    fn new() -> MessageChannel;

    #[wasm_bindgen(method, getter)]
    fn port1(this: &MessageChannel) -> MessagePort;

    #[wasm_bindgen(method, getter)]
    fn port2(this: &MessageChannel) -> MessagePort;

    type MessagePort;

    #[wasm_bindgen(method, setter)]
    fn set_onmessage(this: &MessagePort, handler: &js_sys::Function);

    #[wasm_bindgen(method, js_name = postMessage)]
    fn post_message(this: &MessagePort, message: &JsValue);

    #[wasm_bindgen(method)]
    fn close(this: &MessagePort);
}

/// Resolves once the page's queued tasks ran, a message an abort sends among them.
async fn yielded() {
    let promise = js_sys::Promise::new(&mut |resolve, _| {
        let channel = MessageChannel::new();
        let (port, other) = (channel.port1(), channel.port2());
        let ends: [MessagePort; 2] = [
            port.clone().unchecked_into(),
            other.clone().unchecked_into(),
        ];
        let done = wasm_bindgen::closure::Closure::once_into_js(move || {
            ends.iter().for_each(MessagePort::close);
            let _ = resolve.call0(&JsValue::UNDEFINED);
        });
        port.set_onmessage(done.unchecked_ref());
        other.post_message(&JsValue::UNDEFINED);
    });
    let _ = wasm_bindgen_futures::JsFuture::from(promise).await;
}

/// A page's `AbortSignal`, read between blocks after the page has had its turn.
struct Signal(JsValue);

impl Abandon for Signal {
    fn abandoned(&self) -> Pin<Box<dyn Future<Output = bool> + '_>> {
        Box::pin(async move {
            yielded().await;
            aborted(&self.0)
        })
    }
}

fn aborted(signal: &JsValue) -> bool {
    js_sys::Reflect::get(signal, &JsValue::from_str("aborted"))
        .ok()
        .and_then(|aborted| aborted.as_bool())
        .unwrap_or(false)
}

/// A failing store fails no call: why goes to the console, and memory runs alone.
fn logged(why: &str) {
    warn(&format!("sva: {why}"));
}

fn unstaged(stats: Option<&CacheStats>) {
    if let Some(why) = stats.and_then(|stats| stats.unstaged.as_deref()) {
        logged(&format!("writing what this call computed stopped: {why}"));
    }
    if let Some(why) = stats.and_then(|stats| stats.unslotted.as_deref()) {
        logged(&format!("each knob value keeps its own entry: {why}"));
    }
}

/// One crossing for every refusal: `name` is the CLI's own error code and `refusal` the
/// envelope it would have printed, never null.
fn crossed(code: &str, message: &str, diagnostics: &[Diagnostic]) -> JsValue {
    let error = JsErr::new(message);
    error.set_name(code);
    let envelope = error_envelope(code, message, diagnostics);
    error.set_refusal(parse(&envelope).unwrap_or_else(|_| JsValue::from_str(&envelope)));
    error.into()
}

fn thrown(refusal: &CliError) -> JsValue {
    crossed(refusal.code(), &refusal.message(), &refusal.diagnostics())
}

/// A refusal raised here carries its own repair: a page has no `--help`.
pub(crate) fn refuse(message: String, help: &str) -> JsValue {
    let diagnostic = Diagnostic::new("wasm.bad_argument", message.clone()).helped(help);
    crossed("validation_error", &message, &[diagnostic])
}

#[derive(Default)]
struct Options {
    rate: Option<u32>,
    bits: Option<i32>,
    until: Option<String>,
    volatile: Vec<String>,
    live: bool,
    readings: Vec<String>,
    at: Option<Placed>,
    channels: Option<usize>,
    out: Out,
    signal: Option<JsValue>,
    store_max_bytes: Option<u64>,
    threads: Option<NonZeroUsize>,
    set: Vec<(String, String)>,
    limit: Limit,
    max_renders: Option<usize>,
    step_tol: Option<f64>,
    cost_tol: Option<f64>,
    progress: Option<js_sys::Function>,
    prox: bool,
    continuation: Option<sva_core::continuation::Scheme>,
    steps: Option<usize>,
    draws: Option<usize>,
    write: bool,
}

/// Values per array: stdout's cap, another, or every one (`limit: null`).
#[derive(Clone, Copy)]
struct Limit(Option<usize>);

impl Default for Limit {
    fn default() -> Self {
        Limit(Some(SAMPLE_LIMIT))
    }
}

/// `keys` are the options this call reads; any other is refused by name.
fn options_of(options: &JsValue, keys: &[&str]) -> Result<Options, JsValue> {
    let mut held = Options::default();
    if options.is_undefined() || options.is_null() {
        return Ok(held);
    }
    let object = options.dyn_ref::<js_sys::Object>().ok_or_else(|| {
        refuse(
            "`options` is not an object".into(),
            "pass an object, such as { rate: 48000 }",
        )
    })?;
    for entry in js_sys::Object::entries(object).iter() {
        let pair = js_sys::Array::from(&entry);
        let key = pair.get(0).as_string().unwrap_or_default();
        let value = pair.get(1);
        if !keys.contains(&key.as_str()) {
            return Err(refuse(
                format!("`{key}` is no option here"),
                &format!("the options are {}", keys.join(", ")),
            ));
        }
        match key.as_str() {
            "rate" => held.rate = Some(whole(&key, &value)?),
            "bits" => held.bits = Some(whole(&key, &value)?),
            "channels" => held.channels = Some(whole(&key, &value)?),
            "store_max_bytes" => held.store_max_bytes = Some(whole(&key, &value)?),
            "threads" => {
                held.threads = Some(NonZeroUsize::new(whole(&key, &value)?).ok_or_else(|| {
                    refuse(
                        "`threads` is 0".into(),
                        "pass 1 to compute on one thread, or more",
                    )
                })?)
            }
            "limit" if value.is_null() => held.limit = Limit(None),
            "limit" => {
                let cap = whole::<usize>(&key, &value)?;
                if cap == 0 {
                    return Err(refuse(
                        "`limit` is 0".into(),
                        "pass how many values each array holds, or `limit: null` for every value",
                    ));
                }
                held.limit = Limit(Some(cap))
            }
            "until" => held.until = Some(text(&key, &value)?),
            "max_renders" => held.max_renders = Some(whole(&key, &value)?),
            "step_tol" | "cost_tol" => {
                let tol = value.as_f64().filter(|v| v.is_finite() && *v >= 0.0);
                let tol = tol.ok_or_else(|| {
                    refuse(
                        format!("`{key}` is no number at or above 0"),
                        "pass a number",
                    )
                })?;
                match key.as_str() {
                    "step_tol" => held.step_tol = Some(tol),
                    _ => held.cost_tol = Some(tol),
                }
            }
            "method" => {
                held.prox = match value.as_string().as_deref() {
                    Some("prox") => true,
                    Some("compass") => false,
                    _ => {
                        return Err(refuse(
                            "`method` is neither \"compass\" nor \"prox\"".into(),
                            "pass \"prox\" for the priced fit, or leave it out",
                        ));
                    }
                }
            }
            "continuation" => held.continuation = Some(continuation(&value)?),
            "steps" => held.steps = Some(whole(&key, &value)?),
            "draws" => held.draws = Some(whole(&key, &value)?),
            "write" => {
                held.write = value.as_bool().ok_or_else(|| {
                    refuse("`write` is not a boolean".into(), "pass true or false")
                })?
            }
            "progress" => {
                held.progress = Some(value.dyn_into::<js_sys::Function>().map_err(|_| {
                    refuse(
                        "`progress` is no function".into(),
                        "pass a function of one line",
                    )
                })?)
            }
            "live" => {
                held.live = value
                    .as_bool()
                    .ok_or_else(|| refuse("`live` is not a boolean".into(), "pass true or false"))?
            }
            "readings" => held.readings = names(&key, &value)?,
            "set" => held.set = texts(&key, &value)?,
            "at" => held.at = Some(placed(&text(&key, &value)?)?),
            "out" if value.is_null() => held.out = Out::Dropped,
            "signal" if js_sys::Reflect::has(&value, &"aborted".into()).unwrap_or(false) => {
                held.signal = Some(value)
            }
            "signal" => {
                return Err(refuse(
                    "`signal` is no AbortSignal".into(),
                    "pass an AbortController's `signal`",
                ));
            }
            "out" => {
                return Err(refuse(
                    "`out` is not null".into(),
                    "pass `out: null` to hand back no samples, or leave it out",
                ));
            }
            _ => held.volatile = names(&key, &value)?,
        }
    }
    Ok(held)
}

fn names(key: &str, value: &JsValue) -> Result<Vec<String>, JsValue> {
    let names = value.dyn_ref::<js_sys::Array>().ok_or_else(|| {
        refuse(
            format!("`{key}` is not an array"),
            "pass an array of strings",
        )
    })?;
    names
        .iter()
        .map(|name| {
            name.as_string().ok_or_else(|| {
                refuse(
                    format!("`{key}` holds something that is not a string"),
                    "pass an array of strings",
                )
            })
        })
        .collect()
}

/// `{ "key": "G3" }`: each key a reserved variable's name, each value its text.
fn texts(key: &str, value: &JsValue) -> Result<Vec<(String, String)>, JsValue> {
    let object = value.dyn_ref::<js_sys::Object>().ok_or_else(|| {
        refuse(
            format!("`{key}` is not an object"),
            "pass an object of texts, as { \"key\": \"G3\" }",
        )
    })?;
    js_sys::Object::entries(object)
        .iter()
        .map(|entry| {
            let pair = js_sys::Array::from(&entry);
            let path = pair.get(0).as_string().unwrap_or_default();
            let text = pair.get(1).as_string().ok_or_else(|| {
                refuse(
                    format!("`{key}`'s `{path}` is not a string"),
                    "pass each variable's text as a string, as its file would hold it",
                )
            })?;
            Ok((path, text))
        })
        .collect()
}

/// `{ scheme: "path", prices: [0.01, 0.1], rounds: 6 }` or `{ scheme: "grow", harmonics: 48,
/// lone: 8, rounds: 6 }`, all but `scheme` optional.
fn continuation(value: &JsValue) -> Result<sva_core::continuation::Scheme, JsValue> {
    let help = "pass { scheme: \"path\" }, with `prices` above 0 and `rounds` if wanted, or \
                { scheme: \"grow\" }, with `harmonics`, `lone` and `rounds` if wanted";
    let object = value
        .dyn_ref::<js_sys::Object>()
        .filter(|_| !js_sys::Array::is_array(value))
        .ok_or_else(|| refuse("`continuation` is not an object".into(), help))?;
    let mut scheme = None;
    let mut asked = sva_core::continuation::Asked::default();
    for entry in js_sys::Object::entries(object).iter() {
        let pair = js_sys::Array::from(&entry);
        let key = pair.get(0).as_string().unwrap_or_default();
        let value = pair.get(1);
        match key.as_str() {
            "scheme" => scheme = Some(text("scheme", &value)?),
            "prices" if value.is_null() || value.is_undefined() => {}
            "prices" => {
                let list = value
                    .dyn_ref::<js_sys::Array>()
                    .ok_or_else(|| refuse("`prices` is not an array".into(), help))?;
                asked.prices = Some(
                    list.iter()
                        .map(|p| p.as_f64().unwrap_or(f64::NAN))
                        .collect(),
                );
            }
            "harmonics" => asked.harmonics = Some(whole("harmonics", &value)?),
            "lone" => asked.lone = Some(whole("lone", &value)?),
            "rounds" => asked.rounds = Some(whole("rounds", &value)?),
            _ => {
                return Err(refuse(
                    format!("`continuation` holds `{key}`"),
                    "its keys are scheme, prices, harmonics, lone and rounds",
                ));
            }
        }
    }
    let scheme = scheme.ok_or_else(|| refuse("`continuation` names no `scheme`".into(), help))?;
    sva_core::continuation::Scheme::of(&scheme, asked).map_err(|e| thrown(&e))
}

fn whole<N: TryFrom<u64>>(key: &str, value: &JsValue) -> Result<N, JsValue> {
    value
        .as_f64()
        .filter(|v| v.fract() == 0.0 && *v >= 0.0 && *v <= 2f64.powi(53))
        .and_then(|v| N::try_from(v as u64).ok())
        .ok_or_else(|| {
            refuse(
                format!("`{key}` is not a whole number in range"),
                "pass a whole number",
            )
        })
}

fn text(key: &str, value: &JsValue) -> Result<String, JsValue> {
    value.as_string().ok_or_else(|| {
        refuse(
            format!("`{key}` is not a string"),
            "pass it as a string, as `sva-cli` writes it",
        )
    })
}

/// Each entry one call, as `--representation` writes it: `spectrum(peaks=8)`.
fn representations_of(names: &[String]) -> Result<Vec<Asked>, JsValue> {
    let calls = names
        .iter()
        .map(|name| {
            let call = sva_core::call(name).map_err(|e| thrown(&e))?;
            if call.dest.is_some() {
                return Err(refuse(
                    format!("`{name}` names a destination, and a page writes no file"),
                    "drop the `=path`, and read it from `representations()`",
                ));
            }
            Ok(call)
        })
        .collect::<Result<Vec<_>, _>>()?;
    sva_core::asked_all(&calls).map_err(|e| thrown(&e))
}

/// Nothing in a browser pushes back when memory grows inside the tab's own address space.
const DEFAULT_CACHE_BYTES: u64 = 256 << 20;

/// The page's nodes as they stand at each read, so a stream's later change reads what the
/// page inserted since it opened.
#[derive(Clone, Default)]
struct Live(Rc<RefCell<sva_ast::Composition>>);

impl sva_ast::Source for Live {
    fn paths(&self) -> Result<sva_ast::Listing, String> {
        self.0.borrow().paths()
    }

    fn get(&self, path: &str) -> Result<Option<Cow<'_, str>>, String> {
        let held = self.0.borrow();
        Ok(held.get(path)?.map(|text| Cow::Owned(text.into_owned())))
    }

    fn generation(&self, path: &str) -> Option<u64> {
        self.0.borrow().generation(path)
    }

    fn name(&self) -> String {
        self.0.borrow().name()
    }

    fn recording(&self, path: &str) -> Result<Option<Arc<sva_ast::Recording>>, String> {
        self.0.borrow().recording(path)
    }
}

#[wasm_bindgen]
pub struct Composition {
    inner: Live,
    tier: Rc<Tier<opfs::Opfs>>,
    /// A render in flight holds it; one beside it types in a session of its own.
    session: Cell<Session>,
    threads: Option<NonZeroUsize>,
    subscription: Rc<stats::Subscription>,
}

fn unstored(why: String) -> JsValue {
    let message = format!("the store could not be read or written: {why}");
    let diagnostic = Diagnostic::new("wasm.store", message.clone())
        .helped("check the directory handle is writable, or open with none for memory only");
    crossed("internal_error", &message, &[diagnostic])
}

#[wasm_bindgen]
impl Composition {
    #[wasm_bindgen(constructor)]
    pub fn new(name: Option<String>) -> Composition {
        Composition::over(name, Tier::alone(DEFAULT_CACHE_BYTES))
    }

    fn over(name: Option<String>, tier: Tier<opfs::Opfs>) -> Composition {
        let inner = sva_ast::Composition::new();
        let (tier, subscription) = (Rc::new(tier), Rc::default());
        stats::joined(&tier, &subscription);
        Composition {
            inner: Live(Rc::new(RefCell::new(match name {
                Some(name) => inner.named(name),
                None => inner,
            }))),
            tier,
            session: Cell::default(),
            threads: None,
            subscription,
        }
    }

    /// `callback`, in place of the last one and of the module's, hears an `EngineStats` at
    /// once, then from every call of this composition.
    pub fn subscribe(
        &self,
        #[wasm_bindgen(unchecked_param_type = "(stats: EngineStats) => void")]
        callback: js_sys::Function,
    ) {
        stats::heard_by(&self.tier, &self.subscription, Some(callback));
    }

    /// The module's subscriber, where there is one, hears it from here on.
    pub fn unsubscribe(&self) {
        stats::heard_by(&self.tier, &self.subscription, None);
    }

    /// Memory over the store in `dir`, an origin-private file system directory; alone where
    /// none opens. `options.store_max_bytes`: the store's budget, evicting the least recently
    /// used past it at each persist. `options.threads`: the most threads a render computes on.
    pub async fn open(
        name: Option<String>,
        dir: Option<DirectoryHandle>,
        options: JsValue,
    ) -> Result<Composition, JsValue> {
        let options = options_of(&options, &["store_max_bytes", "threads"])?;
        let mut opened = None;
        let tier = match dir {
            None => Tier::alone(DEFAULT_CACHE_BYTES),
            Some(dir) => {
                let budget = options.store_max_bytes.unwrap_or(DEFAULT_STORE_BYTES);
                let opening = Cell::new(StoreBoot {
                    max_bytes: budget,
                    read: None,
                });
                let told = |step| {
                    let (phase, store) = stats::stepped(opening.get(), step);
                    opening.set(store);
                    stats::booted(phase, None, Some(store));
                };
                match Store::open_told(opfs::Opfs { dir }, budget, &told).await {
                    Ok(store) => {
                        opened = Some(opening.get());
                        Tier::over(store, DEFAULT_CACHE_BYTES)
                    }
                    Err(why) => {
                        logged(&format!("the store would not open, so none is kept: {why}"));
                        stats::booted(BootPhase::StoreFailed, None, Some(opening.get()));
                        Tier::alone(DEFAULT_CACHE_BYTES)
                    }
                }
            }
        };
        stats::booted(BootPhase::Opened, None, opened);
        Ok(Composition {
            threads: options.threads,
            ..Composition::over(name, tier)
        })
    }

    /// The only commit to the directory `open` was handed.
    pub async fn persist(&self) -> Result<usize, JsValue> {
        let done = self.tier.persist().await.map_err(unstored)?;
        if done.refused > 0 {
            logged(&format!(
                "{} value(s) were larger than the store's whole budget, so none was stored",
                done.refused
            ));
        }
        Ok(done.written)
    }

    pub fn insert(&mut self, path: &str, text: &str) {
        self.inner.0.borrow_mut().insert(path, text);
    }

    /// `sva-cli solve`'s envelope. `cost` is a `SolveCost` tree, every reference a value:
    /// `{ norm: { sub: [{ repr: "bands", target: "@lead([0, 5s])" }, { value: data }] } }`.
    /// `vary`: `SolveVary`s. `options` as the CLI's flags, `signal` and `progress(line)`;
    /// `continuation` as `--continuation`: `{ scheme: "grow", harmonics: 48, lone: 8 }`.
    pub async fn solve(
        &self,
        #[wasm_bindgen(unchecked_param_type = "SolveCost")] cost: JsValue,
        #[wasm_bindgen(unchecked_param_type = "SolveVary[]")] vary: js_sys::Array,
        options: JsValue,
    ) -> Result<JsValue, JsValue> {
        let keys = [
            "max_renders",
            "step_tol",
            "cost_tol",
            "set",
            "rate",
            "threads",
            "signal",
            "progress",
            "method",
            "continuation",
            "steps",
            "draws",
            "write",
        ];
        let options = options_of(&options, &keys)?;
        let tree = cost::tree_of(&cost, "cost")?;
        let cost = sva_core::solve::built(tree).map_err(|e| thrown(&e))?;
        let vary = cost::vary_of(&vary)?;
        let stops = sva_core::solve::Stops {
            max_renders: options
                .max_renders
                .unwrap_or(sva_core::solve::DEFAULT_MAX_RENDERS),
            step_tol: options
                .step_tol
                .unwrap_or(sva_core::solve::DEFAULT_STEP_TOL),
            cost_tol: options
                .cost_tol
                .unwrap_or(sva_core::solve::DEFAULT_COST_TOL),
        };
        let problem = sva_core::solve::Problem {
            source: &self.inner,
            texts: &sva_core::solve::NoFiles,
            cost,
            vary: vary.clone(),
            sets: options.set.clone(),
            rate: options.rate,
            threads: Some(threads::limit(options.threads.or(self.threads))),
            stops,
        };
        let signal = options.signal.clone().map(Signal);
        let abandon = signal
            .as_ref()
            .map_or(&Never as &dyn Abandon, |signal| signal as &dyn Abandon);
        if options.prox {
            return self.priced(&problem, &options, abandon).await;
        }
        let mut heard = |point: &sva_core::solve::Point| {
            if let Some(progress) = &options.progress {
                let line = sva_core::solve::point_json(point);
                let _ = progress.call1(&JsValue::NULL, &parse(&line).unwrap_or(line.into()));
            }
        };
        let solved = sva_core::solve::solve(&problem, &*self.tier, &mut heard, abandon)
            .await
            .map_err(|e| thrown(&e))?;
        let found = sva_core::solve::findings(&solved, &vary);
        let data = sva_core::solve::solved_data(&solved, &vary, "");
        if solved.stopped_by == sva_core::solve::Stopped::Cancelled {
            let message = "the solve was aborted; `refusal.data` holds the best so far";
            let error = JsErr::new(message);
            error.set_name("AbortError");
            let envelope = sva_core::error_envelope_with("cancelled", message, &data, &found);
            error.set_refusal(parse(&envelope)?);
            return Err(error.into());
        }
        parse(&sva_core::success_envelope(&data, &found))
    }

    async fn priced(
        &self,
        problem: &sva_core::solve::Problem<'_>,
        options: &Options,
        abandon: &dyn Abandon,
    ) -> Result<JsValue, JsValue> {
        use sva_core::fit::{capped, fitted, fitted_data, writable};
        let settings = sva_core::solve::prox::Settings {
            steps: options
                .steps
                .unwrap_or(sva_core::solve::prox::DEFAULT_STEPS),
            draws: options
                .draws
                .unwrap_or(sva_core::solve::prox::DEFAULT_DRAWS),
        };
        if options.write {
            writable(&settings, ("`write`", "`steps`")).map_err(|e| thrown(&e))?;
        }
        let mut heard = |line: &str| {
            if let Some(progress) = &options.progress {
                let _ = progress.call1(&JsValue::NULL, &parse(line).unwrap_or(line.into()));
            }
        };
        let scheme = options.continuation.as_ref();
        let fitted = fitted(problem, &settings, scheme, &*self.tier, &mut heard, abandon)
            .await
            .map_err(|e| thrown(&e))?;
        let cost = &problem.cost.text;
        let warnings = capped(&fitted, &settings, ("`steps`", "`draws`"));
        if fitted.ran.cancelled {
            let data = fitted_data(&fitted, &settings, cost, "");
            let message = "the priced solve was aborted; `refusal.data.stages` holds the stages it \
                           finished and the one it stopped in, `refusal.data.chosen` the finished \
                           one of least predicted length at the cap's draws, and nothing was \
                           written";
            let error = JsErr::new(message);
            error.set_name("AbortError");
            let envelope = sva_core::error_envelope_with("cancelled", message, &data, &warnings);
            error.set_refusal(parse(&envelope)?);
            return Err(error.into());
        }
        let mut extra = String::new();
        if options.write {
            for (path, text) in &fitted.files {
                self.inner.0.borrow_mut().insert(path, text);
            }
            let paths = sva_core::json::list(&fitted.files, |(path, _)| {
                format!("\"{}\"", sva_core::json::escape(path))
            });
            extra = format!(", \"written\": {paths}");
        }
        let data = fitted_data(&fitted, &settings, cost, &extra);
        parse(&sva_core::success_envelope(&data, &warnings))
    }

    /// `target` and `representations` as `sva-cli render` takes them. `options`: `rate`,
    /// `bits`, `until`, `volatile`, `out: null` (no samples), `threads` (as `open` takes it),
    /// `set` (`--set`, as `{ "key": "G3" }`), and `signal`: aborted, it throws an
    /// `AbortError` before its next block.
    pub async fn render(
        &self,
        target: &str,
        representations: Option<Vec<String>>,
        options: JsValue,
    ) -> Result<Rendering, JsValue> {
        let keys = [
            "rate", "bits", "until", "volatile", "out", "signal", "threads", "set",
        ];
        let options = options_of(&options, &keys)?;
        let out = options.out;
        let names = representations.unwrap_or_else(|| match out {
            Out::Kept => vec!["samples".to_string()],
            Out::Dropped => Vec::new(),
        });
        let asked = representations_of(&names)?;
        let samples = asked
            .iter()
            .find(|asked| asked.name == Representation::Samples.name());
        if let (Out::Dropped, Some(samples)) = (out, samples) {
            return Err(refuse(
                format!(
                    "`{}` reads samples, and `out: null` hands back none",
                    samples.name
                ),
                "render without `out` to read its samples",
            ));
        }
        let signal = options.signal.map(Signal);
        let source = sva_core::overlaid(&self.inner, &options.set).map_err(|e| thrown(&e))?;
        let job = Job {
            until: options.until.as_deref(),
            rate: options.rate,
            bits: options.bits,
            asked: &asked,
            volatile: &options.volatile,
            out,
            abandon: signal
                .as_ref()
                .map_or(&Never, |signal| signal as &dyn Abandon),
            threads: Some(threads::limit(options.threads.or(self.threads))),
            ..Job::over(&source, target)
        };
        let mut session = self.session.take();
        let inner = execute_over(job, &*self.tier, &mut session).await;
        self.session.set(session);
        let inner = inner.map_err(|e| match e {
            CliError::Engine(EngineError::Abandoned) => {
                crossed("AbortError", &e.message(), &e.diagnostics())
            }
            e => thrown(&e),
        })?;
        unstaged(inner.render.cache_stats.as_ref());
        Ok(Rendering { inner, asked })
    }

    /// `target`'s placed notes as `sva-cli outline @score/song` lists them, under
    /// `options.set` as `render` takes it: `{ song, events }`, each with `at`, `sounds`,
    /// `len` (or null), `node`, `hz`, `file`, `line`, `params`. Nothing renders.
    pub fn song(&self, target: &str, options: JsValue) -> Result<JsValue, JsValue> {
        let options = options_of(&options, &["set"])?;
        let source = sva_core::overlaid(&self.inner, &options.set).map_err(|e| thrown(&e))?;
        parse(&sva_core::events_data(&source, target).map_err(|e| thrown(&e))?)
    }

    /// `target` block by block. `options`: `rate`, `bits`, `until`, `live`, `channels`,
    /// `threads` (as `open` takes it).
    pub async fn stream(
        &self,
        target: &str,
        block: usize,
        options: JsValue,
    ) -> Result<Stream, JsValue> {
        let keys = ["rate", "bits", "until", "live", "channels", "threads"];
        let options = options_of(&options, &keys)?;
        let job = Job {
            until: options.until.as_deref(),
            rate: options.rate,
            bits: options.bits,
            threads: Some(threads::limit(options.threads.or(self.threads))),
            ..Job::over(&self.inner, target)
        };
        let opened = sva_core::stream(&job, (block, options.channels), &*self.tier).await;
        let mut inner = opened.map_err(|e| thrown(&e))?;
        if options.live {
            inner.go_live();
        }
        let edits = Edits {
            inner: Rc::new(RefCell::new(inner)),
            source: self.inner.clone(),
            tier: Rc::clone(&self.tier),
            freed: Rc::default(),
        };
        let scratch = RefCell::new(Vec::with_capacity(edits.inner.borrow().width() * block));
        Ok(Stream { edits, scratch })
    }

    /// Memory's disk traffic since it opened.
    pub fn counters(&self) -> Result<JsValue, JsValue> {
        parse(&counters_json(&self.tier.counters()))
    }

    #[wasm_bindgen(getter)]
    pub fn cache_bytes(&self) -> f64 {
        self.tier.bytes() as f64
    }

    #[wasm_bindgen(getter)]
    pub fn cache_max_bytes(&self) -> f64 {
        self.tier.max_bytes() as f64
    }

    #[wasm_bindgen(setter)]
    pub fn set_cache_max_bytes(&self, max_bytes: f64) {
        self.tier.set_max_bytes(max_bytes.max(0.0) as u64);
    }

    /// What the store holds by its index; undefined without one.
    #[wasm_bindgen(getter)]
    pub fn store_bytes(&self) -> Option<f64> {
        self.tier.disk().map(|disk| disk.bytes() as f64)
    }

    #[wasm_bindgen(getter)]
    pub fn store_max_bytes(&self) -> Option<f64> {
        self.tier.disk().map(|disk| disk.max_bytes() as f64)
    }

    #[wasm_bindgen(getter)]
    pub fn cache_entries(&self) -> usize {
        self.tier.entries()
    }

    #[wasm_bindgen(getter)]
    pub fn cache_evictions(&self) -> f64 {
        self.tier.counters().evictions() as f64
    }
}

fn placement(options: Option<JsValue>) -> Result<Placed, JsValue> {
    let options = options.unwrap_or(JsValue::UNDEFINED);
    Ok(options_of(&options, &["at"])?.at.unwrap_or(Placed::Written))
}

fn placed(name: &str) -> Result<Placed, JsValue> {
    match name {
        "written" => Ok(Placed::Written),
        "landing" => Ok(Placed::Landing),
        _ => Err(refuse(
            format!("`{name}` names no placement"),
            "pass \"written\" or \"landing\"",
        )),
    }
}

#[wasm_bindgen]
pub struct Rendering {
    inner: Rendered,
    asked: Vec<Asked>,
}

#[wasm_bindgen]
impl Rendering {
    #[wasm_bindgen(getter)]
    pub fn sample_rate(&self) -> u32 {
        self.inner.config.rate
    }

    #[wasm_bindgen(getter)]
    pub fn duration_secs(&self) -> f64 {
        let rate = self.inner.config.rate;
        self.inner.render.range.map_or(0.0, |r| r.span_secs(rate))
    }

    #[wasm_bindgen(getter)]
    pub fn channels(&self) -> usize {
        let render = &self.inner.render;
        render.buffer(render.root).map_or(0, |b| b.width())
    }

    pub fn samples(&self, channel: usize) -> Result<Vec<f32>, JsValue> {
        let render = &self.inner.render;
        if self.inner.config.out == Out::Dropped {
            return Err(refuse(
                format!("`{}` was rendered with `out: null`", self.inner.expression),
                "render without `out` to read its samples",
            ));
        }
        if render.buffer(render.root).is_none() {
            return Err(refuse(
                format!("`{}` read no samples of its root", self.inner.expression),
                "ask for `samples` among the representations",
            ));
        }
        let buffer = render
            .output(render.root)
            .map_err(|e| thrown(&CliError::Engine(e)))?;
        if channel >= buffer.width() {
            return Err(refuse(
                format!(
                    "`{}` is {} component(s), so there is no component {channel}",
                    self.inner.expression,
                    buffer.width()
                ),
                "read a component this node holds, counting from zero",
            ));
        }
        sva_core::encode::float32(&self.inner.expression, buffer.plane(channel))
            .map_err(|e| thrown(&e))
    }

    /// Every lookup this render made of memory, and what each came to.
    pub fn stats(&self) -> Result<JsValue, JsValue> {
        let none = CacheStats::default();
        parse(&stats_json(
            self.inner.render.cache_stats.as_ref().unwrap_or(&none),
        ))
    }

    pub fn work(&self) -> Result<JsValue, JsValue> {
        parse(&work_json(&self.inner.render.work()))
    }

    /// The object `sva-cli render` puts under `data`, one reading per representation asked,
    /// arrays capped as it caps unless `options.limit` says otherwise.
    pub fn representations(&self, options: JsValue) -> Result<JsValue, JsValue> {
        let options = options_of(&options, &["limit"])?;
        answered(&self.inner, &self.asked, options.limit)
    }
}

fn answered(rendered: &Rendered, asked: &[Asked], limit: Limit) -> Result<JsValue, JsValue> {
    let mut answers = Vec::with_capacity(asked.len());
    for asked in asked {
        let answer = rendered.answered(asked).map_err(|e| thrown(&e))?;
        let printed = Printed::of(
            asked,
            answer,
            rendered.config.rate,
            rendered.seconds_per_bar(),
        );
        answers.push(printed.map_err(|e| thrown(&e))?);
    }
    let rate = rendered.config.rate;
    let interval = rendered
        .render
        .range
        .map(|r| (r.start_secs(rate), r.end as f64 / f64::from(rate)));
    parse(&query_data(&Report {
        target: &rendered.expression,
        rate,
        bits: Some(rendered.config.profile.precision_bits),
        interval,
        profile: rendered.config.profile.name,
        label: rendered.label(),
        written: &[],
        played: None,
        answers: &answers,
        limit: limit.0,
    }))
}

/// A target block by block, reading `@notes` as the sum of its terms, as `@hall(t, x=@notes)`.
/// A key-up replaces a term with one whose release is a number. A term leaves with its handle
/// once the stream passes its support. An edit lands at the first block boundary
/// once memory answered it; `read` and getters answer meanwhile.
#[wasm_bindgen]
pub struct Stream {
    edits: Edits,
    scratch: RefCell<Vec<f32>>,
}

/// What an edit in flight holds: never the page's object, which the page may free meanwhile.
#[derive(Clone)]
struct Edits {
    inner: Rc<RefCell<sva_core::Stream>>,
    source: Live,
    tier: Rc<Tier<opfs::Opfs>>,
    freed: Rc<Cell<bool>>,
}

impl Drop for Stream {
    fn drop(&mut self) {
        self.edits.freed.set(true);
    }
}

impl Edits {
    /// Refused where the page freed the stream meanwhile: no one hears what it did.
    fn answered<T>(&self, done: Result<T, CliError>) -> Result<T, JsValue> {
        if self.freed.get() {
            let message = "the stream was freed while this edit was in flight".to_string();
            let diagnostic = Diagnostic::new("wasm.stream_freed", message.clone())
                .helped("make the edit on the stream that replaced it");
            return Err(crossed("conflict", &message, &[diagnostic]));
        }
        done.map_err(|e| thrown(&e))
    }
}

/// The futures behind the page's promises, holding no borrow of the stream.
impl Stream {
    pub fn edit(&self, expr: &str) -> impl Future<Output = Result<(), JsValue>> + 'static {
        let (edits, expr) = (self.edits.clone(), expr.to_string());
        async move {
            let edited = sva_core::edit(&edits.inner, &edits.source, &expr, &*edits.tier).await;
            edits.answered(edited)
        }
    }

    pub fn add(
        &self,
        term: &str,
        options: Option<JsValue>,
    ) -> impl Future<Output = Result<u32, JsValue>> + 'static {
        let (edits, term) = (self.edits.clone(), term.to_string());
        async move {
            let at = placement(options)?;
            let (inner, source) = (&edits.inner, &edits.source);
            let added = sva_core::add(inner, source, (&term, at), &*edits.tier).await;
            edits.answered(added.map(|handle| handle.0))
        }
    }

    pub fn replace(
        &self,
        handle: u32,
        term: &str,
        options: Option<JsValue>,
    ) -> impl Future<Output = Result<bool, JsValue>> + 'static {
        let (edits, term) = (self.edits.clone(), term.to_string());
        async move {
            let replaced = (Handle(handle), term.as_str(), placement(options)?);
            let (inner, source) = (&edits.inner, &edits.source);
            let replaced = sva_core::replace(inner, source, replaced, &*edits.tier).await;
            edits.answered(replaced)
        }
    }

    pub fn remove(&self, handle: u32) -> impl Future<Output = Result<bool, JsValue>> + 'static {
        let edits = self.edits.clone();
        async move {
            let removed = sva_core::remove(&edits.inner, Handle(handle), &*edits.tier).await;
            edits.answered(removed)
        }
    }

    pub fn fetch(&self) -> impl Future<Output = ()> + 'static {
        let edits = self.edits.clone();
        async move { sva_core::fetch(&edits.inner, &*edits.tier).await }
    }
}

fn promised<T: Into<JsValue>>(
    done: impl Future<Output = Result<T, JsValue>> + 'static,
) -> js_sys::Promise {
    wasm_bindgen_futures::future_to_promise(async move { done.await.map(Into::into) })
}

#[wasm_bindgen]
impl Stream {
    /// Writes frames from sample `at` into `out` (a SharedArrayBuffer view too) interleaved,
    /// `out[i * channels + c]`, and returns how many: `out.length / channels`, fewer at the end.
    /// A later `at` skips; live, what needed the span starts silent, in `stats().dropped`; only
    /// `until` ends it.
    pub fn read(&self, at: f64, out: &js_sys::Float32Array) -> Result<usize, JsValue> {
        if !(at.is_finite() && at >= 0.0 && at.fract() == 0.0) {
            return Err(refuse(
                format!("`at` is {at}, not a whole sample"),
                "pass a whole sample, the stream's `position` or later",
            ));
        }
        let inner = &mut self.edits.inner.borrow_mut();
        let (width, room) = (inner.width(), out.length() as usize);
        if room % width != 0 {
            return Err(refuse(
                format!("`out` holds {room} samples, not whole frames of {width} component(s)"),
                "pass a Float32Array of channels * frames samples",
            ));
        }
        let Some(held) = inner
            .read(at as i64, room / width)
            .map_err(|e| thrown(&CliError::Engine(e)))?
        else {
            return Ok(0);
        };
        let scratch = &mut self.scratch.borrow_mut();
        scratch.clear();
        for i in 0..held.len() {
            for c in 0..width {
                let sample = held.plane(c)[i];
                let sample = sva_core::encode::float32_sample(sva_engine::STREAMED, sample);
                scratch.push(sample.map_err(|e| thrown(&e))?);
            }
        }
        out.subarray(0, scratch.len() as u32).copy_from(scratch);
        Ok(held.len())
    }

    /// `expr` in place of the target. Each edit's promise rejects as a `conflict` where the
    /// page frees the stream first.
    #[wasm_bindgen(js_name = edit, unchecked_return_type = "Promise<void>")]
    pub fn edit_promise(&self, expr: String) -> js_sys::Promise {
        let edited = self.edit(&expr);
        promised(async move { edited.await.map(|()| JsValue::UNDEFINED) })
    }

    /// `term` summed into `@notes`. `options.at`: `"written"` (the stream's `t`) or
    /// `"landing"`, its sample 0 the sample it lands at.
    #[wasm_bindgen(js_name = add, unchecked_return_type = "Promise<number>")]
    pub fn add_promise(&self, term: String, options: Option<JsValue>) -> js_sys::Promise {
        promised(self.add(&term, options))
    }

    /// False where `handle` is gone. `"landing"`: where its add landed.
    #[wasm_bindgen(js_name = replace, unchecked_return_type = "Promise<boolean>")]
    pub fn replace_promise(
        &self,
        handle: u32,
        term: String,
        options: Option<JsValue>,
    ) -> js_sys::Promise {
        promised(self.replace(handle, &term, options))
    }

    pub fn landed(&self, handle: u32) -> Option<f64> {
        let landed = self.edits.inner.borrow().landed(Handle(handle));
        landed.map(|at| at as f64)
    }

    #[wasm_bindgen(js_name = remove, unchecked_return_type = "Promise<boolean>")]
    pub fn remove_promise(&self, handle: u32) -> js_sys::Promise {
        promised(self.remove(handle))
    }

    /// Reads what the next second plays, a few disk reads a call at most.
    #[wasm_bindgen(js_name = fetch, unchecked_return_type = "Promise<void>")]
    pub fn fetch_promise(&self) -> js_sys::Promise {
        let fetched = self.fetch();
        promised(async move {
            fetched.await;
            Ok(JsValue::UNDEFINED)
        })
    }

    /// `{ samples, computed_samples }` since it opened.
    pub fn work(&self) -> Result<JsValue, JsValue> {
        parse(&work_json(&self.edits.inner.borrow().work()))
    }

    /// Each list the latest; its `pagination.count` counts all. `pruned`: the silence threshold
    /// below which a term leaves the sum for good, and each node cut, at the sample it is
    /// treated as silent from.
    pub fn stats(&self) -> Result<JsValue, JsValue> {
        let inner = self.edits.inner.borrow();
        let made = inner.counts().dropped;
        parse(&stream_stats_json(
            &inner.stats(),
            (&inner.dropped(), made),
            &inner.cutting_below_silence_threshold(),
        ))
    }

    /// `late`: edits landed late; `demands`: reads working out the note sum's asks; `built`:
    /// what the latest change did anew; `tier`: memory's disk traffic since it opened.
    pub fn counts(&self) -> Result<JsValue, JsValue> {
        let counts = self.edits.inner.borrow().counts();
        let whole = |pairs: &[(&str, usize)]| -> Result<js_sys::Object, JsValue> {
            let out = js_sys::Object::new();
            for (key, value) in pairs {
                js_sys::Reflect::set(&out, &(*key).into(), &(*value as f64).into())?;
            }
            Ok(out)
        };
        let built = counts.built;
        let out = whole(&[
            ("dropped", counts.dropped),
            ("late", counts.late),
            ("terms", counts.terms),
            ("demands", counts.demands),
        ])?;
        let built = whole(&[
            ("parsed", built.parsed),
            ("instances", built.instances),
            ("visited", built.visited),
            ("typed", built.typed),
            ("values", built.values),
            ("copied", built.copied),
            ("lookups", built.lookups),
        ])?;
        js_sys::Reflect::set(&out, &"built".into(), &built)?;
        let tier = parse(&counters_json(&counts.tier))?;
        js_sys::Reflect::set(&out, &"tier".into(), &tier)?;
        Ok(out.into())
    }

    #[wasm_bindgen(getter)]
    pub fn held_bytes(&self) -> f64 {
        self.edits.inner.borrow().held_bytes() as f64
    }

    #[wasm_bindgen(getter)]
    pub fn channels(&self) -> usize {
        self.edits.inner.borrow().width()
    }

    #[wasm_bindgen(getter)]
    pub fn sample_rate(&self) -> u32 {
        self.edits.inner.borrow().config().render.rate
    }

    #[wasm_bindgen(getter)]
    pub fn block(&self) -> usize {
        self.edits.inner.borrow().config().block
    }

    #[wasm_bindgen(getter)]
    pub fn position(&self) -> f64 {
        self.edits.inner.borrow().position() as f64
    }

    #[wasm_bindgen(getter)]
    pub fn end(&self) -> Option<f64> {
        self.edits.inner.borrow().end().map(|end| end as f64)
    }
}
