//! physis-core tui — the epistemic control panel, in-process.
//!
//! Built additively behind the `tui` feature (`cargo build --features tui`);
//! nothing else in the crate depends on it. Four tabs, re-scoped from the
//! pro TUI's MACHINES/CONFIG to what the ledger thesis actually uses
//! (see `../docs/WHAT_PHYSIS_IS.md` §3½):
//!
//!   STATUS  — engine counts, coherence index, ledger sizes, observation log tail
//!   LEDGER  — hypotheses by fitness (evidence counts, open predictions) and
//!             contradictions, both parties, never netted
//!   ASK     — the grounded-answer pipeline (`service::ask`): compiled context
//!             under a token budget, draft-and-fill or oracle synthesis; every
//!             ask is appended to the observation log
//!   WATCH   — tail of the observation log: what the machine saw
//!
//! Keys: Tab / ← → switch · ↑ ↓ select · type in ASK, Enter runs the ask ·
//!       r refresh · q quit.
//!
//! Runs in-process over `physis_core::service` — no HTTP, no server required;
//! it reads the same `nodes.json` / observation log the CLI and studio write.

use std::io::stdout;
use std::path::PathBuf;
use std::sync::mpsc::Receiver;

use crossterm::{
    event::{self, Event, KeyCode},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, List, ListItem, ListState, Paragraph, Tabs, Wrap},
    Terminal,
};

use physis_core::{observe::Observation, service, store};

const TABS: [&str; 4] = ["STATUS", "LEDGER", "ASK", "WATCH"];

fn usage() -> ! {
    eprintln!(
        "usage: tui [--corpus <dir>] [--budget <tokens>] [--limit <lines>]\n\
         \n\
         in-process epistemic control panel over the persisted core.\n\
         corpus defaults to `examples`; budget to 1200; watch tail to 60."
    );
    std::process::exit(2);
}

struct App {
    tab: usize,
    corpus: String,
    budget: usize,
    watch_limit: usize,
    data_dir: PathBuf,
    sel: usize,
    status: Option<service::StatusSummary>,
    ledger: Option<service::LedgerSnapshot>,
    watch: Vec<Observation>,
    ask_input: String,
    ask_rx: Option<Receiver<Result<service::AskOutcome, String>>>,
    ask_view: String,
    msg: String,
}

impl App {
    fn new(corpus: String, budget: usize, watch_limit: usize) -> Self {
        Self {
            tab: 0,
            corpus,
            budget,
            watch_limit,
            data_dir: store::data_dir(),
            sel: 0,
            status: None,
            ledger: None,
            watch: Vec::new(),
            ask_input: String::new(),
            ask_rx: None,
            ask_view: String::new(),
            msg: String::new(),
        }
    }

    /// Reload everything the current tab needs. Read-only — the TUI never
    /// writes to the core; only `ask` appends to the observation log.
    fn refresh(&mut self) {
        match self.tab {
            0 => match service::status(&self.data_dir) {
                Ok(s) => {
                    self.msg = format!("status refreshed — {} node(s)", s.total_nodes);
                    self.status = Some(s);
                }
                Err(e) => self.msg = format!("status failed: {e}"),
            },
            1 => match service::ledger(&self.data_dir) {
                Ok(l) => {
                    self.msg = format!(
                        "ledger refreshed — {} hypotheses, {} contradiction(s), {} open prediction(s)",
                        l.hypotheses.len(),
                        l.contradictions.len(),
                        l.open_predictions
                    );
                    if self.sel >= l.hypotheses.len().max(1) {
                        self.sel = 0;
                    }
                    self.ledger = Some(l);
                }
                Err(e) => self.msg = format!("ledger failed: {e}"),
            },
            3 => match service::watch_tail(self.watch_limit) {
                Ok(w) => {
                    self.msg = format!("{} observation(s) loaded", w.len());
                    self.watch = w;
                }
                Err(e) => self.msg = format!("watch failed: {e}"),
            },
            _ => {}
        }
    }

    /// Run the ask on a worker thread so the UI stays responsive; the result
    /// is drained from the channel in the event loop below.
    fn run_ask(&mut self) {
        if self.ask_rx.is_some() {
            self.msg = "ask already running…".into();
            return;
        }
        let query = self.ask_input.trim().to_string();
        if query.is_empty() {
            self.msg = "type a question first".into();
            return;
        }
        let req = service::AskRequest {
            query,
            corpus: self.corpus.clone(),
            budget: Some(self.budget),
            draft: false,
            confidence: None,
        };
        let (tx, rx) = std::sync::mpsc::channel();
        self.ask_rx = Some(rx);
        self.ask_view = "… compiling context".into();
        self.msg = format!("asking (corpus `{}`, budget {})", self.corpus, self.budget);
        std::thread::spawn(move || {
            let res = service::ask(&req).map_err(|e| e.to_string());
            let _ = tx.send(res);
        });
    }

    fn poll_ask(&mut self) {
        if let Some(rx) = &self.ask_rx {
            match rx.try_recv() {
                Ok(Ok(outcome)) => {
                    self.ask_view = render_outcome(&outcome);
                    self.msg = format!(
                        "answered · embedder {} · {} ms{}",
                        outcome.embedder,
                        outcome.elapsed_ms as u64,
                        outcome
                            .recorded_seq
                            .map(|s| format!(" · recorded #{}", s))
                            .unwrap_or_else(|| " · NOT recorded".into())
                    );
                    self.ask_rx = None;
                    if self.tab == 0 {
                        self.refresh(); // the observation log grew
                    }
                }
                Ok(Err(e)) => {
                    self.ask_view = format!("ask failed: {e}");
                    self.msg = "ask failed".into();
                    self.ask_rx = None;
                }
                Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                    self.ask_view = "ask worker died".into();
                    self.ask_rx = None;
                }
                Err(std::sync::mpsc::TryRecvError::Empty) => {}
            }
        }
    }
}

/// Render an ask outcome the way `notebook`'s own CLI renders it — same
/// wording, so a browser answer, a TUI answer and a terminal answer read alike.
fn render_outcome(o: &service::AskOutcome) -> String {
    if let Some(d) = &o.draft {
        return format!(
            "── DRAFT (table over the retrieved context) ──\n\n{}\n\ntable supplied {} token(s), {} gap(s) left for a model\ntable_share {:.2} (table entries: {})",
            d.render_with_gaps().trim(),
            d.drafted_tokens,
            d.gaps,
            o.table_share.unwrap_or(0.0),
            d.table_entries
        );
    }
    match &o.answer {
        Some(a) => a.render(),
        None => "(no answer, no draft — this should not happen)".into(),
    }
}

fn main() {
    let mut corpus = "examples".to_string();
    let mut budget = 1200usize;
    let mut watch_limit = 60usize;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "-h" | "--help" => usage(),
            "--corpus" => corpus = args.next().unwrap_or_else(|| usage()),
            "--budget" => budget = args.next().and_then(|v| v.parse().ok()).unwrap_or_else(|| usage()),
            "--limit" => watch_limit = args.next().and_then(|v| v.parse().ok()).unwrap_or_else(|| usage()),
            other => {
                eprintln!("unknown argument `{other}`");
                usage();
            }
        }
    }

    // Terminal setup. The guard restores the screen even on error paths.
    enable_raw_mode().expect("tui needs a terminal (raw mode)");
    let _guard = TermGuard;
    execute!(stdout(), EnterAlternateScreen).expect("alternate screen");
    let backend = CrosstermBackend::new(stdout());
    let mut terminal = Terminal::new(backend).expect("terminal");

    let mut app = App::new(corpus, budget, watch_limit);
    app.refresh();

    loop {
        app.poll_ask();
        terminal.draw(|f| draw(f, &mut app)).expect("draw failed");

        if !event::poll(std::time::Duration::from_millis(150)).expect("poll failed") {
            continue;
        }
        if let Event::Key(key) = event::read().expect("read failed") {
            let ctrl = key.modifiers.contains(crossterm::event::KeyModifiers::CONTROL);
            match key.code {
                KeyCode::Char('q') | KeyCode::Char('Q') => break,
                KeyCode::Char('c') if ctrl => break,
                KeyCode::Tab | KeyCode::Right => {
                    app.tab = (app.tab + 1) % TABS.len();
                    app.sel = 0;
                    app.refresh();
                }
                KeyCode::Left => {
                    app.tab = (app.tab + TABS.len() - 1) % TABS.len();
                    app.sel = 0;
                    app.refresh();
                }
                KeyCode::Char('r') | KeyCode::Char('R') => app.refresh(),
                KeyCode::Up => app.sel = app.sel.saturating_sub(1),
                KeyCode::Down => app.sel = app.sel.saturating_add(1),
                KeyCode::Backspace => {
                    if app.tab == 2 {
                        app.ask_input.pop();
                    }
                }
                KeyCode::Enter => {
                    if app.tab == 2 {
                        app.run_ask();
                    }
                }
                KeyCode::Char(c) => {
                    if app.tab == 2 && !ctrl {
                        app.ask_input.push(c);
                    }
                }
                _ => {}
            }
        }
    }
}

/// Restores the terminal on drop — including panic unwinds between
/// `enable_raw_mode` and the explicit restore.
struct TermGuard;
impl Drop for TermGuard {
    fn drop(&mut self) {
        let _ = disable_raw_mode();
        let _ = execute!(stdout(), LeaveAlternateScreen);
    }
}

// ---- drawing ----

fn draw(f: &mut ratatui::Frame, app: &mut App) {
    let root = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(1),
            Constraint::Min(0),
            Constraint::Length(1),
        ])
        .split(f.size());
    f.render_widget(tabs(app.tab), root[0]);
    match app.tab {
        0 => draw_status(f, app, root[1]),
        1 => draw_ledger(f, app, root[1]),
        2 => draw_ask(f, app, root[1]),
        _ => draw_watch(f, app, root[1]),
    }
    f.render_widget(
        Paragraph::new(app.msg.as_str()).style(Style::default().fg(Color::DarkGray)),
        root[2],
    );
}

fn tabs(selected: usize) -> Tabs<'static> {
    let titles: Vec<Line> = TABS.iter().map(|t| Line::from(*t)).collect();
    Tabs::new(titles)
        .select(selected)
        .block(Block::default().borders(Borders::BOTTOM))
        .highlight_style(Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD))
}

fn block(title: &str) -> Block<'_> {
    Block::default()
        .borders(Borders::ALL)
        .title(Span::styled(title, Style::default().fg(Color::Cyan)))
}

fn draw_status(f: &mut ratatui::Frame, app: &App, area: Rect) {
    let text = match &app.status {
        None => vec![Line::from("press r to refresh")],
        Some(s) => vec![
            Line::from(vec![
                Span::styled("data dir   ", Style::default().fg(Color::DarkGray)),
                Span::raw(s.data_dir.clone()),
            ]),
            Line::from(format!(
                "nodes      {} total · high {} · mid {} · low {} · index {:.3}",
                s.total_nodes, s.high_coherence, s.mid_coherence, s.low_coherence, s.coherence_index
            )),
            Line::from(format!(
                "branches   certified {} · isolated {} · dream cycles {}",
                s.certified_branches, s.isolated_branches, s.dream_cycles
            )),
            Line::from(format!(
                "structure  clusters {} · outliers {} · asserted success {} · inert {}",
                s.cluster_count, s.outlier_count, s.asserted_success, s.asserted_inert
            )),
            Line::from(format!(
                "ledger     {} hypotheses · {} contradictions · {} edges · {} open prediction(s)",
                s.hypotheses, s.contradictions, s.edges, s.open_predictions
            )),
            Line::from(format!(
                "watch      last observation seq {}",
                s.last_observation_seq
                    .map(|v| v.to_string())
                    .unwrap_or_else(|| "— (log empty)".into())
            )),
        ],
    };
    f.render_widget(
        Paragraph::new(text).block(block("STATUS")).wrap(Wrap { trim: false }),
        area,
    );
}

fn draw_ledger(f: &mut ratatui::Frame, app: &mut App, area: Rect) {
    let cols = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(45), Constraint::Percentage(55)])
        .split(area);
    let ledger = match &app.ledger {
        Some(l) => l,
        None => {
            f.render_widget(Paragraph::new("press r to refresh"), area);
            return;
        }
    };
    let items: Vec<ListItem> = ledger
        .hypotheses
        .iter()
        .map(|h| {
            ListItem::new(Line::from(vec![
                Span::styled(format!("{:.2} ", h.fitness), Style::default().fg(Color::Yellow)),
                Span::styled(format!("{:<10} ", h.status), Style::default().fg(Color::Blue)),
                Span::raw(shorten(&h.statement, 60)),
            ]))
        })
        .collect();
    let mut state = ListState::default();
    state.select(Some(app.sel.min(items.len().saturating_sub(1))));
    f.render_stateful_widget(
        List::new(items)
            .block(block("HYPOTHESES (fitness ↓)"))
            .highlight_symbol("▶ "),
        cols[0],
        &mut state,
    );

    let detail = match ledger.hypotheses.get(app.sel) {
        None => vec![Line::from(format!(
            "{} contradiction(s), both parties kept open:\n\n{}",
            ledger.contradictions.len(),
            ledger
                .contradictions
                .iter()
                .map(|c| format!(
                    "· {}  [{}]\n  vs {}  [{}]  — {}",
                    c.claim_a, c.source_a, c.claim_b, c.source_b, c.resolution
                ))
                .collect::<Vec<_>>()
                .join("\n")
        ))],
        Some(h) => {
            let mut lines = vec![
                Line::from(Span::styled(h.statement.clone(), Style::default().fg(Color::White))),
                Line::from(""),
                Line::from(format!(
                    "status {} · fitness {:.3} · evidence +{} / −{} · created {}",
                    h.status, h.fitness, h.supporting, h.contradicting, h.created_at
                )),
                Line::from(""),
            ];
            if h.open_predictions.is_empty() {
                lines.push(Line::from(Span::styled(
                    "no open predictions",
                    Style::default().fg(Color::DarkGray),
                )));
            } else {
                lines.push(Line::from(Span::styled(
                    format!("OPEN PREDICTIONS ({}):", h.open_predictions.len()),
                    Style::default().fg(Color::Yellow),
                )));
                for p in &h.open_predictions {
                    lines.push(Line::from(format!("· {p}")));
                }
            }
            lines
        }
    };
    f.render_widget(
        Paragraph::new(detail)
            .block(block("DETAIL / CONTRADICTIONS"))
            .wrap(Wrap { trim: false }),
        cols[1],
    );
}

fn draw_ask(f: &mut ratatui::Frame, app: &App, area: Rect) {
    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(3), Constraint::Min(0)])
        .split(area);
    let input = Line::from(vec![
        Span::styled("query › ", Style::default().fg(Color::Cyan)),
        Span::raw(app.ask_input.clone()),
        Span::styled("▌", Style::default().fg(Color::Cyan)),
    ]);
    f.render_widget(
        Paragraph::new(input).block(block(&format!(
            "ASK · corpus `{}` · budget {} tokens",
            app.corpus, app.budget
        ))),
        rows[0],
    );
    let view = if app.ask_view.is_empty() {
        "type a question and press Enter — the answer is grounded in the corpus,\nand the ask is appended to the observation log."
    } else {
        app.ask_view.as_str()
    };
    f.render_widget(
        Paragraph::new(view).block(block("ANSWER")).wrap(Wrap { trim: false }),
        rows[1],
    );
}

fn draw_watch(f: &mut ratatui::Frame, app: &App, area: Rect) {
    let items: Vec<ListItem> = app
        .watch
        .iter()
        .rev()
        .map(|o| {
            let mut spans = vec![
                Span::styled(format!("#{:<5} ", o.seq), Style::default().fg(Color::DarkGray)),
                Span::styled(format!("{:<8} ", o.source), Style::default().fg(Color::Blue)),
                Span::raw(shorten(&o.subject, 70)),
            ];
            if !o.body.is_empty() {
                spans.push(Span::styled(
                    format!("  ·  {}", shorten(&o.body, 50)),
                    Style::default().fg(Color::DarkGray),
                ));
            }
            ListItem::new(Line::from(spans))
        })
        .collect();
    let empty = items.is_empty();
    let list = List::new(items).block(block("WATCH (most recent first)"));
    if empty {
        f.render_widget(
            Paragraph::new(
                "the observation log is empty — run `physis-core watch`, `act`, or `ask`",
            )
            .block(block("WATCH")),
            area,
        );
    } else {
        f.render_widget(list, area);
    }
}

fn shorten(s: &str, max: usize) -> String {
    let one = s.replace('\n', " ");
    let cut = one.char_indices().nth(max).map(|(i, _)| i);
    match cut {
        None => one,
        Some(i) => format!("{}…", &one[..i]),
    }
}
