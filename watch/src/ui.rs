//! Drawing the screens. Each screen is a list of lines laid out in columns, like the --profile
//! report, so it reads the same in any terminal at 80 columns or more.

use std::sync::OnceLock;

use ratatui::Frame;
use ratatui::layout::Rect;
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Clear, Paragraph, Wrap};

use crate::app::{Action, App, Dialog, Hit, Screen, Source, Target};
use crate::fmt::{ago, bar, clean, clock, fit, hms, rfit, secs, sparkline, width};
use crate::state::{Call, Link, LogKind, Run, short_name};
use crate::summary::{self, Kind};

const SPIN: [char; 10] = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'];

fn no_color() -> bool {
    static NO_COLOR: OnceLock<bool> = OnceLock::new();
    *NO_COLOR.get_or_init(|| std::env::var_os("NO_COLOR").is_some_and(|v| !v.is_empty()))
}

/// 256-color palette, so the colors are right in terminals without 24-bit color too.
#[derive(Clone, Copy)]
enum C {
    Fg,
    Accent,
    Ok,
    Warn,
    Err,
    Blue,
    Dim,
    Faint,
}

fn st(c: C) -> Style {
    if no_color() {
        return match c {
            C::Accent => Style::default().add_modifier(Modifier::BOLD),
            C::Err | C::Warn => Style::default().add_modifier(Modifier::BOLD),
            C::Dim | C::Faint => Style::default().add_modifier(Modifier::DIM),
            _ => Style::default(),
        };
    }
    let i = match c {
        C::Fg => return Style::default(),
        C::Accent => 215,
        C::Ok => 114,
        C::Warn => 221,
        C::Err => 210,
        C::Blue => 117,
        C::Dim => 245,
        C::Faint => 239,
    };
    Style::default().fg(Color::Indexed(i))
}

fn heading() -> Style {
    st(C::Dim).add_modifier(Modifier::BOLD)
}

fn selected() -> Style {
    if no_color() {
        Style::default().add_modifier(Modifier::REVERSED)
    } else {
        Style::default().bg(Color::Indexed(236))
    }
}

/// A line built from pieces: `L::new().t("text", C::Dim).t(...)`.
struct L(Vec<Span<'static>>);

impl L {
    fn new() -> Self {
        L(Vec::new())
    }
    fn t(mut self, text: impl Into<String>, c: C) -> Self {
        self.0.push(Span::styled(text.into(), st(c)));
        self
    }
    fn s(mut self, text: impl Into<String>, style: Style) -> Self {
        self.0.push(Span::styled(text.into(), style));
        self
    }
    /// The tool and target of the step in progress: the model, or the tool call it asked for.
    fn pending(self, r: &Run, target_w: usize) -> Self {
        match &r.current {
            Some((tool, target)) => {
                self.t(fit(tool, 9), C::Accent).t(fit(target, target_w), C::Fg).t(fit("running", 20), C::Accent)
            }
            None => self.t(fit("model", 9), C::Blue).t(fit("waiting for reply…", target_w + 20), C::Dim),
        }
    }
    fn line(self) -> Line<'static> {
        Line::from(self.0)
    }
    fn sel(self, on: bool) -> Line<'static> {
        let line = Line::from(self.0);
        if on { line.style(selected()) } else { line }
    }
}

fn blank() -> Line<'static> {
    Line::from("")
}

fn title(text: &str, note: &str) -> Line<'static> {
    L::new().t(" ", C::Fg).s(text.to_string(), heading()).t(format!("  {note}"), C::Dim).line()
}

/// A screen's lines, and which of them can be clicked or selected.
struct Out {
    lines: Vec<Line<'static>>,
    links: Vec<(usize, Link)>,
}

impl Out {
    fn new() -> Self {
        Out { lines: Vec::new(), links: Vec::new() }
    }
    fn push(&mut self, line: Line<'static>) {
        self.lines.push(line);
    }
    /// A row that selects and opens `link`.
    fn link(&mut self, line: Line<'static>, link: Link) {
        self.links.push((self.lines.len(), link));
        self.lines.push(line);
    }
    fn len(&self) -> usize {
        self.lines.len()
    }
    fn pop(&mut self) {
        self.lines.pop();
        let n = self.lines.len();
        self.links.retain(|(i, _)| *i < n);
    }
    fn resize(&mut self, n: usize) {
        self.lines.truncate(n);
        self.links.retain(|(i, _)| *i < n);
        self.lines.resize(n, blank());
    }
}

/// A line of buttons and labels that remembers where each button is.
struct Bar {
    spans: Vec<Span<'static>>,
    x: u16,
    y: u16,
    hits: Vec<Hit>,
}

impl Bar {
    fn new(y: u16) -> Self {
        Bar { spans: Vec::new(), x: 0, y, hits: Vec::new() }
    }
    fn text(&mut self, text: impl Into<String>, style: Style) {
        let text = text.into();
        self.x += width(&text) as u16;
        self.spans.push(Span::styled(text, style));
    }
    fn button(&mut self, text: impl Into<String>, style: Style, target: Target) {
        let x0 = self.x;
        self.text(text, style);
        self.hits.push(Hit { y: self.y, x0, x1: self.x, target });
    }
    /// A button drawn as a key and a label: " p Pause ".
    fn key_button(&mut self, key: &str, label: &str, on: bool, target: Target) {
        let x0 = self.x;
        let (ks, ls) = if on { (button_on(), button_on()) } else { (button().patch(st(C::Accent)), button()) };
        self.text(format!(" {key} "), ks.add_modifier(Modifier::BOLD));
        self.text(format!("{label} "), ls);
        self.hits.push(Hit { y: self.y, x0, x1: self.x, target });
        self.text(" ", Style::default());
    }
    fn shift(&mut self, dx: u16) {
        self.x += dx;
        for h in &mut self.hits {
            h.x0 += dx;
            h.x1 += dx;
        }
    }
}

fn button() -> Style {
    if no_color() {
        Style::default().add_modifier(Modifier::REVERSED)
    } else {
        Style::default().bg(Color::Indexed(237))
    }
}

fn button_on() -> Style {
    if no_color() {
        Style::default().add_modifier(Modifier::REVERSED | Modifier::BOLD)
    } else {
        Style::default().bg(Color::Indexed(215)).fg(Color::Indexed(16))
    }
}

/// Left and right parts of a bar, joined with the space between them.
fn join(mut left: Bar, mut right: Bar, w: usize, hits: &mut Vec<Hit>) -> Line<'static> {
    let gap = (w as u16).saturating_sub(left.x + right.x);
    right.shift(left.x + gap);
    left.spans.push(Span::raw(" ".repeat(gap as usize)));
    left.spans.extend(right.spans);
    hits.extend(left.hits);
    hits.extend(right.hits);
    Line::from(left.spans)
}

/// The rows of `items` that fit in `rows`, scrolled so `selected` is visible.
fn window<T>(items: &[T], selected: usize, rows: usize) -> (usize, &[T]) {
    if items.len() <= rows {
        return (0, items);
    }
    let start = selected.saturating_sub(rows.saturating_sub(1).min(rows / 2)).min(items.len() - rows);
    (start, &items[start..start + rows])
}

/// Draw the current screen, and return where its clickable parts are.
pub fn draw(f: &mut Frame, app: &App) -> Vec<Hit> {
    let area = f.area();
    if area.height < 10 || area.width < 50 {
        f.render_widget(Paragraph::new("thunc watch needs a terminal at least 50×10.").wrap(Wrap { trim: true }), area);
        return vec![];
    }
    let w = area.width as usize;
    let top = 3; // header, tabs, rule
    let body_h = area.height as usize - top - 2;
    let mut hits = Vec::new();
    let mut lines = vec![header(app, w), tab_bar(app, w, 1, &mut hits), rule(w)];
    let mut body = match app.screen {
        Screen::Overview => overview(app, w, body_h),
        Screen::Agents => agents(app, w, body_h),
        Screen::Calls => calls(app, w, body_h),
        Screen::Summary => summary_screen(app),
        Screen::Output => output(app, w, body_h),
    };
    body.resize(body_h);
    for (i, link) in body.links {
        hits.push(Hit { y: (top + i) as u16, x0: 0, x1: area.width, target: Target::Link(link) });
    }
    lines.extend(body.lines);
    lines.push(rule(w));
    lines.push(footer(app, w, area.height - 1, &mut hits));
    f.render_widget(Paragraph::new(lines), area);
    if let Some(d) = app.dialog {
        hits.clear(); // only the dialog answers clicks; a click anywhere else closes it
        dialog(f, app, d, &mut hits);
    }
    hits
}

fn rule(w: usize) -> Line<'static> {
    L::new().t("─".repeat(w), C::Faint).line()
}

fn header(app: &App, w: usize) -> Line<'static> {
    let (what, status, status_c): (String, String, C) = match &app.source {
        Source::Program(p) => {
            let (s, c) = if p.running() {
                ("running".to_string(), C::Ok)
            } else {
                (p.exit_text(), if p.exit_code() == 0 { C::Ok } else { C::Err })
            };
            (p.description.clone(), s, c)
        }
        Source::Events(t) => (t.path.display().to_string(), "following".into(), C::Ok),
        Source::Agents(a) => {
            let running = app.state.runs.iter().filter(|r| r.running()).count();
            (format!("{}  ·  {} agents, {running} running", a.root.display(), a.agents), "following".into(), C::Ok)
        }
        Source::Replay(r) => {
            let s = if r.done() { "replayed".to_string() } else { format!("replay ×{}", r.speed) };
            (format!("replay  ·  {} of {} events", r.played(), r.total), s, C::Blue)
        }
    };
    let (status, status_c) =
        if app.paused { (format!("paused, {} waiting", app.held()), C::Warn) } else { (status, status_c) };
    let models: Vec<String> = {
        let mut m: Vec<String> = app
            .state
            .calls
            .iter()
            .filter(|c| !c.backend.is_empty())
            .map(|c| if c.model.is_empty() { c.backend.clone() } else { format!("{}/{}", c.backend, c.model) })
            .collect();
        m.sort();
        m.dedup();
        m
    };
    let models = if models.is_empty() { String::new() } else { format!("{}   ", models.join(", ")) };
    let time = match app.source {
        Source::Agents(_) | Source::Events(_) => hms(app.state.now),
        _ => clock(app.state.wall()),
    };
    let right_dim = format!("{models}{time}   ");
    let right_status = format!("● {status} ");
    let left_w = w.saturating_sub(width(&right_dim) + width(&right_status) + 15);
    L::new()
        .t(" thunc watch  ", C::Accent)
        .s(fit(&what, left_w), Style::default().add_modifier(Modifier::BOLD))
        .t(" ", C::Fg)
        .t(right_dim, C::Dim)
        .t(right_status, status_c)
        .line()
}

fn tab_bar(app: &App, w: usize, y: u16, hits: &mut Vec<Hit>) -> Line<'static> {
    let mut left = Bar::new(y);
    left.text(" ", Style::default());
    for (i, screen) in app.screens().into_iter().enumerate() {
        let name = match screen {
            Screen::Overview => "Overview",
            Screen::Agents => "Agents",
            Screen::Calls => "Calls",
            Screen::Summary => "Summary",
            Screen::Output => "Output",
        };
        let on = app.screen == screen;
        let style = if on { button_on().add_modifier(Modifier::BOLD) } else { st(C::Dim) };
        left.button(format!(" {} {name} ", i + 1), style, Target::Tab(screen));
        left.text(" ", Style::default());
    }
    let mut right = Bar::new(y);
    if let Some(f) = &app.fn_filter {
        right.button(format!(" function {} ✕ ", short_name(f)), button(), Target::Action(Action::ClearFunction));
        right.text(" ", Style::default());
    }
    if app.failures_only {
        right.button(" only failures ✕ ".to_string(), button(), Target::Action(Action::Failures));
        right.text(" ", Style::default());
    }
    join(left, right, w, hits)
}

fn footer(app: &App, w: usize, y: u16, hits: &mut Vec<Hit>) -> Line<'static> {
    let mut right = Bar::new(y);
    right.key_button("p", if app.paused { "Resume" } else { "Pause" }, app.paused, Target::Action(Action::Pause));
    right.key_button("f", "Failures", app.failures_only, Target::Action(Action::Failures));
    right.key_button("?", "Help", false, Target::Action(Action::Help));
    right.key_button("q", "Quit", false, Target::Action(Action::Quit));
    let room = w.saturating_sub(right.x as usize + 1);
    let mut left = Bar::new(y);
    if let Some((text, _)) = &app.notice {
        left.text(fit(&format!(" {text}"), room), st(C::Warn));
    } else {
        let back = if app.screen == Screen::Overview { "" } else { "  esc back" };
        let hint = match app.screen {
            Screen::Overview | Screen::Summary => format!(" ↑↓ or click to select · ⏎ or click again to open{back}"),
            Screen::Agents => format!(" ↑↓ or click a run to see its steps · ←→ screens{back}"),
            Screen::Calls => format!(" ↑↓ or click a call to see its attempts · ←→ screens{back}"),
            Screen::Output => format!(" ↑↓ or the wheel to scroll · Home oldest · End newest{back}"),
        };
        left.text(fit(&hint, room), st(C::Dim));
    }
    join(left, right, w, hits)
}

/// Help and the quit question, drawn over the screen.
fn dialog(f: &mut Frame, app: &App, d: Dialog, hits: &mut Vec<Hit>) {
    let area = f.area();
    let (title, mut lines, buttons): (&str, Vec<Line<'static>>, Vec<(&str, Target)>) = match d {
        Dialog::Help => {
            let row = |k: &str, what: &str| {
                L::new().t(format!(" {}", fit(k, 16)), C::Accent).t(what.to_string(), C::Fg).line()
            };
            let mut l = vec![
                L::new().s(" Mouse", heading()).line(),
                row("click", "select a row, or switch tabs and press buttons"),
                row("click again", "open the selected row"),
                row("wheel", "move the selection, or scroll the output"),
                blank(),
                L::new().s(" Keys", heading()).line(),
                row("1–5  ← →  tab", "switch screens"),
                row("↑ ↓  j k", "move the selection (PgUp PgDn, Home End)"),
                row("⏎", "open it: a function's calls, a call, an agent run"),
                row("esc", "go back, or clear the filters"),
                row("f", "show only retries and failures"),
                row("p  space", "pause the display (the program keeps running)"),
                row("o", "what the program printed"),
                row("q", "quit; Ctrl+C stops the program"),
            ];
            if app.mouse {
                l.push(blank());
                l.push(L::new().t(" To select text, hold Shift (Option on macOS) while you drag.", C::Dim).line());
            }
            l.push(blank());
            ("Help", l, vec![(" Close ", Target::Action(Action::CloseDialog))])
        }
        Dialog::Quit => (
            "Quit",
            vec![
                blank(),
                L::new().t(" The program is still running.", C::Fg).line(),
                L::new().t(" Quitting stops it, as Ctrl+C would, and prints its report.", C::Dim).line(),
                blank(),
            ],
            vec![
                (" ⏎ Stop it and quit ", Target::Action(Action::StopAndQuit)),
                (" esc Keep watching ", Target::Action(Action::CloseDialog)),
            ],
        ),
    };
    let inner_w = lines.iter().map(|l| l.width()).max().unwrap_or(20).max(40) + 2;
    let bw = (inner_w as u16 + 2).min(area.width.saturating_sub(2));
    let bh = (lines.len() as u16 + 4).min(area.height.saturating_sub(2));
    let rect = Rect { x: (area.width - bw) / 2, y: (area.height - bh) / 2, width: bw, height: bh };
    // The buttons go on the last inner line, right-aligned.
    let by = rect.y + bh - 2;
    let mut bar = Bar::new(by);
    for (i, (label, target)) in buttons.into_iter().enumerate() {
        let style = if i == 0 { button_on() } else { button() };
        bar.button(label.to_string(), style, target);
        bar.text("  ", Style::default());
    }
    let pad = (bw - 2).saturating_sub(bar.x);
    bar.shift(rect.x + 1 + pad);
    lines.resize(bh as usize - 3, blank());
    let mut spans = vec![Span::raw(" ".repeat(pad as usize))];
    spans.append(&mut bar.spans);
    lines.push(Line::from(spans));
    // Clicks inside the dialog do nothing, except on its buttons; clicks outside close it.
    for y in rect.y..rect.y + bh {
        hits.push(Hit { y, x0: rect.x, x1: rect.x + bw, target: Target::Action(Action::Nothing) });
    }
    hits.extend(bar.hits);
    f.render_widget(Clear, rect);
    let block = Block::bordered().title(format!(" {title} ")).border_style(st(C::Accent));
    f.render_widget(Paragraph::new(lines).block(block), rect);
}

fn spin(app: &App) -> String {
    SPIN[app.frame as usize % SPIN.len()].to_string()
}

fn is_selected(app: &App, link: &Link) -> bool {
    matches!(&app.selected, Some(Target::Link(l)) if l == link)
}

fn marker(on: bool) -> &'static str {
    if on { " ▸ " } else { "   " }
}

fn overview(app: &App, w: usize, h: usize) -> Out {
    let s = &app.state;
    let mut out = Out::new();

    // In flight
    let flying: Vec<&Call> = s.in_flight().collect();
    let shown = flying.len().min(6);
    let more = if flying.len() > shown { format!(", {} more", flying.len() - shown) } else { String::new() };
    out.push(title("IN FLIGHT", &format!("{} running{more}", flying.len())));
    if flying.is_empty() {
        out.push(L::new().t("   nothing waiting on a model", C::Faint).line());
    }
    let input_w = w.saturating_sub(4 + 18 + 11 + 8 + 2 + 12 + 2).max(10);
    let mut flying = flying;
    flying.sort_by(|a, b| a.started.total_cmp(&b.started));
    for c in flying.iter().take(shown) {
        let el = s.now - c.started;
        let p95 = s.fn_stats(&c.function).and_then(|f| f.percentile(95.0)).unwrap_or(10.0).max(1.0);
        let attempt = c.attempts.len() + 1;
        let retry = attempt > 1;
        let link = Link::Call(c.key.clone());
        let on = is_selected(app, &link);
        out.link(
            L::new()
                .t(marker(on), C::Accent)
                .t(spin(app), C::Accent)
                .t(" ", C::Fg)
                .s(fit(short_name(&c.function), 17), Style::default().add_modifier(Modifier::BOLD))
                .t(" ", C::Fg)
                .t(fit(&c.input_line(), input_w), C::Dim)
                .t(fit(&format!(" attempt {attempt}"), 11), if retry { C::Warn } else { C::Dim })
                .t(rfit(&secs(el), 7), C::Fg)
                .t("  ", C::Fg)
                .t(bar(el / p95, 12), if retry { C::Warn } else { C::Accent })
                .sel(on),
            link,
        );
    }
    out.push(blank());

    // Functions
    out.push(
        L::new()
            .s(format!("   {}", fit("FUNCTION", 17)), heading())
            .s(" CALLS  CACHED  RETRIES  FAILED      MEAN       P95     MODEL   RECENT", heading())
            .line(),
    );
    if s.functions.is_empty() {
        out.push(L::new().t("   no calls yet", C::Faint).line());
    }
    let rows = s.functions.len().min(8);
    let at = s.functions.iter().position(|f| is_selected(app, &Link::Function(f.name.clone()))).unwrap_or(0);
    let (_, funcs) = window(&s.functions, at, rows);
    for f in funcs {
        let link = Link::Function(f.name.clone());
        let sel = is_selected(app, &link);
        let opt = |v: Option<f64>| v.map(secs).unwrap_or_else(|| "-".into());
        let recent = sparkline(&f.recent.iter().map(|(secs, _)| *secs).collect::<Vec<_>>());
        let all_ok = f.recent.iter().all(|(_, ok)| *ok);
        out.link(
            L::new()
                .t(marker(sel), C::Accent)
                .s(fit(short_name(&f.name), 17), Style::default().add_modifier(Modifier::BOLD))
                .t(rfit(&f.calls.to_string(), 6), C::Fg)
                .t(rfit(&f.cached.to_string(), 8), if f.cached > 0 { C::Blue } else { C::Faint })
                .t(rfit(&f.retries.to_string(), 9), if f.retries > 0 { C::Warn } else { C::Faint })
                .t(rfit(&f.failed.to_string(), 8), if f.failed > 0 { C::Err } else { C::Faint })
                .t(rfit(&opt(f.mean()), 10), C::Fg)
                .t(rfit(&opt(f.percentile(95.0)), 10), C::Fg)
                .t(rfit(&secs(f.model), 10), C::Dim)
                .t("   ", C::Fg)
                .t(recent, if all_ok { C::Accent } else { C::Warn })
                .sel(sel),
            link,
        );
    }
    out.push(blank());

    // Agents
    let mut runs: Vec<&Run> = s.runs.iter().filter(|r| r.running()).collect();
    let running = runs.len();
    if runs.is_empty() {
        runs = s.runs.iter().rev().take(1).collect();
    }
    out.push(title("AGENTS", &format!("{running} running")));
    if runs.is_empty() {
        out.push(L::new().t("   no agent runs yet", C::Faint).line());
    }
    for r in runs.iter().take(3) {
        let link = Link::Run(r.key.clone());
        out.link(run_line(app, r, w, is_selected(app, &link)), link);
    }
    out.push(blank());

    // Activity
    let n = w.saturating_sub(62).clamp(10, 90);
    let (busy, summed) = s.busy();
    let overlap = if busy > 0.0 { format!("   overlap {:.1}x", summed / busy) } else { String::new() };
    out.push(
        L::new()
            .t(" ", C::Fg)
            .s("ACTIVITY", heading())
            .t(format!("  results per second, last {n}s  "), C::Dim)
            .t(sparkline(&s.activity_series(n)), C::Accent)
            .t(overlap, C::Dim)
            .line(),
    );
    out.push(blank());

    // Events
    out.push(title("EVENTS", ""));
    let room = h.saturating_sub(out.len());
    let log: Vec<_> = s.log.iter().rev().take(room).collect();
    for l in log.into_iter().rev() {
        match &l.link {
            Some(link) => out.link(log_line(l, w, is_selected(app, link)), link.clone()),
            None => out.push(log_line(l, w, false)),
        }
    }
    out
}

fn run_line(app: &App, r: &Run, w: usize, on: bool) -> Line<'static> {
    let (sym, sym_c) = match &r.end {
        None => (spin(app), C::Accent),
        Some(e) if e.ok => ("✓".into(), C::Ok),
        Some(_) => ("✗".into(), C::Err),
    };
    let (action, action_c) = match (&r.end, r.on_model, r.steps.last()) {
        (Some(e), _, _) if e.ok => (format!("finished · {} steps", e.steps), C::Ok),
        (Some(e), _, _) => (format!("failed: {}", e.error.clone().unwrap_or_default()), C::Err),
        (None, true, _) => ("model   waiting for reply".to_string(), C::Blue),
        (None, false, _) => match &r.current {
            Some((tool, target)) => (format!("{tool:<7} {target}"), C::Accent),
            None => ("working".to_string(), C::Dim),
        },
    };
    let since = if r.running() { app.state.now - r.last_t } else { r.seconds(app.state.now) };
    let action_w = w.saturating_sub(4 + 16 + 32 + 9 + 9).max(10);
    L::new()
        .t(marker(on), C::Accent)
        .t(sym, sym_c)
        .t(" ", C::Fg)
        .s(fit(&r.agent, 15), Style::default().add_modifier(Modifier::BOLD))
        .t(" ", C::Fg)
        .t(fit(&r.call_line(), 31), C::Dim)
        .t(fit(&if r.running() { format!(" step {}", r.steps.len() + 1) } else { String::new() }, 9), C::Fg)
        .t(fit(&action, action_w), action_c)
        .t(rfit(&secs(since), 8), C::Dim)
        .sel(on)
}

fn log_line(l: &crate::state::LogLine, w: usize, on: bool) -> Line<'static> {
    let (sym, c) = match l.kind {
        LogKind::Ok => ("✓", C::Ok),
        LogKind::Cached => ("◆", C::Blue),
        LogKind::Retry => ("↻", C::Warn),
        LogKind::Fail => ("✗", C::Err),
        LogKind::Start => ("▶", C::Accent),
        LogKind::Step => ("·", C::Dim),
        LogKind::Denied => ("⊘", C::Err),
        LogKind::RunOk => ("✓", C::Ok),
        LogKind::RunFail => ("✗", C::Err),
    };
    let text_c = match l.kind {
        LogKind::Fail | LogKind::RunFail | LogKind::Denied => C::Err,
        LogKind::Retry => C::Warn,
        _ => C::Fg,
    };
    let note = match l.kind {
        LogKind::Cached => "cached".to_string(),
        _ => l.seconds.map(secs).unwrap_or_default(),
    };
    let text_w = w.saturating_sub(3 + 8 + 2 + 2 + 16 + 9).max(10);
    L::new()
        .t(marker(on), C::Accent)
        .t(hms(l.t), C::Dim)
        .t("  ", C::Fg)
        .t(sym, c)
        .t(" ", C::Fg)
        .s(fit(short_name(&l.who), 15), Style::default().add_modifier(Modifier::BOLD))
        .t(" ", C::Fg)
        .t(fit(&l.text, text_w), text_c)
        .t(rfit(&note, 8), if l.kind == LogKind::Cached { C::Blue } else { C::Dim })
        .sel(on)
}

fn agents(app: &App, w: usize, h: usize) -> Out {
    let s = &app.state;
    let mut out = Out::new();
    let runs = app.runs_shown();
    let sel = app.selected_run();
    let filter = if app.failures_only { "  ·  only failures and denials" } else { "" };
    out.push(
        L::new()
            .s(format!("   {}", fit("AGENT", 18)), heading())
            .s(fit("TASK", w.saturating_sub(18 + 3 + 54).max(16)), heading())
            .s("  STEPS      TIME   RESULT          STARTED", heading())
            .t(filter, C::Warn)
            .line(),
    );
    if runs.is_empty() {
        out.push(L::new().t("   no agent runs yet", C::Faint).line());
        out.push(L::new().t("   Runs appear here as soon as an agent starts one.", C::Faint).line());
        return out;
    }
    let at = sel.and_then(|x| runs.iter().position(|r| r.key == x.key)).unwrap_or(0);
    let rows = (h / 3).clamp(3, 10);
    let (_, visible) = window(&runs, at, rows);
    let task_w = w.saturating_sub(18 + 3 + 54).max(16);
    for r in visible {
        let is_sel = sel.is_some_and(|x| x.key == r.key);
        let (sym, sym_c) = match &r.end {
            None => (spin(app), C::Accent),
            Some(e) if e.ok => ("✓".into(), C::Ok),
            Some(_) => ("✗".into(), C::Err),
        };
        let (result, result_c) = match &r.end {
            None => ("running".to_string(), C::Accent),
            Some(e) if e.ok => ("ok".to_string(), C::Ok),
            Some(e) if e.error.as_deref().is_some_and(|x| x.contains("process ended")) => {
                ("interrupted".into(), C::Warn)
            }
            Some(_) => ("failed".into(), C::Err),
        };
        out.link(
            L::new()
                .t(marker(is_sel), C::Accent)
                .t(sym, sym_c)
                .t(" ", C::Fg)
                .s(fit(&r.agent, 16), Style::default().add_modifier(Modifier::BOLD))
                .t(fit(&r.call_line(), task_w), C::Dim)
                .t(rfit(&r.step_count().to_string(), 7), C::Fg)
                .t(rfit(&secs(r.seconds(s.now)), 10), C::Fg)
                .t("   ", C::Fg)
                .t(fit(&result, 16), result_c)
                .t(ago(s.now - r.started), C::Dim)
                .sel(is_sel),
            Link::Run(r.key.clone()),
        );
    }
    out.push(rule(w).style(st(C::Faint)));
    let Some(r) = sel else { return out };

    let returns = if r.returns.is_empty() { String::new() } else { format!(" -> {}", r.returns) };
    out.push(
        L::new()
            .t(" ", C::Fg)
            .t(r.agent.clone(), C::Accent)
            .t("  ·  ", C::Faint)
            .s(format!("{}{returns}", r.call_line()), Style::default().add_modifier(Modifier::BOLD))
            .line(),
    );
    if let Some(session) = &r.session {
        out.push(L::new().t(" session  ", C::Dim).t(clean(session), C::Blue).line());
    }
    out.push(blank());
    out.push(
        L::new()
            .s(format!("   {}", "STEP  TOOL     "), heading())
            .s(fit("TARGET", w.saturating_sub(3 + 15 + 30).max(12)), heading())
            .s(fit("RESULT", 20), heading())
            .s("     TIME", heading())
            .line(),
    );

    let tail_rows = 4;
    let room = h.saturating_sub(out.len() + tail_rows).max(1);
    let target_w = w.saturating_sub(3 + 15 + 30).max(12);
    let waiting = r.running() && (r.on_model || r.current.is_some());
    let steps = &r.steps;
    let show = room.saturating_sub(usize::from(waiting));
    let skip = steps.len().saturating_sub(show);
    if skip > 0 {
        out.pop();
        out.push(L::new().t(format!("   … {skip} earlier steps",), C::Faint).line());
    }
    for step in &steps[skip..] {
        let (tc, rc) = if step.denied {
            (C::Err, C::Err)
        } else if step.tool == "finish" {
            (C::Ok, C::Ok)
        } else if step.result.starts_with("error") {
            (C::Accent, C::Warn)
        } else {
            (C::Accent, C::Dim)
        };
        let result = if step.denied { format!("denied: {}", step.result) } else { step.result.clone() };
        out.push(
            L::new()
                .t(rfit(&step.n.to_string(), 7), C::Dim)
                .t("  ", C::Fg)
                .t(fit(&step.tool, 9), tc)
                .t(fit(&step.target, target_w), C::Fg)
                .t(fit(&result, 20), rc)
                .t(rfit(&secs(step.seconds), 9), C::Dim)
                .line(),
        );
    }
    if waiting {
        out.push(
            L::new()
                .t(rfit(&(steps.len() + 1).to_string(), 7), C::Fg)
                .t("  ", C::Fg)
                .pending(r, target_w)
                .t(rfit(&secs(s.now - r.last_t), 9), C::Fg)
                .t(" ", C::Fg)
                .t(spin(app), C::Accent)
                .line(),
        );
    }
    out.resize(h.saturating_sub(tail_rows));

    let model = r.model;
    let total = (model + r.tools).max(0.001);
    out.push(if !r.measured() {
        L::new()
            .t(" ", C::Fg)
            .s("TIME     ", heading())
            .t(format!("{} in all", secs(r.seconds(s.now))), C::Fg)
            .t("   the run record doesn't time the model and tools apart; step times are between its lines", C::Faint)
            .line()
    } else {
        L::new()
            .t(" ", C::Fg)
            .s("TIME     ", heading())
            .t("model ", C::Dim)
            .t(bar(model / total, 18), C::Blue)
            .t(rfit(&format!("{:.0}%", 100.0 * model / total), 5), C::Fg)
            .t("    tools ", C::Dim)
            .t(bar(r.tools / total, 18), C::Accent)
            .t(rfit(&format!("{:.0}%", 100.0 * r.tools / total), 5), C::Fg)
            .line()
    });
    let changed = r.end.as_ref().map(|e| e.files_changed.clone()).unwrap_or_default();
    out.push(
        L::new()
            .t(" ", C::Fg)
            .s("CHANGED  ", heading())
            .t(
                if changed.is_empty() { "no files".to_string() } else { changed.join(", ") },
                if changed.is_empty() { C::Dim } else { C::Fg },
            )
            .line(),
    );
    let denied: Vec<_> = r.denied().collect();
    let mut l = L::new().t(" ", C::Fg).s("DENIED   ", heading());
    l = if denied.is_empty() {
        l.t("none", C::Dim)
    } else {
        l.t(format!("{}  ", denied.len()), C::Err)
            .t(denied.iter().map(|d| format!("{} {}", d.tool, d.target)).collect::<Vec<_>>().join(", "), C::Fg)
    };
    out.push(l.line());
    let end = match &r.end {
        None => L::new().t(" ", C::Fg).s("RESULT   ", heading()).t("still running", C::Dim),
        Some(e) if e.ok => {
            L::new().t(" ", C::Fg).s("RESULT   ", heading()).t(e.value.clone().unwrap_or_default(), C::Ok)
        }
        Some(e) => L::new().t(" ", C::Fg).s("ERROR    ", heading()).t(e.error.clone().unwrap_or_default(), C::Err),
    };
    out.push(end.line());
    out
}

fn calls(app: &App, w: usize, h: usize) -> Out {
    let s = &app.state;
    let mut out = Out::new();
    let calls = app.calls_shown();
    let sel = app.selected_call();
    let mut note = Vec::new();
    if let Some(f) = &app.fn_filter {
        note.push(format!("function {}", short_name(f)));
    }
    if app.failures_only {
        note.push("only retries and failures".into());
    }
    let note = if note.is_empty() {
        format!("{} calls", calls.len())
    } else {
        format!("{} calls  ·  {}", calls.len(), note.join(", "))
    };
    out.push(title("CALLS", &note));
    let input_w = w.saturating_sub(3 + 2 + 17 + 10 + 9 + 24).max(10);
    if calls.is_empty() {
        out.push(L::new().t("   no calls match", C::Faint).line());
        return out;
    }
    let at = sel.and_then(|x| calls.iter().position(|c| c.key == x.key)).unwrap_or(0);
    let rows = (h * 2 / 5).clamp(3, 14);
    let (_, visible) = window(&calls, at, rows);
    for c in visible {
        let is_sel = sel.is_some_and(|x| x.key == c.key);
        let (sym, sc, result) = match &c.end {
            None => (spin(app), C::Accent, "waiting".to_string()),
            Some(e) if !e.ok => ("✗".into(), C::Err, e.error.clone().unwrap_or_default()),
            Some(e) if e.cached => ("◆".into(), C::Blue, format!("cached → {}", e.value.clone().unwrap_or_default())),
            Some(e) => (
                "✓".into(),
                if c.retried() { C::Warn } else { C::Ok },
                format!("→ {}", e.value.clone().unwrap_or_default()),
            ),
        };
        let attempts = c.end.as_ref().map(|e| e.attempts as usize).unwrap_or(c.attempts.len() + 1);
        out.link(
            L::new()
                .t(marker(is_sel), C::Accent)
                .t(sym, sc)
                .t(" ", C::Fg)
                .s(fit(short_name(&c.function), 17), Style::default().add_modifier(Modifier::BOLD))
                .t(fit(&c.input_line(), input_w), C::Dim)
                .t(rfit(&format!("{attempts}×"), 4), if attempts > 1 { C::Warn } else { C::Faint })
                .t(rfit(&secs(c.seconds(s.now)), 8), C::Fg)
                .t("  ", C::Fg)
                .t(fit(&result, 22), if c.failed() { C::Err } else { C::Fg })
                .sel(is_sel),
            Link::Call(c.key.clone()),
        );
    }
    out.push(rule(w).style(st(C::Faint)));
    let Some(c) = sel else { return out };

    let args = c.inputs.iter().map(|(k, v)| format!("{k}={v:?}")).collect::<Vec<_>>().join(", ");
    out.push(
        L::new()
            .t(" ", C::Fg)
            .t(short_name(&c.function).to_string(), C::Accent)
            .s(format!("({args})"), Style::default().add_modifier(Modifier::BOLD))
            .line(),
    );
    let model = if c.model.is_empty() { c.backend.clone() } else { format!("{}/{}", c.backend, c.model) };
    let (status, sc) = match &c.end {
        None => ("waiting".to_string(), C::Accent),
        Some(e) if !e.ok => ("raised ThuncError".into(), C::Err),
        Some(e) if e.cached => ("answered from the cache".into(), C::Blue),
        Some(_) => ("ok".into(), C::Ok),
    };
    out.push(
        L::new()
            .t(
                match &c.end {
                    None => format!(
                        " {model}  ·  waiting for attempt {}  ·  {}  ·  ",
                        c.attempts.len() + 1,
                        secs(c.seconds(s.now))
                    ),
                    Some(e) => format!(" {model}  ·  {} attempts  ·  {}  ·  ", e.attempts, secs(e.seconds)),
                },
                C::Dim,
            )
            .t(status, sc)
            .line(),
    );
    out.push(blank());
    if !c.attempts.is_empty() {
        let reply_w = w.saturating_sub(3 + 9 + 8 + 4).max(20) / 2;
        out.push(
            L::new()
                .s(
                    format!("   {}{}{}{}", fit("ATTEMPT", 9), fit("TIME", 8), fit("REPLY", reply_w), "RESULT"),
                    heading(),
                )
                .line(),
        );
        for a in &c.attempts {
            let (result, rc) = if a.ok {
                ("✓ accepted".to_string(), C::Ok)
            } else {
                (format!("✗ {}", a.problem.clone().unwrap_or_default()), C::Err)
            };
            out.push(
                L::new()
                    .t(format!("   {}", fit(&a.n.to_string(), 9)), C::Fg)
                    .t(fit(&secs(a.seconds), 8), C::Dim)
                    .t(fit(&a.reply.clone().unwrap_or_default(), reply_w), C::Fg)
                    .t(clean(&result), rc)
                    .line(),
            );
        }
        out.push(blank());
    }
    if let Some(e) = &c.end {
        if let Some(err) = &e.error {
            out.push(L::new().t(" ", C::Fg).s("RAISED", heading()).line());
            out.push(L::new().t(format!("   ThuncError: {}", clean(err)), C::Err).line());
            out.push(blank());
        } else if let Some(v) = &e.value {
            out.push(L::new().t(" ", C::Fg).s("RETURNED  ", heading()).t(clean(v), C::Ok).line());
            out.push(blank());
        }
    }
    if !c.inputs.is_empty() {
        out.push(
            L::new()
                .t(" ", C::Fg)
                .s("INPUTS", heading())
                .t("  previews; run with --capture to see them whole", C::Faint)
                .line(),
        );
        for (k, v) in &c.inputs {
            out.push(L::new().t(format!("   {k}: "), C::Blue).t(clean(v), C::Fg).line());
        }
    }
    out
}

fn summary_screen(app: &App) -> Out {
    let status = match app.program() {
        Some(p) if !p.running() => format!("  ·  the program {}", p.exit_text()),
        Some(_) => "  ·  so far; the program is still running".to_string(),
        None => String::new(),
    };
    let one_program = matches!(app.source, Source::Program(_) | Source::Replay(_));
    let mut out = Out::new();
    for (kind, text, link) in summary::lines(&app.state, &status, one_program) {
        let style = match kind {
            Kind::Title => Style::default().add_modifier(Modifier::BOLD),
            Kind::Section => st(C::Accent).add_modifier(Modifier::BOLD),
            Kind::Header => heading(),
            Kind::Row => Style::default(),
            Kind::Warn => st(C::Warn),
            Kind::Note => st(C::Dim),
            Kind::Blank => Style::default(),
        };
        match link {
            Some(link) => {
                let on = is_selected(app, &link);
                out.link(L::new().t(marker(on), C::Accent).s(text, style).sel(on), link);
            }
            None => out.push(L::new().t("   ", C::Fg).s(text, style).line()),
        }
    }
    out
}

fn output(app: &App, w: usize, h: usize) -> Out {
    let mut out = Out::new();
    let Some(p) = app.program() else {
        return out;
    };
    let Ok(o) = p.output.lock() else {
        return out;
    };
    out.push(title(
        "OUTPUT",
        &format!("what the program printed ({} lines; stderr in blue)", o.lines.len() + o.dropped),
    ));
    let room = h.saturating_sub(1);
    let end = o.lines.len().saturating_sub(app.out_scroll.min(o.lines.len().saturating_sub(room)));
    let start = end.saturating_sub(room);
    for (err, text) in o.lines.range(start..end) {
        out.push(
            L::new()
                .t(" ", C::Fg)
                .t(fit(text, w.saturating_sub(2)).trim_end().to_string(), if *err { C::Blue } else { C::Fg })
                .line(),
        );
    }
    if o.lines.is_empty() {
        out.push(L::new().t("   nothing yet", C::Faint).line());
    }
    out
}

/// The current screen as plain text, `w` columns by `h` rows.
pub fn render_text(app: &App, w: u16, h: u16) -> String {
    let mut t = ratatui::Terminal::new(ratatui::backend::TestBackend::new(w, h)).expect("an in-memory terminal");
    t.draw(|f| {
        draw(f, app);
    })
    .expect("drawing in memory");
    let buf = t.backend().buffer();
    (0..h)
        .map(|y| (0..w).map(|x| buf[(x, y)].symbol()).collect::<String>().trim_end().to_string())
        .collect::<Vec<_>>()
        .join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::app::Dialog;
    use crate::event::parse_event_line;
    use crate::source::Tail;

    fn render(app: &App, w: u16, h: u16) -> String {
        render_text(app, w, h)
    }

    fn sample() -> App {
        // A source that never yields anything: the test feeds the state itself.
        let mut app = App::new(Source::Events(Tail::new("/nonexistent/events.jsonl")));
        for l in [
            r#"{"v":1,"event":"call.start","t":10,"pid":1,"id":1,"function":"urgency","backend":"claude-code","model":"sonnet","inputs":{"ticket":"I was charged twice"}}"#,
            r#"{"v":1,"event":"call.attempt","t":12,"pid":1,"id":1,"n":1,"seconds":2,"ok":false,"problem":"the value 7 was rejected by the program's validation check","reply":"7"}"#,
            r#"{"v":1,"event":"call.attempt","t":13,"pid":1,"id":1,"n":2,"seconds":1,"ok":true,"reply":"4"}"#,
            r#"{"v":1,"event":"call.end","t":13,"pid":1,"id":1,"ok":true,"cached":false,"attempts":2,"seconds":3,"value":"4"}"#,
            r#"{"v":1,"event":"call.start","t":14,"pid":1,"id":2,"function":"draft_reply","backend":"claude-code","model":"sonnet","inputs":{"ticket":"Where is order A-1043?"}}"#,
            r#"{"v":1,"event":"agent.start","t":10,"pid":1,"id":3,"agent":"repo-guide","task":"tests_for","returns":"list","session":"/tmp/s.jsonl","inputs":{"feature":"caching"}}"#,
            r#"{"v":1,"event":"agent.reply","t":12,"pid":1,"id":3,"n":1,"seconds":2}"#,
            r#"{"v":1,"event":"agent.step","t":15,"pid":1,"id":3,"n":1,"tool":"run","target":"pytest tests/test_cache.py -q","result":"exit 0","seconds":3}"#,
            r#"{"v":1,"event":"agent.reply","t":16,"pid":1,"id":3,"n":2,"seconds":1}"#,
            r#"{"v":1,"event":"agent.step","t":16,"pid":1,"id":3,"n":2,"tool":"edit","target":"README.md","result":"not allowed","seconds":0,"denied":true}"#,
        ] {
            app.state.apply(parse_event_line(l).unwrap());
        }
        app.state.now = 17.0;
        app
    }

    #[test]
    fn overview_shows_calls_functions_agents_and_events() {
        let mut app = sample();
        app.screen = Screen::Overview;
        let text = render(&app, 110, 32);
        for needle in [
            "IN FLIGHT  1 running",
            "draft_reply",
            "attempt 1",
            "urgency",
            "AGENTS  1 running",
            "repo-guide",
            "model   waiting for reply",
            "EVENTS",
            "attempt 1: the value 7",
            "→ 4",
        ] {
            assert!(text.contains(needle), "missing {needle:?} in\n{text}");
        }
    }

    #[test]
    fn agents_screen_shows_steps_and_denials() {
        let mut app = sample();
        app.screen = Screen::Agents;
        let text = render(&app, 110, 30);
        for needle in [
            "tests_for(feature=\"caching\")",
            "pytest tests/test_cache.py -q",
            "denied: not allowed",
            "waiting for reply",
            "DENIED   1  edit README.md",
            "session  /tmp/s.jsonl",
        ] {
            assert!(text.contains(needle), "missing {needle:?} in\n{text}");
        }
    }

    #[test]
    fn call_screen_explains_each_attempt() {
        let mut app = sample();
        app.screen = Screen::Calls;
        app.fn_filter = Some("urgency".into());
        let text = render(&app, 110, 30);
        for needle in [
            "function urgency",
            "claude-code/sonnet",
            "✗ the value 7 was rejected",
            "✓ accepted",
            "RETURNED  4",
            "ticket: I was charged twice",
        ] {
            assert!(text.contains(needle), "missing {needle:?} in\n{text}");
        }
    }

    /// Draw a frame as the dashboard does (so clicks use its hit map), and return its text.
    fn frame(app: &mut App, w: u16, h: u16) -> String {
        let mut t = ratatui::Terminal::new(ratatui::backend::TestBackend::new(w, h)).unwrap();
        let mut hits = Vec::new();
        t.draw(|f| hits = draw(f, app)).unwrap();
        app.hits = hits;
        let buf = t.backend().buffer();
        (0..h).map(|y| (0..w).map(|x| buf[(x, y)].symbol()).collect::<String>()).collect::<Vec<_>>().join("\n")
    }

    /// Click the first place `needle` is drawn.
    fn click_on(app: &mut App, needle: &str) {
        let text = frame(app, 120, 34);
        let (y, line) = text
            .lines()
            .enumerate()
            .find(|(_, l)| l.contains(needle))
            .unwrap_or_else(|| panic!("{needle:?} isn't on screen:\n{text}"));
        let x = line[..line.find(needle).unwrap()].chars().count();
        app.click(x as u16, y as u16);
    }

    fn key(app: &mut App, code: ratatui::crossterm::event::KeyCode) {
        app.key(ratatui::crossterm::event::KeyEvent::from(code));
        frame(app, 120, 34);
    }

    #[test]
    fn tabs_and_buttons_are_clickable() {
        let mut app = sample();
        click_on(&mut app, "2 Agents");
        assert_eq!(app.screen, Screen::Agents);
        click_on(&mut app, "3 Calls");
        assert_eq!(app.screen, Screen::Calls);
        click_on(&mut app, " f Failures");
        assert!(app.failures_only);
        click_on(&mut app, "only failures ✕");
        assert!(!app.failures_only);
        click_on(&mut app, " p Pause");
        assert!(app.paused);
        assert!(frame(&mut app, 120, 34).contains(" p Resume"));
    }

    #[test]
    fn click_selects_a_row_and_a_second_click_opens_it() {
        let mut app = sample();
        click_on(&mut app, "urgency    ");
        assert_eq!(app.screen, Screen::Overview);
        assert_eq!(app.selected, Some(Target::Link(Link::Function("urgency".into()))));
        assert!(frame(&mut app, 120, 34).contains(" ▸ urgency"));
        click_on(&mut app, "urgency    ");
        assert_eq!((app.screen, app.fn_filter.as_deref()), (Screen::Calls, Some("urgency")));
        assert!(frame(&mut app, 120, 34).contains("function urgency ✕"));

        key(&mut app, ratatui::crossterm::event::KeyCode::Esc);
        assert_eq!((app.screen, app.fn_filter.as_deref()), (Screen::Overview, None));
        assert_eq!(app.selected, Some(Target::Link(Link::Function("urgency".into()))), "back keeps the selection");
    }

    #[test]
    fn events_and_agents_open_where_they_lead() {
        let mut app = sample();
        click_on(&mut app, "attempt 1: the value 7");
        click_on(&mut app, "attempt 1: the value 7");
        assert_eq!((app.screen, app.call_key.as_deref()), (Screen::Calls, Some("1:1")));

        key(&mut app, ratatui::crossterm::event::KeyCode::Esc);
        click_on(&mut app, "model   waiting for reply");
        key(&mut app, ratatui::crossterm::event::KeyCode::Enter);
        assert_eq!((app.screen, app.run_key.as_deref()), (Screen::Agents, Some("1:3")));
    }

    #[test]
    fn keys_walk_the_rows_of_the_overview() {
        let mut app = sample();
        frame(&mut app, 120, 34);
        key(&mut app, ratatui::crossterm::event::KeyCode::Down);
        assert_eq!(app.selected, Some(Target::Link(Link::Call("1:2".into()))), "the call in flight comes first");
        key(&mut app, ratatui::crossterm::event::KeyCode::Down);
        assert_eq!(app.selected, Some(Target::Link(Link::Function("urgency".into()))));
        key(&mut app, ratatui::crossterm::event::KeyCode::Right);
        assert_eq!(app.screen, Screen::Agents);
    }

    #[test]
    fn help_opens_and_a_click_outside_closes_it() {
        let mut app = sample();
        click_on(&mut app, " ? Help");
        assert_eq!(app.dialog, Some(Dialog::Help));
        let text = frame(&mut app, 120, 34);
        assert!(text.contains("click again") && text.contains("Ctrl+C stops the program"), "{text}");
        click_on(&mut app, "click again"); // inside: stays open
        assert_eq!(app.dialog, Some(Dialog::Help));
        app.click(0, 33);
        assert_eq!(app.dialog, None);
        key(&mut app, ratatui::crossterm::event::KeyCode::Char('?'));
        click_on(&mut app, " Close ");
        assert_eq!(app.dialog, None);
    }

    #[test]
    fn the_quit_question_has_clickable_answers() {
        let mut app = sample();
        app.dialog = Some(Dialog::Quit);
        click_on(&mut app, "esc Keep watching");
        assert_eq!((app.dialog, app.quit), (None, false));
        app.dialog = Some(Dialog::Quit);
        click_on(&mut app, "Stop it and quit");
        assert!(app.quit);
    }

    #[test]
    fn every_screen_draws_at_80_columns_and_in_tiny_terminals() {
        let mut app = sample();
        for screen in [Screen::Overview, Screen::Agents, Screen::Calls, Screen::Summary] {
            app.screen = screen;
            render(&app, 80, 24);
            render(&app, 50, 10);
            assert!(render(&app, 20, 5).replace('\n', " ").contains("needs a"));
        }
    }
}
