//! The dashboard's state between frames: which screen, what's selected, and what keys and clicks do.
//!
//! The rule everywhere: a click or ↑↓ selects a row, and clicking the selected row again (or ⏎)
//! opens it. esc goes back to where you opened it from.

use std::time::Instant;

use ratatui::crossterm::event::{
    KeyCode, KeyEvent, KeyEventKind, KeyModifiers, MouseButton, MouseEvent, MouseEventKind,
};

use crate::event::{Ev, Key};
use crate::source::{AgentsDir, Program, Replay, Tail, unix_now};
use crate::state::{Call, Link, Run, State};

pub enum Source {
    Program(Program),
    Events(Tail),
    Agents(AgentsDir),
    Replay(Replay),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Screen {
    Overview,
    Agents,
    Calls,
    Summary,
    Output,
}

/// What a click on part of the screen does.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Target {
    Tab(Screen),
    Link(Link),
    Action(Action),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Action {
    Pause,
    Failures,
    ClearFunction,
    Help,
    Quit,
    StopAndQuit,
    CloseDialog,
    Nothing,
}

/// A clickable area, as drawn in the last frame.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Hit {
    pub y: u16,
    pub x0: u16,
    pub x1: u16, // exclusive
    pub target: Target,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dialog {
    Help,
    Quit,
}

/// Where esc goes back to.
struct Back {
    screen: Screen,
    fn_filter: Option<String>,
    failures_only: bool,
    selected: Option<Target>,
}

pub struct App {
    pub state: State,
    pub source: Source,
    pub screen: Screen,
    pub paused: bool,
    held: Vec<Ev>,
    pub selected: Option<Target>, // the selected row on the overview and summary
    pub call_key: Option<Key>,
    pub fn_filter: Option<String>,
    pub failures_only: bool,
    pub run_key: Option<Key>,
    pub out_scroll: usize, // lines up from the newest
    pub notice: Option<(String, Instant)>,
    pub dialog: Option<Dialog>,
    pub hits: Vec<Hit>,
    pub mouse: bool,
    history: Vec<Back>,
    pub quit: bool,
    pub frame: u64,
    ended_at: Option<f64>,
}

impl App {
    pub fn new(source: Source) -> Self {
        let screen = if matches!(source, Source::Agents(_)) { Screen::Agents } else { Screen::Overview };
        let mut app = App {
            state: State::new(unix_now()),
            source,
            screen,
            paused: false,
            held: Vec::new(),
            selected: None,
            call_key: None,
            fn_filter: None,
            failures_only: false,
            run_key: None,
            out_scroll: 0,
            notice: None,
            dialog: None,
            hits: Vec::new(),
            mouse: false,
            history: Vec::new(),
            quit: false,
            frame: 0,
            ended_at: None,
        };
        if let Source::Program(p) = &app.source {
            app.state.first_t = Some(p.started);
        }
        app
    }

    pub fn program(&self) -> Option<&Program> {
        match &self.source {
            Source::Program(p) => Some(p),
            _ => None,
        }
    }

    pub fn program_running(&self) -> bool {
        self.program().is_some_and(Program::running)
    }

    /// Read new events and move the clock. Called once per frame.
    pub fn tick(&mut self) {
        self.frame += 1;
        let (events, clock) = match &mut self.source {
            Source::Program(p) => {
                let ev = p.poll();
                (ev, unix_now())
            }
            Source::Events(t) => {
                (t.poll().iter().filter_map(|l| crate::event::parse_event_line(l)).collect(), unix_now())
            }
            Source::Agents(a) => (a.poll(), unix_now()),
            Source::Replay(r) => {
                let ev = r.poll();
                (ev, r.clock)
            }
        };
        if self.paused {
            self.held.extend(events);
        } else {
            for e in events {
                self.state.apply(e);
            }
        }
        let finished = match &self.source {
            Source::Program(p) => !p.running(),
            Source::Replay(r) => r.done(),
            _ => false,
        };
        if finished && self.ended_at.is_none() && self.held.is_empty() {
            self.ended_at = Some(if clock.is_finite() { clock } else { self.state.last_t });
            if !matches!(self.source, Source::Replay(_)) || self.state.calls.len() + self.state.runs.len() > 0 {
                self.go(Screen::Summary);
                if self.dialog == Some(Dialog::Quit) {
                    self.dialog = None; // nothing left to stop
                }
                self.say("The program has ended: here is its report. Press q to quit.");
            }
        }
        self.state.now = self.ended_at.unwrap_or(if clock.is_finite() { clock } else { self.state.last_t });
        if self.notice.as_ref().is_some_and(|(_, at)| at.elapsed().as_secs() >= 5) {
            self.notice = None;
        }
    }

    fn say(&mut self, text: &str) {
        self.notice = Some((text.to_string(), Instant::now()));
    }

    pub fn calls_shown(&self) -> Vec<&Call> {
        let mut calls: Vec<&Call> = self
            .state
            .calls
            .iter()
            .filter(|c| self.fn_filter.as_ref().is_none_or(|f| &c.function == f))
            .filter(|c| !self.failures_only || c.failed() || c.retried())
            .collect();
        calls.sort_by(|a, b| b.started.total_cmp(&a.started));
        calls
    }

    pub fn runs_shown(&self) -> Vec<&Run> {
        let mut runs: Vec<&Run> = self
            .state
            .runs
            .iter()
            .filter(|r| !self.failures_only || r.end.as_ref().is_some_and(|e| !e.ok) || r.denied().count() > 0)
            .collect();
        runs.sort_by(|a, b| b.running().cmp(&a.running()).then(b.started.total_cmp(&a.started)));
        runs
    }

    pub fn selected_call(&self) -> Option<&Call> {
        let calls = self.calls_shown();
        let key = self.call_key.as_ref();
        calls.iter().find(|c| Some(&c.key) == key).or(calls.first()).copied()
    }

    pub fn selected_run(&self) -> Option<&Run> {
        let runs = self.runs_shown();
        let key = self.run_key.as_ref();
        runs.iter().find(|r| Some(&r.key) == key).or(runs.first()).copied()
    }

    /// The screens in tab order: the output tab only when thunc-watch runs the program.
    pub fn screens(&self) -> Vec<Screen> {
        let mut s = vec![Screen::Overview, Screen::Agents, Screen::Calls, Screen::Summary];
        if self.program().is_some() {
            s.push(Screen::Output);
        }
        s
    }

    fn go(&mut self, screen: Screen) {
        if screen != self.screen {
            self.history.clear(); // a tab is a fresh start: esc goes back only from something opened
            self.screen = screen;
        }
    }

    fn next_screen(&mut self, step: isize) {
        let s = self.screens();
        let at = s.iter().position(|x| *x == self.screen).unwrap_or(0) as isize;
        self.go(s[(at + step).rem_euclid(s.len() as isize) as usize]);
    }

    /// The rows of the current screen that can be selected, top to bottom, from the last frame.
    fn rows(&self) -> Vec<Target> {
        let mut rows: Vec<&Hit> = self.hits.iter().filter(|h| matches!(h.target, Target::Link(_))).collect();
        rows.sort_by_key(|h| h.y);
        rows.into_iter().map(|h| h.target.clone()).collect()
    }

    fn move_selection(&mut self, down: bool, by: usize) {
        match self.screen {
            Screen::Overview | Screen::Summary => {
                let rows = self.rows();
                if rows.is_empty() {
                    return;
                }
                let at = self.selected.as_ref().and_then(|s| rows.iter().position(|r| r == s));
                let to = match at {
                    None => {
                        if down {
                            0
                        } else {
                            rows.len() - 1
                        }
                    }
                    Some(i) if down => (i + by).min(rows.len() - 1),
                    Some(i) => i.saturating_sub(by),
                };
                self.selected = Some(rows[to].clone());
            }
            Screen::Calls => {
                let calls = self.calls_shown();
                let at = self.selected_call().and_then(|s| calls.iter().position(|c| c.key == s.key)).unwrap_or(0);
                let to = if down { (at + by).min(calls.len().saturating_sub(1)) } else { at.saturating_sub(by) };
                self.call_key = calls.get(to).map(|c| c.key.clone());
            }
            Screen::Agents => {
                let runs = self.runs_shown();
                let at = self.selected_run().and_then(|s| runs.iter().position(|r| r.key == s.key)).unwrap_or(0);
                let to = if down { (at + by).min(runs.len().saturating_sub(1)) } else { at.saturating_sub(by) };
                self.run_key = runs.get(to).map(|r| r.key.clone());
            }
            Screen::Output => {
                self.out_scroll = if down { self.out_scroll.saturating_sub(by) } else { self.out_scroll + by };
            }
        }
    }

    fn select_end(&mut self, last: bool) {
        self.move_selection(last, usize::MAX / 2);
        if self.screen == Screen::Output && !last {
            self.out_scroll = usize::MAX / 2; // the oldest line; drawing clamps it
        }
    }

    pub fn key(&mut self, k: KeyEvent) {
        if k.kind == KeyEventKind::Release {
            return;
        }
        if k.modifiers.contains(KeyModifiers::CONTROL) && matches!(k.code, KeyCode::Char('c')) {
            return self.interrupt();
        }
        if let Some(dialog) = self.dialog {
            return self.dialog_key(dialog, k.code);
        }
        match k.code {
            KeyCode::Char(c @ '1'..='5') => {
                if let Some(s) = self.screens().get(c as usize - '1' as usize) {
                    self.go(*s);
                }
            }
            KeyCode::Char('o') if self.program().is_some() => self.go(Screen::Output),
            KeyCode::Right | KeyCode::Tab | KeyCode::Char('l') => self.next_screen(1),
            KeyCode::Left | KeyCode::BackTab | KeyCode::Char('h') => self.next_screen(-1),
            KeyCode::Up | KeyCode::Char('k') => self.move_selection(false, 1),
            KeyCode::Down | KeyCode::Char('j') => self.move_selection(true, 1),
            KeyCode::PageUp => self.move_selection(false, 10),
            KeyCode::PageDown => self.move_selection(true, 10),
            KeyCode::Home | KeyCode::Char('g') => self.select_end(false),
            KeyCode::End | KeyCode::Char('G') => self.select_end(true),
            KeyCode::Enter => {
                if let Some(t) = self.selected.clone()
                    && matches!(self.screen, Screen::Overview | Screen::Summary)
                {
                    self.open(t);
                }
            }
            KeyCode::Esc | KeyCode::Backspace => self.back(),
            KeyCode::Char('f') => self.act(Action::Failures),
            KeyCode::Char('p') | KeyCode::Char(' ') => self.act(Action::Pause),
            KeyCode::Char('?') => self.act(Action::Help),
            KeyCode::Char('q') => self.act(Action::Quit),
            _ => {}
        }
    }

    fn dialog_key(&mut self, dialog: Dialog, code: KeyCode) {
        match (dialog, code) {
            (Dialog::Quit, KeyCode::Enter | KeyCode::Char('y') | KeyCode::Char('s') | KeyCode::Char('q')) => {
                self.act(Action::StopAndQuit)
            }
            (Dialog::Help, KeyCode::Char('q')) => self.dialog = None,
            (_, KeyCode::Esc | KeyCode::Enter | KeyCode::Char('n') | KeyCode::Char('?')) => self.dialog = None,
            _ => {}
        }
    }

    pub fn mouse(&mut self, m: MouseEvent) {
        match m.kind {
            MouseEventKind::Down(MouseButton::Left) => self.click(m.column, m.row),
            MouseEventKind::ScrollUp if self.dialog.is_none() => {
                self.move_selection(false, if self.screen == Screen::Output { 3 } else { 1 })
            }
            MouseEventKind::ScrollDown if self.dialog.is_none() => {
                self.move_selection(true, if self.screen == Screen::Output { 3 } else { 1 })
            }
            _ => {}
        }
    }

    pub fn click(&mut self, x: u16, y: u16) {
        let hit = self.hits.iter().rev().find(|h| h.y == y && h.x0 <= x && x < h.x1).map(|h| h.target.clone());
        match hit {
            Some(Target::Tab(s)) => self.go(s),
            Some(Target::Action(a)) => self.act(a),
            Some(Target::Link(link)) => self.click_link(link),
            None if self.dialog.is_some() => self.dialog = None, // a click outside a dialog closes it
            None => {}
        }
    }

    fn click_link(&mut self, link: Link) {
        match (self.screen, &link) {
            (Screen::Calls, Link::Call(key)) => self.call_key = Some(key.clone()),
            (Screen::Agents, Link::Run(key)) => self.run_key = Some(key.clone()),
            _ => {
                let target = Target::Link(link);
                if self.selected.as_ref() == Some(&target) {
                    self.open(target);
                } else {
                    self.selected = Some(target);
                }
            }
        }
    }

    /// Open a row: a function's calls, one call, or one agent run. esc comes back here.
    pub fn open(&mut self, target: Target) {
        let Target::Link(link) = target.clone() else { return };
        self.history.push(Back {
            screen: self.screen,
            fn_filter: self.fn_filter.clone(),
            failures_only: self.failures_only,
            selected: Some(target),
        });
        self.failures_only = false;
        match link {
            Link::Function(name) => {
                self.fn_filter = Some(name);
                self.call_key = None;
                self.screen = Screen::Calls;
            }
            Link::Call(key) => {
                self.fn_filter = None;
                self.call_key = Some(key);
                self.screen = Screen::Calls;
            }
            Link::Run(key) => {
                self.run_key = Some(key);
                self.screen = Screen::Agents;
            }
        }
    }

    fn back(&mut self) {
        if let Some(b) = self.history.pop() {
            self.screen = b.screen;
            self.fn_filter = b.fn_filter;
            self.failures_only = b.failures_only;
            self.selected = b.selected;
        } else if self.fn_filter.is_some() || self.failures_only {
            self.fn_filter = None;
            self.failures_only = false;
        } else if self.selected.is_some() {
            self.selected = None;
        } else {
            self.go(Screen::Overview);
        }
    }

    pub fn act(&mut self, a: Action) {
        match a {
            Action::Pause => self.toggle_pause(),
            Action::Failures => {
                self.failures_only = !self.failures_only;
                self.say(if self.failures_only { "Showing only retries and failures." } else { "Showing everything." });
            }
            Action::ClearFunction => self.fn_filter = None,
            Action::Help => self.dialog = if self.dialog == Some(Dialog::Help) { None } else { Some(Dialog::Help) },
            Action::Quit => {
                if self.program_running() {
                    self.dialog = Some(Dialog::Quit);
                } else {
                    self.quit = true;
                }
            }
            Action::StopAndQuit => {
                if let Source::Program(p) = &mut self.source {
                    p.stop();
                }
                self.quit = true;
            }
            Action::CloseDialog => self.dialog = None,
            Action::Nothing => {}
        }
    }

    fn toggle_pause(&mut self) {
        self.paused = !self.paused;
        if !self.paused {
            for e in std::mem::take(&mut self.held) {
                self.state.apply(e);
            }
            self.say("Resumed.");
        } else {
            self.say("Paused: the program keeps running and its events wait here. Press p to resume.");
        }
    }

    fn interrupt(&mut self) {
        match &mut self.source {
            Source::Program(p) if p.running() => {
                p.stop();
                self.dialog = None;
                self.say("Stopping the program (Ctrl+C again to force it)…");
            }
            _ => self.quit = true,
        }
    }

    pub fn held(&self) -> usize {
        self.held.len()
    }
}
