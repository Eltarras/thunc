//! Formatting shared by the dashboard and the plain output: durations, clocks and text that has to
//! fit a column without breaking the terminal.

use unicode_width::UnicodeWidthChar;

/// A duration the way the dashboard shows it: 840ms, 4.2s, 1m02.4s.
pub fn secs(s: f64) -> String {
    if !s.is_finite() || s < 0.0 {
        return "-".into();
    }
    if s < 1.0 {
        return format!("{:.0}ms", s * 1000.0);
    }
    if s < 60.0 {
        return format!("{s:.1}s");
    }
    let m = (s / 60.0).floor();
    format!("{}m{:04.1}s", m as u64, s - m * 60.0)
}

/// Time since the program started, as 01:02.4.
pub fn clock(elapsed: f64) -> String {
    let e = elapsed.max(0.0);
    let m = (e / 60.0).floor();
    format!("{:02}:{:04.1}", m as u64, e - m * 60.0)
}

/// How long ago, roughly: 12s ago, 4m ago, 3h ago, 2d ago.
pub fn ago(dt: f64) -> String {
    let dt = dt.max(0.0);
    if dt < 5.0 {
        "just now".into()
    } else if dt < 60.0 {
        format!("{}s ago", dt as u64)
    } else if dt < 3600.0 {
        format!("{}m ago", (dt / 60.0) as u64)
    } else if dt < 86400.0 {
        format!("{}h ago", (dt / 3600.0) as u64)
    } else {
        format!("{}d ago", (dt / 86400.0) as u64)
    }
}

/// Text that is safe to draw: no newlines, tabs or terminal escape sequences, which would move the
/// cursor or change colors when a program's output or a model's reply contains them.
pub fn clean(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut chars = s.chars().peekable();
    while let Some(c) = chars.next() {
        match c {
            '\x1b' => {
                // Skip an escape sequence: ESC [ ... final byte, or ESC and one character.
                if chars.peek() == Some(&'[') {
                    chars.next();
                    for c in chars.by_ref() {
                        if ('\x40'..='\x7e').contains(&c) {
                            break;
                        }
                    }
                } else {
                    chars.next();
                }
            }
            '\n' | '\r' | '\t' => out.push(' '),
            c if c.is_control() => {}
            c => out.push(c),
        }
    }
    out
}

/// Display width of a string, in terminal cells.
pub fn width(s: &str) -> usize {
    s.chars().map(|c| c.width().unwrap_or(0)).sum()
}

/// `s` cut or padded to exactly `n` cells, ending in … when cut.
pub fn fit(s: &str, n: usize) -> String {
    let s = clean(s);
    if width(&s) <= n {
        let pad = n - width(&s);
        return s + &" ".repeat(pad);
    }
    if n == 0 {
        return String::new();
    }
    let mut out = String::new();
    let mut w = 0;
    for c in s.chars() {
        let cw = c.width().unwrap_or(0);
        if w + cw > n - 1 {
            break;
        }
        out.push(c);
        w += cw;
    }
    out.push('…');
    out + &" ".repeat(n - 1 - w)
}

/// `s` right-aligned in `n` cells.
pub fn rfit(s: &str, n: usize) -> String {
    let w = width(s);
    if w >= n { fit(s, n) } else { " ".repeat(n - w) + s }
}

/// A bar of `n` cells, filled to `frac`.
pub fn bar(frac: f64, n: usize) -> String {
    let filled = ((frac.clamp(0.0, 1.0)) * n as f64).round() as usize;
    "█".repeat(filled) + &"░".repeat(n - filled)
}

pub const SPARK: [char; 8] = ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];

/// Values as a sparkline, scaled to the largest.
pub fn sparkline(values: &[f64]) -> String {
    let max = values.iter().cloned().fold(0.0_f64, f64::max);
    values
        .iter()
        .map(|v| if max <= 0.0 { SPARK[0] } else { SPARK[((v / max) * 7.0).round().clamp(0.0, 7.0) as usize] })
        .collect()
}

/// A Unix time as local HH:MM:SS.
pub fn hms(t: f64) -> String {
    let (h, m, s) = local_hms(t as i64);
    format!("{h:02}:{m:02}:{s:02}")
}

#[cfg(unix)]
fn local_hms(t: i64) -> (i32, i32, i32) {
    // SAFETY: localtime_r only writes into the tm we pass it.
    unsafe {
        let mut tm: libc::tm = std::mem::zeroed();
        let tt = t as libc::time_t;
        if libc::localtime_r(&tt, &mut tm).is_null() {
            return utc_hms(t);
        }
        (tm.tm_hour, tm.tm_min, tm.tm_sec)
    }
}

#[cfg(not(unix))]
fn local_hms(t: i64) -> (i32, i32, i32) {
    utc_hms(t)
}

fn utc_hms(t: i64) -> (i32, i32, i32) {
    let day = t.rem_euclid(86400);
    ((day / 3600) as i32, (day % 3600 / 60) as i32, (day % 60) as i32)
}

/// A session record's "time" ("2026-10-05T14:02:11+0200", Python's %Y-%m-%dT%H:%M:%S%z) as Unix time.
pub fn parse_session_time(s: &str) -> Option<f64> {
    let b = s.as_bytes();
    if b.len() < 19 || b[4] != b'-' || b[7] != b'-' || b[10] != b'T' || b[13] != b':' || b[16] != b':' {
        return None;
    }
    let num = |r: std::ops::Range<usize>| s.get(r)?.parse::<i64>().ok();
    let (y, mo, d) = (num(0..4)?, num(5..7)?, num(8..10)?);
    let (h, mi, se) = (num(11..13)?, num(14..16)?, num(17..19)?);
    let rest = &s[19..];
    let offset = if rest.is_empty() || rest == "Z" {
        0
    } else {
        let sign = match rest.as_bytes()[0] {
            b'+' => 1,
            b'-' => -1,
            _ => return None,
        };
        let digits: String = rest[1..].chars().filter(|c| c.is_ascii_digit()).collect();
        if digits.len() < 4 {
            return None;
        }
        let oh: i64 = digits[0..2].parse().ok()?;
        let om: i64 = digits[2..4].parse().ok()?;
        sign * (oh * 3600 + om * 60)
    };
    Some((days_from_civil(y, mo, d) * 86400 + h * 3600 + mi * 60 + se - offset) as f64)
}

/// Days since 1970-01-01 for a proleptic Gregorian date (Howard Hinnant's algorithm).
fn days_from_civil(y: i64, m: i64, d: i64) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = y - era * 400;
    let doy = (153 * (m + if m > 2 { -3 } else { 9 }) + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146097 + doe - 719468
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn durations() {
        assert_eq!(secs(0.0123), "12ms");
        assert_eq!(secs(4.24), "4.2s");
        assert_eq!(secs(62.4), "1m02.4s");
        assert_eq!(clock(62.4), "01:02.4");
    }

    #[test]
    fn fitting_text() {
        assert_eq!(fit("urgency", 10), "urgency   ");
        assert_eq!(fit("draft_reply_long", 8), "draft_r…");
        assert_eq!(rfit("4.2s", 6), "  4.2s");
        assert_eq!(width(&fit("日本語のテキスト", 7)), 7);
    }

    #[test]
    fn escapes_and_newlines_never_reach_the_terminal() {
        assert_eq!(clean("a\x1b[31mred\x1b[0m\nb\tc\x07"), "ared b c");
    }

    #[test]
    fn session_times() {
        assert_eq!(parse_session_time("1970-01-01T00:00:00+0000"), Some(0.0));
        assert_eq!(parse_session_time("2026-10-05T14:02:11+0200"), Some(1_791_201_731.0));
        assert_eq!(parse_session_time("2026-10-05T12:02:11Z"), Some(1_791_201_731.0));
        assert_eq!(parse_session_time("garbage"), None);
    }
}
