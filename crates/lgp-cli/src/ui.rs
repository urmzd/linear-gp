//! Styled terminal UI helpers following sr's patterns.
//!
//! All output goes to stderr so stdout remains clean for data.

use std::io::Write;
use std::time::Duration;

use crossterm::style::{Attribute, Color, ResetColor, SetAttribute, SetForegroundColor};
use indicatif::{ProgressBar, ProgressStyle};

/// Print a cyan bold header with a 40-char dim horizontal rule beneath it.
pub fn header(text: &str) {
    let mut stderr = std::io::stderr();
    let _ = writeln!(
        stderr,
        "  {}{}{}{}",
        SetForegroundColor(Color::Cyan),
        SetAttribute(Attribute::Bold),
        text,
        ResetColor,
    );
    let _ = writeln!(
        stderr,
        "  {}{}{}{}",
        SetForegroundColor(Color::DarkGrey),
        SetAttribute(Attribute::Dim),
        "\u{2500}".repeat(40),
        ResetColor,
    );
}

/// Print a green bold checkmark with a message (phase/step completion).
pub fn phase_ok(text: &str) {
    let mut stderr = std::io::stderr();
    let _ = writeln!(
        stderr,
        "  {}{}\u{2713}{} {}",
        SetForegroundColor(Color::Green),
        SetAttribute(Attribute::Bold),
        ResetColor,
        text,
    );
}

/// Print a yellow bold warning.
pub fn warn(text: &str) {
    let mut stderr = std::io::stderr();
    let _ = writeln!(
        stderr,
        "  {}{}\u{26A0}{} {}",
        SetForegroundColor(Color::Yellow),
        SetAttribute(Attribute::Bold),
        ResetColor,
        text,
    );
}

/// Print a cyan info icon with dim text.
pub fn info(text: &str) {
    let mut stderr = std::io::stderr();
    let _ = writeln!(
        stderr,
        "  {}\u{2139}{} {}{}{}",
        SetForegroundColor(Color::Cyan),
        ResetColor,
        SetAttribute(Attribute::Dim),
        text,
        ResetColor,
    );
}

/// Print a plain 2-space-indented line to stderr.
pub fn line(text: &str) {
    let mut stderr = std::io::stderr();
    let _ = writeln!(stderr, "  {}", text);
}

/// Create a braille spinner (cyan, 80ms tick) with the given message.
/// Returns the ProgressBar handle; call `.finish_and_clear()` when done.
pub fn spinner(msg: &str) -> ProgressBar {
    let pb = ProgressBar::new_spinner();
    pb.set_style(
        ProgressStyle::default_spinner()
            .tick_chars(
                "\u{280B}\u{2819}\u{2839}\u{2838}\u{283C}\u{2834}\u{2826}\u{2827}\u{2807}\u{280F}",
            )
            .template("  {spinner:.cyan} {msg}")
            .expect("valid spinner template"),
    );
    pb.set_message(msg.to_string());
    pb.enable_steady_tick(Duration::from_millis(80));
    pb
}
