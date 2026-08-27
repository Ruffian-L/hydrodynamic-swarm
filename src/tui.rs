//! Small live-control TUI used by the chat REPL.
//!
//! `/tui` enters a temporary alternate-screen panel. Arrow keys select and move
//! sliders; every change is applied to the already-loaded model session immediately.
//! The panel deliberately owns no model or physics state: the caller supplies the
//! current values and a setter callback, which keeps this UI reusable and testable.

use crossterm::cursor::{Hide, MoveTo, Show};
use crossterm::event::{self, Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers};
use crossterm::execute;
use crossterm::queue;
use crossterm::style::{Attribute, Color, Print, ResetColor, SetAttribute, SetForegroundColor};
use crossterm::terminal::{self, Clear, ClearType, EnterAlternateScreen, LeaveAlternateScreen};
use std::io::{self, Stdout, Write};

/// One mutable scalar rendered as a slider.
#[derive(Debug, Clone)]
pub struct Slider {
    pub name: String,
    pub value: f32,
    pub min: f32,
    pub max: f32,
    pub step: f32,
    initial: f32,
}

impl Slider {
    pub fn live(name: &str, value: f32, min: f32, max: f32) -> Self {
        let min = min.min(max);
        let max = max.max(min);
        let value = value.clamp(min, max);
        Self {
            name: name.to_string(),
            value,
            min,
            max,
            step: step_for(name, min, max),
            initial: value,
        }
    }

    pub fn fraction(&self) -> f32 {
        let span = self.max - self.min;
        if span <= f32::EPSILON {
            0.0
        } else {
            ((self.value - self.min) / span).clamp(0.0, 1.0)
        }
    }

    pub fn nudge(&mut self, ticks: f32) -> bool {
        self.set(self.value + self.step * ticks)
    }

    pub fn set(&mut self, value: f32) -> bool {
        let mut next = value.clamp(self.min, self.max);
        if self.step >= 1.0 {
            next = next.round();
        }
        if (next - self.value).abs() <= f32::EPSILON {
            return false;
        }
        self.value = next;
        true
    }

    fn reset(&mut self) -> bool {
        self.set(self.initial)
    }

    pub fn value_label(&self) -> String {
        if self.name.ends_with(".on") {
            if self.value >= 0.5 {
                "ON".to_string()
            } else {
                "OFF".to_string()
            }
        } else if self.name == "hook.site" {
            match self.value.round() as i32 {
                0 => "pre_layer".to_string(),
                1 => "post_attn".to_string(),
                2 => "post_mlp".to_string(),
                _ => "final_norm".to_string(),
            }
        } else if self.step < 0.001 {
            format!("{:.5}", self.value)
        } else if self.step < 1.0 {
            format!("{:.3}", self.value)
        } else {
            format!("{:.0}", self.value)
        }
    }
}

fn step_for(name: &str, min: f32, max: f32) -> f32 {
    match name {
        "residual.dt" => 0.001,
        "hook.fraction" => 0.0001,
        "hook.start" | "hook.end" | "sample.temp" | "sample.rep" => 0.05,
        "gov.velocity" | "gov.visc_thresh" => 0.005,
        "splat.scale" => 0.005,
        "splat.top_m" | "splat.top_k" | "hook.site" => 1.0,
        name if name.ends_with(".on") => 1.0,
        "gov.visc_gain" => 0.25,
        "residual.field" | "residual.splat" | "residual.goal" | "field.alpha" => 0.01,
        _ => ((max - min) / 100.0).max(0.001),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum UiIntent {
    None,
    Move(i32),
    Adjust(i32),
    NextTab,
    PreviousTab,
    SelectTab(usize),
    Min,
    Max,
    ResetOne,
    ResetAll,
    Quit,
}

fn intent_for_key(key: KeyEvent) -> UiIntent {
    let fast = key
        .modifiers
        .intersects(KeyModifiers::SHIFT | KeyModifiers::CONTROL);
    let amount = if fast { 10 } else { 1 };
    match key.code {
        KeyCode::Up | KeyCode::Char('k') => UiIntent::Move(-1),
        KeyCode::Down | KeyCode::Char('j') => UiIntent::Move(1),
        KeyCode::Tab | KeyCode::Char(']') => UiIntent::NextTab,
        KeyCode::BackTab | KeyCode::Char('[') => UiIntent::PreviousTab,
        KeyCode::Char('1') => UiIntent::SelectTab(0),
        KeyCode::Char('2') => UiIntent::SelectTab(1),
        KeyCode::Char('3') => UiIntent::SelectTab(2),
        KeyCode::Char('4') => UiIntent::SelectTab(3),
        KeyCode::Left | KeyCode::Char('h') | KeyCode::Char('-') => UiIntent::Adjust(-amount),
        KeyCode::Right | KeyCode::Char('l') | KeyCode::Char('+') | KeyCode::Char('=') => {
            UiIntent::Adjust(amount)
        }
        KeyCode::PageDown => UiIntent::Adjust(-10),
        KeyCode::PageUp => UiIntent::Adjust(10),
        KeyCode::Home => UiIntent::Min,
        KeyCode::End => UiIntent::Max,
        KeyCode::Char('r') => UiIntent::ResetOne,
        KeyCode::Char('R') => UiIntent::ResetAll,
        KeyCode::Esc | KeyCode::Enter | KeyCode::Char('q') => UiIntent::Quit,
        _ => UiIntent::None,
    }
}

const TAB_NAMES: [&str; 4] = ["Residual", "Logit", "Hook", "Sampling"];

fn tab_for_slider(name: &str) -> usize {
    if name.starts_with("residual.") {
        0
    } else if name.starts_with("hook.") {
        2
    } else if name.starts_with("sample.") {
        3
    } else {
        1
    }
}

fn tab_indices(sliders: &[Slider], tab: usize) -> Vec<usize> {
    sliders
        .iter()
        .enumerate()
        .filter_map(|(index, slider)| (tab_for_slider(&slider.name) == tab).then_some(index))
        .collect()
}

fn move_tab(tab: usize, delta: i32) -> usize {
    (tab as i32 + delta).rem_euclid(TAB_NAMES.len() as i32) as usize
}

struct TerminalGuard;

impl TerminalGuard {
    fn enter(stdout: &mut Stdout) -> io::Result<Self> {
        terminal::enable_raw_mode()?;
        if let Err(error) = execute!(stdout, EnterAlternateScreen, Hide) {
            let _ = terminal::disable_raw_mode();
            return Err(error);
        }
        Ok(Self)
    }
}

impl Drop for TerminalGuard {
    fn drop(&mut self) {
        let mut stdout = io::stdout();
        let _ = execute!(stdout, Show, LeaveAlternateScreen, ResetColor);
        let _ = terminal::disable_raw_mode();
    }
}

/// Open the live slider panel and apply every edit immediately.
///
/// Returns the final value of each parameter changed during this visit.
pub fn run_slider_tui(
    title: &str,
    sliders: &mut [Slider],
    mut apply: impl FnMut(&str, f32) -> bool,
) -> io::Result<Vec<(String, f32)>> {
    if sliders.is_empty() {
        return Ok(Vec::new());
    }

    let mut stdout = io::stdout();
    let _guard = TerminalGuard::enter(&mut stdout)?;
    let mut tab = 0usize;
    let mut selected = 0usize;
    let mut offset = 0usize;
    let mut changed: Vec<(String, f32)> = Vec::new();
    let mut status = String::from("Changes apply immediately");

    loop {
        render(
            &mut stdout,
            title,
            sliders,
            tab,
            selected,
            &mut offset,
            &status,
            changed.len(),
        )?;

        let Event::Key(key) = event::read()? else {
            continue;
        };
        if !matches!(key.kind, KeyEventKind::Press | KeyEventKind::Repeat) {
            continue;
        }

        match intent_for_key(key) {
            UiIntent::None => {}
            UiIntent::Quit => break,
            UiIntent::NextTab | UiIntent::PreviousTab => {
                let delta = if matches!(intent_for_key(key), UiIntent::NextTab) {
                    1
                } else {
                    -1
                };
                tab = move_tab(tab, delta);
                selected = 0;
                offset = 0;
                status = format!("{} controls", TAB_NAMES[tab]);
            }
            UiIntent::SelectTab(next) => {
                tab = next;
                selected = 0;
                offset = 0;
                status = format!("{} controls", TAB_NAMES[tab]);
            }
            UiIntent::Move(delta) => {
                let indices = tab_indices(sliders, tab);
                selected = move_selection(selected, indices.len(), delta);
                status = format!("Selected {}", sliders[indices[selected]].name);
            }
            UiIntent::Adjust(ticks) => {
                let indices = tab_indices(sliders, tab);
                let index = indices[selected];
                if sliders[index].nudge(ticks as f32) {
                    apply_slider(&mut sliders[index], &mut apply, &mut changed, &mut status);
                }
            }
            UiIntent::Min => {
                let indices = tab_indices(sliders, tab);
                let index = indices[selected];
                let min = sliders[index].min;
                if sliders[index].set(min) {
                    apply_slider(&mut sliders[index], &mut apply, &mut changed, &mut status);
                }
            }
            UiIntent::Max => {
                let indices = tab_indices(sliders, tab);
                let index = indices[selected];
                let max = sliders[index].max;
                if sliders[index].set(max) {
                    apply_slider(&mut sliders[index], &mut apply, &mut changed, &mut status);
                }
            }
            UiIntent::ResetOne => {
                let indices = tab_indices(sliders, tab);
                let index = indices[selected];
                if sliders[index].reset() {
                    apply_slider(&mut sliders[index], &mut apply, &mut changed, &mut status);
                }
            }
            UiIntent::ResetAll => {
                for slider in sliders.iter_mut() {
                    if slider.reset() {
                        apply_slider(slider, &mut apply, &mut changed, &mut status);
                    }
                }
                status = String::from("Reset all controls to panel-entry values");
            }
        }
    }

    Ok(changed)
}

fn apply_slider(
    slider: &mut Slider,
    apply: &mut impl FnMut(&str, f32) -> bool,
    changed: &mut Vec<(String, f32)>,
    status: &mut String,
) {
    if apply(&slider.name, slider.value) {
        if let Some((_, value)) = changed.iter_mut().find(|(name, _)| *name == slider.name) {
            *value = slider.value;
        } else {
            changed.push((slider.name.clone(), slider.value));
        }
        *status = format!("Applied {} = {}", slider.name, slider.value_label());
    } else {
        *status = format!("Could not apply {}", slider.name);
    }
}

fn move_selection(current: usize, len: usize, delta: i32) -> usize {
    if len == 0 {
        return 0;
    }
    (current as i32 + delta).rem_euclid(len as i32) as usize
}

fn render(
    stdout: &mut Stdout,
    title: &str,
    sliders: &[Slider],
    tab: usize,
    selected: usize,
    offset: &mut usize,
    status: &str,
    changed_count: usize,
) -> io::Result<()> {
    let (width, height) = terminal::size().unwrap_or((100, 30));
    let visible = height.saturating_sub(7).max(1) as usize;
    let indices = tab_indices(sliders, tab);
    if indices.is_empty() {
        return Ok(());
    }
    if selected < *offset {
        *offset = selected;
    } else if selected >= *offset + visible {
        *offset = selected + 1 - visible;
    }
    let end = (*offset + visible).min(indices.len());
    let bar_width = (width as usize).saturating_sub(47).max(8).min(42);

    queue!(
        stdout,
        MoveTo(0, 0),
        Clear(ClearType::All),
        SetForegroundColor(Color::Cyan),
        SetAttribute(Attribute::Bold),
        Print(format!("  {title}\r\n")),
        ResetColor,
        SetAttribute(Attribute::Reset),
        SetForegroundColor(Color::Yellow),
        Print("  "),
        Print(
            TAB_NAMES
                .iter()
                .enumerate()
                .map(|(index, name)| {
                    if index == tab {
                        format!("[{} {}]", index + 1, name)
                    } else {
                        format!(" {} {} ", index + 1, name)
                    }
                })
                .collect::<Vec<_>>()
                .join("  "),
        ),
        Print("\r\n"),
        ResetColor,
        SetForegroundColor(Color::DarkGrey),
        Print(
            "  live model controls · Tab/Shift-Tab or [/] switches groups · arrows move/adjust\r\n"
        ),
        ResetColor,
        Print("\r\n")
    )?;

    for (row, &idx) in indices[*offset..end].iter().enumerate() {
        let slider = &sliders[idx];
        let position = *offset + row;
        let filled = (slider.fraction() * bar_width as f32).round() as usize;
        let empty = bar_width.saturating_sub(filled);
        let marker = if position == selected { "▶" } else { " " };
        let name_color = if position == selected {
            Color::White
        } else {
            Color::Grey
        };
        queue!(
            stdout,
            SetForegroundColor(name_color),
            SetAttribute(if position == selected {
                Attribute::Bold
            } else {
                Attribute::Reset
            }),
            Print(format!(" {marker} {:<18} [", slider.name)),
            SetForegroundColor(Color::Cyan),
            Print("█".repeat(filled)),
            SetForegroundColor(Color::DarkGrey),
            Print("·".repeat(empty)),
            SetForegroundColor(name_color),
            Print(format!("] {:>10}\r\n", slider.value_label())),
            ResetColor,
            SetAttribute(Attribute::Reset)
        )?;
    }

    let scroll = if indices.len() > visible {
        format!("rows {}–{} of {} · ", *offset + 1, end, indices.len())
    } else {
        String::new()
    };
    queue!(
        stdout,
        Print("\r\n"),
        SetForegroundColor(Color::Yellow),
        Print(format!("  {status}\r\n")),
        SetForegroundColor(Color::DarkGrey),
        Print(format!(
            "  {scroll}←/→ adjust · PgUp/PgDn ×10 · Home/End · r reset · R reset all\r\n"
        )),
        Print(format!(
            "  Enter/Esc/q close · {} tab · {changed_count} parameter(s) changed\r\n",
            TAB_NAMES[tab]
        )),
        ResetColor
    )?;
    stdout.flush()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slider_nudges_clamps_and_resets() {
        let mut slider = Slider::live("field.alpha", 0.15, 0.0, 1.0);
        assert_eq!(slider.step, 0.01);
        assert!(slider.nudge(2.0));
        assert!((slider.value - 0.17).abs() < 1e-6);
        slider.set(99.0);
        assert_eq!(slider.value, 1.0);
        slider.reset();
        assert!((slider.value - 0.15).abs() < 1e-6);
    }

    #[test]
    fn enum_and_boolean_labels_are_human_readable() {
        let mut enabled = Slider::live("hook.on", 1.0, 0.0, 1.0);
        assert_eq!(enabled.value_label(), "ON");
        enabled.set(0.0);
        assert_eq!(enabled.value_label(), "OFF");

        let site = Slider::live("hook.site", 2.0, 0.0, 3.0);
        assert_eq!(site.value_label(), "post_mlp");
    }

    #[test]
    fn selection_wraps() {
        assert_eq!(move_selection(0, 4, -1), 3);
        assert_eq!(move_selection(3, 4, 1), 0);
        assert_eq!(move_selection(1, 4, 2), 3);
    }

    #[test]
    fn keymap_supports_fast_adjust_and_exit() {
        let fast_right = KeyEvent::new(KeyCode::Right, KeyModifiers::SHIFT);
        assert_eq!(intent_for_key(fast_right), UiIntent::Adjust(10));
        let escape = KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE);
        assert_eq!(intent_for_key(escape), UiIntent::Quit);
    }

    #[test]
    fn tabs_group_controls_and_switch_directly() {
        let sliders = vec![
            Slider::live("residual.cap", 1.0, 0.0, 20.0),
            Slider::live("field.alpha", 0.1, 0.0, 1.0),
            Slider::live("hook.on", 1.0, 0.0, 1.0),
            Slider::live("sample.temp", 0.8, 0.0, 2.0),
        ];
        assert_eq!(tab_indices(&sliders, 0), vec![0]);
        assert_eq!(tab_indices(&sliders, 1), vec![1]);
        assert_eq!(tab_indices(&sliders, 2), vec![2]);
        assert_eq!(tab_indices(&sliders, 3), vec![3]);
        assert_eq!(
            intent_for_key(KeyEvent::new(KeyCode::Tab, KeyModifiers::NONE)),
            UiIntent::NextTab
        );
        assert_eq!(
            intent_for_key(KeyEvent::new(KeyCode::Char('4'), KeyModifiers::NONE)),
            UiIntent::SelectTab(3)
        );
    }
}
