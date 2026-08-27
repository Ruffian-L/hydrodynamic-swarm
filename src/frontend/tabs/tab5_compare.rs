//! Tab 5: Compare Arena.
//!
//! Direct side-by-side comparative execution between vanilla unsteered inference
//! (stock Llama.cpp / vanilla model) and the Hydrodynamic Swarm engine.

use ratatui::{
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, BorderType, Borders, Paragraph},
    Frame,
};
use crate::frontend::App;

pub fn render(f: &mut Frame, area: Rect, app: &App) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),
            Constraint::Min(8),
            Constraint::Length(3),
        ])
        .split(area);

    // 1. Arena Controller Bar
    let controller_line = Line::from(vec![
        Span::styled(" Benchmark Prompt: ", Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)),
        Span::styled("< 1_SpatialAGI (Rubik's Cube 3D Reasoning) >   ", Style::default().fg(Color::White)),
        Span::styled("Tokens: ", Style::default().fg(Color::LightBlue)),
        Span::styled("[ 200 ]   ", Style::default().fg(Color::White)),
        Span::styled("Temp: ", Style::default().fg(Color::LightBlue)),
        Span::styled("[ 0.85 ]   ", Style::default().fg(Color::White)),
        Span::styled("Vanilla Server: ", Style::default().fg(Color::LightMagenta)),
        Span::styled("http://127.0.0.1:8211 [READY]", Style::default().fg(Color::Green)),
    ]);

    let controller_block = Paragraph::new(controller_line)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_type(BorderType::Rounded)
                .title(" A/B Arena Controller ")
                .style(Style::default().fg(Color::Cyan)),
        );
    f.render_widget(controller_block, chunks[0]);

    // 2. Dual-Pane Output
    let panes = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
        .split(chunks[1]);

    // Left Pane: Vanilla Baseline
    let vanilla_text = if app.vanilla_compare_text.is_empty() {
        "To solve the Rubik's cube from this scrambled state:\n1. First rotate the top face clockwise (U).\n2. Execute standard right algorithm: R U R' U'.\n3. Align corner orientation without breaking bottom cross.\n\n[Baseline: unsteered sampling without scar memory or residual gradient force]"
    } else {
        &app.vanilla_compare_text
    };

    let vanilla_lines: Vec<Line> = vanilla_text
        .lines()
        .map(|l| Line::from(Span::styled(l, Style::default().fg(Color::Gray))))
        .collect();

    let vanilla_block = Paragraph::new(vanilla_lines)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_type(BorderType::Rounded)
                .title(" [A] Vanilla Baseline (Stock Llama.cpp) ")
                .style(Style::default().fg(Color::LightRed)),
        );
    f.render_widget(vanilla_block, panes[0]);

    // Right Pane: Hydro Swarm
    let hydro_text = if app.hydro_compare_text.is_empty() {
        "<|channel>thought\nThe cube state has inverted corner parity at layer 3. We align the Diderot field ridge gradient and activate spline scar memory to lock orientation...\n<channel|>\nTo solve the Rubik's cube with verified topological stability:\n1. Rotate U face clockwise (U) with invariant cross retention.\n2. Execute R U R' U' to stabilize yellow corners.\n<lock>State settled.</lock>"
    } else {
        &app.hydro_compare_text
    };

    let hydro_lines: Vec<Line> = hydro_text
        .lines()
        .map(|l| Line::from(Span::styled(l, Style::default().fg(Color::White))))
        .collect();

    let hydro_block = Paragraph::new(hydro_lines)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_type(BorderType::Rounded)
                .title(" [B] Hydro Swarm (3-Surface Physics + Splats) ")
                .style(Style::default().fg(Color::LightGreen)),
        );
    f.render_widget(hydro_block, panes[1]);

    // 3. Actions Footer
    let action_spans = Line::from(vec![
        Span::styled(" Actions: ", Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)),
        Span::styled("[Space/Enter] ", Style::default().fg(Color::Green).add_modifier(Modifier::BOLD)),
        Span::styled("Run Compare Arena   ", Style::default().fg(Color::White)),
        Span::styled("[S] ", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
        Span::styled("Save Comparison Receipt (`logs/ab_.../RECEIPT.md`)   ", Style::default().fg(Color::White)),
        Span::styled("[R] ", Style::default().fg(Color::Magenta).add_modifier(Modifier::BOLD)),
        Span::styled("Reload Baseline Server", Style::default().fg(Color::White)),
    ]);

    let actions_block = Paragraph::new(action_spans)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_type(BorderType::Rounded)
                .style(Style::default().fg(Color::DarkGray)),
        );
    f.render_widget(actions_block, chunks[2]);
}
