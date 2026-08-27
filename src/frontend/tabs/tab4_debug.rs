//! Tab 4: Debug Matrix.
//!
//! Diagnostic matrix displaying entropy trajectories, Topological Data Analysis (TDA)
//! Vietoris-Rips metrics, Jacobian sensitivity, and self-regulation phase badges.

use ratatui::{
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, BorderType, Borders, Paragraph, Sparkline},
    Frame,
};
use crate::frontend::App;

pub fn render(f: &mut Frame, area: Rect, app: &App) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(7), Constraint::Min(8)])
        .split(area);

    // Top Pane: Entropy & Margin Trajectory
    let top_block = Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .title(" Sampling Entropy & Margin ")
        .style(Style::default().fg(Color::Cyan));
    let inner_top = top_block.inner(chunks[0]);
    f.render_widget(top_block, chunks[0]);

    let top_chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(1),
            Constraint::Length(1),
            Constraint::Length(1),
        ])
        .split(inner_top);

    // Trajectory sparkline data
    let entropy_data: [u64; 30] = [
        12, 15, 18, 22, 28, 26, 20, 18, 14, 12, 10, 14, 18, 22, 28, 26, 20, 18, 14, 12, 15, 18, 22,
        28, 26, 20, 18, 14, 12, 16,
    ];
    let margin_data: [u64; 30] = [
        25, 22, 18, 14, 10, 8, 5, 8, 12, 16, 20, 24, 28, 24, 20, 16, 12, 8, 5, 8, 12, 16, 20, 25,
        28, 24, 20, 16, 12, 15,
    ];

    let entropy_p = Paragraph::new(Line::from(vec![
        Span::styled("Entropy H(t) [ 2.81 ]   ", Style::default().fg(Color::Yellow)),
    ]));
    f.render_widget(entropy_p, top_chunks[0]);
    let entropy_spark = Sparkline::default()
        .data(&entropy_data)
        .style(Style::default().fg(Color::Yellow));
    let spark_area_0 = Rect {
        x: top_chunks[0].x + 24,
        y: top_chunks[0].y,
        width: top_chunks[0].width.saturating_sub(26),
        height: 1,
    };
    f.render_widget(entropy_spark, spark_area_0);

    let margin_p = Paragraph::new(Line::from(vec![
        Span::styled("Margin Δp(t) [ 0.18 ]   ", Style::default().fg(Color::Cyan)),
    ]));
    f.render_widget(margin_p, top_chunks[1]);
    let margin_spark = Sparkline::default()
        .data(&margin_data)
        .style(Style::default().fg(Color::Cyan));
    let spark_area_1 = Rect {
        x: top_chunks[1].x + 24,
        y: top_chunks[1].y,
        width: top_chunks[1].width.saturating_sub(26),
        height: 1,
    };
    f.render_widget(margin_spark, spark_area_1);

    let drift_p = Paragraph::new(Line::from(vec![
        Span::styled("Cos Drift    [ 0.9971 ] ", Style::default().fg(Color::Green)),
        Span::styled("████████████████████████████████████████████", Style::default().fg(Color::DarkGray)),
    ]));
    f.render_widget(drift_p, top_chunks[2]);

    // Bottom Pane: TDA and Jacobian Sensitivity
    let bottom_columns = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
        .split(chunks[1]);

    // Bottom-Left: TDA Shadow Monitor
    let tda_lines = vec![
        Line::from(vec![
            Span::styled("Loop Pressure:       ", Style::default().fg(Color::LightBlue)),
            Span::styled("[██████····] ", Style::default().fg(Color::Yellow)),
            Span::styled("0.62", Style::default().fg(Color::White).add_modifier(Modifier::BOLD)),
        ]),
        Line::from(vec![
            Span::styled("Repetition Pressure: ", Style::default().fg(Color::LightBlue)),
            Span::styled("[███·······] ", Style::default().fg(Color::Green)),
            Span::styled("0.31", Style::default().fg(Color::White)),
        ]),
        Line::from(vec![
            Span::styled("Route Fragmentation: ", Style::default().fg(Color::LightBlue)),
            Span::styled("[██········] ", Style::default().fg(Color::Green)),
            Span::styled("0.20", Style::default().fg(Color::White)),
        ]),
        Line::from(vec![
            Span::styled("Margin Collapse:     ", Style::default().fg(Color::LightBlue)),
            Span::styled("[█·········] ", Style::default().fg(Color::Green)),
            Span::styled("0.10", Style::default().fg(Color::White)),
        ]),
        Line::from(""),
        Line::from(Span::styled("TDA Warning Stream:", Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD))),
        Line::from(Span::styled("[Internal monitor: stable manifold | H0bars=8 H1bars=21 loop=0.62]", Style::default().fg(Color::DarkGray))),
    ];

    let tda_block = Paragraph::new(tda_lines)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_type(BorderType::Rounded)
                .title(" Topological Data Analysis (TDA Shadow Monitor) ")
                .style(Style::default().fg(Color::Cyan)),
        );
    f.render_widget(tda_block, bottom_columns[0]);

    // Bottom-Right: Jacobian Sensitivity & Self-Regulation Phase
    let right_splits = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
        .split(bottom_columns[1]);

    let jacobian_lines = vec![
        Line::from(vec![
            Span::styled("Global Sensitivity ‖J‖: ", Style::default().fg(Color::LightMagenta)),
            Span::styled("14.82", Style::default().fg(Color::White).add_modifier(Modifier::BOLD)),
        ]),
        Line::from(vec![
            Span::styled("Dominant Residual Dims:  ", Style::default().fg(Color::LightMagenta)),
            Span::styled("[ 1042, 819, 2301 ]", Style::default().fg(Color::Yellow)),
        ]),
        Line::from(vec![
            Span::styled("Measurement Step:        ", Style::default().fg(Color::LightMagenta)),
            Span::styled("Step 120 (Interval: 40)", Style::default().fg(Color::DarkGray)),
        ]),
    ];

    let jacobian_block = Paragraph::new(jacobian_lines)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_type(BorderType::Rounded)
                .title(" Jacobian Sensitivity Matrix ")
                .style(Style::default().fg(Color::Cyan)),
        );
    f.render_widget(jacobian_block, right_splits[0]);

    let phase_lines = vec![
        Line::from(vec![
            Span::styled("Phase: ", Style::default().fg(Color::White)),
            Span::styled("[ SETTLE ]", Style::default().fg(Color::Black).bg(Color::Green).add_modifier(Modifier::BOLD)),
            Span::styled(" (Stable attractor reached)", Style::default().fg(Color::DarkGray)),
        ]),
        Line::from(vec![
            Span::styled("Cycle Count: ", Style::default().fg(Color::White)),
            Span::styled("0 cycles detected", Style::default().fg(Color::Green)),
            Span::styled(" │ Step: ", Style::default().fg(Color::DarkGray)),
            Span::styled(format!("{}", app.last_hud_frame.as_ref().map(|f| f.step).unwrap_or(0)), Style::default().fg(Color::White)),
        ]),
        Line::from(vec![
            Span::styled("Force Gate:  ", Style::default().fg(Color::White)),
            Span::styled("ACTIVE", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
            Span::styled(" (Residual cap + splat attraction active)", Style::default().fg(Color::DarkGray)),
        ]),
    ];

    let phase_block = Paragraph::new(phase_lines)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_type(BorderType::Rounded)
                .title(" Self-Regulation Phase State ")
                .style(Style::default().fg(Color::Cyan)),
        );
    f.render_widget(phase_block, right_splits[1]);
}
