//! Tab 3: System Deck.
//!
//! Live System Prompt editing, multi-turn format inspection, and control channel
//! protocol tags (<spike>, <lock>, <focus>, <remember>).

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
        .constraints([Constraint::Min(10), Constraint::Length(3)])
        .split(area);

    let columns = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
        .split(chunks[0]);

    // Left Pane: System Prompt Editor
    let prompt_text = if app.system_prompt_buffer.is_empty() {
        "You are an autonomous agent equipped with a Choice-Driven KV Cache and Niodoo 3-surface physics steering.\n\nAvailable control tags:\n- <spike>: fork reality timeline preview (10 tokens)\n- <lock>: commit answer stream\n- <focus>: tighten goal attractor\n- <remember>k=v</remember>: persist key-value pair to memory"
    } else {
        &app.system_prompt_buffer
    };

    let editor_lines: Vec<Line> = vec![
        Line::from(vec![
            Span::styled("Template: ", Style::default().fg(Color::Yellow)),
            Span::styled("< Gemma 4 Control Channel Protocol >", Style::default().fg(Color::Green).add_modifier(Modifier::BOLD)),
        ]),
        Line::from(""),
    ]
    .into_iter()
    .chain(prompt_text.lines().map(|l| Line::from(Span::styled(l, Style::default().fg(Color::White)))))
    .collect();

    let editor_block = Paragraph::new(editor_lines)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_type(BorderType::Rounded)
                .title(" Active System Prompt Deck ")
                .style(Style::default().fg(Color::Cyan)),
        );
    f.render_widget(editor_block, columns[0]);

    // Right Pane: Packed Prompt Inspection
    let packed_lines = vec![
        Line::from(vec![
            Span::styled("Status: ", Style::default().fg(Color::Yellow)),
            Span::styled("[ PRESENT ]", Style::default().fg(Color::Green).add_modifier(Modifier::BOLD)),
            Span::styled(" (Gemma 4 Control Channel Active)", Style::default().fg(Color::DarkGray)),
        ]),
        Line::from(""),
        Line::from(Span::styled("<bos><|turn>system", Style::default().fg(Color::Cyan))),
        Line::from(Span::styled("You are an autonomous agent equipped with a Choice-Driven KV Cache...", Style::default().fg(Color::DarkGray))),
        Line::from(Span::styled("Available control tags in Control Tag Quick-Palette:", Style::default().fg(Color::DarkGray))),
        Line::from(Span::styled("  - <spike>: fork timeline preview", Style::default().fg(Color::Yellow))),
        Line::from(Span::styled("  - <lock>: commit answer stream", Style::default().fg(Color::Yellow))),
        Line::from(Span::styled("  - <focus>: tighten goal attractor", Style::default().fg(Color::Yellow))),
        Line::from(Span::styled("  - <remember>k=v</remember>: persistent store", Style::default().fg(Color::Yellow))),
        Line::from(Span::styled("<turn|>", Style::default().fg(Color::Cyan))),
        Line::from(Span::styled("<|turn>user", Style::default().fg(Color::LightBlue))),
        Line::from(Span::styled(if app.prompt_buffer.is_empty() { "Explain the physics of self-regulation." } else { &app.prompt_buffer }, Style::default().fg(Color::White))),
        Line::from(Span::styled("<turn|>", Style::default().fg(Color::LightBlue))),
        Line::from(Span::styled("<|turn>model", Style::default().fg(Color::Magenta))),
        Line::from(Span::styled("<|channel>thought", Style::default().fg(Color::Magenta))),
        Line::from(Span::styled("<channel|>", Style::default().fg(Color::Magenta))),
    ];

    let packed_block = Paragraph::new(packed_lines)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_type(BorderType::Rounded)
                .title(" Packed Prompt & Control Tag Quick-Palette ")
                .style(Style::default().fg(Color::Cyan)),
        );
    f.render_widget(packed_block, columns[1]);

    // Bottom Controls Bar
    let control_spans = Line::from(vec![
        Span::styled(" Controls: ", Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)),
        Span::styled("[Ctrl+S] ", Style::default().fg(Color::Green).add_modifier(Modifier::BOLD)),
        Span::styled("Apply System Prompt   ", Style::default().fg(Color::White)),
        Span::styled("[Ctrl+R] ", Style::default().fg(Color::Red).add_modifier(Modifier::BOLD)),
        Span::styled("Reset Default   ", Style::default().fg(Color::White)),
        Span::styled("[T] ", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
        Span::styled("Toggle Control Tags (ON)", Style::default().fg(Color::White)),
    ]);

    let controls_block = Paragraph::new(control_spans)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_type(BorderType::Rounded)
                .style(Style::default().fg(Color::DarkGray)),
        );
    f.render_widget(controls_block, chunks[1]);
}
