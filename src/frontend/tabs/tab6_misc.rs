//! Tab 6: Misc (KV Cache & Remember Store).
//!
//! Management of Choice-Driven KV Cache snapshots, timeline rollback,
//! and persistent Remember Store JSONL entries.

use ratatui::{
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, BorderType, Borders, List, ListItem, Paragraph},
    Frame,
};
use crate::frontend::App;

pub fn render(f: &mut Frame, area: Rect, app: &App) {
    let columns = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
        .split(area);

    // Left Column: Choice-Driven KV Cache
    let left_splits = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Min(8), Constraint::Length(4)])
        .split(columns[0]);

    let kv_timeline_items = vec![
        ListItem::new(Line::from(vec![
            Span::styled(" ▶ [0] ", Style::default().fg(Color::Green)),
            Span::styled("init_prefill", Style::default().fg(Color::White).add_modifier(Modifier::BOLD)),
            Span::styled(" (Tokens: 128 - Prefill baseline)", Style::default().fg(Color::DarkGray)),
        ])),
        ListItem::new(Line::from(vec![
            Span::styled("   [1] ", Style::default().fg(Color::Gray)),
            Span::styled("snap_prefill_end", Style::default().fg(Color::White)),
            Span::styled(" (Tokens: 256 - Sinks & Pins active)", Style::default().fg(Color::DarkGray)),
        ])),
        ListItem::new(Line::from(vec![
            Span::styled("   [2] ", Style::default().fg(Color::Yellow)),
            Span::styled("spike_fork_1", Style::default().fg(Color::Yellow)),
            Span::styled(" (Tokens: 266 - Rollback preview)", Style::default().fg(Color::DarkGray)),
        ])),
    ];

    let kv_block = List::new(kv_timeline_items)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_type(BorderType::Rounded)
                .title(" Choice-Driven KV Cache Management ")
                .style(Style::default().fg(Color::Cyan)),
        );
    f.render_widget(kv_block, left_splits[0]);

    let kv_actions = Paragraph::new(vec![
        Line::from(vec![
            Span::styled("KV Actions: ", Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)),
            Span::styled("[S] ", Style::default().fg(Color::Green).add_modifier(Modifier::BOLD)),
            Span::styled("Snapshot KV   ", Style::default().fg(Color::White)),
            Span::styled("[R] ", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
            Span::styled("Restore Snapshot   ", Style::default().fg(Color::White)),
            Span::styled("[C] ", Style::default().fg(Color::Red).add_modifier(Modifier::BOLD)),
            Span::styled("Clear All", Style::default().fg(Color::White)),
        ]),
        Line::from(vec![
            Span::styled("Total Footprint: ", Style::default().fg(Color::Gray)),
            Span::styled("1024 / 4096 tokens (384 MB zero-copy)", Style::default().fg(Color::Green)),
        ]),
    ])
    .block(
        Block::default()
            .borders(Borders::ALL)
            .border_type(BorderType::Rounded)
            .style(Style::default().fg(Color::DarkGray)),
    );
    f.render_widget(kv_actions, left_splits[1]);

    // Right Column: Persistent Remember Store
    let right_splits = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Min(8), Constraint::Length(4)])
        .split(columns[1]);

    let remember_items: Vec<ListItem> = if app.remember_items.is_empty() {
        vec![
            ListItem::new(Line::from(vec![
                Span::styled(" ▶ tuesday-boy       ", Style::default().fg(Color::Yellow)),
                Span::styled("│ 13/27                ", Style::default().fg(Color::White)),
                Span::styled("│ Loaded", Style::default().fg(Color::Green)),
            ])),
            ListItem::new(Line::from(vec![
                Span::styled("   lumina-basin-7    ", Style::default().fg(Color::Gray)),
                Span::styled("│ 0.882                ", Style::default().fg(Color::White)),
                Span::styled("│ Synced", Style::default().fg(Color::Green)),
            ])),
            ListItem::new(Line::from(vec![
                Span::styled("   loop-seat         ", Style::default().fg(Color::Gray)),
                Span::styled("│ alive                ", Style::default().fg(Color::White)),
                Span::styled("│ Synced", Style::default().fg(Color::Green)),
            ])),
        ]
    } else {
        app.remember_items
            .iter()
            .map(|(k, v)| {
                ListItem::new(Line::from(vec![
                    Span::styled(format!("   {:<18} ", k), Style::default().fg(Color::Yellow)),
                    Span::styled(format!("│ {:<20} ", v), Style::default().fg(Color::White)),
                    Span::styled("│ Synced", Style::default().fg(Color::Green)),
                ]))
            })
            .collect()
    };

    let remember_block = List::new(remember_items)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_type(BorderType::Rounded)
                .title(" Persistent Remember Store (data/seat_remember.jsonl) ")
                .style(Style::default().fg(Color::Cyan)),
        );
    f.render_widget(remember_block, right_splits[0]);

    let remember_actions = Paragraph::new(vec![
        Line::from(vec![
            Span::styled("Store Actions: ", Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)),
            Span::styled("[Enter] ", Style::default().fg(Color::Green).add_modifier(Modifier::BOLD)),
            Span::styled("Upsert Entry   ", Style::default().fg(Color::White)),
            Span::styled("[D] ", Style::default().fg(Color::Red).add_modifier(Modifier::BOLD)),
            Span::styled("Delete Entry   ", Style::default().fg(Color::White)),
            Span::styled("[U] ", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
            Span::styled("Reload JSONL", Style::default().fg(Color::White)),
        ]),
        Line::from(vec![
            Span::styled("Active Seat: ", Style::default().fg(Color::Gray)),
            Span::styled("local-research-seat (Tracked in JSONL)", Style::default().fg(Color::LightBlue)),
        ]),
    ])
    .block(
        Block::default()
            .borders(Borders::ALL)
            .border_type(BorderType::Rounded)
            .style(Style::default().fg(Color::DarkGray)),
    );
    f.render_widget(remember_actions, right_splits[1]);
}
