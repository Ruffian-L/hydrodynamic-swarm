//! Tab System & Rendering Components.
//!
//! Provides the `Tab` enumeration, top tab bar navigation, footer status bar,
//! and tab-specific render dispatchers.

pub mod tab1_model;
pub mod tab2_physics;
pub mod tab3_system;
pub mod tab4_debug;
pub mod tab5_compare;
pub mod tab6_misc;

use ratatui::{
    layout::{Alignment, Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, BorderType, Borders, Paragraph, Tabs},
    Frame,
};
use crate::frontend::App;

/// The 6 canonical tabs in the unified Hydrodynamic Swarm UI.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Tab {
    #[default]
    ModelLoader = 0,
    PhysicsBoard = 1,
    SystemDeck = 2,
    DebugMatrix = 3,
    CompareArena = 4,
    Misc = 5,
}

impl Tab {
    /// All tabs in display order.
    pub const ALL: [Tab; 6] = [
        Tab::ModelLoader,
        Tab::PhysicsBoard,
        Tab::SystemDeck,
        Tab::DebugMatrix,
        Tab::CompareArena,
        Tab::Misc,
    ];

    /// Zero-based index of this tab.
    pub fn index(&self) -> usize {
        *self as usize
    }

    /// Construct tab from zero-based index.
    pub fn from_index(idx: usize) -> Option<Self> {
        match idx {
            0 => Some(Tab::ModelLoader),
            1 => Some(Tab::PhysicsBoard),
            2 => Some(Tab::SystemDeck),
            3 => Some(Tab::DebugMatrix),
            4 => Some(Tab::CompareArena),
            5 => Some(Tab::Misc),
            _ => None,
        }
    }

    /// Display title with keyboard shortcut number.
    pub fn title(&self) -> &'static str {
        match self {
            Tab::ModelLoader => "[1 Model & Config]",
            Tab::PhysicsBoard => "[2 Physics Board]",
            Tab::SystemDeck => "[3 System Deck]",
            Tab::DebugMatrix => "[4 Debug Matrix]",
            Tab::CompareArena => "[5 Compare Arena]",
            Tab::Misc => "[6 Misc & KV]",
        }
    }

    /// Navigate to the next tab in circular order.
    pub fn next(&self) -> Self {
        Self::from_index((self.index() + 1) % Self::ALL.len()).unwrap_or(Tab::ModelLoader)
    }

    /// Navigate to the previous tab in circular order.
    pub fn prev(&self) -> Self {
        Self::from_index((self.index() + Self::ALL.len() - 1) % Self::ALL.len())
            .unwrap_or(Tab::ModelLoader)
    }
}

/// Renders the top tab bar header.
pub fn render_tab_header(f: &mut Frame, area: Rect, active_tab: Tab) {
    let titles: Vec<Line> = Tab::ALL
        .iter()
        .map(|t| {
            if *t == active_tab {
                Line::from(vec![
                    Span::styled(
                        t.title(),
                        Style::default()
                            .fg(Color::Yellow)
                            .add_modifier(Modifier::BOLD | Modifier::UNDERLINED),
                    ),
                ])
            } else {
                Line::from(vec![
                    Span::styled(t.title(), Style::default().fg(Color::Gray)),
                ])
            }
        })
        .collect();

    let tabs = Tabs::new(titles)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_type(BorderType::Rounded)
                .title(" Hydrodynamic Swarm v0.2.0 — 3-Surface Physics Engine ")
                .title_alignment(Alignment::Left)
                .style(Style::default().fg(Color::Cyan)),
        )
        .select(active_tab.index())
        .highlight_style(
            Style::default()
                .fg(Color::Yellow)
                .add_modifier(Modifier::BOLD),
        );

    f.render_widget(tabs, area);
}

/// Renders the status bar footer.
pub fn render_status_footer(f: &mut Frame, area: Rect, app: &App) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(1), Constraint::Length(1)])
        .split(area);

    // Line 1: Live engine state & physics scalars
    let model_str = if app.model_name.is_empty() {
        "No Model Loaded"
    } else {
        &app.model_name
    };

    let cap_val = app.last_hud_frame.as_ref().map(|fr| fr.force_cap).unwrap_or(5.0);
    let goal_val = app.last_hud_frame.as_ref().map(|fr| fr.goal_force_scale).unwrap_or(0.15);
    let scars_val = app.last_hud_frame.as_ref().map(|fr| fr.scars).unwrap_or(0);
    let temp_val = app.last_hud_frame.as_ref().map(|fr| fr.temperature).unwrap_or(0.85);

    let status_line = Line::from(vec![
        Span::styled(" STATUS: ", Style::default().fg(Color::Black).bg(Color::Cyan).add_modifier(Modifier::BOLD)),
        Span::styled(format!(" {} ", model_str), Style::default().fg(Color::White).add_modifier(Modifier::BOLD)),
        Span::styled("│ ", Style::default().fg(Color::DarkGray)),
        Span::styled(format!("MSG: {} ", app.status_message), Style::default().fg(Color::Yellow)),
        Span::styled("│ ", Style::default().fg(Color::DarkGray)),
        Span::styled(format!("σ: {:.2} ", cap_val), Style::default().fg(Color::Green)),
        Span::styled(format!("θ: {:.2} ", goal_val), Style::default().fg(Color::LightBlue)),
        Span::styled(format!("β: {:.2} ", temp_val), Style::default().fg(Color::Magenta)),
        Span::styled(format!("Scars: {}", scars_val), Style::default().fg(Color::Yellow)),
    ]);

    let status_widget = Paragraph::new(status_line)
        .style(Style::default().bg(Color::Rgb(20, 20, 30)));
    f.render_widget(status_widget, chunks[0]);

    // Line 2: Keybinding shortcuts
    let keybindings_line = Line::from(vec![
        Span::styled(" [Tab/1-6]", Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)),
        Span::styled(" Switch Tab │", Style::default().fg(Color::Gray)),
        Span::styled(" [↑↓/jk]", Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)),
        Span::styled(" Select │", Style::default().fg(Color::Gray)),
        Span::styled(" [←→/hl]", Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)),
        Span::styled(" Adjust (Shift x10) │", Style::default().fg(Color::Gray)),
        Span::styled(" [Enter/Space]", Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)),
        Span::styled(" Action │", Style::default().fg(Color::Gray)),
        Span::styled(" [Esc/q]", Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)),
        Span::styled(" Quit │", Style::default().fg(Color::Gray)),
        Span::styled(" [Ctrl+C]", Style::default().fg(Color::Red).add_modifier(Modifier::BOLD)),
        Span::styled(" Exit", Style::default().fg(Color::Gray)),
    ]);

    let keys_widget = Paragraph::new(keybindings_line)
        .style(Style::default().bg(Color::Rgb(15, 15, 25)));
    f.render_widget(keys_widget, chunks[1]);
}

/// Dispatches active tab rendering to the appropriate tab module.
pub fn render_active_tab(f: &mut Frame, area: Rect, app: &App) {
    match app.active_tab {
        Tab::ModelLoader => tab1_model::render(f, area, app),
        Tab::PhysicsBoard => tab2_physics::render(f, area, app),
        Tab::SystemDeck => tab3_system::render(f, area, app),
        Tab::DebugMatrix => tab4_debug::render(f, area, app),
        Tab::CompareArena => tab5_compare::render(f, area, app),
        Tab::Misc => tab6_misc::render(f, area, app),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tab_indices_and_cycling() {
        assert_eq!(Tab::ModelLoader.index(), 0);
        assert_eq!(Tab::Misc.index(), 5);

        assert_eq!(Tab::ModelLoader.next(), Tab::PhysicsBoard);
        assert_eq!(Tab::Misc.next(), Tab::ModelLoader);

        assert_eq!(Tab::ModelLoader.prev(), Tab::Misc);
        assert_eq!(Tab::PhysicsBoard.prev(), Tab::ModelLoader);
    }

    #[test]
    fn test_tab_from_index() {
        assert_eq!(Tab::from_index(0), Some(Tab::ModelLoader));
        assert_eq!(Tab::from_index(5), Some(Tab::Misc));
        assert_eq!(Tab::from_index(6), None);
    }

    #[test]
    fn test_render_all_tabs_without_panic() {
        use ratatui::backend::TestBackend;
        use ratatui::Terminal;
        use crate::config::Config;
        use crate::frontend::engine_bridge::EngineBridge;

        let backend = TestBackend::new(120, 40);
        let mut terminal = Terminal::new(backend).expect("Terminal init");

        let config = Config::default();
        let bridge = EngineBridge::spawn(config, None, true);
        let mut app = App::new(bridge, Some("gemma-4-it.gguf".to_string()));

        for tab in Tab::ALL {
            app.active_tab = tab;
            terminal
                .draw(|f| {
                    let size = f.area();
                    render_tab_header(f, Rect { x: 0, y: 0, width: size.width, height: 3 }, app.active_tab);
                    render_active_tab(f, Rect { x: 0, y: 3, width: size.width, height: size.height.saturating_sub(5) }, &app);
                    render_status_footer(f, Rect { x: 0, y: size.height.saturating_sub(2), width: size.width, height: 2 }, &app);
                })
                .expect("Draw tab must succeed");
        }
    }
}
