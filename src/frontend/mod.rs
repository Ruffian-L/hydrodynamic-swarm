//! Unified 6-Tab Ratatui Frontend for Hydrodynamic Swarm.
//!
//! Provides a decoupled multi-threaded terminal user interface for model management,
//! 3-surface physics monitoring, system prompt control channels, TDA diagnostics,
//! A/B comparison arena, and Choice-Driven KV cache snapshotting.

pub mod channel;
pub mod engine_bridge;
pub mod event;
pub mod tabs;

use std::io::{self, stdout, Stdout};
use std::panic;
use std::time::Duration;

use anyhow::Result;
use crossterm::{
    cursor::{Hide, Show},
    event::{DisableMouseCapture, EnableMouseCapture, KeyEvent, KeyEventKind},
    execute,
    terminal::{
        disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen,
    },
};
use ratatui::{
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout},
    Frame, Terminal,
};

use crate::config::Config;
use crate::frontend::channel::{EngineToUiMsg, UiToEngineMsg};
use crate::frontend::engine_bridge::EngineBridge;
use crate::frontend::event::{poll_crossterm_event, AppEvent, KeyRouter};
use crate::frontend::tabs::{render_active_tab, render_status_footer, render_tab_header, Tab};
use crate::hud::HudFrame;

/// RAII Terminal Guard ensuring the terminal is always restored cleanly upon exit or panic.
pub struct TerminalGuard {
    terminal: Terminal<CrosstermBackend<Stdout>>,
}

impl TerminalGuard {
    /// Initialize raw mode, alternate screen, and panic hook.
    pub fn init() -> io::Result<Self> {
        enable_raw_mode()?;
        let mut out = stdout();
        execute!(out, EnterAlternateScreen, EnableMouseCapture, Hide)?;

        let backend = CrosstermBackend::new(out);
        let terminal = Terminal::new(backend)?;

        // Install panic hook to restore terminal if a panic occurs
        let default_panic = panic::take_hook();
        panic::set_hook(Box::new(move |info| {
            let _ = disable_raw_mode();
            let _ = execute!(
                stdout(),
                LeaveAlternateScreen,
                DisableMouseCapture,
                Show
            );
            default_panic(info);
        }));

        Ok(Self { terminal })
    }

    /// Access the underlying Ratatui terminal.
    pub fn terminal_mut(&mut self) -> &mut Terminal<CrosstermBackend<Stdout>> {
        &mut self.terminal
    }
}

impl Drop for TerminalGuard {
    fn drop(&mut self) {
        let _ = disable_raw_mode();
        let _ = execute!(
            self.terminal.backend_mut(),
            LeaveAlternateScreen,
            DisableMouseCapture,
            Show
        );
    }
}

/// Central UI application state.
pub struct App {
    /// Currently active navigation tab (0..5)
    pub active_tab: Tab,
    /// Flag signaling main loop termination
    pub should_quit: bool,
    /// Human-readable status notification
    pub status_message: String,
    /// Background engine communication bridge
    pub engine_bridge: EngineBridge,
    /// Loaded model name or identifier
    pub model_name: String,
    /// Latest scalar HUD telemetry snapshot
    pub last_hud_frame: Option<HudFrame>,
    /// Whether generation is currently active
    pub is_generating: bool,
    /// Streamed generated text output buffer
    pub generated_text: String,
    /// User input prompt text buffer
    pub prompt_buffer: String,
    /// Live system prompt text buffer
    pub system_prompt_buffer: String,
    /// RememberStore key-value pairs
    pub remember_items: Vec<(String, String)>,
    /// Vanilla comparison baseline text
    pub vanilla_compare_text: String,
    /// Hydro swarm comparison text
    pub hydro_compare_text: String,
    /// Selected index for in-tab lists/sliders
    pub selected_index: usize,
    /// Tab 1 (Model Loader & Config) state
    pub tab1_state: crate::frontend::tabs::tab1_model::Tab1State,
    /// Tab 2 (Physics Board & HookControls) state
    pub tab2_state: crate::frontend::tabs::tab2_physics::Tab2State,
}

impl App {
    /// Create a new application instance.
    pub fn new(engine_bridge: EngineBridge, model_name: Option<String>) -> Self {
        let tab1_state = crate::frontend::tabs::tab1_model::Tab1State::default();
        let tab2_state = crate::frontend::tabs::tab2_physics::Tab2State::default();
        let m_name = model_name.unwrap_or_default();

        Self {
            active_tab: Tab::ModelLoader,
            should_quit: false,
            status_message: "Engine initialized. Ready.".to_string(),
            engine_bridge,
            model_name: m_name,
            last_hud_frame: None,
            is_generating: false,
            generated_text: String::new(),
            prompt_buffer: "Explain the physics of self-regulation.".to_string(),
            system_prompt_buffer: String::new(),
            remember_items: Vec::new(),
            vanilla_compare_text: String::new(),
            hydro_compare_text: String::new(),
            selected_index: 0,
            tab1_state,
            tab2_state,
        }
    }

    /// Process a key event.
    pub fn handle_key_event(&mut self, key: KeyEvent) {
        if key.kind != KeyEventKind::Press {
            return;
        }

        // Global quit check
        if KeyRouter::is_quit(&key) {
            self.should_quit = true;
            return;
        }

        // Direct tab switching (1-6 or F1-F6)
        if let Some(tab_idx) = KeyRouter::direct_tab(&key) {
            if let Some(tab) = Tab::from_index(tab_idx) {
                self.active_tab = tab;
                self.status_message = format!("Switched to {}", tab.title());
                return;
            }
        }

        // Tab cycling
        if KeyRouter::is_next_tab(&key) {
            self.active_tab = self.active_tab.next();
            self.status_message = format!("Switched to {}", self.active_tab.title());
            return;
        }
        if KeyRouter::is_prev_tab(&key) {
            self.active_tab = self.active_tab.prev();
            self.status_message = format!("Switched to {}", self.active_tab.title());
            return;
        }

        // Escape / Cancel / Abort
        if KeyRouter::is_cancel(&key) {
            if self.is_generating {
                let _ = self.engine_bridge.send(UiToEngineMsg::AbortGeneration);
                self.status_message = "Generation aborted".to_string();
            } else {
                self.status_message = "Unfocused / idle".to_string();
            }
            return;
        }

        // Vertical navigation
        if let Some(delta) = KeyRouter::vertical_nav(&key) {
            if delta < 0 {
                self.selected_index = self.selected_index.saturating_sub(1);
            } else {
                self.selected_index = self.selected_index.saturating_add(1);
            }
            self.tab1_state.selected_field_idx = self.selected_index % crate::frontend::tabs::tab1_model::Tab1State::TOTAL_FIELDS;
            self.tab2_state.selected_field_idx = self.selected_index % crate::frontend::tabs::tab2_physics::Tab2State::TOTAL_FIELDS;
            return;
        }

        // Tab-specific key routing
        match self.active_tab {
            Tab::ModelLoader => {
                // Horizontal adjustment
                if let Some(delta) = KeyRouter::horizontal_adjust(&key) {
                    if let Some(msg) = self.tab1_state.adjust_field(delta) {
                        let _ = self.engine_bridge.send(msg);
                    }
                    let field_name = match self.tab1_state.selected_field_idx {
                        0 => "Model Selection",
                        1 => "Temperature",
                        2 => "Repetition Penalty",
                        3 => "Max Tokens",
                        4 => "Top-P",
                        5 => "Top-K",
                        6 => "Model Size B",
                        7 => "Model Archetype",
                        8 => "Scaling Rule",
                        _ => "Config",
                    };
                    self.status_message = format!("Adjusted {}", field_name);
                    return;
                }

                // Confirm / Action
                if KeyRouter::is_confirm(&key) {
                    if let Some(msg) = self.tab1_state.toggle_or_action() {
                        let _ = self.engine_bridge.send(msg);
                        self.status_message = "Loaded selected model".to_string();
                    }
                    return;
                }

                // Tab 1 Hotkeys
                match key.code {
                    crossterm::event::KeyCode::Char('l') | crossterm::event::KeyCode::Char('L') => {
                        if let Some(msg) = self.tab1_state.load_selected_msg() {
                            let _ = self.engine_bridge.send(msg);
                            if let Some(m) = self.tab1_state.current_model() {
                                self.status_message = format!("Loading model: {}", m.name);
                            }
                        }
                    }
                    crossterm::event::KeyCode::Char('u') | crossterm::event::KeyCode::Char('U') => {
                        self.tab1_state.loaded_model_idx = None;
                        self.model_name = String::new();
                        self.status_message = "Unloaded model".to_string();
                    }
                    crossterm::event::KeyCode::Char('s') | crossterm::event::KeyCode::Char('S') => {
                        self.status_message = "Config saved to active profile".to_string();
                    }
                    crossterm::event::KeyCode::Char('c') | crossterm::event::KeyCode::Char('C') => {
                        let _ = self.engine_bridge.send(UiToEngineMsg::ClearKv);
                        self.status_message = "KV Cache cleared".to_string();
                    }
                    _ => {}
                }
            }

            Tab::PhysicsBoard => {
                // Horizontal adjustment
                if let Some(delta) = KeyRouter::horizontal_adjust(&key) {
                    if let Some(msg) = self.tab2_state.adjust_field(delta) {
                        let _ = self.engine_bridge.send(msg);
                    }
                    self.status_message = "Applied physics adjustment".to_string();
                    return;
                }

                // Confirm / Action
                if KeyRouter::is_confirm(&key) {
                    if let Some(msg) = self.tab2_state.toggle_or_action() {
                        let _ = self.engine_bridge.send(msg);
                        self.status_message = "Applied physics adjustment".to_string();
                    }
                    return;
                }

                // Tab 2 Hotkeys
                match key.code {
                    crossterm::event::KeyCode::Char('h') | crossterm::event::KeyCode::Char('H') => {
                        self.tab2_state.hook_on = !self.tab2_state.hook_on;
                        let _ = self.engine_bridge.send(self.tab2_state.hook_msg());
                        self.status_message = format!("Hook: {}", if self.tab2_state.hook_on { "ENABLED" } else { "DISABLED" });
                    }
                    crossterm::event::KeyCode::Char('g') | crossterm::event::KeyCode::Char('G') => {
                        self.tab2_state.gov_on = !self.tab2_state.gov_on;
                        let _ = self.engine_bridge.send(UiToEngineMsg::SetLiveParam {
                            key: "gov.on".to_string(),
                            val: if self.tab2_state.gov_on { 1.0 } else { 0.0 },
                        });
                        self.status_message = format!("Governor: {}", if self.tab2_state.gov_on { "ON" } else { "OFF" });
                    }
                    crossterm::event::KeyCode::Char('r') | crossterm::event::KeyCode::Char('R') => {
                        let pred = self.tab1_state.compute_scaling_preview();
                        let seat = self.tab1_state.predicted_hydro_seat();
                        self.tab2_state.residual_cap = seat.physics.force_cap;
                        self.tab2_state.residual_goal = seat.physics.goal_force_scale;
                        self.tab2_state.residual_field = seat.physics.field_wake_scale;
                        self.tab2_state.residual_splat = seat.physics.splat_force_scale;
                        self.tab2_state.residual_field_max = seat.physics.field_wake_max;
                        self.tab2_state.residual_splat_max = seat.physics.splat_force_max;
                        self.tab2_state.residual_goal_max = seat.physics.goal_force_max;
                        self.tab2_state.force_ramp_len = seat.physics.force_ramp_tokens;
                        self.tab2_state.force_ramp_str = seat.physics.force_ramp_start;
                        let _ = self.engine_bridge.send(UiToEngineMsg::SetLiveParam {
                            key: "residual.cap".to_string(),
                            val: seat.physics.force_cap,
                        });
                        let _ = self.engine_bridge.send(UiToEngineMsg::SetLiveParam {
                            key: "residual.goal".to_string(),
                            val: seat.physics.goal_force_scale,
                        });
                        let _ = self.engine_bridge.send(UiToEngineMsg::SetLiveParam {
                            key: "residual.field".to_string(),
                            val: seat.physics.field_wake_scale,
                        });
                        let _ = self.engine_bridge.send(UiToEngineMsg::SetLiveParam {
                            key: "residual.splat".to_string(),
                            val: seat.physics.splat_force_scale,
                        });
                        if pred.temperature_coupled {
                            self.tab1_state.temperature = pred.predicted_temperature;
                            let _ = self.engine_bridge.send(UiToEngineMsg::SetLiveParam {
                                key: "sample.temp".to_string(),
                                val: pred.predicted_temperature,
                            });
                        }
                        self.status_message = "Reset residual seat via apply_to_hydro_profile (not σ→cap)".to_string();
                    }
                    _ => {}
                }
            }

            Tab::SystemDeck => {
                if KeyRouter::is_confirm(&key) {
                    let _ = self.engine_bridge.send(UiToEngineMsg::SetSystemPrompt(self.system_prompt_buffer.clone()));
                    self.status_message = "System prompt applied to engine".to_string();
                }
            }

            Tab::DebugMatrix => {
                if KeyRouter::is_confirm(&key) {
                    self.status_message = "Debug snapshot refreshed".to_string();
                }
            }

            Tab::CompareArena => {
                if KeyRouter::is_confirm(&key) {
                    let _ = self.engine_bridge.send(UiToEngineMsg::CompareVanilla {
                        prompt: self.prompt_buffer.clone(),
                        endpoint: "http://127.0.0.1:8211".to_string(),
                    });
                    self.status_message = "Comparing Vanilla vs Hydro Swarm...".to_string();
                }
            }

            Tab::Misc => {
                if KeyRouter::is_confirm(&key) {
                    let _ = self.engine_bridge.send(UiToEngineMsg::SnapshotKv);
                    self.status_message = "KV Snapshot triggered".to_string();
                }
            }
        }
    }

    /// Process a message from the background engine worker.
    pub fn handle_engine_msg(&mut self, msg: EngineToUiMsg) {
        match msg {
            EngineToUiMsg::EngineReady => {
                self.status_message = "Engine ready".to_string();
            }
            EngineToUiMsg::ModelLoading { status, .. } => {
                self.status_message = status;
            }
            EngineToUiMsg::ModelLoaded { name, .. } => {
                self.model_name = name.clone();
                if let Some(pos) = self.tab1_state.models.iter().position(|m| m.path == name || m.name == name) {
                    self.tab1_state.loaded_model_idx = Some(pos);
                    self.tab1_state.selected_model_idx = pos;
                }
                self.status_message = format!("Model loaded: {}", name);
            }
            EngineToUiMsg::TokenGenerated { text, frame } => {
                self.is_generating = true;
                self.generated_text.push_str(&text);
                self.last_hud_frame = Some(frame);
            }
            EngineToUiMsg::GenerationComplete { total_tokens, elapsed_sec } => {
                self.is_generating = false;
                self.status_message = format!("Complete: {} tokens in {:.2}s ({:.1} tok/s)", total_tokens, elapsed_sec, total_tokens as f32 / elapsed_sec.max(0.001));
            }
            EngineToUiMsg::Error(err) => {
                self.is_generating = false;
                self.status_message = format!("Error: {}", err);
            }
            EngineToUiMsg::CompareResult { vanilla_text, hydro_text } => {
                self.vanilla_compare_text = vanilla_text;
                self.hydro_compare_text = hydro_text;
                self.status_message = "Comparison complete".to_string();
            }
            EngineToUiMsg::RememberStoreUpdated(items) => {
                self.remember_items = items;
                self.status_message = "RememberStore updated".to_string();
            }
            EngineToUiMsg::KvSnapshotStatus { state } => {
                self.status_message = state;
            }
            EngineToUiMsg::TelemetryUpdate(frame) => {
                self.tab2_state.residual_cap = frame.force_cap;
                self.tab2_state.residual_goal = frame.goal_force_scale;
                self.tab1_state.temperature = frame.temperature;
                self.last_hud_frame = Some(frame);
            }
        }
    }

    /// Drain all pending engine messages.
    pub fn update(&mut self) {
        while let Some(msg) = self.engine_bridge.try_recv() {
            self.handle_engine_msg(msg);
        }
    }

    /// Render the overall application layout.
    pub fn render(&mut self, f: &mut Frame) {
        let size = f.area();
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(3), // Top Tab Bar
                Constraint::Min(5),    // Active Tab Viewport
                Constraint::Length(2), // Bottom Status & Shortcuts Footer
            ])
            .split(size);

        render_tab_header(f, chunks[0], self.active_tab);
        render_active_tab(f, chunks[1], self);
        render_status_footer(f, chunks[2], self);
    }
}

/// Main entrypoint to run the Ratatui frontend.
pub fn run_ratatui_frontend(
    config: Config,
    model_path: Option<String>,
    dry_run: bool,
) -> Result<()> {
    let mut guard = TerminalGuard::init()?;
    let engine_bridge = EngineBridge::spawn(config, model_path.clone(), dry_run);
    let mut app = App::new(engine_bridge, model_path);

    // Event loop running at ~60 FPS
    let tick_rate = Duration::from_millis(16);

    while !app.should_quit {
        // 1. Drain background engine messages
        app.update();

        // 2. Render frame
        guard.terminal_mut().draw(|f| {
            app.render(f);
        })?;

        // 3. Poll and process user inputs
        if let Some(event) = poll_crossterm_event(tick_rate)? {
            match event {
                AppEvent::Key(key) => {
                    app.handle_key_event(key);
                }
                AppEvent::Resize(_, _) => {
                    // Terminal resize is handled automatically on next draw
                }
                AppEvent::Engine(msg) => {
                    app.handle_engine_msg(msg);
                }
                AppEvent::Mouse(_) | AppEvent::Tick => {}
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crossterm::event::{KeyCode, KeyModifiers};

    #[test]
    fn test_app_state_and_tab_switching() {
        let config = Config::default();
        let bridge = EngineBridge::spawn(config, None, true);
        let mut app = App::new(bridge, Some("test_model.gguf".to_string()));

        assert_eq!(app.active_tab, Tab::ModelLoader);
        assert_eq!(app.model_name, "test_model.gguf");

        // Test direct tab key switch to 3 (System Deck, index 2)
        let key_3 = KeyEvent::new(KeyCode::Char('3'), KeyModifiers::empty());
        app.handle_key_event(key_3);
        assert_eq!(app.active_tab, Tab::SystemDeck);

        // Test Tab key next cycle
        let key_tab = KeyEvent::new(KeyCode::Tab, KeyModifiers::empty());
        app.handle_key_event(key_tab);
        assert_eq!(app.active_tab, Tab::DebugMatrix);

        // Test quit key
        let key_q = KeyEvent::new(KeyCode::Char('q'), KeyModifiers::empty());
        app.handle_key_event(key_q);
        assert!(app.should_quit);
    }

    #[test]
    fn test_app_message_handling_and_render() {
        use ratatui::backend::TestBackend;
        let config = Config::default();
        let bridge = EngineBridge::spawn(config, None, true);
        let mut app = App::new(bridge, None);

        // Test ModelLoaded message
        app.handle_engine_msg(EngineToUiMsg::ModelLoaded {
            name: "gemma-4-it".to_string(),
            n_layers: 36,
        });
        assert_eq!(app.model_name, "gemma-4-it");

        // Test TokenGenerated message
        let frame = HudFrame {
            step: 5,
            temperature: 0.85,
            force_cap: 5.0,
            ..Default::default()
        };
        app.handle_engine_msg(EngineToUiMsg::TokenGenerated {
            text: " hello".to_string(),
            frame,
        });
        assert!(app.is_generating);
        assert_eq!(app.generated_text, " hello");
        assert!(app.last_hud_frame.is_some());

        // Test GenerationComplete message
        app.handle_engine_msg(EngineToUiMsg::GenerationComplete {
            total_tokens: 1,
            elapsed_sec: 0.05,
        });
        assert!(!app.is_generating);

        // Test CompareResult message
        app.handle_engine_msg(EngineToUiMsg::CompareResult {
            vanilla_text: "vanilla".to_string(),
            hydro_text: "hydro".to_string(),
        });
        assert_eq!(app.vanilla_compare_text, "vanilla");
        assert_eq!(app.hydro_compare_text, "hydro");

        // Test RememberStoreUpdated message
        app.handle_engine_msg(EngineToUiMsg::RememberStoreUpdated(vec![
            ("key1".to_string(), "val1".to_string()),
        ]));
        assert_eq!(app.remember_items.len(), 1);

        // Test KvSnapshotStatus message
        app.handle_engine_msg(EngineToUiMsg::KvSnapshotStatus {
            state: "Snapshot saved".to_string(),
        });
        assert_eq!(app.status_message, "Snapshot saved");

        // Test Error message
        app.handle_engine_msg(EngineToUiMsg::Error("Test error".to_string()));
        assert_eq!(app.status_message, "Error: Test error");

        // Test app render with TestBackend
        let backend = TestBackend::new(100, 30);
        let mut terminal = Terminal::new(backend).expect("Terminal creation");
        terminal.draw(|f| app.render(f)).expect("App render draw");
    }
}
