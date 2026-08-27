//! Tab 1: Model Loader & Config.
//!
//! Interactive controls for hot-swapping GGUF model files, architecture inspection,
//! sampling parameter sliders (Temperature, Top-P, Repetition penalty, Max tokens),
//! and real-time model-size scaling preview (`algo_scale.rs`).

use ratatui::{
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, BorderType, Borders, List, ListItem, Paragraph},
    Frame,
};

use crate::algo_scale::{self, ModelType, SizeRule, TransformPrediction};
use crate::frontend::channel::UiToEngineMsg;
use crate::frontend::App;

/// A discoverable model definition for the Model Browser.
#[derive(Debug, Clone)]
pub struct ModelEntry {
    pub name: String,
    pub path: String,
    pub size_gb: f32,
    pub arch: String,
    pub params_b: f32,
    pub model_type: ModelType,
    pub tokenizer_path: String,
    pub hidden_dim: usize,
    pub n_layers: usize,
    pub swa_window: Option<usize>,
    pub device: String,
}

/// Interactive state for Tab 1.
#[derive(Debug, Clone)]
pub struct Tab1State {
    /// Available models in the browser
    pub models: Vec<ModelEntry>,
    /// Currently selected index in the model list
    pub selected_model_idx: usize,
    /// Currently loaded model index, if any
    pub loaded_model_idx: Option<usize>,
    /// Currently focused interactive field index (0..8)
    pub selected_field_idx: usize,

    // Sampling configuration
    pub temperature: f32,
    pub repetition_penalty: f32,
    pub max_tokens: usize,
    pub top_p: f32,
    pub top_k: usize,

    // Model size scaling preview parameters
    pub scaling_params_b: f32,
    pub scaling_model_type: ModelType,
    pub scaling_size_rule: SizeRule,
}

impl Default for Tab1State {
    fn default() -> Self {
        let models = vec![
            ModelEntry {
                name: "gemma-4-9b-it-Q4_K_M.gguf".to_string(),
                path: "models/gemma-4-9b-it-Q4_K_M.gguf".to_string(),
                size_gb: 5.2,
                arch: "gemma4".to_string(),
                params_b: 9.0,
                model_type: ModelType::Instruct,
                tokenizer_path: "data/google/gemma4_assets/tokenizer.json".to_string(),
                hidden_dim: 3840,
                n_layers: 36,
                swa_window: Some(1024),
                device: "CUDA 0".to_string(),
            },
            ModelEntry {
                name: "gemma-3-4b-it-Q4_K_M.gguf".to_string(),
                path: "models/gemma-3-4b-it-Q4_K_M.gguf".to_string(),
                size_gb: 2.8,
                arch: "gemma".to_string(),
                params_b: 4.0,
                model_type: ModelType::Instruct,
                tokenizer_path: "data/google/gemma3_assets/tokenizer.json".to_string(),
                hidden_dim: 2560,
                n_layers: 32,
                swa_window: None,
                device: "CUDA 0".to_string(),
            },
            ModelEntry {
                name: "qwen2.5-7b-instruct.gguf".to_string(),
                path: "models/qwen2.5-7b-instruct.gguf".to_string(),
                size_gb: 4.5,
                arch: "qwen25".to_string(),
                params_b: 7.0,
                model_type: ModelType::Instruct,
                tokenizer_path: "data/qwen/tokenizer.json".to_string(),
                hidden_dim: 3584,
                n_layers: 28,
                swa_window: None,
                device: "CUDA 0".to_string(),
            },
            ModelEntry {
                name: "llama-3.1-8b-instruct.gguf".to_string(),
                path: "models/llama-3.1-8b-instruct.gguf".to_string(),
                size_gb: 4.9,
                arch: "llama".to_string(),
                params_b: 8.0,
                model_type: ModelType::Instruct,
                tokenizer_path: "data/llama3/tokenizer.json".to_string(),
                hidden_dim: 4096,
                n_layers: 32,
                swa_window: None,
                device: "CUDA 0".to_string(),
            },
        ];

        Self {
            models,
            selected_model_idx: 0,
            loaded_model_idx: Some(0),
            selected_field_idx: 0,
            temperature: 0.85,
            repetition_penalty: 1.15,
            max_tokens: 512,
            top_p: 0.95,
            top_k: 64,
            scaling_params_b: 9.0,
            scaling_model_type: ModelType::Instruct,
            scaling_size_rule: SizeRule::Piecewise,
        }
    }
}

impl Tab1State {
    /// Number of focusable fields in Tab 1
    pub const TOTAL_FIELDS: usize = 9;

    /// Navigate to previous field
    pub fn prev_field(&mut self) {
        if self.selected_field_idx == 0 {
            self.selected_field_idx = Self::TOTAL_FIELDS - 1;
        } else {
            self.selected_field_idx -= 1;
        }
    }

    /// Navigate to next field
    pub fn next_field(&mut self) {
        self.selected_field_idx = (self.selected_field_idx + 1) % Self::TOTAL_FIELDS;
    }

    /// Selected model entry
    pub fn current_model(&self) -> Option<&ModelEntry> {
        self.models.get(self.selected_model_idx)
    }

    /// Adjust the currently focused field by a signed delta.
    /// Returns an optional `UiToEngineMsg` if an engine parameter was updated.
    pub fn adjust_field(&mut self, delta: f32) -> Option<UiToEngineMsg> {
        match self.selected_field_idx {
            0 => {
                // Model selection
                if delta < 0.0 {
                    if self.selected_model_idx == 0 {
                        self.selected_model_idx = self.models.len().saturating_sub(1);
                    } else {
                        self.selected_model_idx -= 1;
                    }
                } else if delta > 0.0 {
                    self.selected_model_idx = (self.selected_model_idx + 1) % self.models.len().max(1);
                }
                // Sync scaling preview size to selected model
                if let Some(m) = self.models.get(self.selected_model_idx) {
                    let pb = m.params_b;
                    let mt = m.model_type;
                    self.scaling_params_b = pb;
                    self.scaling_model_type = mt;
                }
                None
            }
            1 => {
                // Temperature: 0.00 .. 2.00, step 0.05
                let step = if delta.abs() >= 5.0 { 0.20 } else { 0.05 };
                self.temperature = (self.temperature + delta.signum() * step).clamp(0.0, 2.0);
                Some(UiToEngineMsg::SetLiveParam {
                    key: "sample.temp".to_string(),
                    val: self.temperature,
                })
            }
            2 => {
                // Repetition Penalty: 1.00 .. 2.00, step 0.05
                let step = if delta.abs() >= 5.0 { 0.20 } else { 0.05 };
                self.repetition_penalty = (self.repetition_penalty + delta.signum() * step).clamp(1.0, 2.0);
                Some(UiToEngineMsg::SetLiveParam {
                    key: "sample.rep".to_string(),
                    val: self.repetition_penalty,
                })
            }
            3 => {
                // Max Tokens: 16 .. 8192, step 64
                let step = if delta.abs() >= 5.0 { 512 } else { 64 };
                if delta < 0.0 {
                    self.max_tokens = self.max_tokens.saturating_sub(step).max(16);
                } else {
                    self.max_tokens = (self.max_tokens + step).min(8192);
                }
                Some(UiToEngineMsg::SetLiveParam {
                    key: "sample.max".to_string(),
                    val: self.max_tokens as f32,
                })
            }
            4 => {
                // Top-P: 0.00 .. 1.00, step 0.05
                let step = if delta.abs() >= 5.0 { 0.20 } else { 0.05 };
                self.top_p = (self.top_p + delta.signum() * step).clamp(0.0, 1.0);
                Some(UiToEngineMsg::SetLiveParam {
                    key: "sample.top_p".to_string(),
                    val: self.top_p,
                })
            }
            5 => {
                // Top-K: 1 .. 256, step 8
                let step = if delta.abs() >= 5.0 { 32 } else { 8 };
                if delta < 0.0 {
                    self.top_k = self.top_k.saturating_sub(step).max(1);
                } else {
                    self.top_k = (self.top_k + step).min(256);
                }
                Some(UiToEngineMsg::SetLiveParam {
                    key: "sample.top_k".to_string(),
                    val: self.top_k as f32,
                })
            }
            6 => {
                // Scaling params_b: 0.5 .. 140.0, step 0.5
                let step = if delta.abs() >= 5.0 { 5.0 } else { 0.5 };
                self.scaling_params_b = (self.scaling_params_b + delta.signum() * step).clamp(0.5, 140.0);
                None
            }
            7 => {
                // Scaling ModelType cycling
                self.scaling_model_type = cycle_model_type(self.scaling_model_type, delta > 0.0);
                None
            }
            8 => {
                // Scaling SizeRule cycling
                self.scaling_size_rule = cycle_size_rule(self.scaling_size_rule, delta > 0.0);
                None
            }
            _ => None,
        }
    }

    /// Trigger action on Enter/Space
    pub fn toggle_or_action(&mut self) -> Option<UiToEngineMsg> {
        match self.selected_field_idx {
            0 => {
                // Load selected model
                self.loaded_model_idx = Some(self.selected_model_idx);
                if let Some(m) = self.current_model() {
                    Some(UiToEngineMsg::LoadModel {
                        path: m.path.clone(),
                        tokenizer: Some(m.tokenizer_path.clone()),
                    })
                } else {
                    None
                }
            }
            7 => {
                self.scaling_model_type = cycle_model_type(self.scaling_model_type, true);
                None
            }
            8 => {
                self.scaling_size_rule = cycle_size_rule(self.scaling_size_rule, true);
                None
            }
            _ => None,
        }
    }

    /// Create LoadModel message for currently selected model
    pub fn load_selected_msg(&mut self) -> Option<UiToEngineMsg> {
        self.loaded_model_idx = Some(self.selected_model_idx);
        self.current_model().map(|m| UiToEngineMsg::LoadModel {
            path: m.path.clone(),
            tokenizer: Some(m.tokenizer_path.clone()),
        })
    }

    /// Compute live scaling prediction (formula-native σ/θ/β — not Hydro knobs).
    pub fn compute_scaling_preview(&self) -> TransformPrediction {
        algo_scale::transform_prediction(
            self.scaling_params_b,
            self.scaling_model_type,
            self.scaling_size_rule,
            self.temperature,
        )
    }

    /// Hydro residual seat after adapting the selected transform onto Config defaults.
    ///
    /// SizeRule::Off leaves the default seat (manual gain ladder). This is *not*
    /// `pred.sigma` written onto `residual.cap`.
    pub fn predicted_hydro_seat(&self) -> crate::config::Config {
        let pred = self.compute_scaling_preview();
        let mut cfg = crate::config::Config::default();
        if self.scaling_size_rule != SizeRule::Off {
            algo_scale::apply_to_hydro_profile(&mut cfg, &pred, 1.0, true);
        }
        cfg
    }
}

fn cycle_model_type(current: ModelType, forward: bool) -> ModelType {
    match current {
        ModelType::Standard => if forward { ModelType::Instruct } else { ModelType::Coding },
        ModelType::Instruct => if forward { ModelType::Chat } else { ModelType::Standard },
        ModelType::Chat => if forward { ModelType::Thinking } else { ModelType::Instruct },
        ModelType::Thinking => if forward { ModelType::Coding } else { ModelType::Chat },
        ModelType::Coding => if forward { ModelType::Standard } else { ModelType::Thinking },
    }
}

fn cycle_size_rule(current: SizeRule, forward: bool) -> SizeRule {
    match current {
        SizeRule::Legacy => if forward { SizeRule::EightBSqrt } else { SizeRule::Off },
        SizeRule::EightBSqrt => if forward { SizeRule::Piecewise } else { SizeRule::Legacy },
        SizeRule::Piecewise => if forward { SizeRule::Off } else { SizeRule::EightBSqrt },
        SizeRule::Off => if forward { SizeRule::Legacy } else { SizeRule::Piecewise },
    }
}

/// Helper to render an ASCII slider bar `[████····]`
pub fn make_slider(value: f32, min: f32, max: f32, width: usize) -> String {
    let span = (max - min).max(1e-6);
    let frac = ((value - min) / span).clamp(0.0, 1.0);
    let filled = (frac * width as f32).round() as usize;
    let mut bar = String::with_capacity(width);
    for i in 0..width {
        if i < filled {
            bar.push('█');
        } else {
            bar.push('·');
        }
    }
    bar
}

pub fn render(f: &mut Frame, area: Rect, app: &App) {
    let main_chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Min(8), Constraint::Length(3)])
        .split(area);

    let upper_chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
        .split(main_chunks[0]);

    let left_chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
        .split(upper_chunks[0]);

    let right_chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
        .split(upper_chunks[1]);

    let state = &app.tab1_state;
    let focus = state.selected_field_idx;

    // 1. Model Browser (Top-Left)
    let model_items: Vec<ListItem> = state
        .models
        .iter()
        .enumerate()
        .map(|(idx, m)| {
            let is_selected_in_browser = idx == state.selected_model_idx;
            let is_loaded = state.loaded_model_idx == Some(idx)
                || (!app.model_name.is_empty() && (app.model_name.contains(&m.arch) || app.model_name.contains(&m.name)));

            let cursor = if focus == 0 && is_selected_in_browser {
                " ▶ "
            } else if is_selected_in_browser {
                " → "
            } else {
                "   "
            };

            let load_tag = if is_loaded {
                Span::styled("[x] ", Style::default().fg(Color::Green).add_modifier(Modifier::BOLD))
            } else {
                Span::styled("[ ] ", Style::default().fg(Color::DarkGray))
            };

            let name_style = if is_selected_in_browser {
                Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)
            } else if is_loaded {
                Style::default().fg(Color::White).add_modifier(Modifier::BOLD)
            } else {
                Style::default().fg(Color::Gray)
            };

            ListItem::new(Line::from(vec![
                Span::styled(cursor, Style::default().fg(Color::Yellow)),
                load_tag,
                Span::styled(format!("{:<26}", m.name), name_style),
                Span::styled(
                    format!(" ({:.1} GB - {})", m.size_gb, m.device),
                    Style::default().fg(Color::DarkGray),
                ),
            ]))
        })
        .collect();

    let browser_border_style = if focus == 0 {
        Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)
    } else {
        Style::default().fg(Color::Cyan)
    };

    let browser_block = List::new(model_items).block(
        Block::default()
            .borders(Borders::ALL)
            .border_type(BorderType::Rounded)
            .title(" Model Browser (GGUF Discovery) [Select: ↑↓ / ←→, Load: Enter/L] ")
            .style(browser_border_style),
    );
    f.render_widget(browser_block, left_chunks[0]);

    // 2. Active Model Architecture (Top-Right)
    let cur_model = state.current_model();
    let arch_lines = vec![
        Line::from(vec![
            Span::styled("Model:       ", Style::default().fg(Color::Yellow)),
            Span::styled(
                if !app.model_name.is_empty() {
                    &app.model_name
                } else if let Some(m) = cur_model {
                    &m.name
                } else {
                    "No Model Loaded"
                },
                Style::default().fg(Color::White).add_modifier(Modifier::BOLD),
            ),
        ]),
        Line::from(vec![
            Span::styled("Arch:        ", Style::default().fg(Color::Yellow)),
            Span::styled(
                cur_model.map(|m| m.arch.as_str()).unwrap_or("gemma4"),
                Style::default().fg(Color::White),
            ),
            Span::styled(" / transformer stack", Style::default().fg(Color::DarkGray)),
        ]),
        Line::from(vec![
            Span::styled("Hidden Dim:  ", Style::default().fg(Color::Yellow)),
            Span::styled(
                format!("{}", cur_model.map(|m| m.hidden_dim).unwrap_or(3840)),
                Style::default().fg(Color::White),
            ),
            Span::styled("   Layers: ", Style::default().fg(Color::Yellow)),
            Span::styled(
                format!("{}", cur_model.map(|m| m.n_layers).unwrap_or(36)),
                Style::default().fg(Color::White),
            ),
        ]),
        Line::from(vec![
            Span::styled("Tokenizer:   ", Style::default().fg(Color::Yellow)),
            Span::styled(
                cur_model.map(|m| m.tokenizer_path.as_str()).unwrap_or("data/google/gemma4_assets/tokenizer.json"),
                Style::default().fg(Color::DarkGray),
            ),
        ]),
        Line::from(vec![
            Span::styled("SWA Window:  ", Style::default().fg(Color::Yellow)),
            Span::styled(
                cur_model
                    .and_then(|m| m.swa_window)
                    .map(|w| format!("{}", w))
                    .unwrap_or_else(|| "None (Full Attn)".to_string()),
                Style::default().fg(Color::White),
            ),
            Span::styled("   Device: ", Style::default().fg(Color::Yellow)),
            Span::styled(
                cur_model.map(|m| m.device.as_str()).unwrap_or("CUDA 0"),
                Style::default().fg(Color::Green),
            ),
        ]),
    ];

    let arch_block = Paragraph::new(arch_lines).block(
        Block::default()
            .borders(Borders::ALL)
            .border_type(BorderType::Rounded)
            .title(" Active Model Architecture ")
            .style(Style::default().fg(Color::Cyan)),
    );
    f.render_widget(arch_block, right_chunks[0]);

    // 3. Model Size Scaling & Formula Preview (Bottom-Left)
    let pred = state.compute_scaling_preview();
    let seat = state.predicted_hydro_seat();
    let sel_f6 = focus == 6;
    let sel_f7 = focus == 7;
    let sel_f8 = focus == 8;

    let scaling_lines = vec![
        Line::from(vec![
            Span::styled(if sel_f6 { " ▶ " } else { "   " }, Style::default().fg(Color::Yellow)),
            Span::styled("Model Size (B):   ", if sel_f6 { Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD) } else { Style::default().fg(Color::LightBlue) }),
            Span::styled(format!("[ {:>5.2} ] ", state.scaling_params_b), Style::default().fg(Color::White).add_modifier(Modifier::BOLD)),
            Span::styled(format!("(Gain: {:.2}x)", pred.force_intensity), Style::default().fg(Color::DarkGray)),
        ]),
        Line::from(vec![
            Span::styled(if sel_f7 { " ▶ " } else { "   " }, Style::default().fg(Color::Yellow)),
            Span::styled("Model Type:       ", if sel_f7 { Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD) } else { Style::default().fg(Color::LightBlue) }),
            Span::styled(format!("< {} >", state.scaling_model_type.as_str()), Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)),
            Span::styled(format!(" (mult: {:.2}x)", state.scaling_model_type.multiplier()), Style::default().fg(Color::DarkGray)),
        ]),
        Line::from(vec![
            Span::styled(if sel_f8 { " ▶ " } else { "   " }, Style::default().fg(Color::Yellow)),
            Span::styled("Scale Mode:       ", if sel_f8 { Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD) } else { Style::default().fg(Color::LightBlue) }),
            Span::styled(
                match state.scaling_size_rule {
                    SizeRule::Legacy => "< Legacy (3B √ coupled T) >",
                    SizeRule::EightBSqrt => "< 8B-Sqrt (July 8B √ coupled T) >",
                    SizeRule::Piecewise => "< Piecewise (√ to 8B, log-soft >8B, decoupled T) >",
                    SizeRule::Off => "< Off (Manual Gain Ladder) >",
                },
                Style::default().fg(Color::Green).add_modifier(Modifier::BOLD),
            ),
        ]),
        Line::from(vec![
            Span::styled("   Predicted Knobs:  ", Style::default().fg(Color::LightBlue)),
            Span::styled(
                format!(
                    "formula σ:{:.3} θ:{:.3} β:{:.1}  Hydro cap:{:.2} goal:{:.3}",
                    pred.sigma,
                    pred.theta,
                    pred.beta,
                    seat.physics.force_cap,
                    seat.physics.goal_force_scale,
                ),
                Style::default().fg(Color::Magenta).add_modifier(Modifier::BOLD),
            ),
        ]),
        Line::from(vec![
            Span::styled("   T Coupling:       ", Style::default().fg(Color::DarkGray)),
            Span::styled(
                if pred.temperature_coupled {
                    format!("Coupled (T_pred: {:.2})", pred.predicted_temperature)
                } else {
                    format!("Decoupled (T: {:.2})", state.temperature)
                },
                if pred.temperature_coupled { Style::default().fg(Color::LightCyan) } else { Style::default().fg(Color::Green) },
            ),
        ]),
    ];

    let scaling_border_style = if sel_f6 || sel_f7 || sel_f8 {
        Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)
    } else {
        Style::default().fg(Color::Cyan)
    };

    let scaling_block = Paragraph::new(scaling_lines).block(
        Block::default()
            .borders(Borders::ALL)
            .border_type(BorderType::Rounded)
            .title(" Sampling & Size Scaling (Algo Scale Preview) ")
            .style(scaling_border_style),
    );
    f.render_widget(scaling_block, left_chunks[1]);

    // 4. Sampling & Context Configuration (Bottom-Right)
    let sel_f1 = focus == 1;
    let sel_f2 = focus == 2;
    let sel_f3 = focus == 3;
    let sel_f4 = focus == 4;
    let sel_f5 = focus == 5;

    let temp_bar = make_slider(state.temperature, 0.0, 2.0, 16);
    let rep_bar = make_slider(state.repetition_penalty, 1.0, 2.0, 16);
    let max_bar = make_slider(state.max_tokens as f32, 16.0, 8192.0, 16);
    let topp_bar = make_slider(state.top_p, 0.0, 1.0, 10);

    let sampling_lines = vec![
        Line::from(vec![
            Span::styled(if sel_f1 { " ▶ " } else { "   " }, Style::default().fg(Color::Yellow)),
            Span::styled("Temperature:    ", if sel_f1 { Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD) } else { Style::default().fg(Color::LightMagenta) }),
            Span::styled(format!("[{}] ", temp_bar), Style::default().fg(Color::LightMagenta)),
            Span::styled(format!("{:.2}", state.temperature), Style::default().fg(Color::White).add_modifier(Modifier::BOLD)),
        ]),
        Line::from(vec![
            Span::styled(if sel_f2 { " ▶ " } else { "   " }, Style::default().fg(Color::Yellow)),
            Span::styled("Repetition Pen: ", if sel_f2 { Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD) } else { Style::default().fg(Color::LightMagenta) }),
            Span::styled(format!("[{}] ", rep_bar), Style::default().fg(Color::LightMagenta)),
            Span::styled(format!("{:.2}", state.repetition_penalty), Style::default().fg(Color::White).add_modifier(Modifier::BOLD)),
        ]),
        Line::from(vec![
            Span::styled(if sel_f3 { " ▶ " } else { "   " }, Style::default().fg(Color::Yellow)),
            Span::styled("Max Tokens:     ", if sel_f3 { Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD) } else { Style::default().fg(Color::LightMagenta) }),
            Span::styled(format!("[{}] ", max_bar), Style::default().fg(Color::LightMagenta)),
            Span::styled(format!("{:<4}", state.max_tokens), Style::default().fg(Color::White).add_modifier(Modifier::BOLD)),
        ]),
        Line::from(vec![
            Span::styled(if sel_f4 { " ▶ " } else { "   " }, Style::default().fg(Color::Yellow)),
            Span::styled("Top-P:          ", if sel_f4 { Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD) } else { Style::default().fg(Color::LightMagenta) }),
            Span::styled(format!("[{}] ", topp_bar), Style::default().fg(Color::LightMagenta)),
            Span::styled(format!("{:.2}", state.top_p), Style::default().fg(Color::White).add_modifier(Modifier::BOLD)),
        ]),
        Line::from(vec![
            Span::styled(if sel_f5 { " ▶ " } else { "   " }, Style::default().fg(Color::Yellow)),
            Span::styled("Top-K:          ", if sel_f5 { Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD) } else { Style::default().fg(Color::LightMagenta) }),
            Span::styled(format!("[ {:>3} ]", state.top_k), Style::default().fg(Color::White).add_modifier(Modifier::BOLD)),
        ]),
    ];

    let sampling_border_style = if sel_f1 || sel_f2 || sel_f3 || sel_f4 || sel_f5 {
        Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)
    } else {
        Style::default().fg(Color::Cyan)
    };

    let sampling_block = Paragraph::new(sampling_lines).block(
        Block::default()
            .borders(Borders::ALL)
            .border_type(BorderType::Rounded)
            .title(" Sampling & Context Configuration ")
            .style(sampling_border_style),
    );
    f.render_widget(sampling_block, right_chunks[1]);

    // 5. Actions Footer Bar
    let action_spans = Line::from(vec![
        Span::styled(" Actions: ", Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)),
        Span::styled("[Enter/L] ", Style::default().fg(Color::Green).add_modifier(Modifier::BOLD)),
        Span::styled("Load Selected Model   ", Style::default().fg(Color::White)),
        Span::styled("[U] ", Style::default().fg(Color::Red).add_modifier(Modifier::BOLD)),
        Span::styled("Unload Model   ", Style::default().fg(Color::White)),
        Span::styled("[S] ", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
        Span::styled("Save Config TOML   ", Style::default().fg(Color::White)),
        Span::styled("[C] ", Style::default().fg(Color::Magenta).add_modifier(Modifier::BOLD)),
        Span::styled("Clear KV Cache", Style::default().fg(Color::White)),
    ]);

    let actions_block = Paragraph::new(action_spans).block(
        Block::default()
            .borders(Borders::ALL)
            .border_type(BorderType::Rounded)
            .style(Style::default().fg(Color::DarkGray)),
    );
    f.render_widget(actions_block, main_chunks[1]);
}
