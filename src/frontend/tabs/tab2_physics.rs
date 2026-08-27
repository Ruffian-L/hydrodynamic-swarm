//! Tab 2: Physics Board.
//!
//! Control flight deck for the 3-surface physics steering engine:
//! 1. Residual Forces (Niodoo 3-force model, scar memory, & force ceilings)
//! 2. Logit Chain (Continuous field, Splat vocabulary bias, Fluid Governor, & Hands)
//! 3. Layer Hook (Mid-stack transformer HookControls, layer band, & norm fractions)

use ratatui::{
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, BorderType, Borders, Paragraph},
    Frame,
};

use crate::frontend::channel::UiToEngineMsg;
use crate::frontend::tabs::tab1_model::make_slider;
use crate::frontend::App;
use crate::hooks::HookSite;

/// Interactive state for Tab 2 (Physics Board).
#[derive(Debug, Clone)]
pub struct Tab2State {
    /// Currently focused field index (0..27)
    pub selected_field_idx: usize,

    // Surface 1: Residual Forces
    pub residual_cap: f32,
    pub residual_goal: f32,
    pub residual_field: f32,
    pub residual_splat: f32,
    pub residual_dt: f32,
    pub force_ramp_len: usize,
    pub force_ramp_str: f32,
    pub residual_field_max: f32,
    pub residual_splat_max: f32,
    pub residual_goal_max: f32,

    // Surface 2: Logit Biases & Fluid Governor & Hands
    pub field_alpha: f32,
    pub splat_scale: f32,
    pub splat_top_m: usize,
    pub splat_top_k: usize,
    pub gov_on: bool,
    pub gov_velocity: f32,
    pub gov_brake: f32,
    pub gov_visc_gain: f32,
    pub gov_max_bias: f32,
    pub backslash_pen: f32,
    pub hands_repulsion: f32,
    pub hands_beta: f32,
    pub hands_blend: f32,

    // Surface 3: Layer Hook (HookControls)
    pub hook_on: bool,
    pub hook_site: HookSite,
    pub hook_norm_fraction: f32,
    pub hook_start_frac: f32,
    pub hook_end_frac: f32,
}

impl Default for Tab2State {
    fn default() -> Self {
        Self {
            selected_field_idx: 0,

            // Surface 1
            residual_cap: 5.00,
            residual_goal: 0.15,
            residual_field: 0.10,
            residual_splat: 0.05,
            residual_dt: 0.05,
            force_ramp_len: 15,
            force_ramp_str: 0.20,
            residual_field_max: 40.0,
            residual_splat_max: 28.0,
            residual_goal_max: 40.0,

            // Surface 2
            field_alpha: 0.15,
            splat_scale: 0.05,
            splat_top_m: 4,
            splat_top_k: 32,
            gov_on: true,
            gov_velocity: 0.85,
            gov_brake: 3.0,
            gov_visc_gain: 12.0,
            gov_max_bias: 2.0,
            backslash_pen: 2.0,
            hands_repulsion: 0.0,
            hands_beta: 0.85,
            hands_blend: 1.0,

            // Surface 3
            hook_on: true,
            hook_site: HookSite::PostMlp,
            hook_norm_fraction: 0.0050,
            hook_start_frac: 0.50,
            hook_end_frac: 1.00,
        }
    }
}

impl Tab2State {
    pub const TOTAL_FIELDS: usize = 28;

    pub fn prev_field(&mut self) {
        if self.selected_field_idx == 0 {
            self.selected_field_idx = Self::TOTAL_FIELDS - 1;
        } else {
            self.selected_field_idx -= 1;
        }
    }

    pub fn next_field(&mut self) {
        self.selected_field_idx = (self.selected_field_idx + 1) % Self::TOTAL_FIELDS;
    }

    /// Convert HookSite to float representation for message contract
    pub fn site_to_f32(site: HookSite) -> f32 {
        match site {
            HookSite::PreLayer => 0.0,
            HookSite::PostAttn => 1.0,
            HookSite::PostMlp => 2.0,
            HookSite::FinalNorm => 3.0,
        }
    }

    /// Convert f32 to HookSite
    pub fn f32_to_site(val: f32) -> HookSite {
        match val.round() as i32 {
            0 => HookSite::PreLayer,
            1 => HookSite::PostAttn,
            3 => HookSite::FinalNorm,
            _ => HookSite::PostMlp,
        }
    }

    /// Cycle hook site
    pub fn cycle_hook_site(&mut self, forward: bool) {
        self.hook_site = match self.hook_site {
            HookSite::PreLayer => if forward { HookSite::PostAttn } else { HookSite::FinalNorm },
            HookSite::PostAttn => if forward { HookSite::PostMlp } else { HookSite::PreLayer },
            HookSite::PostMlp => if forward { HookSite::FinalNorm } else { HookSite::PostAttn },
            HookSite::FinalNorm => if forward { HookSite::PreLayer } else { HookSite::PostMlp },
        };
    }

    /// Construct HookControl message
    pub fn hook_msg(&self) -> UiToEngineMsg {
        UiToEngineMsg::SetHookControl {
            enabled: self.hook_on,
            site: Self::site_to_f32(self.hook_site),
            start_frac: self.hook_start_frac,
            end_frac: self.hook_end_frac,
            norm_fraction: self.hook_norm_fraction,
        }
    }

    /// Adjust the currently focused field by a signed delta.
    pub fn adjust_field(&mut self, delta: f32) -> Option<UiToEngineMsg> {
        let is_shift = delta.abs() >= 5.0;
        let sign = delta.signum();

        match self.selected_field_idx {
            // Surface 1: Residual Forces
            0 => {
                let step = if is_shift { 1.0 } else { 0.1 };
                self.residual_cap = (self.residual_cap + sign * step).clamp(0.0, 20.0);
                Some(UiToEngineMsg::SetLiveParam {
                    key: "residual.cap".to_string(),
                    val: self.residual_cap,
                })
            }
            1 => {
                let step = if is_shift { 0.1 } else { 0.01 };
                self.residual_goal = (self.residual_goal + sign * step).clamp(0.0, 2.0);
                Some(UiToEngineMsg::SetLiveParam {
                    key: "residual.goal".to_string(),
                    val: self.residual_goal,
                })
            }
            2 => {
                let step = if is_shift { 0.1 } else { 0.01 };
                self.residual_field = (self.residual_field + sign * step).clamp(0.0, 2.0);
                Some(UiToEngineMsg::SetLiveParam {
                    key: "residual.field".to_string(),
                    val: self.residual_field,
                })
            }
            3 => {
                let step = if is_shift { 0.1 } else { 0.01 };
                self.residual_splat = (self.residual_splat + sign * step).clamp(0.0, 2.0);
                Some(UiToEngineMsg::SetLiveParam {
                    key: "residual.splat".to_string(),
                    val: self.residual_splat,
                })
            }
            4 => {
                let step = if is_shift { 0.02 } else { 0.005 };
                self.residual_dt = (self.residual_dt + sign * step).clamp(0.001, 0.20);
                Some(UiToEngineMsg::SetLiveParam {
                    key: "residual.dt".to_string(),
                    val: self.residual_dt,
                })
            }
            5 => {
                let step = if is_shift { 10 } else { 1 };
                if sign < 0.0 {
                    self.force_ramp_len = self.force_ramp_len.saturating_sub(step);
                } else {
                    self.force_ramp_len = (self.force_ramp_len + step).min(100);
                }
                Some(UiToEngineMsg::SetLiveParam {
                    key: "force_ramp_tokens".to_string(),
                    val: self.force_ramp_len as f32,
                })
            }
            6 => {
                let step = if is_shift { 0.2 } else { 0.05 };
                self.force_ramp_str = (self.force_ramp_str + sign * step).clamp(0.0, 1.0);
                Some(UiToEngineMsg::SetLiveParam {
                    key: "force_ramp_start".to_string(),
                    val: self.force_ramp_str,
                })
            }
            7 => {
                let step = if is_shift { 10.0 } else { 1.0 };
                self.residual_field_max = (self.residual_field_max + sign * step).clamp(0.0, 100.0);
                Some(UiToEngineMsg::SetLiveParam {
                    key: "residual.field_max".to_string(),
                    val: self.residual_field_max,
                })
            }
            8 => {
                let step = if is_shift { 10.0 } else { 1.0 };
                self.residual_splat_max = (self.residual_splat_max + sign * step).clamp(0.0, 100.0);
                Some(UiToEngineMsg::SetLiveParam {
                    key: "residual.splat_max".to_string(),
                    val: self.residual_splat_max,
                })
            }
            9 => {
                let step = if is_shift { 10.0 } else { 1.0 };
                self.residual_goal_max = (self.residual_goal_max + sign * step).clamp(0.0, 100.0);
                Some(UiToEngineMsg::SetLiveParam {
                    key: "residual.goal_max".to_string(),
                    val: self.residual_goal_max,
                })
            }

            // Surface 2: Logit Biases & Fluid Governor & Hands
            10 => {
                let step = if is_shift { 0.1 } else { 0.01 };
                self.field_alpha = (self.field_alpha + sign * step).clamp(0.0, 1.0);
                Some(UiToEngineMsg::SetLiveParam {
                    key: "field.alpha".to_string(),
                    val: self.field_alpha,
                })
            }
            11 => {
                let step = if is_shift { 0.1 } else { 0.01 };
                self.splat_scale = (self.splat_scale + sign * step).clamp(0.0, 1.0);
                Some(UiToEngineMsg::SetLiveParam {
                    key: "splat.scale".to_string(),
                    val: self.splat_scale,
                })
            }
            12 => {
                let step = if is_shift { 4 } else { 1 };
                if sign < 0.0 {
                    self.splat_top_m = self.splat_top_m.saturating_sub(step).max(1);
                } else {
                    self.splat_top_m = (self.splat_top_m + step).min(16);
                }
                Some(UiToEngineMsg::SetLiveParam {
                    key: "splat.top_m".to_string(),
                    val: self.splat_top_m as f32,
                })
            }
            13 => {
                let step = if is_shift { 32 } else { 4 };
                if sign < 0.0 {
                    self.splat_top_k = self.splat_top_k.saturating_sub(step).max(1);
                } else {
                    self.splat_top_k = (self.splat_top_k + step).min(128);
                }
                Some(UiToEngineMsg::SetLiveParam {
                    key: "splat.top_k".to_string(),
                    val: self.splat_top_k as f32,
                })
            }
            14 => {
                self.gov_on = !self.gov_on;
                Some(UiToEngineMsg::SetLiveParam {
                    key: "gov.on".to_string(),
                    val: if self.gov_on { 1.0 } else { 0.0 },
                })
            }
            15 => {
                let step = if is_shift { 0.20 } else { 0.05 };
                self.gov_velocity = (self.gov_velocity + sign * step).clamp(0.5, 1.0);
                Some(UiToEngineMsg::SetLiveParam {
                    key: "gov.velocity".to_string(),
                    val: self.gov_velocity,
                })
            }
            16 => {
                let step = if is_shift { 2.0 } else { 0.5 };
                self.gov_brake = (self.gov_brake + sign * step).clamp(0.0, 15.0);
                Some(UiToEngineMsg::SetLiveParam {
                    key: "gov.brake".to_string(),
                    val: self.gov_brake,
                })
            }
            17 => {
                let step = if is_shift { 5.0 } else { 1.0 };
                self.gov_visc_gain = (self.gov_visc_gain + sign * step).clamp(0.0, 35.0);
                Some(UiToEngineMsg::SetLiveParam {
                    key: "gov.visc_gain".to_string(),
                    val: self.gov_visc_gain,
                })
            }
            18 => {
                let step = if is_shift { 1.0 } else { 0.2 };
                self.gov_max_bias = (self.gov_max_bias + sign * step).clamp(0.0, 10.0);
                Some(UiToEngineMsg::SetLiveParam {
                    key: "gov.max_bias".to_string(),
                    val: self.gov_max_bias,
                })
            }
            19 => {
                let step = if is_shift { 1.0 } else { 0.2 };
                self.backslash_pen = (self.backslash_pen + sign * step).clamp(0.0, 10.0);
                Some(UiToEngineMsg::SetLiveParam {
                    key: "backslash.penalty".to_string(),
                    val: self.backslash_pen,
                })
            }
            20 => {
                let step = if is_shift { 1.0 } else { 0.1 };
                self.hands_repulsion = (self.hands_repulsion + sign * step).clamp(-5.0, 5.0);
                Some(UiToEngineMsg::SetLiveParam {
                    key: "hands.repulsion".to_string(),
                    val: self.hands_repulsion,
                })
            }
            21 => {
                let step = if is_shift { 0.5 } else { 0.05 };
                self.hands_beta = (self.hands_beta + sign * step).clamp(0.0, 5.0);
                Some(UiToEngineMsg::SetLiveParam {
                    key: "hands.beta".to_string(),
                    val: self.hands_beta,
                })
            }
            22 => {
                let step = if is_shift { 1.0 } else { 0.1 };
                self.hands_blend = (self.hands_blend + sign * step).clamp(0.0, 10.0);
                Some(UiToEngineMsg::SetLiveParam {
                    key: "hands.blend".to_string(),
                    val: self.hands_blend,
                })
            }

            // Surface 3: Layer Hook (HookControls)
            23 => {
                self.hook_on = !self.hook_on;
                Some(self.hook_msg())
            }
            24 => {
                self.cycle_hook_site(delta > 0.0);
                Some(self.hook_msg())
            }
            25 => {
                let step = if is_shift { 0.0050 } else { 0.0005 };
                self.hook_norm_fraction = (self.hook_norm_fraction + sign * step).clamp(0.0, 0.01);
                Some(self.hook_msg())
            }
            26 => {
                let step = if is_shift { 0.10 } else { 0.02 };
                self.hook_start_frac = (self.hook_start_frac + sign * step).clamp(0.0, self.hook_end_frac);
                Some(self.hook_msg())
            }
            27 => {
                let step = if is_shift { 0.10 } else { 0.02 };
                self.hook_end_frac = (self.hook_end_frac + sign * step).clamp(self.hook_start_frac, 1.0);
                Some(self.hook_msg())
            }

            _ => None,
        }
    }

    /// Toggle or trigger action on Enter / Space
    pub fn toggle_or_action(&mut self) -> Option<UiToEngineMsg> {
        match self.selected_field_idx {
            14 => {
                // Governor toggle
                self.gov_on = !self.gov_on;
                Some(UiToEngineMsg::SetLiveParam {
                    key: "gov.on".to_string(),
                    val: if self.gov_on { 1.0 } else { 0.0 },
                })
            }
            23 => {
                // Hook enable toggle
                self.hook_on = !self.hook_on;
                Some(self.hook_msg())
            }
            24 => {
                // Hook site cycle
                self.cycle_hook_site(true);
                Some(self.hook_msg())
            }
            _ => None,
        }
    }
}

pub fn render(f: &mut Frame, area: Rect, app: &App) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Min(10), Constraint::Length(3)])
        .split(area);

    let columns = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(33),
            Constraint::Percentage(34),
            Constraint::Percentage(33),
        ])
        .split(chunks[0]);

    let state = &app.tab2_state;
    let focus = state.selected_field_idx;

    // Telemetry from last hud frame
    let grad_mag = app.last_hud_frame.as_ref().map(|fr| fr.grad_mag).unwrap_or(0.0);
    let splat_mag = app.last_hud_frame.as_ref().map(|fr| fr.splat_mag).unwrap_or(0.0);
    let goal_mag = app.last_hud_frame.as_ref().map(|fr| fr.goal_mag).unwrap_or(0.0);
    let scars = app.last_hud_frame.as_ref().map(|fr| fr.scars).unwrap_or(0);
    let hook_apps = app.last_hud_frame.as_ref().and_then(|fr| fr.hook_applications).unwrap_or(0);

    // ========================================================================
    // Column 1: Surface 1 - Residual Forces
    // ========================================================================
    let bar_cap = make_slider(state.residual_cap, 0.0, 20.0, 8);
    let bar_goal = make_slider(state.residual_goal, 0.0, 2.0, 8);
    let bar_field = make_slider(state.residual_field, 0.0, 2.0, 8);
    let bar_splat = make_slider(state.residual_splat, 0.0, 2.0, 8);
    let bar_dt = make_slider(state.residual_dt, 0.001, 0.20, 8);
    let bar_ramp_l = make_slider(state.force_ramp_len as f32, 0.0, 100.0, 8);
    let bar_ramp_s = make_slider(state.force_ramp_str, 0.0, 1.0, 8);
    let bar_fmax = make_slider(state.residual_field_max, 0.0, 100.0, 6);
    let bar_smax = make_slider(state.residual_splat_max, 0.0, 100.0, 6);
    let bar_gmax = make_slider(state.residual_goal_max, 0.0, 100.0, 6);

    let col1_lines = vec![
        Line::from(vec![
            Span::styled(if focus == 0 { " ▶ " } else { "   " }, Style::default().fg(Color::Yellow)),
            Span::styled("residual.cap   ", if focus == 0 { Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD) } else { Style::default().fg(Color::LightGreen) }),
            Span::styled(format!("[{}] ", bar_cap), Style::default().fg(Color::Green)),
            Span::styled(format!("{:.2}", state.residual_cap), Style::default().fg(Color::White).add_modifier(Modifier::BOLD)),
        ]),
        Line::from(vec![
            Span::styled(if focus == 1 { " ▶ " } else { "   " }, Style::default().fg(Color::Yellow)),
            Span::styled("residual.goal  ", if focus == 1 { Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD) } else { Style::default().fg(Color::Gray) }),
            Span::styled(format!("[{}] ", bar_goal), Style::default().fg(Color::Cyan)),
            Span::styled(format!("{:.2}", state.residual_goal), Style::default().fg(Color::White)),
        ]),
        Line::from(vec![
            Span::styled(if focus == 2 { " ▶ " } else { "   " }, Style::default().fg(Color::Yellow)),
            Span::styled("residual.field ", if focus == 2 { Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD) } else { Style::default().fg(Color::Gray) }),
            Span::styled(format!("[{}] ", bar_field), Style::default().fg(Color::Cyan)),
            Span::styled(format!("{:.2}", state.residual_field), Style::default().fg(Color::White)),
        ]),
        Line::from(vec![
            Span::styled(if focus == 3 { " ▶ " } else { "   " }, Style::default().fg(Color::Yellow)),
            Span::styled("residual.splat ", if focus == 3 { Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD) } else { Style::default().fg(Color::Gray) }),
            Span::styled(format!("[{}] ", bar_splat), Style::default().fg(Color::Cyan)),
            Span::styled(format!("{:.2}", state.residual_splat), Style::default().fg(Color::White)),
        ]),
        Line::from(vec![
            Span::styled(if focus == 4 { " ▶ " } else { "   " }, Style::default().fg(Color::Yellow)),
            Span::styled("residual.dt    ", if focus == 4 { Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD) } else { Style::default().fg(Color::Gray) }),
            Span::styled(format!("[{}] ", bar_dt), Style::default().fg(Color::LightBlue)),
            Span::styled(format!("{:.3}", state.residual_dt), Style::default().fg(Color::White)),
        ]),
        Line::from(vec![
            Span::styled(if focus == 5 { " ▶ " } else { "   " }, Style::default().fg(Color::Yellow)),
            Span::styled("force_ramp_len ", if focus == 5 { Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD) } else { Style::default().fg(Color::Gray) }),
            Span::styled(format!("[{}] ", bar_ramp_l), Style::default().fg(Color::LightBlue)),
            Span::styled(format!("{:<3}", state.force_ramp_len), Style::default().fg(Color::White)),
        ]),
        Line::from(vec![
            Span::styled(if focus == 6 { " ▶ " } else { "   " }, Style::default().fg(Color::Yellow)),
            Span::styled("force_ramp_str ", if focus == 6 { Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD) } else { Style::default().fg(Color::Gray) }),
            Span::styled(format!("[{}] ", bar_ramp_s), Style::default().fg(Color::LightBlue)),
            Span::styled(format!("{:.2}", state.force_ramp_str), Style::default().fg(Color::White)),
        ]),
        Line::from(vec![
            Span::styled(if focus == 7 { " ▶ " } else { "   " }, Style::default().fg(Color::Yellow)),
            Span::styled("field_max_mag  ", if focus == 7 { Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD) } else { Style::default().fg(Color::Gray) }),
            Span::styled(format!("[{}] ", bar_fmax), Style::default().fg(Color::DarkGray)),
            Span::styled(format!("{:.1}", state.residual_field_max), Style::default().fg(Color::White)),
        ]),
        Line::from(vec![
            Span::styled(if focus == 8 { " ▶ " } else { "   " }, Style::default().fg(Color::Yellow)),
            Span::styled("splat_max_mag  ", if focus == 8 { Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD) } else { Style::default().fg(Color::Gray) }),
            Span::styled(format!("[{}] ", bar_smax), Style::default().fg(Color::DarkGray)),
            Span::styled(format!("{:.1}", state.residual_splat_max), Style::default().fg(Color::White)),
        ]),
        Line::from(vec![
            Span::styled(if focus == 9 { " ▶ " } else { "   " }, Style::default().fg(Color::Yellow)),
            Span::styled("goal_max_mag   ", if focus == 9 { Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD) } else { Style::default().fg(Color::Gray) }),
            Span::styled(format!("[{}] ", bar_gmax), Style::default().fg(Color::DarkGray)),
            Span::styled(format!("{:.1}", state.residual_goal_max), Style::default().fg(Color::White)),
        ]),
        Line::from(""),
        Line::from(vec![
            Span::styled(" Force Monitors:", Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)),
        ]),
        Line::from(vec![
            Span::styled("   F_grad:  ", Style::default().fg(Color::DarkGray)),
            Span::styled(format!("{:.2} / {:.1}", grad_mag, state.residual_field_max), Style::default().fg(Color::LightGreen)),
            Span::styled("   Scars: ", Style::default().fg(Color::DarkGray)),
            Span::styled(format!("{}", scars), Style::default().fg(Color::Yellow)),
        ]),
        Line::from(vec![
            Span::styled("   F_splat: ", Style::default().fg(Color::DarkGray)),
            Span::styled(format!("{:.2} / {:.1}", splat_mag, state.residual_splat_max), Style::default().fg(Color::LightYellow)),
        ]),
        Line::from(vec![
            Span::styled("   F_goal:  ", Style::default().fg(Color::DarkGray)),
            Span::styled(format!("{:.2} / {:.1}", goal_mag, state.residual_goal_max), Style::default().fg(Color::LightBlue)),
        ]),
    ];

    let col1_border_style = if (0..=9).contains(&focus) {
        Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)
    } else {
        Style::default().fg(Color::Cyan)
    };

    let col1_block = Paragraph::new(col1_lines).block(
        Block::default()
            .borders(Borders::ALL)
            .border_type(BorderType::Rounded)
            .title(" Surface 1: Residual Forces ")
            .style(col1_border_style),
    );
    f.render_widget(col1_block, columns[0]);

    // ========================================================================
    // Column 2: Surface 2 - Logit Biases & Fluid Governor
    // ========================================================================
    let bar_alpha = make_slider(state.field_alpha, 0.0, 1.0, 8);
    let bar_s_scale = make_slider(state.splat_scale, 0.0, 1.0, 8);
    let bar_top_m = make_slider(state.splat_top_m as f32, 1.0, 32.0, 8);
    let bar_top_k = make_slider(state.splat_top_k as f32, 1.0, 256.0, 8);
    let bar_g_vel = make_slider(state.gov_velocity, 0.0, 2.0, 8);
    let bar_g_brk = make_slider(state.gov_brake, 0.0, 20.0, 8);
    let bar_g_vgn = make_slider(state.gov_visc_gain, 0.0, 50.0, 8);
    let bar_g_mb = make_slider(state.gov_max_bias, 0.0, 10.0, 8);
    let bar_bs_pen = make_slider(state.backslash_pen, 0.0, 10.0, 8);
    let bar_h_rep = make_slider(state.hands_repulsion, -5.0, 5.0, 8);
    let bar_h_beta = make_slider(state.hands_beta, 0.0, 5.0, 8);
    let bar_h_bld = make_slider(state.hands_blend, 0.0, 10.0, 8);

    let col2_lines = vec![
        Line::from(vec![
            Span::styled(if focus == 10 { " ▶ " } else { "   " }, Style::default().fg(Color::Yellow)),
            Span::styled("field.alpha    ", if focus == 10 { Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD) } else { Style::default().fg(Color::Gray) }),
            Span::styled(format!("[{}] ", bar_alpha), Style::default().fg(Color::LightCyan)),
            Span::styled(format!("{:.2}", state.field_alpha), Style::default().fg(Color::White)),
        ]),
        Line::from(vec![
            Span::styled(if focus == 11 { " ▶ " } else { "   " }, Style::default().fg(Color::Yellow)),
            Span::styled("splat.scale    ", if focus == 11 { Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD) } else { Style::default().fg(Color::Gray) }),
            Span::styled(format!("[{}] ", bar_s_scale), Style::default().fg(Color::LightCyan)),
            Span::styled(format!("{:.2}", state.splat_scale), Style::default().fg(Color::White)),
        ]),
        Line::from(vec![
            Span::styled(if focus == 12 { " ▶ " } else { "   " }, Style::default().fg(Color::Yellow)),
            Span::styled("splat.top_m    ", if focus == 12 { Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD) } else { Style::default().fg(Color::Gray) }),
            Span::styled(format!("[{}] ", bar_top_m), Style::default().fg(Color::LightCyan)),
            Span::styled(format!("{:<3}", state.splat_top_m), Style::default().fg(Color::White)),
        ]),
        Line::from(vec![
            Span::styled(if focus == 13 { " ▶ " } else { "   " }, Style::default().fg(Color::Yellow)),
            Span::styled("splat.top_k    ", if focus == 13 { Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD) } else { Style::default().fg(Color::Gray) }),
            Span::styled(format!("[{}] ", bar_top_k), Style::default().fg(Color::LightCyan)),
            Span::styled(format!("{:<3}", state.splat_top_k), Style::default().fg(Color::White)),
        ]),
        Line::from(vec![
            Span::styled(if focus == 14 { " ▶ " } else { "   " }, Style::default().fg(Color::Yellow)),
            Span::styled("gov.on         ", if focus == 14 { Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD) } else { Style::default().fg(Color::Gray) }),
            Span::styled(
                if state.gov_on { "< ON >" } else { "< OFF >" },
                if state.gov_on { Style::default().fg(Color::Green).add_modifier(Modifier::BOLD) } else { Style::default().fg(Color::Red).add_modifier(Modifier::BOLD) },
            ),
        ]),
        Line::from(vec![
            Span::styled(if focus == 15 { " ▶ " } else { "   " }, Style::default().fg(Color::Yellow)),
            Span::styled("gov.velocity   ", if focus == 15 { Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD) } else { Style::default().fg(Color::Gray) }),
            Span::styled(format!("[{}] ", bar_g_vel), Style::default().fg(Color::Magenta)),
            Span::styled(format!("{:.2}", state.gov_velocity), Style::default().fg(Color::White)),
        ]),
        Line::from(vec![
            Span::styled(if focus == 16 { " ▶ " } else { "   " }, Style::default().fg(Color::Yellow)),
            Span::styled("gov.brake      ", if focus == 16 { Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD) } else { Style::default().fg(Color::Gray) }),
            Span::styled(format!("[{}] ", bar_g_brk), Style::default().fg(Color::Magenta)),
            Span::styled(format!("{:.1}", state.gov_brake), Style::default().fg(Color::White)),
        ]),
        Line::from(vec![
            Span::styled(if focus == 17 { " ▶ " } else { "   " }, Style::default().fg(Color::Yellow)),
            Span::styled("gov.visc_gain  ", if focus == 17 { Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD) } else { Style::default().fg(Color::Gray) }),
            Span::styled(format!("[{}] ", bar_g_vgn), Style::default().fg(Color::Magenta)),
            Span::styled(format!("{:.1}", state.gov_visc_gain), Style::default().fg(Color::White)),
        ]),
        Line::from(vec![
            Span::styled(if focus == 18 { " ▶ " } else { "   " }, Style::default().fg(Color::Yellow)),
            Span::styled("gov.max_bias   ", if focus == 18 { Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD) } else { Style::default().fg(Color::Gray) }),
            Span::styled(format!("[{}] ", bar_g_mb), Style::default().fg(Color::Magenta)),
            Span::styled(format!("{:.1}", state.gov_max_bias), Style::default().fg(Color::White)),
        ]),
        Line::from(vec![
            Span::styled(if focus == 19 { " ▶ " } else { "   " }, Style::default().fg(Color::Yellow)),
            Span::styled("backslash.pen  ", if focus == 19 { Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD) } else { Style::default().fg(Color::Gray) }),
            Span::styled(format!("[{}] ", bar_bs_pen), Style::default().fg(Color::LightRed)),
            Span::styled(format!("{:.1}", state.backslash_pen), Style::default().fg(Color::White)),
        ]),
        Line::from(vec![
            Span::styled(if focus == 20 { " ▶ " } else { "   " }, Style::default().fg(Color::Yellow)),
            Span::styled("hands.repulsion", if focus == 20 { Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD) } else { Style::default().fg(Color::Gray) }),
            Span::styled(format!("[{}] ", bar_h_rep), Style::default().fg(Color::LightYellow)),
            Span::styled(format!("{:.2}", state.hands_repulsion), Style::default().fg(Color::White)),
        ]),
        Line::from(vec![
            Span::styled(if focus == 21 { " ▶ " } else { "   " }, Style::default().fg(Color::Yellow)),
            Span::styled("hands.beta     ", if focus == 21 { Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD) } else { Style::default().fg(Color::Gray) }),
            Span::styled(format!("[{}] ", bar_h_beta), Style::default().fg(Color::LightYellow)),
            Span::styled(format!("{:.2}", state.hands_beta), Style::default().fg(Color::White)),
        ]),
        Line::from(vec![
            Span::styled(if focus == 22 { " ▶ " } else { "   " }, Style::default().fg(Color::Yellow)),
            Span::styled("hands.blend    ", if focus == 22 { Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD) } else { Style::default().fg(Color::Gray) }),
            Span::styled(format!("[{}] ", bar_h_bld), Style::default().fg(Color::LightYellow)),
            Span::styled(format!("{:.2}", state.hands_blend), Style::default().fg(Color::White)),
        ]),
    ];

    let col2_border_style = if (10..=22).contains(&focus) {
        Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)
    } else {
        Style::default().fg(Color::Cyan)
    };

    let col2_block = Paragraph::new(col2_lines).block(
        Block::default()
            .borders(Borders::ALL)
            .border_type(BorderType::Rounded)
            .title(" Surface 2: Logit Biases & Hands ")
            .style(col2_border_style),
    );
    f.render_widget(col2_block, columns[1]);

    // ========================================================================
    // Column 3: Surface 3 - Layer Hook (HookControls)
    // ========================================================================
    let bar_h_frac = make_slider(state.hook_norm_fraction, 0.0, 0.10, 8);
    let bar_h_start = make_slider(state.hook_start_frac, 0.0, 1.0, 8);
    let bar_h_end = make_slider(state.hook_end_frac, 0.0, 1.0, 8);

    let n_layers: usize = 36; // Default / loaded layer count
    let start_layer = ((state.hook_start_frac * n_layers as f32).floor() as usize).min(n_layers.saturating_sub(1));
    let end_layer = ((state.hook_end_frac * n_layers as f32).ceil() as usize).min(n_layers.saturating_sub(1));

    let col3_lines = vec![
        Line::from(vec![
            Span::styled(if focus == 23 { " ▶ " } else { "   " }, Style::default().fg(Color::Yellow)),
            Span::styled("hook.on        ", if focus == 23 { Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD) } else { Style::default().fg(Color::Gray) }),
            Span::styled(
                if state.hook_on { "< ON >" } else { "< OFF >" },
                if state.hook_on { Style::default().fg(Color::Green).add_modifier(Modifier::BOLD) } else { Style::default().fg(Color::Red).add_modifier(Modifier::BOLD) },
            ),
        ]),
        Line::from(vec![
            Span::styled(if focus == 24 { " ▶ " } else { "   " }, Style::default().fg(Color::Yellow)),
            Span::styled("hook.site      ", if focus == 24 { Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD) } else { Style::default().fg(Color::Gray) }),
            Span::styled(
                format!("< {:?} >", state.hook_site),
                Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD),
            ),
        ]),
        Line::from(vec![
            Span::styled(if focus == 25 { " ▶ " } else { "   " }, Style::default().fg(Color::Yellow)),
            Span::styled("hook.fraction  ", if focus == 25 { Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD) } else { Style::default().fg(Color::Gray) }),
            Span::styled(format!("[{}] ", bar_h_frac), Style::default().fg(Color::LightGreen)),
            Span::styled(format!("{:.4}", state.hook_norm_fraction), Style::default().fg(Color::White)),
        ]),
        Line::from(vec![
            Span::styled(if focus == 26 { " ▶ " } else { "   " }, Style::default().fg(Color::Yellow)),
            Span::styled("hook.start     ", if focus == 26 { Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD) } else { Style::default().fg(Color::Gray) }),
            Span::styled(format!("[{}] ", bar_h_start), Style::default().fg(Color::LightGreen)),
            Span::styled(format!("{:.2}", state.hook_start_frac), Style::default().fg(Color::White)),
        ]),
        Line::from(vec![
            Span::styled(if focus == 27 { " ▶ " } else { "   " }, Style::default().fg(Color::Yellow)),
            Span::styled("hook.end       ", if focus == 27 { Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD) } else { Style::default().fg(Color::Gray) }),
            Span::styled(format!("[{}] ", bar_h_end), Style::default().fg(Color::LightGreen)),
            Span::styled(format!("{:.2}", state.hook_end_frac), Style::default().fg(Color::White)),
        ]),
        Line::from(""),
        Line::from(vec![
            Span::styled(" Hook Telemetry:", Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)),
        ]),
        Line::from(vec![
            Span::styled("   Applications:  ", Style::default().fg(Color::DarkGray)),
            Span::styled(format!("{} layers", if state.hook_on { end_layer.saturating_sub(start_layer) + 1 } else { 0 }), Style::default().fg(Color::White)),
        ]),
        Line::from(vec![
            Span::styled("   Resolved Band: ", Style::default().fg(Color::DarkGray)),
            Span::styled(format!("L{}..L{}", start_layer, end_layer), Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
        ]),
        Line::from(vec![
            Span::styled("   Scale-Free:    ", Style::default().fg(Color::DarkGray)),
            Span::styled(format!("‖Δh‖/‖h‖ ≤ {:.4}", state.hook_norm_fraction), Style::default().fg(Color::Green)),
        ]),
        Line::from(vec![
            Span::styled("   Recent Apps:   ", Style::default().fg(Color::DarkGray)),
            Span::styled(format!("{} hits", hook_apps), Style::default().fg(Color::LightGreen)),
        ]),
    ];

    let col3_border_style = if (23..=27).contains(&focus) {
        Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)
    } else {
        Style::default().fg(Color::Cyan)
    };

    let col3_block = Paragraph::new(col3_lines).block(
        Block::default()
            .borders(Borders::ALL)
            .border_type(BorderType::Rounded)
            .title(" Surface 3: Layer Hook (HookControls) ")
            .style(col3_border_style),
    );
    f.render_widget(col3_block, columns[2]);

    // ========================================================================
    // Bottom: Stability Verdicts — Hydro knobs vs adapted Hydro seat.
    // Formula-native σ/θ/β are not these sliders.
    // ========================================================================
    let seat = app.tab1_state.predicted_hydro_seat();
    let pred_cap = seat.physics.force_cap;
    let pred_goal = seat.physics.goal_force_scale;
    let pred_field = seat.physics.field_wake_scale;
    let cap_zone = crate::algo_scale::zone_ratio(state.residual_cap, pred_cap);
    let goal_zone = crate::algo_scale::zone_ratio(state.residual_goal, pred_goal);
    let field_zone = crate::algo_scale::zone_ratio(state.residual_field, pred_field);

    let zone_color = |z: crate::algo_scale::Zone| match z {
        crate::algo_scale::Zone::Hot => Color::Red,
        crate::algo_scale::Zone::Cold => Color::Cyan,
        crate::algo_scale::Zone::In => Color::Green,
    };

    let verdict_line = Line::from(vec![
        Span::styled(" Stability Verdicts:  ", Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)),
        Span::styled(format!("cap {:.2}/{:.2} ", state.residual_cap, pred_cap), Style::default().fg(Color::Green)),
        Span::styled(format!("[{}]  │  ", cap_zone.as_str()), Style::default().fg(zone_color(cap_zone)).add_modifier(Modifier::BOLD)),
        Span::styled(format!("goal {:.3}/{:.3} ", state.residual_goal, pred_goal), Style::default().fg(Color::Cyan)),
        Span::styled(format!("[{}]  │  ", goal_zone.as_str()), Style::default().fg(zone_color(goal_zone)).add_modifier(Modifier::BOLD)),
        Span::styled(format!("field {:.3}/{:.3} ", state.residual_field, pred_field), Style::default().fg(Color::Magenta)),
        Span::styled(format!("[{}]  │  ", field_zone.as_str()), Style::default().fg(zone_color(field_zone)).add_modifier(Modifier::BOLD)),
        Span::styled("Hook: ", Style::default().fg(Color::DarkGray)),
        Span::styled(if state.hook_on { "ACTIVE" } else { "BYPASSED" }, if state.hook_on { Style::default().fg(Color::Green) } else { Style::default().fg(Color::Red) }),
    ]);

    let verdict_block = Paragraph::new(verdict_line).block(
        Block::default()
            .borders(Borders::ALL)
            .border_type(BorderType::Rounded)
            .title(" Stability Verdicts (Live vs Predicted) ")
            .style(Style::default().fg(Color::DarkGray)),
    );
    f.render_widget(verdict_block, chunks[1]);
}
