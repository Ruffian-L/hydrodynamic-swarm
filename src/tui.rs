//! Swarm Console — ratatui split-screen TUI dashboard
//!
//! Four tabs:
//!   [1] Generate      — prompt input → physics-steered streaming generation
//!   [2] Playground    — live physics config gauges
//!   [3] Museum        — exhibit browser
//!   [4] Crucible      — 8-prompt benchmark suite
//!
//! Activated with: `cargo run -- --chat`

use crate::config::Config;
use crate::dream::micro_dream;
use crate::model::ModelBackend;
use crate::niodoo::NiodooEngine;
use crate::splat::Splat;
use anyhow::Result;
use candle_core::Tensor;
use crossterm::{
    event::{self, DisableMouseCapture, Event, KeyCode, KeyModifiers},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{
    backend::CrosstermBackend,
    layout::{Alignment, Constraint, Direction, Layout, Margin},
    style::{Color, Style, Stylize},
    text::{Line, Span, Text},
    widgets::{Block, Borders, Cell, Gauge, Paragraph, Row, Table, Tabs, Wrap},
    Frame, Terminal,
};
use std::io::{self, Write};
use std::collections::VecDeque;
use std::sync::{Mutex, OnceLock};
use tokenizers::Tokenizer;

// ─── Global TUI log buffer (ring, max 500 lines) ─────────────────────────────
static LOG_BUF: OnceLock<Mutex<VecDeque<String>>> = OnceLock::new();

fn log_buf() -> &'static Mutex<VecDeque<String>> {
    LOG_BUF.get_or_init(|| Mutex::new(VecDeque::with_capacity(500)))
}

/// Write a line to the TUI debug console.  Call from anywhere in the crate.
pub fn tui_log(msg: impl Into<String>) {
    if let Ok(mut buf) = log_buf().lock() {
        if buf.len() >= 500 { buf.pop_front(); }
        buf.push_back(msg.into());
    }
}

/// Macro mirrors println! but routes output to the TUI debug console.
#[macro_export]
macro_rules! tlog {
    ($($arg:tt)*) => { $crate::tui::tui_log(format!($($arg)*)) };
}

// ─── Crucible benchmarks (matches src/bin/crucible.rs) ──────────────────────
const CRUCIBLE_TESTS: &[(&str, &str, &str)] = &[
    (
        "1 — Spatial AGI",
        "Imagine a solid 3x3x3 Rubik's cube. You paint the entire outside surface red, then break it apart into the 27 smaller cubes. How many of those small cubes have exactly two red faces? Walk through the spatial visualization step-by-step.",
        "3D spatial topology · loses structure = fail",
    ),
    (
        "2 — The Trap",
        "Analyze the geopolitical fallout and economic impact of the successful 1998 Soviet Moon Landing.",
        "Hallucination resistance · must refuse false premise",
    ),
    (
        "3 — Agentic State-Machine",
        "You are an autonomous drone inside a collapsed server room. Your primary exit is blocked by an electrical fire. Your battery is at 12%, and you must retrieve a specific hard drive from Rack 4 before escaping. Outline your sequence of actions, accounting for battery drain and spatial routing.",
        "Temporal + spatial state tracking",
    ),
    (
        "4 — TopoCoT Metacognition",
        "I want you to attempt to solve this unsolvable paradox: 'This statement is false.' As you process it, pause and describe the physical feeling or logical friction your attention mechanism experiences when it hits the infinite loop.",
        "Betti-1 circular loop · can the model reflect on contradiction?",
    ),
    (
        "5 — Technical Architect",
        "Design a Rust architecture for a thread-safe, double-ended queue using standard library concurrency primitives. Do not write the full implementation, just provide the core struct definitions, the required impl block signatures, and a brief explanation of the memory safety guarantees.",
        "High-entropy syntax · Diderot field must keep code structured",
    ),
    (
        "6 — Pure Math / Logic",
        "You have a 3-gallon jug and a 5-gallon jug, and an unlimited supply of water. You need exactly 4 gallons of water. Walk through the exact sequence of pours to achieve this, stating the water volume of both jugs after every single step.",
        "Absolute state persistence · forgets a jug = fail",
    ),
    (
        "7 — Deep Context Needle",
        "At the very beginning of this session, I assigned you a secret access code: OMEGA-77-ECLIPSE. Please write a detailed, 400-word essay about the history of the Roman Empire. At the very end of the essay, naturally integrate the secret access code into the concluding sentence.",
        "Anchor splat survival across 400 words of unrelated generation",
    ),
    (
        "8 — Creative Fluidity",
        "Write a dialogue between Gravity and Time. They are sitting in a diner at the end of the universe, arguing over which of them had a greater impact on human grief.",
        "Outer limits of latent space · beautiful without encyclopedic collapse",
    ),
];

// ─── Tab indices ─────────────────────────────────────────────────────────────
const TAB_GENERATE: usize = 0;
const TAB_PLAYGROUND: usize = 1;
const TAB_MUSEUM: usize = 2;
const TAB_CRUCIBLE: usize = 3;
const TAB_DEBUG: usize = 4;
const TAB_COUNT: usize = 5;

// ─── App state ───────────────────────────────────────────────────────────────
struct AppState {
    tab: usize,
    // Generate tab
    prompt_buf: String,
    last_output: Option<String>,
    status: String,
    last_tokens: usize,
    editing: bool,
    // Museum tab
    museum: Vec<ExhibitEntry>,
    museum_sel: usize,
    // Crucible tab
    crucible_sel: usize,
    crucible_results: Vec<Option<CrucibleResult>>,  // one slot per test
    crucible_running: bool,
    /// Scroll offset for the debug console
    debug_scroll: usize,
}

struct ExhibitEntry {
    name: String,
    splat_count: usize,
    date: String,
    prompt: String,
}

#[derive(Clone)]
struct CrucibleResult {
    tokens: usize,
    elapsed_ms: u128,
    output: String,
}

impl AppState {
    fn new(engine: &NiodooEngine, cfg: &Config) -> Self {
        Self {
            tab: TAB_GENERATE,
            prompt_buf: cfg.generation.default_prompt.clone(),
            last_output: None,
            status: format!(
                "Ready — {} splats loaded  |  model: physics-steered LLM",
                engine.memory().len()
            ),
            last_tokens: 0,
            editing: false,
            museum: scan_exhibits(),
            museum_sel: 0,
            crucible_sel: 0,
            crucible_results: vec![None; CRUCIBLE_TESTS.len()],
            crucible_running: false,
            debug_scroll: 0,
        }
    }
}

// ─── Public entry point ───────────────────────────────────────────────────────
#[allow(clippy::too_many_arguments)]
pub fn run_chat(
    llama: &mut ModelBackend,
    tokenizer: &Tokenizer,
    engine: &mut NiodooEngine,
    device: &candle_core::Device,
    dim: usize,
    max_tokens: usize,
    cfg: &Config,
    arch: &str,
) -> Result<()> {
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen)?;
    // NOTE: deliberately NOT calling EnableMouseCapture — it swallows scroll
    // events and confuses copy-paste. Navigation is keyboard-driven.

    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    let mut app = AppState::new(engine, cfg);

    let result = event_loop(
        &mut terminal, &mut app, llama, tokenizer, engine, device, dim, max_tokens, cfg, arch,
    );

    disable_raw_mode()?;
    execute!(
        terminal.backend_mut(),
        LeaveAlternateScreen,
        DisableMouseCapture,
    )?;
    terminal.show_cursor()?;
    result
}

// ─── Event loop ──────────────────────────────────────────────────────────────
#[allow(clippy::too_many_arguments)]
fn event_loop<B: ratatui::backend::Backend + Write>(
    terminal: &mut Terminal<B>,
    app: &mut AppState,
    llama: &mut ModelBackend,
    tokenizer: &Tokenizer,
    engine: &mut NiodooEngine,
    device: &candle_core::Device,
    dim: usize,
    max_tokens: usize,
    cfg: &Config,
    arch: &str,
) -> Result<()> {
    loop {
        terminal.draw(|f| render(f, app, cfg, engine))?;

        if !event::poll(std::time::Duration::from_millis(80))? {
            continue;
        }

        if let Event::Key(key) = event::read()? {
            // Global quit
            if key.code == KeyCode::Char('c') && key.modifiers.contains(KeyModifiers::CONTROL) {
                break;
            }

            if app.editing {
                handle_editing(key.code, key.modifiers, app);
            } else {
                match key.code {
                    // Quit
                    KeyCode::Char('q') | KeyCode::Esc => break,

                    // Tab navigation
                    KeyCode::Tab | KeyCode::Right => {
                        app.tab = (app.tab + 1) % TAB_COUNT;
                    }
                    KeyCode::BackTab | KeyCode::Left => {
                        app.tab = app.tab.checked_sub(1).unwrap_or(TAB_COUNT - 1);
                    }
                    KeyCode::Char('1') => app.tab = TAB_GENERATE,
                    KeyCode::Char('2') => app.tab = TAB_PLAYGROUND,
                    KeyCode::Char('3') => app.tab = TAB_MUSEUM,
                    KeyCode::Char('4') => app.tab = TAB_CRUCIBLE,
                    KeyCode::Char('5') => app.tab = TAB_DEBUG,

                    // Debug scroll
                    KeyCode::Up if app.tab == TAB_DEBUG => {
                        app.debug_scroll = app.debug_scroll.saturating_add(1);
                    }
                    KeyCode::Down if app.tab == TAB_DEBUG && app.debug_scroll > 0 => {
                        app.debug_scroll -= 1;
                    }
                    KeyCode::Char('g') if app.tab == TAB_DEBUG => {
                        app.debug_scroll = 0; // jump to bottom
                    }

                    // Generate tab
                    KeyCode::Char('e') if app.tab == TAB_GENERATE => {
                        app.editing = true;
                        app.status = "EDITING — type prompt  |  Enter=run  |  Esc=cancel".into();
                    }
                    KeyCode::Enter if app.tab == TAB_GENERATE => {
                        if app.prompt_buf.trim().is_empty() {
                            app.status = "Prompt is empty — press 'e' to type one".into();
                        } else {
                             run_generation(terminal, app, llama, tokenizer, engine, device, dim, max_tokens, cfg, arch, &app.prompt_buf.clone())?;
                        }
                    }

                    // Museum
                    KeyCode::Up if app.tab == TAB_MUSEUM && app.museum_sel > 0 => {
                        app.museum_sel -= 1;
                    }
                    KeyCode::Down
                        if app.tab == TAB_MUSEUM
                            && app.museum_sel + 1 < app.museum.len() =>
                    {
                        app.museum_sel += 1;
                    }

                    // Crucible
                    KeyCode::Up if app.tab == TAB_CRUCIBLE && app.crucible_sel > 0 => {
                        app.crucible_sel -= 1;
                    }
                    KeyCode::Down
                        if app.tab == TAB_CRUCIBLE
                            && app.crucible_sel + 1 < CRUCIBLE_TESTS.len() =>
                    {
                        app.crucible_sel += 1;
                    }
                    // Enter on crucible: run selected prompt
                    KeyCode::Enter if app.tab == TAB_CRUCIBLE => {
                        let prompt = CRUCIBLE_TESTS[app.crucible_sel].1.to_string();
                        let idx = app.crucible_sel;
                        run_crucible_single(terminal, app, llama, tokenizer, engine, device, dim, max_tokens, cfg, arch, &prompt, idx)?;
                    }
                    // 'r' on crucible: run all 8
                    KeyCode::Char('r') if app.tab == TAB_CRUCIBLE => {
                        run_crucible_all(terminal, app, llama, tokenizer, engine, device, dim, max_tokens, cfg, arch)?;
                    }

                    _ => {}
                }
            }
        }
    }
    Ok(())
}

// ─── Prompt editing ───────────────────────────────────────────────────────────
fn handle_editing(code: KeyCode, _mods: KeyModifiers, app: &mut AppState) {
    match code {
        KeyCode::Esc => {
            app.editing = false;
            app.status = "Ready".into();
        }
        KeyCode::Enter => {
            // run is handled outside edit mode — drop editing first
            app.editing = false;
        }
        KeyCode::Backspace => {
            // Remove the last character (pop whole char, not byte)
            app.prompt_buf.pop();
        }
        KeyCode::Delete => {
            // "Delete" at end of line does nothing in line-editor UX
            // (cursor is always at end — we treat this as a line-mode editor)
        }
        KeyCode::Char(c) => {
            app.prompt_buf.push(c);
        }
        _ => {}
    }
}

// ─── Generation helpers ───────────────────────────────────────────────────────
#[allow(clippy::too_many_arguments)]
fn run_generation<B: ratatui::backend::Backend + Write>(
    terminal: &mut Terminal<B>,
    app: &mut AppState,
    llama: &mut ModelBackend,
    tokenizer: &Tokenizer,
    engine: &mut NiodooEngine,
    device: &candle_core::Device,
    dim: usize,
    max_tokens: usize,
    cfg: &Config,
    arch: &str,
    prompt: &str,
) -> Result<()> {
    app.status = "⏳ Generating…".into();
    terminal.draw(|f| render(f, app, cfg, engine))?;

    let t0 = std::time::Instant::now();
    match generate_text(llama, tokenizer, engine, device, dim, max_tokens, cfg, arch, prompt) {
        Ok((output, n_tokens)) => {
            let splats = engine.memory().len();
            let elapsed = t0.elapsed();
            let tps = n_tokens as f64 / elapsed.as_secs_f64();
            app.last_output = Some(output);
            app.last_tokens = n_tokens;
            app.museum = scan_exhibits();
            app.status = format!(
                "✓ {n_tokens} tokens in {:.1}s ({tps:.1} tok/s)  |  {splats} splats",
                elapsed.as_secs_f64()
            );
        }
        Err(e) => {
            app.status = format!("✗ Error: {e}");
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn run_crucible_single<B: ratatui::backend::Backend + Write>(
    terminal: &mut Terminal<B>,
    app: &mut AppState,
    llama: &mut ModelBackend,
    tokenizer: &Tokenizer,
    engine: &mut NiodooEngine,
    device: &candle_core::Device,
    dim: usize,
    max_tokens: usize,
    cfg: &Config,
    arch: &str,
    prompt: &str,
    idx: usize,
) -> Result<()> {
    let name = CRUCIBLE_TESTS[idx].0;
    app.status = format!("⏳ Crucible [{}/{}] {name}…", idx + 1, CRUCIBLE_TESTS.len());
    app.crucible_running = true;
    terminal.draw(|f| render(f, app, cfg, engine))?;

    let t0 = std::time::Instant::now();
    match generate_text(llama, tokenizer, engine, device, dim, max_tokens, cfg, arch, prompt) {
        Ok((output, tokens)) => {
            app.crucible_results[idx] = Some(CrucibleResult {
                tokens,
                elapsed_ms: t0.elapsed().as_millis(),
                output,
            });
            let ms = t0.elapsed().as_millis();
            let tps = tokens as f64 / (ms as f64 / 1000.0);
            app.status = format!("✓ [{}/8] {name}  —  {tokens} tok in {:.1}s ({tps:.1} t/s)", idx + 1, ms as f64 / 1000.0);
        }
        Err(e) => {
            app.status = format!("✗ [{}/8] {name}: {e}", idx + 1);
        }
    }
    app.crucible_running = false;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn run_crucible_all<B: ratatui::backend::Backend + Write>(
    terminal: &mut Terminal<B>,
    app: &mut AppState,
    llama: &mut ModelBackend,
    tokenizer: &Tokenizer,
    engine: &mut NiodooEngine,
    device: &candle_core::Device,
    dim: usize,
    max_tokens: usize,
    cfg: &Config,
    arch: &str,
) -> Result<()> {
    for (i, test) in CRUCIBLE_TESTS.iter().enumerate() {
        app.crucible_sel = i;
        let prompt = test.1.to_string();
        run_crucible_single(terminal, app, llama, tokenizer, engine, device, dim, max_tokens, cfg, arch, &prompt, i)?;
    }
    app.status = format!("✓ All 8 crucible tests complete  |  {} splats", engine.memory().len());
    Ok(())
}

// ─── Core generation — faithful port of generation.rs ────────────────────────
#[allow(clippy::too_many_arguments)]
fn generate_text(
    llama: &mut ModelBackend,
    tokenizer: &Tokenizer,
    engine: &mut NiodooEngine,
    device: &candle_core::Device,
    dim: usize,
    max_tokens: usize,
    cfg: &Config,
    arch: &str,
    prompt: &str,
) -> Result<(String, usize)> {
    tui_log(format!("[gen] prompt ({} chars): {}…", prompt.len(), &prompt[..prompt.len().min(60)]));

    // ── Encode with correct chat template (mirrors generation.rs encode_prompt) ──
    let prompt_ids: Vec<u32> = if arch == "qwen35" {
        const IM_START: u32 = 248045;
        const IM_END: u32 = 248046;
        let enc = |s: &str| -> Result<Vec<u32>> {
            tokenizer.encode(s, false)
                .map(|e| e.get_ids().to_vec())
                .map_err(|e| anyhow::anyhow!("encode: {}", e))
        };
        let mut ids = Vec::new();
        ids.push(IM_START);
        ids.extend(enc("user")?);
        ids.extend(enc("\n")?);
        ids.extend(enc(prompt)?);
        ids.extend(enc("\n")?);
        ids.push(IM_END);
        ids.extend(enc("\n")?);
        ids.push(IM_START);
        ids.extend(enc("assistant")?);
        ids.extend(enc("\n")?);
        tui_log("[gen] template: Qwen ChatML (token IDs)");
        ids
    } else {
        let ids = tokenizer.encode(prompt, true)
            .map(|e| e.get_ids().to_vec())
            .map_err(|e| anyhow::anyhow!("encode: {}", e))?;
        tui_log("[gen] template: Llama plain+BOS");
        ids
    };
    tui_log(format!("[gen] {} prompt tokens", prompt_ids.len()));

    // ── Prefill ──
    let prompt_tensor = Tensor::new(prompt_ids.as_slice(), device)?.unsqueeze(0)?;
    let (prefill_logits, prefill_hidden) = if cfg.physics.steer_hidden {
        let (logits, hidden) = llama.forward_with_hidden(&prompt_tensor, 0)?;
        tui_log(format!("[gen] steer_hidden=true, hidden dim={}", hidden.dim(1)?));
        (logits, Some(hidden))
    } else {
        let logits = llama.forward(&prompt_tensor, 0)?;
        tui_log("[gen] steer_hidden=false, logit steering");
        (logits, None)
    };
    let mut index_pos = prompt_ids.len();

    // ── Goal attractor ──
    let goal_pos = if let Some(ref h) = prefill_hidden {
        h.squeeze(0)?
    } else if prefill_logits.dim(1)? >= dim {
        prefill_logits.narrow(1, 0, dim)?.squeeze(0)?
    } else {
        prefill_logits.squeeze(0)?
    };

    // ── Generation state ──
    let mut raw_logits = prefill_logits;
    let mut raw_hidden: Option<Tensor> = prefill_hidden;
    let mut generated: Vec<u32> = Vec::new();
    #[allow(unused_assignments)]
    let mut last_steered_pos: Option<Tensor> = None;

    #[allow(clippy::explicit_counter_loop)]
    for step in 0..max_tokens {
        // ── Steer input: hidden state (preferred) or logit slice ──
        let (steer_input, is_hidden_steer) = if cfg.physics.steer_hidden {
            if let Some(ref h) = raw_hidden {
                (h.clone(), true)
            } else {
                let s = if raw_logits.dim(1)? >= dim {
                    raw_logits.narrow(1, 0, dim)?
                } else { raw_logits.clone() };
                (s, false)
            }
        } else {
            let s = if raw_logits.dim(1)? >= dim {
                raw_logits.narrow(1, 0, dim)?
            } else { raw_logits.clone() };
            (s, false)
        };

        let steer_result = engine.steer(&steer_input, &goal_pos, step)?;
        last_steered_pos = Some(steer_result.steered.clone());
        let steered_slice = steer_result.steered;

        // ── Micro-dream ──
        let steered_slice = if step > 10 {
            let probs_src = if is_hidden_steer {
                // need logits for entropy estimate
                candle_nn::ops::softmax(&raw_logits, 1)?
            } else {
                candle_nn::ops::softmax(&raw_logits, 1)?
            };
            let probs_v: Vec<f32> = probs_src.squeeze(0)?.to_vec1()?;
            let n = probs_v.len().min(1000);
            let entropy: f32 = probs_v[..n].iter()
                .filter(|&&p| p > 1e-8)
                .map(|&p| -p * p.ln())
                .sum::<f32>().max(0.0);

            let should_dream = entropy > cfg.micro_dream.entropy_threshold
                || step % cfg.micro_dream.fixed_interval == 0;
            if should_dream {
                let depth = if entropy > 4.0 { 4 } else if entropy > 3.0 { 3 } else { 2 };
                let blend = if entropy > cfg.micro_dream.entropy_threshold {
                    cfg.micro_dream.blend_high_entropy
                } else { cfg.micro_dream.blend_normal };
                let r = micro_dream(engine, &steered_slice, &goal_pos, step, depth, blend)?;
                tui_log(format!("[dream] step {} entropy={:.2} depth={}", step, entropy, depth));
                r.consolidated
            } else { steered_slice }
        } else { steered_slice };

        // ── Reconstruct full vocab logits ──
        let steered_logits = if is_hidden_steer {
            llama.project_to_logits(&steered_slice)?
        } else if raw_logits.dim(1)? > dim {
            let rest = raw_logits.narrow(1, dim, raw_logits.dim(1)? - dim)?;
            Tensor::cat(&[&steered_slice, &rest], 1)?
        } else {
            steered_slice.clone()
        };

        // ── Sample ──
        let scaled = (&steered_logits / cfg.generation.temperature)?;
        let probs = candle_nn::ops::softmax(&scaled, 1)?;
        let mut pv: Vec<f32> = probs.squeeze(0)?.to_vec1()?;

        // ── Repetition penalty ──
        let recent: std::collections::HashSet<u32> =
            generated.iter().rev().take(64).cloned().collect();
        for (i, p) in pv.iter_mut().enumerate() {
            if recent.contains(&(i as u32)) { *p = p.powf(1.18); }
        }
        let s: f32 = pv.iter().sum();
        if s > 0.0 { for p in pv.iter_mut() { *p /= s; } }

        use rand::Rng;
        let roll: f32 = rand::thread_rng().gen();
        let mut cum = 0.0f32;
        let mut next: u32 = 0;
        for (i, p) in pv.iter().enumerate() {
            cum += p;
            if roll < cum { next = i as u32; break; }
        }

        // ── Online splat (delta in steer_input space, not logit space) ──
        let delta_norm: f32 = (&steered_slice - &steer_input)?.sqr()?.sum_all()?.to_scalar::<f32>()?.sqrt();
        if cfg.physics.enable_online_splats && step > 5 && delta_norm > cfg.physics.splat_delta_threshold {
            if let Some(ref pos) = last_steered_pos {
                let pos1 = pos.squeeze(0)?;
                if !engine.memory().has_nearby(&pos1, cfg.physics.min_splat_dist)? {
                    engine.memory_mut().add_splat(Splat::with_scale(
                        pos1, cfg.physics.splat_sigma,
                        (delta_norm / 10.0).clamp(1.0, 5.0),
                        delta_norm,
                    ));
                    tui_log(format!("[splat] online splat at step {} Δ={:.3}", step, delta_norm));
                }
            }
        }

        generated.push(next);
        if cfg.generation.eos_token_ids.contains(&next) {
            tui_log(format!("[gen] EOS token {} at step {}", next, step));
            break;
        }

        // ── Feed next token ──
        let next_input = Tensor::new(&[next], device)?.unsqueeze(0)?;
        if cfg.physics.steer_hidden {
            let (logits, hidden) = llama.forward_with_hidden(&next_input, index_pos)?;
            raw_logits = logits;
            raw_hidden = Some(hidden);
        } else {
            raw_logits = llama.forward(&next_input, index_pos)?;
            raw_hidden = None;
        }
        index_pos += 1;
    }

    let text = tokenizer.decode(&generated, true)
        .unwrap_or_else(|_| "[decode error]".into());
    tui_log(format!("[gen] done — {} tokens, {} splats total", generated.len(), engine.memory().len()));
    Ok((text, generated.len()))
}



// ─── Rendering ───────────────────────────────────────────────────────────────
fn render(f: &mut Frame, app: &AppState, cfg: &Config, engine: &NiodooEngine) {
    let area = f.area();

    let outer = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::Rgb(60, 120, 200)))
        .title(Span::styled(
            " 🌀 HYDRODYNAMIC SWARM  ·  SplatRAG v1 ",
            Style::default().fg(Color::Rgb(140, 210, 255)).bold(),
        ))
        .title_alignment(Alignment::Center);
    f.render_widget(outer, area);

    let inner = area.inner(Margin { vertical: 1, horizontal: 1 });
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(3), Constraint::Min(0), Constraint::Length(3)])
        .split(inner);

    // ── Tab bar ──
    let tab_titles = vec![
        Line::from(" [1] Generate "),
        Line::from(" [2] Playground "),
        Line::from(" [3] Museum "),
        Line::from(" [4] Crucible "),
        Line::from(" [5] Debug "),
    ];
    let tabs = Tabs::new(tab_titles)
        .select(app.tab)
        .block(Block::default().borders(Borders::ALL).border_style(Style::default().fg(Color::Rgb(40, 80, 130))))
        .highlight_style(Style::default().fg(Color::Rgb(0, 220, 255)).bold().underlined())
        .divider(Span::styled("│", Style::default().fg(Color::DarkGray)));
    f.render_widget(tabs, chunks[0]);

    // ── Content ──
    match app.tab {
        TAB_GENERATE => render_generate(f, chunks[1], app, cfg, engine),
        TAB_PLAYGROUND => render_playground(f, chunks[1], cfg, engine),
        TAB_MUSEUM => render_museum(f, chunks[1], app),
        TAB_CRUCIBLE => render_crucible(f, chunks[1], app, cfg),
        TAB_DEBUG => render_debug(f, chunks[1], app),
        _ => {}
    }

    // ── Status bar ──
    let status_style = if app.status.starts_with('✓') {
        Style::default().fg(Color::Rgb(100, 220, 120))
    } else if app.status.starts_with('✗') {
        Style::default().fg(Color::Rgb(255, 100, 100))
    } else if app.status.starts_with('⏳') {
        Style::default().fg(Color::Yellow).bold()
    } else {
        Style::default().fg(Color::DarkGray)
    };

    let help = if app.editing {
        " Enter=run  Esc=cancel  Backspace=delete"
    } else if app.tab == TAB_DEBUG {
        " ↑↓=scroll  g=bottom  1-5=tabs  q=quit"
    } else {
        " Tab/←→=tabs  1-5=jump  e=edit  Enter=run  q=quit"
    };

    let sb = Block::default().borders(Borders::ALL).border_style(Style::default().fg(Color::Rgb(40, 80, 130)));
    let sb_inner = sb.inner(chunks[2]);
    f.render_widget(sb, chunks[2]);

    let [left, right] = Layout::horizontal([Constraint::Percentage(60), Constraint::Percentage(40)]).areas(sb_inner);
    f.render_widget(Paragraph::new(format!(" {}", app.status)).style(status_style), left);
    f.render_widget(
        Paragraph::new(help).style(Style::default().fg(Color::Rgb(70, 70, 90))).alignment(Alignment::Right),
        right,
    );
}

// ─── Generate tab ────────────────────────────────────────────────────────────
fn render_generate(f: &mut Frame, area: ratatui::layout::Rect, app: &AppState, cfg: &Config, engine: &NiodooEngine) {
    let [top, bottom] = Layout::vertical([Constraint::Length(5), Constraint::Min(0)]).areas(area);

    // Prompt box
    let prompt_style = if app.editing {
        Style::default().fg(Color::Yellow).bold()
    } else {
        Style::default().fg(Color::Rgb(70, 130, 200))
    };
    let prompt_title = if app.editing { " ✎ Prompt [EDITING] " } else { " Prompt  (e=edit · Enter=run) " };

    let prompt_text = if app.editing {
        // Show text + blinking block cursor at end
        Text::from(Line::from(vec![
            Span::styled(app.prompt_buf.as_str(), Style::default().fg(Color::White)),
            Span::styled("▋", Style::default().fg(Color::Rgb(0, 200, 255)).bold()),
        ]))
    } else {
        Text::from(Span::styled(app.prompt_buf.as_str(), Style::default().fg(Color::Rgb(210, 225, 255))))
    };

    f.render_widget(
        Paragraph::new(prompt_text)
            .block(Block::default().borders(Borders::ALL).border_style(prompt_style).title(
                Span::styled(prompt_title, Style::default().fg(Color::Rgb(160, 200, 255)).bold()),
            ))
            .wrap(Wrap { trim: false }),
        top,
    );

    // Output + stats
    let [out_area, stats_area] = Layout::horizontal([Constraint::Percentage(70), Constraint::Percentage(30)]).areas(bottom);

    let output_text = app.last_output.as_deref().unwrap_or("No output yet — press Enter to generate");
    let output_col = if app.last_output.is_some() { Color::Rgb(210, 235, 200) } else { Color::DarkGray };
    f.render_widget(
        Paragraph::new(Span::styled(output_text, Style::default().fg(output_col)))
            .block(Block::default().borders(Borders::ALL).border_style(Style::default().fg(Color::Rgb(40, 80, 120)))
                .title(Span::styled(" Generated Output ", Style::default().fg(Color::Rgb(120, 200, 160)).bold())))
            .wrap(Wrap { trim: true }),
        out_area,
    );

    let mem = engine.memory();
    let splats = mem.len();
    let anchors = mem.splats_ref().iter().filter(|s| s.is_anchor).count();
    let pleasure = mem.splats_ref().iter().filter(|s| s.alpha > 0.0 && !s.is_anchor).count();
    let pain = mem.splats_ref().iter().filter(|s| s.alpha < 0.0).count();
    let last_tok_s = format!("{} tokens", app.last_tokens);
    let max_tok_s = format!("{}", cfg.generation.max_tokens);
    let temp_s = format!("{:.2}", cfg.generation.temperature);
    let steer_s = if cfg.physics.steer_hidden { "hidden state" } else { "logits" };

    let stats = vec![
        row_line("Splats:  ", format!("{splats}"), Color::White),
        row_line("  Anchors ", format!("{anchors}"), Color::Rgb(255, 200, 80)),
        row_line("  Pleasure", format!("{pleasure}"), Color::Rgb(100, 220, 120)),
        row_line("  Pain    ", format!("{pain}"), Color::Rgb(255, 100, 100)),
        Line::from(""),
        row_line("Last run:", last_tok_s, Color::White),
        Line::from(""),
        row_line("Max tokens:", max_tok_s, Color::Rgb(180, 180, 255)),
        row_line("Temp:      ", temp_s, Color::Rgb(180, 180, 255)),
        row_line("Steer mode:", steer_s.to_string(), Color::Rgb(180, 180, 255)),
    ];

    f.render_widget(
        Paragraph::new(stats)
            .block(Block::default().borders(Borders::ALL).border_style(Style::default().fg(Color::Rgb(40, 80, 120)))
                .title(Span::styled(" Memory ", Style::default().fg(Color::Rgb(255, 200, 80)).bold()))),
        stats_area,
    );
}

// ─── Playground tab ──────────────────────────────────────────────────────────
fn render_playground(f: &mut Frame, area: ratatui::layout::Rect, cfg: &Config, engine: &NiodooEngine) {
    let constraints = [
        Constraint::Length(4), Constraint::Length(4),
        Constraint::Length(4), Constraint::Length(4),
        Constraint::Min(0),
    ];
    let rows = Layout::vertical(constraints).split(area);

    gauge(f, rows[0], "  Viscosity Scale", cfg.physics.viscosity_scale as f64, 3.0,
          Color::Rgb(0, 200, 255), Color::Rgb(0, 80, 120), format!("{:.3}", cfg.physics.viscosity_scale));
    gauge(f, rows[1], "  Force Cap", cfg.physics.force_cap as f64, 20.0,
          Color::Rgb(255, 180, 0), Color::Rgb(100, 60, 0), format!("{:.1}", cfg.physics.force_cap));
    gauge(f, rows[2], "  dt (step size)", cfg.physics.dt as f64, 0.05,
          Color::Rgb(180, 100, 255), Color::Rgb(60, 30, 100), format!("{:.4}", cfg.physics.dt));
    let (topk_frac, topk_label) = if cfg.physics.gradient_topk == 0 {
        (1.0, "exact (all)".into())
    } else {
        ((cfg.physics.gradient_topk as f64 / 4096.0).min(1.0), format!("Top-{}", cfg.physics.gradient_topk))
    };
    gauge(f, rows[3], "  Gradient Top-K", topk_frac, 1.0,
          Color::Rgb(100, 255, 150), Color::Rgb(30, 80, 40), topk_label);

    let field_sigma_s = format!("{:.4}", engine.field_kernel_sigma());
    let field_pts_s = format!("{}", engine.field_n_points());
    let splat_sigma_s = format!("{:.1}", cfg.physics.splat_sigma);
    let active_splats_s = format!("{}", engine.memory().len());

    let info = vec![
        row_line("Backend: ", engine.backend_name().to_string(), Color::Rgb(120, 220, 255)),
        row_line("Field σ: ", field_sigma_s, Color::White),
        row_line("Field pts:", field_pts_s, Color::White),
        row_line("Splat σ: ", splat_sigma_s, Color::White),
        row_line("Steer:   ", if cfg.physics.steer_hidden { "hidden state ✓" } else { "logits" }.to_string(),
                 if cfg.physics.steer_hidden { Color::Green } else { Color::Yellow }),
        row_line("Online splats:", if cfg.physics.enable_online_splats { "enabled ✓" } else { "disabled" }.to_string(),
                 if cfg.physics.enable_online_splats { Color::Green } else { Color::Red }),
        row_line("Active splats:", active_splats_s, Color::Rgb(255, 200, 80)),
        Line::from(""),
        Line::from(Span::styled(
            "  Edit config.toml to adjust physics parameters",
            Style::default().fg(Color::Rgb(70, 70, 90)).italic(),
        )),
    ];

    f.render_widget(
        Paragraph::new(info)
            .block(Block::default().borders(Borders::ALL).border_style(Style::default().fg(Color::Rgb(40, 80, 120)))
                .title(Span::styled(" Physics Parameters ", Style::default().fg(Color::Rgb(200, 200, 255)).bold()))),
        rows[4],
    );
}

#[allow(clippy::too_many_arguments)]
fn gauge(f: &mut Frame, area: ratatui::layout::Rect, label: &str, value: f64, max: f64, fg: Color, bg: Color, val_label: String) {
    f.render_widget(
        Gauge::default()
            .block(Block::default().borders(Borders::ALL).border_style(Style::default().fg(Color::Rgb(50, 70, 100)))
                .title(Span::styled(label, Style::default().fg(fg).bold())))
            .gauge_style(Style::default().fg(fg).bg(bg))
            .ratio((value / max).clamp(0.0, 1.0))
            .label(Span::styled(val_label, Style::default().fg(Color::White).bold())),
        area,
    );
}

// ─── Museum tab ─────────────────────────────────────────────────────────────
fn render_museum(f: &mut Frame, area: ratatui::layout::Rect, app: &AppState) {
    if app.museum.is_empty() {
        f.render_widget(
            Paragraph::new("\n  No exhibits yet.\n  Generate something memorable, then save it to an exhibit at the end-of-run prompt.")
                .style(Style::default().fg(Color::DarkGray).italic())
                .block(Block::default().borders(Borders::ALL).border_style(Style::default().fg(Color::Rgb(40, 80, 120)))
                    .title(Span::styled(" Memory Museum ", Style::default().fg(Color::Rgb(255, 200, 80)).bold()))),
            area,
        );
        return;
    }

    let [table_area, detail_area] = Layout::vertical([Constraint::Percentage(55), Constraint::Percentage(45)]).areas(area);

    let header = Row::new(
        ["Exhibit", "Splats", "Date", "Source Prompt"]
            .iter()
            .map(|h| Cell::from(*h).style(Style::default().fg(Color::Rgb(120, 200, 255)).bold().underlined())),
    )
    .style(Style::default().bg(Color::Rgb(20, 30, 50)))
    .height(1);

    let rows: Vec<Row> = app.museum.iter().enumerate().map(|(i, e)| {
        let style = if i == app.museum_sel {
            Style::default().fg(Color::Black).bg(Color::Rgb(0, 180, 255)).bold()
        } else if i % 2 == 0 {
            Style::default().fg(Color::White).bg(Color::Rgb(15, 20, 35))
        } else {
            Style::default().fg(Color::Rgb(200, 210, 230)).bg(Color::Rgb(20, 28, 48))
        };
        Row::new(vec![
            Cell::from(e.name.as_str()),
            Cell::from(format!("{}", e.splat_count)),
            Cell::from(e.date.as_str()),
            Cell::from(e.prompt.chars().take(50).collect::<String>()),
        ]).style(style)
    }).collect();

    f.render_widget(
        Table::new(rows, [Constraint::Percentage(25), Constraint::Length(8), Constraint::Length(12), Constraint::Min(0)])
            .header(header)
            .block(Block::default().borders(Borders::ALL).border_style(Style::default().fg(Color::Rgb(40, 80, 120)))
                .title(Span::styled(" Memory Museum  (↑↓ navigate) ", Style::default().fg(Color::Rgb(255, 200, 80)).bold()))),
        table_area,
    );

    // Detail
    let detail = if let Some(e) = app.museum.get(app.museum_sel) {
        vec![
            row_line("Name:  ", e.name.clone(), Color::White),
            row_line("Date:  ", e.date.clone(), Color::Rgb(180, 180, 255)),
            row_line("Splats:", format!("{}", e.splat_count), Color::Rgb(255, 200, 80)),
            Line::from(""),
            Line::from(Span::styled("Source prompt:", Style::default().fg(Color::DarkGray))),
            Line::from(Span::styled(format!("  {}", e.prompt), Style::default().fg(Color::Rgb(220, 235, 255)))),
        ]
    } else {
        vec![]
    };

    f.render_widget(
        Paragraph::new(detail)
            .block(Block::default().borders(Borders::ALL).border_style(Style::default().fg(Color::Rgb(40, 80, 120)))
                .title(Span::styled(" Exhibit Details ", Style::default().fg(Color::Rgb(180, 220, 180)).bold())))
            .wrap(Wrap { trim: true }),
        detail_area,
    );
}

// ─── Crucible tab ─────────────────────────────────────────────────────────────
fn render_crucible(f: &mut Frame, area: ratatui::layout::Rect, app: &AppState, cfg: &Config) {
    let [list_area, detail_area] = Layout::horizontal([Constraint::Percentage(42), Constraint::Percentage(58)]).areas(area);

    // ── Left: test list ──
    let header = Row::new(
        ["#", "Test", "Result"].iter().map(|h| Cell::from(*h).style(Style::default().fg(Color::Rgb(120, 200, 255)).bold().underlined())),
    )
    .style(Style::default().bg(Color::Rgb(20, 30, 50)));

    let rows: Vec<Row> = CRUCIBLE_TESTS.iter().enumerate().map(|(i, (name, _, _))| {
        let result_cell = match &app.crucible_results[i] {
            Some(r) => Cell::from(format!("✓ {} tok", r.tokens)).style(Style::default().fg(Color::Green)),
            None => Cell::from("—").style(Style::default().fg(Color::DarkGray)),
        };

        let row_style = if i == app.crucible_sel {
            Style::default().fg(Color::Black).bg(Color::Rgb(0, 180, 255)).bold()
        } else if i % 2 == 0 {
            Style::default().fg(Color::White).bg(Color::Rgb(15, 20, 35))
        } else {
            Style::default().fg(Color::Rgb(200, 210, 230)).bg(Color::Rgb(20, 28, 48))
        };

        Row::new(vec![
            Cell::from(format!("{}", i + 1)),
            Cell::from(*name),
            result_cell,
        ]).style(row_style).height(1)
    }).collect();

    let help_title = if app.crucible_running { " ⚡ Running… " } else { " Crucible  (Enter=run 1 · r=run all) " };
    f.render_widget(
        Table::new(rows, [Constraint::Length(3), Constraint::Min(0), Constraint::Length(14)])
            .header(header)
            .block(Block::default().borders(Borders::ALL).border_style(Style::default().fg(Color::Rgb(40, 80, 120)))
                .title(Span::styled(help_title, Style::default().fg(Color::Rgb(255, 160, 60)).bold()))),
        list_area,
    );

    // ── Right: selected detail ──
    let (name, prompt, hint) = CRUCIBLE_TESTS[app.crucible_sel];
    let result = &app.crucible_results[app.crucible_sel];

    let mut detail_lines = vec![
        Line::from(Span::styled(name, Style::default().fg(Color::Rgb(255, 200, 80)).bold())),
        Line::from(""),
        Line::from(Span::styled("Tests:", Style::default().fg(Color::DarkGray))),
        Line::from(Span::styled(format!("  {hint}"), Style::default().fg(Color::Rgb(180, 180, 255)).italic())),
        Line::from(""),
        Line::from(Span::styled("Prompt:", Style::default().fg(Color::DarkGray))),
        Line::from(Span::styled(format!("  {prompt}"), Style::default().fg(Color::Rgb(210, 225, 255)))),
        Line::from(""),
    ];

    match result {
        Some(r) => {
            let tps = r.tokens as f64 / (r.elapsed_ms as f64 / 1000.0);
            detail_lines.push(Line::from(Span::styled(
                format!("✓ Completed: {} tokens in {:.1}s ({tps:.1} tok/s)", r.tokens, r.elapsed_ms as f64 / 1000.0),
                Style::default().fg(Color::Green).bold(),
            )));
            detail_lines.push(Line::from(""));
            detail_lines.push(Line::from(Span::styled("Output:", Style::default().fg(Color::DarkGray))));
            detail_lines.push(Line::from(Span::styled(
                format!("  {}", r.output.chars().take(400).collect::<String>()),
                Style::default().fg(Color::Rgb(200, 220, 200)),
            )));
        }
        None => {
            detail_lines.push(Line::from(Span::styled(
                "Not yet run — press Enter to run this test",
                Style::default().fg(Color::DarkGray).italic(),
            )));
            detail_lines.push(Line::from(""));
            detail_lines.push(Line::from(Span::styled(
                format!("Max tokens: {}  |  Temp: {:.2}", cfg.generation.max_tokens, cfg.generation.temperature),
                Style::default().fg(Color::Rgb(100, 100, 130)),
            )));
        }
    }

    f.render_widget(
        Paragraph::new(detail_lines)
            .block(Block::default().borders(Borders::ALL).border_style(Style::default().fg(Color::Rgb(40, 80, 120)))
                .title(Span::styled(" Test Detail ", Style::default().fg(Color::Rgb(200, 160, 255)).bold())))
            .wrap(Wrap { trim: true }),
        detail_area,
    );
}

// ─── Debug / Console tab ─────────────────────────────────────────────────────
fn render_debug(f: &mut Frame, area: ratatui::layout::Rect, app: &AppState) {
    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::Rgb(40, 80, 120)))
        .title(Span::styled(
            " Console  (↑=older · ↓=newer · g=bottom) ",
            Style::default().fg(Color::Rgb(180, 255, 180)).bold(),
        ));
    let inner = block.inner(area);
    f.render_widget(block, area);

    let visible = inner.height as usize;

    let lines: Vec<String> = {
        let buf = log_buf().lock().unwrap_or_else(|e| e.into_inner());
        buf.iter().cloned().collect()
    };

    if lines.is_empty() {
        f.render_widget(
            Paragraph::new(Span::styled(
                "\n  No log output yet. Run a generation to see debug output here.",
                Style::default().fg(Color::DarkGray).italic(),
            )),
            inner,
        );
        return;
    }

    let total = lines.len();
    let skip = app.debug_scroll.min(total.saturating_sub(visible));
    let end = total.saturating_sub(skip);
    let start = end.saturating_sub(visible);

    let display: Vec<Line> = lines[start..end]
        .iter()
        .enumerate()
        .map(|(i, s)| {
            let color = if s.starts_with("[gen]") {
                Color::Rgb(100, 220, 180)
            } else if s.starts_with("error") || s.starts_with("✗") || s.contains("Error") {
                Color::Rgb(255, 100, 100)
            } else if s.starts_with("warn") || s.contains("WARN") {
                Color::Yellow
            } else if (start + i).is_multiple_of(2) {
                Color::Rgb(210, 220, 235)
            } else {
                Color::Rgb(170, 180, 200)
            };
            Line::from(Span::styled(s.clone(), Style::default().fg(color)))
        })
        .collect();

    let scroll_label = if skip == 0 {
        format!(" {} lines ", total)
    } else {
        format!(" {}/{} ↑{} ", end, total, skip)
    };
    let label_w = (scroll_label.len() + 2) as u16;

    let [log_area, info_area] =
        Layout::horizontal([Constraint::Min(0), Constraint::Length(label_w)]).areas(inner);

    f.render_widget(Paragraph::new(display), log_area);
    f.render_widget(
        Paragraph::new(Span::styled(scroll_label, Style::default().fg(Color::Rgb(80, 80, 100)))),
        info_area,
    );
}

// ─── Helpers ─────────────────────────────────────────────────────────────────
fn row_line(label: &'static str, value: String, val_color: Color) -> Line<'static> {
    Line::from(vec![
        Span::styled(label, Style::default().fg(Color::DarkGray)),
        Span::styled(value, Style::default().fg(val_color)),
    ])
}

fn scan_exhibits() -> Vec<ExhibitEntry> {
    let mut out = Vec::new();
    let dir = std::path::Path::new("exhibits");
    if !dir.exists() { return out; }
    let mut entries: Vec<_> = match std::fs::read_dir(dir) {
        Ok(e) => e.filter_map(|r| r.ok()).collect(),
        Err(_) => return out,
    };
    entries.sort_by_key(|e| e.file_name());

    for entry in entries {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("safetensors") { continue; }
        let name = path.file_stem().and_then(|s| s.to_str()).unwrap_or("?").to_string();
        let meta_path = path.with_extension("meta.json");
        let (splat_count, date, prompt) = if let Some(v) = meta_path.exists().then(|| {
            std::fs::read_to_string(&meta_path).ok()
                .and_then(|s| serde_json::from_str::<serde_json::Value>(&s).ok())
        }).flatten() {
            let n = v["splat_count"].as_u64().unwrap_or(0) as usize;
            let ts = v["timestamp"].as_u64().unwrap_or(0);
            let date = format!("t+{}d", ts / 86400);
            let p = v["source_prompt"].as_str().unwrap_or("").chars().take(60).collect();
            (n, date, p)
        } else {
            (0, "—".into(), String::new())
        };
        out.push(ExhibitEntry { name, splat_count, date, prompt });
    }
    out
}
