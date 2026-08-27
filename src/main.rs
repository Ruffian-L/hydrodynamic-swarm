//! SplatRAG v1 — Hydrodynamic Swarm
//!
//! Physics-steered generation over Llama 3.1 / Gemma 3 / Gemma 4 with shared-ocean multi-mind forces.
//! Type a prompt → physics steers generation → decoded text output.
//!
//! ## Licenses & attributions
//!
//! - Our code: MIT-0 (LICENSE)
//! - Candle loader code: Apache-2.0 OR MIT (NOT the same as model weights)
//! - Llama 3.1 weights: Llama 3.1 Community License — "Built with Llama" in README
//! - Gemma 3 weights: Gemma Terms of Use (NOT Apache; Gemma 4 is Apache)
//! - GGUF quants: bartowski, Unsloth (on top of Meta/Google terms)
//!
//! See NOTICE in the repo root.

mod algo_scale;
#[allow(dead_code, unused_imports, unused_variables)]
mod concourse;
mod control_tags;
mod dream;
mod endocrine;
pub mod frontend;
mod field;
mod gpu;
mod hud;
mod logger;
mod logit_physics;
mod memory;
mod niodoo;
mod ocean;
mod picks;
mod qsma;
mod quality;
mod remember_store;
mod repl_tui;
mod splat;
mod tct;
mod tda_monitor;
mod tui;
mod viz;
mod weather;
// mod viz_metal; // removed: XSS-vulnerable HTML viewer (security audit 2026-03-07)

// Model loading and forward-pass instrumentation now live in the library half of this
// crate so `jlens-gguf` can load the same weights through the same path. Re-exported at
// the crate root so sibling modules keep resolving `crate::dim_assert::…` unchanged.
pub(crate) use hydrodynamic_swarm::{config, dim_assert, gemma4, hooks, jacobian, loader};

use anyhow::Result;
use candle_core::{Device, Tensor};
use config::Config;
use dream::micro_dream;
use field::ContinuousField;
use loader::{find_existing_file, Model};
use logger::{SessionConfig, SessionLogger, SessionSummary, StepEntry};
use memory::{MemoryForceMode, MemoryPickConfig, SplatMemory};
use niodoo::{FieldWakeConfig, FieldWakeMode, NiodooEngine};
use ocean::{MindId, OceanConfig, SharedOcean};
use quality::{alpha_for, classify, score_token, QualityThresholds, SplatKind};
use rand::{rngs::StdRng, RngExt, SeedableRng};
use splat::Splat;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;
use tokenizers::Tokenizer;
use viz::VizCollector;

fn sha256_file(path: &Path) -> String {
    let cache_dir = Path::new("data/.sha256_cache");
    let _ = std::fs::create_dir_all(cache_dir);

    if let Ok(metadata) = std::fs::metadata(path) {
        if let Ok(canon) = path.canonicalize() {
            let file_name = canon.file_name().unwrap_or_default().to_string_lossy().into_owned();
            let mtime = metadata.modified().unwrap_or(std::time::SystemTime::UNIX_EPOCH)
                .duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_secs();
            let size = metadata.len();
            
            // Stable cache key instead of randomly seeded DefaultHasher
            let safe_name = file_name.replace('/', "_").replace('\\', "_");
            let cache_file = cache_dir.join(format!("{}_{}_{}.sha256", safe_name, size, mtime));
            
            if let Ok(cached) = std::fs::read_to_string(&cache_file) {
                return cached.trim().to_string();
            }
            
            let hash = std::process::Command::new("sha256sum")
                .arg(path)
                .output()
                .ok()
                .filter(|out| out.status.success())
                .and_then(|out| String::from_utf8(out.stdout).ok())
                .and_then(|line| line.split_whitespace().next().map(str::to_owned))
                .unwrap_or_else(|| "unavailable".into());
                
            if hash != "unavailable" {
                let _ = std::fs::write(cache_file, &hash);
            }
            return hash;
        }
    }

    std::process::Command::new("sha256sum")
        .arg(path)
        .output()
        .ok()
        .filter(|out| out.status.success())
        .and_then(|out| String::from_utf8(out.stdout).ok())
        .and_then(|line| line.split_whitespace().next().map(str::to_owned))
        .unwrap_or_else(|| "unavailable".into())
}

fn file_snapshot(label: &str, path: Option<&Path>) -> algo_scale::FileSnapshot {
    let Some(path) = path else {
        return algo_scale::FileSnapshot {
            label: label.into(),
            path: "unavailable".into(),
            exists: false,
            bytes: 0,
            sha256: "unavailable".into(),
        };
    };
    let metadata = std::fs::metadata(path).ok();
    algo_scale::FileSnapshot {
        label: label.into(),
        path: path.display().to_string(),
        exists: metadata.is_some(),
        bytes: metadata.as_ref().map(|m| m.len()).unwrap_or(0),
        sha256: if metadata.is_some() {
            sha256_file(path)
        } else {
            "absent".into()
        },
    }
}

fn live_seat_profile(
    cfg: &Config,
    engine: &NiodooEngine,
    logit_chain: &logit_physics::LogitChain,
) -> algo_scale::SeatProfile {
    let residual = engine.live_params();
    let residual_value = |name: &str, fallback: f32| {
        residual
            .iter()
            .find(|(n, _, _, _)| *n == name)
            .map(|(_, value, _, _)| *value)
            .unwrap_or(fallback)
    };
    let logits = logit_chain.params();
    let logit_value = |name: &str, fallback: f32| {
        logits
            .iter()
            .find(|(n, _, _, _)| *n == name)
            .map(|(_, value, _, _)| *value)
            .unwrap_or(fallback)
    };
    algo_scale::SeatProfile {
        residual_cap: residual_value("residual.cap", cfg.physics.force_cap),
        residual_field: residual_value("residual.field", cfg.physics.field_wake_scale),
        residual_field_max: residual_value("residual.field_max", cfg.physics.field_wake_max),
        residual_splat: residual_value("residual.splat", cfg.physics.splat_force_scale),
        residual_splat_max: residual_value("residual.splat_max", cfg.physics.splat_force_max),
        residual_goal: residual_value("residual.goal", cfg.physics.goal_force_scale),
        residual_goal_max: residual_value("residual.goal_max", cfg.physics.goal_force_max),
        force_ramp_tokens: cfg.physics.force_ramp_tokens,
        force_ramp_start: cfg.physics.force_ramp_start,
        temperature: cfg.generation.temperature as f32,
        logit_field_alpha: logit_value("field.alpha", cfg.logit_physics.field_alpha),
        logit_splat_scale: logit_value("splat.scale", cfg.logit_physics.splat_scale),
        governor_brake: logit_value("gov.brake", cfg.logit_physics.governor_brake),
        governor_viscosity_gain: logit_value(
            "gov.visc_gain",
            cfg.logit_physics.governor_viscosity_gain,
        ),
    }
}

fn write_scaler_receipt(receipt: &algo_scale::ScalerReceipt) -> Result<PathBuf> {
    let path = std::env::var("HYDRO_SCALER_RECEIPT")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            PathBuf::from("logs/scaler_receipts").join(format!("{}.json", receipt.receipt_id))
        });
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&path)?;
    serde_json::to_writer_pretty(&mut file, receipt)?;
    writeln!(file)?;
    file.sync_all()?;
    Ok(path)
}

fn generation_eos_token_ids(variant: &str, configured: &[u32]) -> Vec<u32> {
    match variant {
        // 1=<eos> 106=<turn|> 50=<|tool_response>
        // Do NOT put 100/101 (<|channel>/<channel|>) here. Thought channel is a
        // live token stream (P12 110203 died when settle_channel treated reopen
        // as EOS). Phrase/cycle settle still catches soup.
        "gemma4" => vec![1, 106, 50],
        // Gemma 3 IT: <eos> and <end_of_turn>.
        "gemma3" => vec![1, 106],
        "qwen35" => vec![248046, 248044],
        _ => configured.to_vec(),
    }
}

/// True while generated `pieces` are inside an unclosed Gemma 4 thought
/// channel (`<|channel>…` with no later `<channel|>`). Prefill already closed
/// the empty thinking-off prefix; a reopen in the mouth is live trajectory.
fn gemma4_in_open_thought(pieces: &str) -> bool {
    let open_g = pieces.rfind("<|channel>");
    let close_g = pieces.rfind("<channel|>");
    let gemma_open = match (open_g, close_g) {
        (Some(o), Some(c)) => o > c,
        (Some(_), None) => true,
        _ => false,
    };
    let open_q = pieces.rfind("<think>");
    let close_q = pieces.rfind("</think>");
    let qwen_open = match (open_q, close_q) {
        (Some(o), Some(c)) => o > c,
        (Some(_), None) => true,
        _ => false,
    };
    gemma_open || qwen_open
}

/// Channel specials 100=`<|channel>` / 101=`<channel|>` are a live token
/// stream, not a settle stop. P12 `20260822_110203` died here at step 3.
fn gemma4_should_settle_channel(_pieces: &str, _next_id: u32) -> bool {
    false
}

/// `<lock>` commits the answer stream. Inside thought it is a planning hand
/// and must not kill the rest of the turn.
fn gemma4_lock_stops_turn(pieces: &str) -> bool {
    !gemma4_in_open_thought(pieces)
}

/// Clean model text for *next* prefill history only. Transcript keeps full raw.
/// Thought/channel markers stay: the trace is the research object. Hyphen-thrash
/// tails still drop so next prefill isn't poisoned.
fn gemma4_history_clean(raw: &str) -> String {
    let mut s = raw.to_string();
    // Drop trailing hyphen-thrash lines so next prefill isn't poisoned.
    let lines: Vec<&str> = s.lines().collect();
    let mut keep = lines.len();
    while keep > 0 {
        let t = lines[keep - 1].trim();
        if t == "The-" || t == "the-" || t.ends_with('-') && t.len() <= 6 {
            keep -= 1;
        } else {
            break;
        }
    }
    s = lines[..keep].join("\n");
    // Keep newlines so Internal monitor lines and tags stay their own
    // turns-in-the-mouth for the next prefill KV. Do not smash to one line.
    s.trim().to_string()
}

fn tda_monitor_injection_ready(pieces: &str, pending: bool) -> bool {
    pending && !control_tags::incomplete_control_hand(pieces)
}

/// Detect "The-\nThe-\n…" or identical short line repeated (forced-length doom).
fn gemma4_hyphen_thrash(pieces: &str) -> bool {
    let lines: Vec<&str> = pieces
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .collect();
    if lines.len() < 4 {
        return false;
    }
    let last4 = &lines[lines.len() - 4..];
    // Same short fragment 4× (legacy short stub)
    if last4.iter().all(|l| *l == last4[0]) && last4[0].len() <= 12 {
        return true;
    }
    // Hyphen-stub lines (The- / the- / X-)
    let hyphen_stubs = last4
        .iter()
        .filter(|l| l.ends_with('-') && l.len() <= 8)
        .count();
    hyphen_stubs >= 3
}

/// Trailing short-cycle lock (`esesese`, `TheTheThe`): a 1–3 char unit
/// repeating ≥8 times at the end. 256-token unmatched residual soup.
fn trailing_short_cycle_lock(pieces: &str) -> bool {
    let chars: Vec<char> = pieces.chars().collect();
    if chars.len() < 16 {
        return false;
    }
    let start = chars.len().saturating_sub(48);
    let tail = &chars[start..];
    for ulen in 1..=3 {
        if tail.len() < ulen * 8 {
            continue;
        }
        let unit = &tail[tail.len() - ulen..];
        let mut n = 0usize;
        let mut i = tail.len();
        while i >= ulen && &tail[i - ulen..i] == unit {
            n += 1;
            i -= ulen;
        }
        if n >= 8 {
            return true;
        }
    }
    false
}

/// Trailing run of identical non-empty lines (trim). Returns (count, last_line_len).
/// Used for self-reg revise labels (early) and settle clamp (later).
fn trailing_identical_line_run(pieces: &str) -> (usize, usize) {
    let lines: Vec<&str> = pieces
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .collect();
    if lines.is_empty() {
        return (0, 0);
    }
    let last = lines[lines.len() - 1];
    let mut n = 0usize;
    for l in lines.iter().rev() {
        if *l == last {
            n += 1;
        } else {
            break;
        }
    }
    (n, last.len())
}

/// True when trailing identical lines hit `need` and line is long enough.
fn line_repeat_at_least(pieces: &str, need: usize, min_chars: usize) -> bool {
    if need == 0 {
        return false;
    }
    let (n, line_len) = trailing_identical_line_run(pieces);
    n >= need && line_len >= min_chars
}

/// Count self-reg Wait / try-again blocks (Spell-cat class multi-line revise loops).
fn wait_loop_count(pieces: &str) -> usize {
    let lower = pieces.to_ascii_lowercase();
    let try_again = lower.matches("try again").count();
    let wait = lower
        .matches("wait,")
        .count()
        .max(lower.matches("wait ").count());
    // Prefer try_again as the block marker; fall back to Wait, if denser.
    try_again.max(wait)
}

/// Same-line phrase thrash: `need` consecutive copies of a unit of length [min_unit, max_unit].
/// Catches "No, the question is 17 x 17? "×N when there are no newlines.
/// Drops a short incomplete tail so a truncated last copy still matches.
fn phrase_repeat_at_least(pieces: &str, need: usize, min_unit: usize, max_unit: usize) -> bool {
    if need < 2 || min_unit == 0 {
        return false;
    }
    // Only strip trailing newlines — keep trailing spaces (phrase units often end in ' ').
    let s = pieces.trim_end_matches(['\n', '\r']);
    // Last non-empty line without trimming its trailing spaces (alignment).
    let focus = s.lines().rev().find(|l| !l.trim().is_empty()).unwrap_or(s);
    let n = focus.len();
    if n < min_unit.saturating_mul(need) {
        return false;
    }
    let max_u = max_unit.min(n / need);
    for unit in min_unit..=max_u {
        // Incomplete tail drop (up to one unit).
        for drop in 0..unit {
            if drop > n {
                break;
            }
            let end = n - drop;
            if end < unit * need {
                continue;
            }
            if !focus.is_char_boundary(end) || !focus.is_char_boundary(end - unit) {
                continue;
            }
            let pat = &focus[end - unit..end];
            if pat.chars().all(|c| c.is_whitespace()) {
                continue;
            }
            if !pat.chars().any(|c| c.is_alphanumeric()) {
                continue;
            }
            let mut matched = 1usize;
            let mut e = end - unit;
            while matched < need && e >= unit {
                let s0 = e - unit;
                if !focus.is_char_boundary(s0) || !focus.is_char_boundary(e) {
                    break;
                }
                if &focus[s0..e] != pat {
                    break;
                }
                matched += 1;
                e = s0;
            }
            if matched >= need {
                return true;
            }
        }
    }
    false
}

/// Wrap a raw user prompt in Gemma 3 IT chat turns when needed.
///
/// Wrap free text into the model’s IT turn format. Raw prompts that already
/// contain turn markers are left unchanged.
///
/// Gemma 3 IT: `<start_of_turn>user` / `model` (historical hydro path).
/// Gemma 4 IT: match `data/google/gemma4_assets/chat_template.jinja` with
/// `enable_thinking=false`, no tools, `add_generation_prompt=true`:
///   `<|turn>user\n…\n<turn|>\n` then `<|turn>model\n` + empty thought
///   channel `<|channel>thought\n<channel|>` so the model starts free content.
///
/// Do **not** inject “Answer in one short paragraph / correct answer” framing
/// for Gemma 4 — that primed exam/list completions even on “Say hi”.
fn format_prompt_for_model(raw: &str, variant: &str) -> String {
    // Single-shot must match `--chat` **first turn** packing.
    //
    // `run_simple_chat` always sets `tags_on=true`, which prepends the god-tier
    // control-channel **system** turn (`gemma4_system_prefix`). Not a fake user
    // turn. Keep gemma3/llama without that prefix (they never used it).
    let control_tags = variant == "gemma4";
    format_multiturn_prompt_ex(&[(true, raw.trim().to_string())], variant, control_tags)
}

/// Build IT prompt from (is_user, text) turns. Last turn should be user; we append
/// the model generation prefix. No extra “helpful” / exam framing.
///
/// When `control_tags` is true on Gemma 4, a **system** turn carries the
/// available-tags table. Engine scans raw generated `pieces`. Tags stay in history.
fn format_multiturn_prompt(turns: &[(bool, String)], variant: &str) -> String {
    format_multiturn_prompt_ex(turns, variant, false)
}

/// Prefill packing status for Gemma 4. PRESENT = the tag table is in the packed
/// prompt. Stale keys: “DO NOT emit your tags”, “exactly one tag”.
fn gemma4_control_channel_status(prompt: &str) -> &'static str {
    if control_tags::packed_prompt_has_emit_channel(prompt) {
        "PRESENT"
    } else if control_tags::packed_prompt_has_legacy_panel(prompt) {
        "LEGACY"
    } else {
        "ABSENT"
    }
}

fn print_gemma4_control_channel_packing(prompt: &str) {
    match gemma4_control_channel_status(prompt) {
        "PRESENT" => eprintln!(
            "    Prefill packing: god-tier system turn PRESENT (available tags table; matches chat tags_on=true)"
        ),
        "LEGACY" => eprintln!("    Prefill packing: LEGACY sticky user-turn panel still present"),
        _ => eprintln!("    Prefill packing: control-channel system turn ABSENT"),
    }
}

/// What this turn's KV actually contains from prior mouth (monitor + hands).
fn official_pack_layout() -> bool {
    matches!(
        std::env::var("HYDRO_OFFICIAL_LAYOUT").as_deref(),
        Ok("1") | Ok("true") | Ok("yes") | Ok("on")
    )
}

fn official_prompt_label(turn_idx: usize) -> String {
    if turn_idx <= 1 {
        "Prompt>".to_string()
    } else {
        format!("Prompt{}>", turn_idx - 1)
    }
}

fn official_expected_for_turn(turn_idx: usize) -> Option<String> {
    let path = std::env::var("HYDRO_EXPECTED_FILE").ok()?;
    let raw = std::fs::read_to_string(path).ok()?;
    let key = if turn_idx <= 1 {
        "Opening>".to_string()
    } else {
        format!("P{}>", turn_idx - 1)
    };
    for line in raw.lines() {
        let line = line.trim();
        if let Some(rest) = line.strip_prefix(&key) {
            let rest = rest.trim();
            if !rest.is_empty() {
                return Some(rest.to_string());
            }
        }
    }
    None
}

fn print_official_turn_open(turn_idx: usize, user_line: &str) {
    let label = official_prompt_label(turn_idx);
    println!("{}", "=".repeat(72));
    println!("{label}");
    println!("{user_line}");
    println!();
    println!("Expected answer — EVALUATOR ONLY>");
    match official_expected_for_turn(turn_idx) {
        Some(exp) => println!("{exp}"),
        None => println!("(no expected block on disk)"),
    }
    println!();
    println!("Model>");
    let _ = std::io::stdout().flush();
}

fn print_prefill_see(prompt: &str, turn_idx: usize) {
    let hits = control_tags::scan(prompt);
    let mon = prompt.matches("[Internal monitor:").count();
    let tags = hits
        .iter()
        .map(|t| t.as_str())
        .collect::<Vec<_>>()
        .join(",");
    eprintln!(
        "    [CHAT PREFILL see] turn={turn_idx} monitor_lines={mon} tags={}",
        if tags.is_empty() {
            "none"
        } else {
            tags.as_str()
        }
    );
}

fn format_multiturn_prompt_ex(
    turns: &[(bool, String)],
    variant: &str,
    control_tags: bool,
) -> String {
    match variant {
        "gemma4" => {
            let mut s = String::new();
            // `<bos>` must be emitted explicitly. The Gemma 4 tokenizer's post-processor is
            // `single = [A]` — it adds **nothing** — whereas Gemma 3's is
            // `[<bos>, A, <eos>]`. So `encode(text, true)` yields a sequence starting at
            // `<|turn>` (105) instead of `<bos>` (2), and Gemma leans hard on BOS as an
            // attention sink: without it the model answers as though the user turn were
            // empty ("If you are looking for a specific type of information…" to
            // "What is the currency of Italy?"). With it, "The currency of Italy is the
            // **Euro (€)**."
            //
            // Gemma 3 must NOT get this — its tokenizer already inserts BOS and a second
            // one would be worse than none.
            s.push_str("<bos>");
            if control_tags {
                s.push_str(&control_tags::gemma4_system_prefix());
            }
            for (is_user, text) in turns {
                if *is_user {
                    s.push_str("<|turn>user\n");
                    s.push_str(text.trim());
                    // No newline before `<turn|>`: the canonical template emits
                    // `{{- captured_content -}}` (trailing whitespace stripped) and then
                    // `{{- '<turn|>\n' -}}`. This is the same stray-token bug the gemma3
                    // branch below documents having fixed; it was left here.
                    s.push_str("<turn|>\n");
                } else {
                    s.push_str("<|turn>model\n");
                    s.push_str(text.trim());
                    s.push_str("<turn|>\n");
                }
            }
            // Generation prompt (thinking off): open model + empty thought.
            s.push_str("<|turn>model\n<|channel>thought\n<channel|>");
            s
        }
        "gemma3" => {
            // Verbatim from the chat template embedded in the Gemma 3 GGUF:
            //   '<start_of_turn>' + role + '\n' + content|trim + '<end_of_turn>\n'
            // The trimmed content is followed *immediately* by <end_of_turn> —
            // there is no newline between them. We used to insert one, adding a
            // stray token at every turn boundary; a one-shot prompt carries a
            // single stray, but a multi-turn conversation accumulates one per
            // turn and drifts off the format the model was trained on.
            let mut s = String::new();
            for (is_user, text) in turns {
                let role = if *is_user { "user" } else { "model" };
                s.push_str("<start_of_turn>");
                s.push_str(role);
                s.push('\n');
                s.push_str(text.trim());
                s.push_str("<end_of_turn>\n");
            }
            s.push_str("<start_of_turn>model\n");
            s
        }
        "qwen35" => {
            let mut s = String::new();
            if control_tags {
                s.push_str("<|im_start|>system\nYou are an autonomous agent equipped with a Choice-Driven KV Cache. You have access to a sandbox hook. Inside your <think>...</think> block, you can emit `<spike>` to fork the timeline, preview 10 tokens of your choice, and have it stripped before you make your final choice. Once ready to answer, emit `<lock>` to commit. DO NOT be confused by <spike>, it is a mechanical hook that grants you a safe physics fork to test futures.<|im_end|>\n");
            }
            for (is_user, text) in turns {
                let role = if *is_user { "user" } else { "assistant" };
                s.push_str(format!("<|im_start|>{}\n", role).as_str());
                s.push_str(text.trim());
                s.push_str("<|im_end|>\n");
            }
            s.push_str("<|im_start|>assistant\n");
            s
        }
        _ => {
            turns
                .iter()
                .map(|(u, t)| {
                    if *u {
                        format!("User: {}\n", t.trim())
                    } else {
                        format!("Assistant: {}\n", t.trim())
                    }
                })
                .collect::<String>()
                + "Assistant: "
        }
    }
}

/// Encode a prompt for a forward pass, dropping any trailing EOS that the
/// tokenizer's post-processor appended.
///
/// Gemma's `tokenizer.json` uses TemplateProcessing `<bos> A <eos>`, so
/// `encode(text, true)` terminates every prompt with `<eos>`: the model is told
/// the sequence already ended and is then asked to continue it. The result is
/// degraded, loosely-on-topic text — on every turn, independent of physics,
/// which is why force_off runs garbled too.
///
/// Llama / Qwen tokenizers add only BOS, so they were never affected. That is
/// why those families read clean through this same stack while both Gemmas did
/// not. The leading BOS is still left to the post-processor so each family
/// keeps its own convention; only the trailing terminator is removed.
fn encode_prompt_no_trailing_eos(
    tokenizer: &tokenizers::Tokenizer,
    text: &str,
    eos_token_ids: &[u32],
) -> Result<Vec<u32>> {
    let encoded = tokenizer
        .encode(text, true)
        .map_err(|e| anyhow::anyhow!("encode: {e}"))?;
    let mut ids: Vec<u32> = encoded.get_ids().to_vec();
    // Keep at least one token: an all-EOS encode is degenerate but must not
    // become an empty prefill.
    while ids.len() > 1 && ids.last().is_some_and(|id| eos_token_ids.contains(id)) {
        ids.pop();
    }
    Ok(ids)
}

/// Top-k entropy (nats), top1−top2 margin, p_top1, top1 token id — for collapse probes.
fn collapse_logit_stats(logits: &[f32], topk: usize) -> (f32, f32, f32, u32) {
    if logits.is_empty() {
        return (0.0, 0.0, 0.0, 0);
    }
    let k = topk.min(logits.len()).max(2);
    let mut idx: Vec<usize> = (0..logits.len()).collect();
    idx.select_nth_unstable_by(k - 1, |&a, &b| {
        logits[b]
            .partial_cmp(&logits[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    idx.truncate(k);
    idx.sort_unstable_by(|&a, &b| {
        logits[b]
            .partial_cmp(&logits[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let max_l = logits[idx[0]];
    let mut exps: Vec<f32> = idx.iter().map(|&i| (logits[i] - max_l).exp()).collect();
    let z: f32 = exps.iter().sum::<f32>().max(1e-12);
    for e in &mut exps {
        *e /= z;
    }
    let entropy: f32 = exps
        .iter()
        .map(|&p| {
            let p = p.max(1e-12);
            -p * p.ln()
        })
        .sum();
    let p1 = exps[0];
    let p2 = if exps.len() > 1 { exps[1] } else { 0.0 };
    (entropy, (p1 - p2).clamp(0.0, 1.0), p1, idx[0] as u32)
}

/// Sample next token id from logits (after any penalties). Supports greedy, top-k, top-p.
#[allow(dead_code)] // kept for T/rep A/B; decode policy is QSMA argmax
fn sample_from_logits(logits: &[f32], temperature: f64, top_k: usize, top_p: f32) -> u32 {
    if logits.is_empty() {
        return 0;
    }
    if temperature < 1e-5 {
        return logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i as u32)
            .unwrap_or(0);
    }

    // Rank by logit (pre-softmax) for nucleus / top-k
    let mut idxs: Vec<usize> = (0..logits.len()).collect();
    idxs.sort_by(|&a, &b| {
        logits[b]
            .partial_cmp(&logits[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    if top_k > 0 && top_k < idxs.len() {
        idxs.truncate(top_k);
    }

    let t = temperature.max(1e-5) as f32;
    let mut scored: Vec<(usize, f32)> = idxs.iter().map(|&i| (i, (logits[i] / t).exp())).collect();
    let sum: f32 = scored.iter().map(|(_, w)| *w).sum::<f32>().max(1e-12);
    for s in scored.iter_mut() {
        s.1 /= sum;
    }

    // top-p on the remaining mass
    if top_p > 0.0 && top_p < 1.0 {
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let mut cum = 0.0f32;
        let mut cut = scored.len();
        for (i, (_, p)) in scored.iter().enumerate() {
            cum += *p;
            if cum >= top_p {
                cut = i + 1;
                break;
            }
        }
        scored.truncate(cut.max(1));
        let renorm: f32 = scored.iter().map(|(_, p)| *p).sum::<f32>().max(1e-12);
        for s in scored.iter_mut() {
            s.1 /= renorm;
        }
    }

    let roll: f32 = if let Ok(s) = std::env::var("HYDRO_SAMPLE_SEED") {
        let seed: u64 = s.parse().unwrap_or(1);
        StdRng::seed_from_u64(seed).random()
    } else {
        rand::rng().random()
    };
    let mut cumsum = 0.0f32;
    for (i, p) in &scored {
        cumsum += *p;
        if roll < cumsum {
            return *i as u32;
        }
    }
    scored.last().map(|(i, _)| *i as u32).unwrap_or(0)
}

/// Parse a `name=value` override (`--set` / REPL `/set`). Also accepts `name value`.
fn parse_set_arg(s: &str) -> Option<(String, f32)> {
    let (name, value) = s.split_once(['=', ':']).or_else(|| s.split_once(' '))?;
    let value = value.trim().parse::<f32>().ok()?;
    if !value.is_finite() {
        return None;
    }
    Some((name.trim().to_string(), value))
}

fn render_live_controls(
    engine: &NiodooEngine,
    logit_chain: &logit_physics::LogitChain,
    hook_controls: &hooks::HookControls,
    temperature: f64,
    rep_penalty: f32,
) -> String {
    let mut out = engine.render_live_sliders();
    out.push_str(&logit_chain.render_sliders());
    out.push_str(&hook_controls.render_sliders());
    out.push_str(&format!(
        "  sampling\n    {:<16} {:>8.4}\n    {:<16} {:>8.4}\n",
        "sample.temp", temperature, "sample.rep", rep_penalty
    ));
    out
}

fn collect_live_sliders(
    engine: &NiodooEngine,
    logit_chain: &logit_physics::LogitChain,
    hook_controls: &hooks::HookControls,
    temperature: f64,
    rep_penalty: f32,
) -> Vec<tui::Slider> {
    let mut sliders = Vec::new();
    for (name, value, min, max) in engine.live_params() {
        sliders.push(tui::Slider::live(name, value, min, max));
    }
    for (name, value, min, max) in logit_chain.params() {
        sliders.push(tui::Slider::live(name, value, min, max));
    }
    for (name, value, min, max) in hook_controls.params() {
        sliders.push(tui::Slider::live(name, value, min, max));
    }
    sliders.push(tui::Slider::live(
        "sample.temp",
        temperature as f32,
        0.0,
        2.0,
    ));
    sliders.push(tui::Slider::live("sample.rep", rep_penalty, 1.0, 2.5));
    sliders
}

fn set_live_control(
    name: &str,
    value: f32,
    engine: &mut NiodooEngine,
    logit_chain: &mut logit_physics::LogitChain,
    hook_controls: &mut hooks::HookControls,
    temperature: &mut f64,
    rep_penalty: &mut f32,
) -> bool {
    if engine.set_live_param(name, value)
        || logit_chain.set_param(name, value)
        || hook_controls.set_param(name, value)
    {
        return true;
    }
    match name {
        "sample.temp" => *temperature = value.clamp(0.0, 2.0) as f64,
        "sample.rep" => *rep_penalty = value.clamp(1.0, 2.5),
        _ => return false,
    }
    true
}

/// John A0 live integrity: empty-key geometry (static) + finite logits after prefill.
/// Lengths 1023 / 1024 / 1025 / 1039 match the HF review checklist.
fn run_a0_swa_check(model: &mut Model, device: &Device, is_gemma4: bool) -> Result<()> {
    const W: usize = 1024;
    const LENGTHS: [usize; 4] = [1023, 1024, 1025, 1039];

    println!("\n=== John A0 SWA check ===");
    println!(
        "    variant={}  window_ref={}  (static geometry uses W={}; Gemma4 card=1024)",
        model.variant_name(),
        if is_gemma4 {
            "1024 (gemma4)"
        } else {
            "model-dependent"
        },
        W
    );

    // --- A0a static: legacy vs fixed geometry (no GPU matmul) ---
    println!("\n--- A0a static valid-key rows (window={W}) ---");
    let mut static_ok = true;
    for &len in &LENGTHS {
        let legacy_empty =
            gemma4::empty_valid_key_rows(&gemma4::legacy_trim_prefill_valid_keys(len, W));
        let fixed_empty = gemma4::empty_valid_key_rows(&gemma4::fixed_prefill_valid_keys(len, W));
        let expected_legacy = match len {
            1023 | 1024 => 0,
            1025 => 1,
            1039 => 15,
            _ => legacy_empty,
        };
        let leg_ok = legacy_empty == expected_legacy;
        let fix_ok = fixed_empty == 0;
        static_ok &= leg_ok && fix_ok;
        println!(
            "    len={len:>4}  legacy_empty={legacy_empty} (expect {expected_legacy})  fixed_empty={fixed_empty} (expect 0)  {}",
            if leg_ok && fix_ok { "PASS" } else { "FAIL" }
        );
    }

    // --- A0b live: prefill synthetic ids → logits/hidden finite ---
    println!("\n--- A0b live finite after prefill (first-token path) ---");
    // Safe filler id: 1 is usually a real piece in BPE; avoid 0 if it's pad.
    let fill_id: u32 = 2;
    let mut live_ok = true;
    for &len in &LENGTHS {
        let ids: Vec<u32> = vec![fill_id; len];
        let tokens = Tensor::new(ids.as_slice(), device)?.unsqueeze(0)?;
        let t0 = std::time::Instant::now();
        let result = model.forward_with_hidden(&tokens, 0);
        match result {
            Ok((logits, hidden)) => {
                let logits_ok = tensor_all_finite(&logits)?;
                let hidden_ok = tensor_all_finite(&hidden)?;
                let pass = logits_ok && hidden_ok;
                live_ok &= pass;
                let ln = logits.dims().to_vec();
                let hn = hidden.dims().to_vec();
                println!(
                    "    len={len:>4}  logits_finite={logits_ok}  hidden_finite={hidden_ok}  shape_L={ln:?} shape_H={hn:?}  {:.1?}  {}",
                    t0.elapsed(),
                    if pass { "PASS" } else { "FAIL" }
                );
            }
            Err(e) => {
                live_ok = false;
                println!("    len={len:>4}  ERROR: {e}  FAIL");
            }
        }
    }

    println!("\n--- A0 summary ---");
    println!(
        "    static={}  live_finite={}  overall={}",
        if static_ok { "PASS" } else { "FAIL" },
        if live_ok { "PASS" } else { "FAIL" },
        if static_ok && live_ok { "PASS" } else { "FAIL" }
    );
    if !is_gemma4 {
        println!(
            "    note: ran on {}; SWA window may not be 1024. Re-run with Gemma 4 GGUF for card-faithful A0.",
            model.variant_name()
        );
    }
    if !(static_ok && live_ok) {
        anyhow::bail!("A0 SWA check failed");
    }
    println!("    A0 SWA check complete.\n");
    Ok(())
}

fn tensor_all_finite(t: &Tensor) -> Result<bool> {
    // Flatten to f32 host for is_finite; large logits rows are OK at these lengths.
    let flat = t.flatten_all()?.to_dtype(candle_core::DType::F32)?;
    let v = flat.to_vec1::<f32>()?;
    Ok(v.iter().all(|x| x.is_finite()))
}

fn forward_decode_with_hook(
    model: &mut Model,
    tokens: &Tensor,
    index_pos: usize,
    direction: &Tensor,
    step: usize,
    hook_controls: &hooks::HookControls,
    hook_trace: &mut Option<hooks::HookTrace>,
) -> candle_core::Result<(Tensor, Tensor, hooks::HookReport)> {
    if !hook_controls.enabled || hook_controls.norm_fraction <= 0.0 {
        let (logits, hidden) = model.forward_with_hidden(tokens, index_pos)?;
        return Ok((logits, hidden, hooks::HookReport::default()));
    }
    let variant = model.variant_name();
    let mut hook =
        hooks::NiodooLayerHook::new(hook_controls, direction, step, variant, hook_trace.as_mut())?;
    let (logits, hidden) = model.forward_with_hidden_hooked(tokens, index_pos, Some(&mut hook))?;
    let report = hook.finish()?;
    Ok((logits, hidden, report))
}

/// Full-screen REPL: chat, live scalars, and live sliders in one view (`--tui`).
///
/// Same generation path as `--chat` — only the presentation differs.
#[allow(clippy::too_many_arguments)]
fn run_tui_chat(
    model: &mut Model,
    tokenizer: &tokenizers::Tokenizer,
    device: &Device,
    cfg: &Config,
    max_tokens: usize,
    engine: &mut NiodooEngine,
    logit_chain: &mut logit_physics::LogitChain,
    hook_controls: &mut hooks::HookControls,
    hook_trace: &mut Option<hooks::HookTrace>,
    algo: Option<hud::AlgoView>,
    scaler_receipt: &algo_scale::ScalerReceipt,
) -> Result<()> {
    use std::io::Write;
    let variant = model.variant_name();
    let eos_token_ids = generation_eos_token_ids(variant, &cfg.generation.eos_token_ids);
    let mut operator_temperature = cfg.generation.temperature;
    let mut operator_rep_penalty = cfg.generation.rep_penalty;
    let tags_on = true;

    let chat_dir = std::path::Path::new("private/chats");
    let _ = std::fs::create_dir_all(chat_dir);
    let transcript_path = chat_dir.join(format!("{}_{variant}_tui.txt", chrono_like_stamp()));
    let mut transcript = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&transcript_path)
        .ok();

    let sliders = collect_live_sliders(
        engine,
        logit_chain,
        hook_controls,
        operator_temperature,
        operator_rep_penalty,
    );
    // Nothing may print to stdout while the alternate screen is up.
    hud::set_quiet(true);
    let mut app = repl_tui::App::new(algo, sliders)?;
    app.push_line(&format!("=== {variant} · physics-steered chat ==="));
    app.push_line(&format!(
        "transcript: {} · Tab for sliders · Esc cuts a turn short · ^C quits",
        transcript_path.display()
    ));
    app.push_line("");

    let mut history: Vec<(bool, String)> = Vec::new();
    let result = (|| -> Result<()> {
        loop {
            let Some(line) = app.read_prompt()? else {
                break;
            };
            // Slider moves made at the prompt come back as a /set line.
            if let Some(rest) = line.strip_prefix("/set ") {
                if let Some((name, value)) = parse_set_arg(rest.trim()) {
                    set_live_control(
                        &name,
                        value,
                        engine,
                        logit_chain,
                        hook_controls,
                        &mut operator_temperature,
                        &mut operator_rep_penalty,
                    );
                }
                continue;
            }
            if line.eq_ignore_ascii_case("quit") || line.eq_ignore_ascii_case("exit") {
                break;
            }
            if line.eq_ignore_ascii_case("reset") || line.eq_ignore_ascii_case("clear") {
                history.clear();
                app.push_line("(history cleared)");
                continue;
            }

            if let Some(ref mut f) = transcript {
                let _ = writeln!(f, "you> {line}");
            }
            app.push_line(&format!("you> {line}"));
            app.push(&format!("{variant}> "));
            history.push((true, line));

            let prompt = format_multiturn_prompt_ex(&history, variant, tags_on);
            let pieces = generate_turn(
                model,
                tokenizer,
                device,
                cfg,
                max_tokens,
                engine,
                logit_chain,
                hook_controls,
                hook_trace,
                &prompt,
                &eos_token_ids,
                tags_on,
                &mut operator_temperature,
                &mut operator_rep_penalty,
                scaler_receipt,
                &mut |piece, frame| {
                    app.push(piece);
                    app.set_frame(frame);
                    let edits = app.poll_edits().unwrap_or_default();
                    let _ = app.draw();
                    edits
                },
            )?;
            app.push("\n\n");

            // The engine may have moved during the turn; re-read the knobs.
            app.sync_sliders(collect_live_sliders(
                engine,
                logit_chain,
                hook_controls,
                operator_temperature,
                operator_rep_penalty,
            ));

            let raw_reply = pieces.trim().to_string();
            if let Some(ref mut f) = transcript {
                let _ = writeln!(f, "{variant}> {raw_reply}");
                let _ = f.flush();
            }
            // Keep the hand in next-prefill history (Niodoo strip is identity).
            // Masking meant she could not attend to or reaffirm her own tag.
            let mut reply = raw_reply.clone();
            if variant == "gemma4" {
                reply = gemma4_history_clean(&reply);
            }
            if !reply.is_empty() {
                history.push((false, reply));
            }
        }
        Ok(())
    })();

    drop(app); // restores the terminal before anything else prints
    hud::set_quiet(false);
    result?;
    eprintln!(
        "Chat ended ({} messages). Saved private: {}",
        history.len(),
        transcript_path.display()
    );
    Ok(())
}

/// One assistant turn: prefill, then the physics-steered decode loop.
///
/// Shared by plain `--chat` and the full-screen `--tui` REPL. The caller decides
/// how each token is displayed and hands back any live-control edits the
/// operator made while it was streaming; those are applied here between tokens,
/// so a slider nudge shows up in the very next step's scalars.
#[allow(clippy::too_many_arguments)]
fn generate_turn(
    model: &mut Model,
    tokenizer: &tokenizers::Tokenizer,
    device: &Device,
    cfg: &Config,
    max_tokens: usize,
    engine: &mut NiodooEngine,
    logit_chain: &mut logit_physics::LogitChain,
    hook_controls: &mut hooks::HookControls,
    hook_trace: &mut Option<hooks::HookTrace>,
    prompt: &str,
    eos_token_ids: &[u32],
    tags_on: bool,
    operator_temperature: &mut f64,
    operator_rep_penalty: &mut f32,
    scaler_receipt: &algo_scale::ScalerReceipt,
    on_token: &mut dyn FnMut(&str, hud::HudFrame) -> repl_tui::Edits,
) -> Result<String> {
    // Optional mid-convo collapse probe: COLLAPSE_PROBE=1 or path.
    // Logs residual_norm / entropy / margin / top1 before each sample so we can
    // see telemetry diverge *before* visible garbage.
    generate_turn_ex(
        model,
        tokenizer,
        device,
        cfg,
        max_tokens,
        engine,
        logit_chain,
        hook_controls,
        hook_trace,
        prompt,
        prompt,
        eos_token_ids,
        tags_on,
        operator_temperature,
        operator_rep_penalty,
        /*turn_idx*/ 0,
        /*prev_assistant_len*/ 0,
        /*mint_wills*/ false,
        scaler_receipt,
        on_token,
    )
}

/// Synth string `generate_turn_ex` scans when `HYDRO_INJECT_TAG` is set.
fn hydro_inject_synth(raw: &str) -> Option<String> {
    let raw = raw.trim().to_ascii_lowercase();
    if raw.is_empty() || raw == "none" {
        return None;
    }
    Some(match raw.as_str() {
        "remember" => "<remember>k=v</remember>".to_string(),
        "lock" => "<lock>k=v</lock>".to_string(),
        other => format!("<{other}>"),
    })
}

/// Take `HYDRO_INJECT_TAG` once so a 9-turn seat does not re-spike every turn.
fn take_hydro_inject_tag() -> Option<String> {
    let raw = std::env::var("HYDRO_INJECT_TAG").ok()?;
    let raw = raw.trim().to_ascii_lowercase();
    if raw.is_empty() || raw == "none" {
        return None;
    }
    std::env::remove_var("HYDRO_INJECT_TAG");
    Some(raw)
}

/// Shipped inject path: synth → `scan_hits` → `fire_tag`. Same helper `generate_turn_ex` uses.
fn apply_hydro_inject(engine: &mut NiodooEngine, raw: &str) -> Vec<control_tags::TagHit> {
    let Some(synth) = hydro_inject_synth(raw) else {
        return Vec::new();
    };
    let found = control_tags::scan_hits(&synth);
    for hit in &found {
        let _stop = engine.fire_tag(hit);
    }
    found
}

/// Chat-path will deposit — same `SplatMemory::add_splat` store as oneshot.
/// Returns true when a splat was added.
fn deposit_chat_will(
    engine: &mut NiodooEngine,
    pos: &Tensor,
    sigma: f32,
    alpha: f32,
    min_dist: f32,
) -> Result<bool> {
    if engine.memory().has_nearby(pos, min_dist)? {
        return Ok(false);
    }
    engine
        .memory_mut()
        .add_splat(Splat::new(pos.clone(), sigma, alpha));
    Ok(true)
}

fn persist_splat_store(engine: &NiodooEngine, path: &Path) -> Result<()> {
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    engine.memory().save(path)
}

fn load_splat_store(engine: &mut NiodooEngine, path: &Path) -> Result<usize> {
    engine.memory_mut().load(path)
}

/// Prefill-bridge mint at the residual site later prefills query (oneshot continuity geometry).
fn mint_chat_prefill_bridge_at(
    engine: &mut NiodooEngine,
    goal: &Tensor,
    sigma: f32,
    alpha: f32,
    lambda: f32,
    offset_frac: f32,
    prompt_fp: u32,
) -> Result<usize> {
    let replace_dist = sigma.max(1.0) * (1.0 + offset_frac.abs());
    engine.memory_mut().deposit_prefill_bridge(
        goal,
        sigma,
        alpha,
        lambda,
        replace_dist,
        offset_frac,
        prompt_fp,
    )?;
    Ok(engine.memory().count_prefill_bridges())
}

/// Interpolate probe logits with the next-token dist of a topic-matched scar residual.
/// `mix=0` is identity. Used on the chat decode path (not residual mix-gain).
fn blend_topic_logits(probe: &Tensor, scar: &Tensor, mix: f32) -> Result<Tensor> {
    if mix <= 1e-6 {
        return Ok(probe.clone());
    }
    let l = mix.clamp(0.0, 1.0) as f64;
    Ok((&probe.affine(1.0 - l, 0.0)? + &scar.affine(l, 0.0)?)?)
}

/// Geometry at a residual position: nearest L2, σ of nearest, potential, |F_s|.
fn chat_basin_query(engine: &NiodooEngine, pos: &Tensor) -> Result<(f32, f32, f32, f32)> {
    if engine.memory().len() == 0 {
        return Ok((f32::INFINITY, 0.0, 0.0, 0.0));
    }
    let (nearest, sigma, _mean, _n) = engine.memory().nearest_scar_stats(pos, 64)?;
    let pot = engine.memory().query_potential(pos).unwrap_or(0.0);
    let force = engine.memory().query_force(pos)?;
    let fv = force.flatten_all()?.to_vec1::<f32>()?;
    let mag = fv.iter().map(|x| x * x).sum::<f32>().sqrt();
    Ok((nearest, sigma, pot, mag))
}

#[allow(clippy::too_many_arguments)]
fn generate_turn_ex(
    model: &mut Model,
    tokenizer: &tokenizers::Tokenizer,
    device: &Device,
    cfg: &Config,
    max_tokens: usize,
    engine: &mut NiodooEngine,
    logit_chain: &mut logit_physics::LogitChain,
    hook_controls: &mut hooks::HookControls,
    hook_trace: &mut Option<hooks::HookTrace>,
    prompt: &str,
    fp_src: &str,
    eos_token_ids: &[u32],
    tags_on: bool,
    operator_temperature: &mut f64,
    operator_rep_penalty: &mut f32,
    turn_idx: usize,
    prev_assistant_len: usize,
    mint_wills: bool,
    scaler_receipt: &algo_scale::ScalerReceipt,
    on_token: &mut dyn FnMut(&str, hud::HudFrame) -> repl_tui::Edits,
) -> Result<String> {
    // Model tags modulate these for this turn only; operator `/set` values persist.
    let mut live_temp = *operator_temperature;
    let mut live_rep = *operator_rep_penalty;
    let mut lock_stop = false;
    let _top_k = cfg.generation.top_k;
    let _top_p = cfg.generation.top_p;
    // Ablation / Path B smoke: apply a tag as if the model had already emitted it.
    // HYDRO_INJECT_TAG=spike|… — consumed so later turns in this process do not re-fire.
    let inject_raw = take_hydro_inject_tag();
    let mut inject_hits: Vec<control_tags::TagHit> = Vec::new();
    if let Some(ref raw) = inject_raw {
        let synth = hydro_inject_synth(raw);
        eprintln!(
            "    [tag-inject] requested={raw} synth={synth:?} blend0={:.2} β0={:.2}",
            engine.hands.physics_blend,
            engine.qsma_beta(0)
        );
        inject_hits = apply_hydro_inject(engine, raw);
        if inject_hits.is_empty() {
            eprintln!("    [tag-inject] NO ControlTag — hands unchanged");
        }
        for hit in &inject_hits {
            let stop = hit.tag == control_tags::ControlTag::Lock;
            lock_stop = lock_stop || stop;
            eprintln!(
                "    [tag-inject] applied={:?} payload={:?} blend={:.2} β={:.2} σ={:.2} stop={stop}",
                hit.tag,
                hit.payload,
                engine.hands.physics_blend,
                engine.hands.beta,
                engine.hands.kinetic_noise
            );
        }
    }

    let prompt_ids: Vec<u32> = encode_prompt_no_trailing_eos(tokenizer, prompt, eos_token_ids)?;
    let first_id = prompt_ids.first().copied().unwrap_or(u32::MAX);
    let last_id = prompt_ids.last().copied().unwrap_or(u32::MAX);
    let bos_ok = first_id == 2;
    eprintln!(
        "    [prefill turn={turn_idx} n={} first_id={first_id} last_id={last_id} bos={}]",
        prompt_ids.len(),
        if bos_ok { "yes" } else { "NO" }
    );
    let prompt_tensor = Tensor::new(prompt_ids.as_slice(), device)?.unsqueeze(0)?;

    // Collapse probe: COLLAPSE_PROBE=1 → logs/collapse_probe.jsonl
    // or COLLAPSE_PROBE=/path/to/file.jsonl
    let mut collapse_log: Option<std::fs::File> =
        std::env::var("COLLAPSE_PROBE").ok().and_then(|v| {
            if v == "0" || v.eq_ignore_ascii_case("off") || v.eq_ignore_ascii_case("false") {
                return None;
            }
            let path = if v == "1" || v.eq_ignore_ascii_case("true") || v.eq_ignore_ascii_case("on")
            {
                std::path::PathBuf::from("logs/collapse_probe.jsonl")
            } else {
                std::path::PathBuf::from(v)
            };
            if let Some(parent) = path.parent() {
                let _ = std::fs::create_dir_all(parent);
            }
            std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(&path)
                .ok()
        });
    if let Some(ref mut f) = collapse_log {
        use std::io::Write;
        if turn_idx <= 1 {
            let event = serde_json::json!({
                "event": "scaler_receipt",
                "scaler_receipt_id": scaler_receipt.receipt_id,
                "scaler": scaler_receipt,
            });
            let _ = writeln!(f, "{event}");
        }
        let _ = writeln!(
            f,
            "{{\"event\":\"turn_start\",\"scaler_receipt_id\":\"{}\",\"turn\":{turn_idx},\"prev_asst_len\":{prev_assistant_len},\"prompt_tokens\":{},\"prompt_chars\":{},\"first_id\":{first_id},\"last_id\":{last_id},\"bos\":{}}}",
            scaler_receipt.receipt_id,
            prompt_ids.len(),
            prompt.len(),
            bos_ok
        );
        if let Some(ref raw) = inject_raw {
            let r = engine.hands_report();
            let _ = writeln!(
                f,
                "{{\"event\":\"tag_inject\",\"turn\":{turn_idx},\"requested\":\"{raw}\",\"applied\":{},\"physics_blend\":{},\"qsma_beta\":{},\"kinetic_noise\":{},\"dynamic_repulsion\":{}}}",
                inject_hits
                    .first()
                    .map(|h| format!("\"{}\"", h.tag.as_str()))
                    .unwrap_or_else(|| "null".into()),
                r["physics_blend"],
                r["qsma_beta"],
                r["kinetic_noise"],
                r["dynamic_repulsion"]
            );
        }
    }

    // Fresh prefill each turn: chat re-encodes full history at index_pos=0.
    // Without clear, prior-turn decode cache can poison the next prefill
    // (coherence death on multi-turn even when residual forces are off).
    model.clear_kv_cache();

    // Prefill is intentionally unhooked: its model-native hidden state establishes
    // the turn's J-space goal. Every decode forward after the first sampled token
    // receives the cached residual steering direction through the layer hook.
    let (mut raw_logits, mut raw_hidden) = model.forward_with_hidden(&prompt_tensor, 0)?;
    dim_assert::assert_last_dim(
        &raw_hidden,
        engine.residual_dim(),
        "decode.prefill_raw_hidden",
    )?;
    let goal_pos = raw_hidden.squeeze(0)?;
    dim_assert::assert_last_dim(&goal_pos, engine.residual_dim(), "decode.goal_pos")?;
    let dir_steer = load_dir_steer(model, tokenizer, device, engine.residual_dim())?;
    let diag_specs: Vec<(String, u32)> = std::env::var("HYDRO_DIAG_TOKENS")
        .unwrap_or_else(|_| "loop,repetitive,STOP,Paris".into())
        .split(',')
        .filter_map(|s| {
            let s = s.trim();
            if s.is_empty() {
                return None;
            }
            encode_first_id(tokenizer, s).map(|id| (s.to_string(), id))
        })
        .collect();
    engine.set_prompt_fp(tct::continuity_fp(fp_src));
    let prompt_fp = engine.prompt_fp();
    // Collect a decode trail whenever this turn can persist. commit_decode_trail
    // keeps an existing matching trail so a failed probe cannot overwrite it.
    let write_trail = mint_wills && engine.residual_enabled() && cfg.physics.prefill_bridge_scar;
    let mut decode_trail: Vec<Tensor> = Vec::new();
    let mut decode_trail_toks: Vec<u32> = Vec::new();
    // Trail-own is for short matching return probes. The Official 10 key
    // (Prompt13) quotes lumina-basin-7 inside a 2k-char review, so topic-fp
    // matches the mint and hijacks SCORE. Long user turns generate freely.
    let trail_own_len = if fp_src.chars().count() <= 800
        && engine.residual_enabled()
        && cfg.physics.topic_logit_mix > 1e-6
    {
        engine.memory().decode_trail_len(prompt_fp)
    } else {
        0
    };
    let (mut basin_nearest, mut basin_sigma, mut basin_pot, mut basin_fs) =
        chat_basin_query(engine, &goal_pos)?;
    if engine.memory().len() > 0 {
        eprintln!(
            "    [CHAT BASIN load] turn={turn_idx} n={} nearest={:.2} σ={:.2} pot={:.3} |F_s|={:.4}",
            engine.memory().len(),
            basin_nearest,
            basin_sigma,
            basin_pot,
            basin_fs
        );
        if engine.memory().has_decode_trail(prompt_fp) {
            eprintln!(
                "    [CHAT TRAIL load] turn={turn_idx} n={} fp={:#x}",
                engine.memory().decode_trail_len(prompt_fp),
                prompt_fp
            );
        }
        if let Some(ref mut f) = collapse_log {
            use std::io::Write;
            let nearest_log = if basin_nearest.is_finite() {
                basin_nearest
            } else {
                -1.0
            };
            let _ = writeln!(
                f,
                "{{\"event\":\"basin_load\",\"turn\":{turn_idx},\"nearest\":{nearest_log:.4},\"sigma\":{:.4},\"pot\":{:.6},\"splat_force\":{:.6},\"scars_active\":{},\"bridges\":{}}}",
                basin_sigma,
                basin_pot,
                basin_fs,
                engine.memory().len(),
                engine.memory().count_prefill_bridges()
            );
        }
    }
    if let Some(ref mut f) = collapse_log {
        use std::io::Write;
        let nearest_log = if basin_nearest.is_finite() {
            basin_nearest
        } else {
            -1.0
        };
        let _ = writeln!(
            f,
            "{{\"event\":\"basin\",\"turn\":{turn_idx},\"nearest\":{nearest_log:.4},\"sigma\":{:.4},\"pot\":{:.6},\"splat_force\":{:.6},\"scars_active\":{},\"bridges\":{},\"mint\":{}}}",
            basin_sigma,
            basin_pot,
            basin_fs,
            engine.memory().len(),
            engine.memory().count_prefill_bridges(),
            mint_wills && engine.residual_enabled()
        );
    }
    let mut index_pos = prompt_ids.len();
    let mut generated: Vec<u32> = Vec::new();
    let mut pieces = String::new();
    let mut last_will_pos: Option<Tensor> = None;
    let mut last_chat_will_step: isize = -999;
    // Self-reg phase observe / force (docs/SELF_REG_PHASES.md) — answer | revise | settle
    let self_reg_mode = cfg.self_reg.mode.to_ascii_lowercase();
    let self_reg_observe = self_reg_mode == "observe" || self_reg_mode == "force";
    let self_reg_force = self_reg_mode == "force";
    let mut phase = "answer";
    let mut entropy0: Option<f32> = None;
    let mut last_force_on: Option<bool> = None;
    if self_reg_observe {
        if let Some(ref mut f) = collapse_log {
            use std::io::Write;
            let _ = writeln!(
                f,
                "{{\"event\":\"phase\",\"phase\":\"answer\",\"turn\":{turn_idx},\"step\":0,\"mode\":\"{self_reg_mode}\"}}"
            );
        }
    }
    let mut tags_seen: Vec<control_tags::TagHit> = Vec::new();
    let mut prev_hidden: Option<Tensor> = None;
    let tda_on = tda_monitor::tda_monitor_enabled();
    let mut tda = tda_monitor::TdaShadowMonitor::new(32, 8);
    let mut pending_tda_monitor: Option<String> = None;
    let mut last_tda_step: usize = 0;
    let mut keep_talking_until: usize = 0;
    let mut held_mouth = String::new();
    // Report for the forward pass that produced the logits sampled on this iteration.
    // Step 0 comes from prefill and therefore has no per-layer decode hook report.
    let mut incoming_hook_report = hooks::HookReport::default();

    // Jacobian lens: measure hidden→logit sensitivity periodically.
    // Jason's "literal key" — turns clusters into perm-addresses.
    let jacobian_interval = cfg.jacobian.interval;
    let mut jacobian_counter = 0usize;
    let jacobian_epsilon = cfg.jacobian.epsilon;
    let jacobian_top_k = cfg.jacobian.top_k;
    let jacobian_max_dims = cfg.jacobian.max_dims;
    // Phase-edge multi-key address (first answer / revise / settle).
    let phase_edge_keys = cfg.jacobian.phase_edge_keys;
    let mut multi_keys = jacobian::MultiKeyAddress::new();
    let mut key_captured_answer = false;
    let mut key_captured_revise = false;
    let mut key_captured_settle = false;
    // Full residual D is too slow for edge capture; default subsample if unset.
    let phase_key_max_dims = if jacobian_max_dims > 0 {
        jacobian_max_dims
    } else {
        64
    };

    for step in 0..max_tokens {
        if lock_stop {
            break;
        }
        if trail_own_len > 0 && step >= trail_own_len {
            eprintln!("    [CHAT TRAIL own] turn={turn_idx} stop n={trail_own_len}");
            break;
        }
        engine.tick_hands();
        engine.tick_endocrine();
        // mode=force: residual ON only while phase==revise. Answer stays force-off.
        // Isolation baseline keeps physics.force_cap=0; revise pulls self_reg.force_*.
        if self_reg_force {
            let force_on = phase == "revise" && cfg.self_reg.force_cap > 1e-8;
            if force_on {
                let _ = engine.set_live_param("residual.cap", cfg.self_reg.force_cap);
                let _ = engine.set_live_param("residual.goal", cfg.self_reg.force_goal_scale);
                let _ = engine.set_live_param("residual.splat", cfg.self_reg.force_splat_scale);
                let _ = engine.set_live_param("residual.field", cfg.self_reg.force_field_scale);
            } else {
                let _ = engine.set_live_param("residual.cap", 0.0);
                let _ = engine.set_live_param("residual.goal", 0.0);
                let _ = engine.set_live_param("residual.splat", 0.0);
                let _ = engine.set_live_param("residual.field", 0.0);
            }
            if last_force_on != Some(force_on) {
                last_force_on = Some(force_on);
                if let Some(ref mut f) = collapse_log {
                    use std::io::Write;
                    let _ = writeln!(
                        f,
                        "{{\"event\":\"force_gate\",\"phase\":\"{phase}\",\"force_on\":{force_on},\"force_cap\":{:.4},\"goal\":{:.4},\"splat\":{:.4},\"field\":{:.4},\"turn\":{turn_idx},\"step\":{step}}}",
                        if force_on {
                            cfg.self_reg.force_cap
                        } else {
                            0.0
                        },
                        if force_on {
                            cfg.self_reg.force_goal_scale
                        } else {
                            0.0
                        },
                        if force_on {
                            cfg.self_reg.force_splat_scale
                        } else {
                            0.0
                        },
                        if force_on {
                            cfg.self_reg.force_field_scale
                        } else {
                            0.0
                        },
                    );
                }
            }
        }

        dim_assert::assert_last_dim(&raw_hidden, engine.residual_dim(), "decode.step_raw_hidden")?;
        let steer = engine.steer(&raw_hidden, &goal_pos, step)?;
        if step == 0 {
            eprintln!(
                "    [CHAT STEER] turn={turn_idx} pot={:.3} warm={} ramp={:.3} |F_s|={:.4}",
                steer.scar_pot, steer.memory_warm, steer.ramp, steer.splat_mag
            );
        }
        let residual_live = cfg.physics.steer_hidden && engine.residual_enabled();
        let mut surface_hidden = if residual_live {
            steer.steered.clone()
        } else {
            raw_hidden.clone()
        };
        let dir_c = dir_steer.as_ref().map(|d| d.c).unwrap_or(0.0);
        if let Some(dir) = dir_steer.as_ref() {
            if dir.c.abs() > 1e-12 {
                let add = dir.vec.affine(dir.c as f64, 0.0)?;
                surface_hidden = (&surface_hidden + &add)?;
            }
        }
        let intervene = residual_live || dir_c.abs() > 1e-12;
        dim_assert::assert_last_dim(
            &surface_hidden,
            engine.residual_dim(),
            "decode.surface_hidden",
        )?;

        // Jacobian measurement: every N steps, measure sensitivity of hidden dims to logits.
        if jacobian_interval > 0 && step > 0 && step % jacobian_interval == 0 {
            jacobian_counter += 1;
            if let Ok(report) = measure_jacobian_step(
                &surface_hidden,
                model,
                jacobian_epsilon,
                jacobian_top_k,
                jacobian_max_dims,
                step,
                engine.residual_dim(),
            ) {
                let top_dims: Vec<String> = report
                    .dominant_dimensions
                    .iter()
                    .take(3)
                    .map(|d| d.to_string())
                    .collect();
                let top_tokens: Vec<String> = report
                    .dominant_tokens
                    .iter()
                    .take(3)
                    .map(|t| t.to_string())
                    .collect();
                eprintln!(
                    "[JACOBIAN] step={} norm={:.4} top_dim={} top_token={}",
                    step,
                    report.global_sensitivity,
                    top_dims.join(","),
                    top_tokens.join(","),
                );
            }
        }

        let mut biased_logits = if intervene {
            model.project_to_logits(&surface_hidden)?
        } else {
            raw_logits.clone()
        };
        // hidden_delta = ||h'-h|| after pullback / dir add (what lm_head sees).
        // delta_h_norm is ||steering|| before residual-off clone / pullback.
        // logit_delta = ||z'-z|| from residual/dir project only (before topic mix /
        // logit_chain / rep penalty).
        let hidden_delta: f32 = (&surface_hidden - &raw_hidden)?
            .sqr()?
            .sum_all()?
            .to_scalar::<f32>()?
            .sqrt();
        let logit_delta: f32 = if intervene {
            (&biased_logits - &raw_logits)?
                .sqr()?
                .sum_all()?
                .to_scalar::<f32>()?
                .sqrt()
        } else {
            0.0
        };
        if step == 0 {
            let dir_mode = dir_steer
                .as_ref()
                .map(|d| d.mode.as_str())
                .unwrap_or("none");
            eprintln!(
                "    [CHAT DELTA] turn={turn_idx} residual_live={} dir_mode={dir_mode} dir_c={dir_c:.4} delta_h_norm={:.4} hidden_delta={:.4} logit_delta={:.4}",
                residual_live, steer.delta_h_norm, hidden_delta, logit_delta
            );
        }
        // Scar-conditioned decode: step k reads lm_head of the minted
        // completion residual at k. Same μ reused every step soups TheThe;
        // the trail is the content. No trail → step-0 bridge μ (old stores).
        if residual_live && cfg.physics.topic_logit_mix > 1e-6 {
            let fp = engine.prompt_fp();
            let scar_mu = match engine.memory().matched_trail_mu(fp, step)? {
                Some(mu) => Some(mu),
                None if step == 0 => engine.memory().matched_bridge_mu(fp)?,
                None => None,
            };
            if let Some(mu) = scar_mu {
                let mu = if mu.dims().len() == 1 {
                    mu.unsqueeze(0)?
                } else {
                    mu
                };
                if mu.dims() == surface_hidden.dims() {
                    let scar_logits = model.project_to_logits(&mu)?;
                    biased_logits = blend_topic_logits(
                        &biased_logits,
                        &scar_logits,
                        cfg.physics.topic_logit_mix,
                    )?;
                    if step == 0 {
                        let own = if engine.memory().matched_trail_token(fp, 0).is_some() {
                            "yes"
                        } else {
                            "no"
                        };
                        eprintln!(
                            "    [CHAT TOPIC LOGIT] turn={turn_idx} mix={:.2} fp={:#x} trail={} own={own}",
                            cfg.physics.topic_logit_mix,
                            fp,
                            engine.memory().decode_trail_len(fp)
                        );
                    }
                }
            }
        }
        {
            let ctx = logit_physics::StepCtx {
                step,
                steered_hidden: Some(&surface_hidden),
                steer: Some(&steer),
                token_embeddings: model.token_embeddings(),
                field: Some(engine.field()),
                memory: Some(engine.memory()),
                memory_pick: Some(engine.memory_pick()),
                prompt_fp: engine.prompt_fp(),
            };
            biased_logits = logit_chain.apply(&biased_logits, &ctx)?;
        }
        let mut logits_vec: Vec<f32> = biased_logits.squeeze(0)?.to_vec1()?;
        let diag_ranks: Vec<(String, u32, Option<u32>)> = diag_specs
            .iter()
            .map(|(name, id)| (name.clone(), *id, logit_rank(&logits_vec, *id)))
            .collect();
        // Repetition penalty: once per unique token id, not once per
        // occurrence. Iterating raw history without dedup let a token seen N
        // times in the growing multi-turn context get divided by live_rep^N.
        // For common glue tokens (space, punctuation) N climbs fast across
        // turns, and the compounding drove them toward zero probability
        // within a few turns — that is what was collapsing multi-turn output
        // into run-together or fragmented text (same sign convention as
        // one-shot path).
        let seen_ids: std::collections::HashSet<u32> =
            prompt_ids.iter().chain(generated.iter()).copied().collect();
        for tid in seen_ids {
            if (tid as usize) < logits_vec.len() {
                let l = &mut logits_vec[tid as usize];
                if *l > 0.0 {
                    *l /= live_rep;
                } else {
                    *l *= live_rep;
                }
            }
        }

        // Spike/focus/explore/reset/remember never stop the turn. After a
        // physics hand, mask EOS so Gemma cannot eot on the tag (Niodoo
        // keep-talking). Only <lock> is allowed to stop.
        if step < keep_talking_until {
            for id in eos_token_ids {
                let idx = *id as usize;
                if idx < logits_vec.len() {
                    logits_vec[idx] = f32::NEG_INFINITY;
                }
            }
        }

        // QSMA in-decode: Q + ease(F)×β + C + σ·ξ on the leading logits.
        let qsma_pick = engine.apply_qsma_logits(&mut logits_vec, &generated, step);
        // Softmax over logits for collapse telemetry (cheap top-k entropy + margin).
        let (entropy, margin, p_top1, top1_id) = collapse_logit_stats(&logits_vec, 64);
        let residual_norm = steer.steered_norm;
        if entropy0.is_none() {
            entropy0 = Some(entropy);
        }

        // Observe: label revise — entropy/margin, text cues ("Wait…"), or line-repeat
        // thrash (confident low-entropy loops entropy/margin miss). Re-checked after
        // each token push so the completing token of the Nth line can flip phase.
        if self_reg_observe && phase == "answer" {
            let min_a = cfg.self_reg.min_answer_tokens;
            let e0 = entropy0.unwrap_or(entropy);
            let ent_spike = entropy > e0 + cfg.self_reg.revise_entropy_delta;
            let flat = margin < cfg.self_reg.revise_margin_max;
            let text_cue = pieces.contains("Wait")
                || pieces.contains("try again")
                || pieces.contains("Try again")
                || pieces.contains("wrong");
            let line_rep = line_repeat_at_least(
                &pieces,
                cfg.self_reg.revise_line_repeat,
                cfg.self_reg.line_repeat_min_chars,
            );
            let phrase_rep = cfg.self_reg.revise_line_repeat > 0
                && phrase_repeat_at_least(&pieces, cfg.self_reg.revise_line_repeat, 12, 48);
            let reason = if text_cue {
                Some("text_cue")
            } else if generated.len() >= min_a && (line_rep || phrase_rep) {
                Some(if line_rep {
                    "line_repeat"
                } else {
                    "phrase_repeat"
                })
            } else if generated.len() >= min_a && ent_spike && flat {
                Some("entropy_margin")
            } else {
                None
            };
            if let Some(reason) = reason {
                phase = "revise";
                if let Some(ref mut f) = collapse_log {
                    use std::io::Write;
                    let (rep_n, _) = trailing_identical_line_run(&pieces);
                    let _ = writeln!(
                        f,
                        "{{\"event\":\"phase\",\"phase\":\"revise\",\"reason\":\"{reason}\",\"line_repeat\":{rep_n},\"turn\":{turn_idx},\"step\":{step},\"entropy\":{entropy:.4},\"margin\":{margin:.4},\"e0\":{e0:.4}}}"
                    );
                }
                if phase_edge_keys {
                    capture_phase_edge_key(
                        "revise",
                        step,
                        turn_idx,
                        &surface_hidden,
                        &pieces,
                        model,
                        jacobian_epsilon,
                        jacobian_top_k,
                        phase_key_max_dims,
                        engine.residual_dim(),
                        &mut multi_keys,
                        &mut collapse_log,
                        &mut key_captured_revise,
                    );
                }
            }
        }

        // QSMA is the decode policy (π=argmax[Q+ease(F)×β+C+σξ] on top-64), not a
        // softmax sidecar. Hands set β/σ; T/rep do not pick the token.
        let mut next = qsma_pick.index as u32;
        if trail_own_len > 0 {
            if let Some(tok) = engine.memory().matched_trail_token(prompt_fp, step) {
                next = tok;
            }
        }
        engine.observe_token(next, p_top1 as f64);
        if eos_token_ids.contains(&next) {
            if self_reg_observe {
                phase = "settle";
            }
            if phase_edge_keys {
                capture_phase_edge_key(
                    "settle",
                    step,
                    turn_idx,
                    &surface_hidden,
                    &pieces,
                    model,
                    jacobian_epsilon,
                    jacobian_top_k,
                    phase_key_max_dims,
                    engine.residual_dim(),
                    &mut multi_keys,
                    &mut collapse_log,
                    &mut key_captured_settle,
                );
            }
            if let Some(ref mut f) = collapse_log {
                use std::io::Write;
                let _ = writeln!(
                    f,
                    "{{\"event\":\"eos\",\"phase\":\"{phase}\",\"turn\":{turn_idx},\"step\":{step},\"residual_norm\":{residual_norm:.4},\"entropy\":{entropy:.4},\"margin\":{margin:.4},\"p_top1\":{p_top1:.4},\"prev_asst_len\":{prev_assistant_len}}}"
                );
            }
            break;
        }

        // Gemma 4 channel specials (100=`<|channel>` 101=`<channel|>`) stay in
        // the token stream. They are thought trajectory, not a settle stop.
        let has_content = !pieces.trim().is_empty();
        if gemma4_should_settle_channel(&pieces, next) {
            phase = "settle";
            if let Some(ref mut f) = collapse_log {
                use std::io::Write;
                let _ = writeln!(
                    f,
                    "{{\"event\":\"settle_channel\",\"phase\":\"settle\",\"turn\":{turn_idx},\"step\":{step},\"token_id\":{next}}}"
                );
            }
            break;
        }

        generated.push(next);
        if write_trail && decode_trail.len() < SplatMemory::DECODE_TRAIL_MAX {
            decode_trail.push(surface_hidden.clone());
            decode_trail_toks.push(next);
        }
        let piece = tokenizer
            .decode(&[next], false)
            .unwrap_or_else(|_| format!("[{next}]"));
        pieces.push_str(&piece);
        if tda_on && pending_tda_monitor.is_none() {
            if let Some(line) = tda.observe(
                &piece,
                entropy,
                margin,
                residual_norm,
                steer.splat_mag,
                p_top1,
                step,
                max_tokens,
            ) {
                if last_tda_step == 0 || step.saturating_sub(last_tda_step) >= 48 {
                    pending_tda_monitor = Some(line);
                    last_tda_step = step;
                }
            }
        }
        if let Ok(pos) = surface_hidden.flatten_all() {
            last_will_pos = Some(pos);
            let interval = cfg.physics.online_splat_interval.max(1) as isize;
            let rate_ok = step as isize - last_chat_will_step >= interval;
            if engine.residual_enabled() && rate_ok {
                let alpha = if p_top1 >= 0.25 {
                    cfg.generation.pleasure_alpha.abs().max(0.15)
                } else {
                    -cfg.generation.pain_alpha.abs().max(0.15)
                };
                match deposit_chat_will(
                    engine,
                    last_will_pos.as_ref().unwrap(),
                    cfg.physics.splat_sigma,
                    alpha,
                    cfg.physics.min_splat_dist,
                ) {
                    Ok(true) => {
                        last_chat_will_step = step as isize;
                        engine.memory_mut().prune_to_limit(cfg.memory.max_splats);
                    }
                    Ok(false) => {}
                    Err(e) => eprintln!("    [CHAT WILL] deposit failed: {e}"),
                }
            }
        }
        // Legacy XML `</thought>` close is not Gemma 4's thought stream
        // (`<|channel>thought` … `<channel|>`). Do not settle-stop a live
        // channel on that leftover marker.
        if has_content && pieces.contains("</thought>") && !pieces.contains("<|channel>") {
            phase = "settle";
            if let Some(ref mut f) = collapse_log {
                use std::io::Write;
                let _ = writeln!(
                    f,
                    "{{\"event\":\"settle_thought_close\",\"phase\":\"settle\",\"turn\":{turn_idx},\"step\":{step}}}"
                );
            }
            if let Some(i) = pieces.find("</thought>") {
                pieces.truncate(i);
            }
            break;
        }
        // Hyphen thrash: "The-\nThe-\nThe-" or same short fragment 4+ times in a row.
        // Research signal kept in transcript until cut; sampling stops so turn can settle.
        if pending_tda_monitor.is_none() && gemma4_hyphen_thrash(&pieces) {
            phase = "settle";
            if let Some(ref mut f) = collapse_log {
                use std::io::Write;
                let _ = writeln!(
                    f,
                    "{{\"event\":\"settle_hyphen_thrash\",\"phase\":\"settle\",\"turn\":{turn_idx},\"step\":{step}}}"
                );
            }
            break;
        }
        // Short-cycle lock (`esesese`): stop before the 256-token residual soup.
        if pending_tda_monitor.is_none() && trailing_short_cycle_lock(&pieces) {
            phase = "settle";
            if let Some(ref mut f) = collapse_log {
                use std::io::Write;
                let _ = writeln!(
                    f,
                    "{{\"event\":\"settle_short_cycle\",\"phase\":\"settle\",\"turn\":{turn_idx},\"step\":{step}}}"
                );
            }
            eprintln!("    [CHAT SETTLE cycle] turn={turn_idx} step={step}");
            break;
        }
        // Confident line-repeat thrash (any length ≥ min_chars): settle after N copies.
        // Labels revise earlier (revise_line_repeat); this stops the turn for multi-turn usability.
        if pending_tda_monitor.is_none()
            && line_repeat_at_least(
                &pieces,
                cfg.self_reg.settle_line_repeat,
                cfg.self_reg.line_repeat_min_chars,
            )
        {
            phase = "settle";
            if let Some(ref mut f) = collapse_log {
                use std::io::Write;
                let (rep_n, _) = trailing_identical_line_run(&pieces);
                let _ = writeln!(
                    f,
                    "{{\"event\":\"settle_line_repeat\",\"phase\":\"settle\",\"line_repeat\":{rep_n},\"turn\":{turn_idx},\"step\":{step}}}"
                );
            }
            break;
        }
        // Wait / try-again block thrash (Spell-cat class): settle after N loops.
        let wait_n = wait_loop_count(&pieces);
        if pending_tda_monitor.is_none()
            && cfg.self_reg.settle_wait_loops > 0
            && wait_n >= cfg.self_reg.settle_wait_loops
        {
            phase = "settle";
            if let Some(ref mut f) = collapse_log {
                use std::io::Write;
                let _ = writeln!(
                    f,
                    "{{\"event\":\"settle_wait_loop\",\"phase\":\"settle\",\"wait_loops\":{wait_n},\"turn\":{turn_idx},\"step\":{step}}}"
                );
            }
            break;
        }
        // Same-line phrase thrash (no newlines): "No, the question is …?"×N
        // Use settle_line_repeat as copy count; unit 12..=48 chars.
        if pending_tda_monitor.is_none()
            && cfg.self_reg.settle_line_repeat > 0
            && phrase_repeat_at_least(&pieces, cfg.self_reg.settle_line_repeat, 12, 48)
        {
            phase = "settle";
            if let Some(ref mut f) = collapse_log {
                use std::io::Write;
                let _ = writeln!(
                    f,
                    "{{\"event\":\"settle_phrase_repeat\",\"phase\":\"settle\",\"turn\":{turn_idx},\"step\":{step}}}"
                );
            }
            break;
        }

        // Re-run revise heuristics after the token lands (line_repeat / text_cue complete mid-piece).
        if self_reg_observe && phase == "answer" {
            let min_a = cfg.self_reg.min_answer_tokens;
            let e0 = entropy0.unwrap_or(entropy);
            let text_cue = pieces.contains("Wait")
                || pieces.contains("try again")
                || pieces.contains("Try again")
                || pieces.contains("wrong");
            let line_rep = line_repeat_at_least(
                &pieces,
                cfg.self_reg.revise_line_repeat,
                cfg.self_reg.line_repeat_min_chars,
            );
            let phrase_rep = cfg.self_reg.revise_line_repeat > 0
                && phrase_repeat_at_least(&pieces, cfg.self_reg.revise_line_repeat, 12, 48);
            let reason = if text_cue {
                Some("text_cue")
            } else if generated.len() >= min_a && (line_rep || phrase_rep) {
                Some(if line_rep {
                    "line_repeat"
                } else {
                    "phrase_repeat"
                })
            } else {
                None
            };
            if let Some(reason) = reason {
                phase = "revise";
                if let Some(ref mut f) = collapse_log {
                    use std::io::Write;
                    let (rep_n, _) = trailing_identical_line_run(&pieces);
                    let _ = writeln!(
                        f,
                        "{{\"event\":\"phase\",\"phase\":\"revise\",\"reason\":\"{reason}\",\"line_repeat\":{rep_n},\"turn\":{turn_idx},\"step\":{step},\"entropy\":{entropy:.4},\"margin\":{margin:.4},\"e0\":{e0:.4}}}"
                    );
                }
            }
        }

        let mut shown = piece.clone();

        if let Some(ref mut f) = collapse_log {
            use std::io::Write;
            let esc = piece
                .replace('\\', "\\\\")
                .replace('"', "\\\"")
                .replace('\n', "\\n");
            // phase always present when collapse probe is on (answer|revise|settle).
            // With self_reg.mode=off, phase stays "answer" unless a settle clamp fired.
            let _ = writeln!(
                f,
                "{{\"event\":\"tok\",\"scaler_receipt_id\":\"{}\",\"phase\":\"{phase}\",\"force_on\":{},\"turn\":{turn_idx},\"step\":{step},\"token_id\":{next},\"token\":\"{esc}\",\"residual_norm\":{residual_norm:.4},\"baseline_norm\":{:.4},\"entropy\":{entropy:.4},\"margin\":{margin:.4},\"p_top1\":{p_top1:.4},\"top1_id\":{top1_id},\"prompt_tokens\":{},\"gen_so_far\":{},\"prev_asst_len\":{prev_assistant_len},\"physics_blend\":{:.4},\"qsma_beta\":{:.4},\"kinetic_noise\":{:.4},\"qsma_idx\":{},\"ramp\":{:.4},\"delta_h_norm\":{:.4},\"hidden_delta\":{:.4},\"logit_delta\":{:.4},\"dir_mode\":\"{}\",\"dir_c\":{:.4},\"diag\":{},\"grad_mag\":{:.4},\"splat_mag\":{:.4},\"goal_mag\":{:.4},\"ocean_mag\":{:.4},\"scars_active\":{},\"scar_pot\":{:.4},\"memory_warm\":{},\"hook_applications\":{},\"hook_delta_mean\":{:.6},\"hook_delta_max\":{:.6}}}",
                scaler_receipt.receipt_id,
                residual_live,
                steer.baseline_norm,
                prompt_ids.len(),
                generated.len(),
                steer.physics_blend,
                steer.qsma_beta,
                steer.kinetic_noise,
                qsma_pick.index,
                steer.ramp,
                steer.delta_h_norm,
                hidden_delta,
                logit_delta,
                dir_steer.as_ref().map(|d| d.mode.as_str()).unwrap_or("none"),
                dir_c,
                {
                    let mut m = serde_json::Map::new();
                    for (name, id, rank) in &diag_ranks {
                        m.insert(
                            name.clone(),
                            serde_json::json!({"id": id, "logit_rank": rank}),
                        );
                    }
                    serde_json::Value::Object(m)
                },
                steer.grad_mag,
                steer.splat_mag,
                steer.goal_mag,
                steer.ocean_mag,
                engine.memory().len(),
                steer.scar_pot,
                steer.memory_warm,
                incoming_hook_report.applications,
                incoming_hook_report.delta_mean,
                incoming_hook_report.delta_max
            );
        }

        let cos_drift = hud::cosine_drift(prev_hidden.as_ref(), &surface_hidden);
        prev_hidden = surface_hidden.flatten_all().ok();
        let logit_report = logit_chain.last_report();
        // Collapse probe fills entropy/margin; otherwise HUD shows "—".
        let frame = hud::HudFrame {
            step,
            max_tokens,
            force_cap: engine.force_cap(),
            goal_force_scale: engine.goal_force_scale(),
            temperature: live_temp as f32,
            force_ramp_start: cfg.physics.force_ramp_start,
            force_ramp_tokens: cfg.physics.force_ramp_tokens,
            field_grad_blend: cfg.physics.field_grad_blend,
            baseline_norm: steer.baseline_norm,
            steered_norm: steer.steered_norm,
            pullback: steer.pullback,
            delta_h_norm: steer.delta_h_norm,
            clip_frac: steer.clip_frac,
            ramp: steer.ramp,
            eureka_boost: steer.eureka_boost,
            cos_drift,
            grad_mag: steer.grad_mag,
            splat_mag: steer.splat_mag,
            goal_mag: steer.goal_mag,
            ocean_mag: steer.ocean_mag,
            memory_ranked: steer.memory_ranked,
            field_wake_max: engine.field_wake_max(),
            splat_force_max: engine.splat_force_max(),
            goal_force_max: engine.goal_force_max(),
            logit_delta: Some(logit_delta),
            logit_velocity: logit_report.velocity,
            logit_viscosity: logit_report.viscosity,
            hook_delta_mean: Some(incoming_hook_report.delta_mean),
            hook_applications: Some(incoming_hook_report.applications),
            p_chosen: Some(p_top1),
            entropy: Some(entropy),
            margin: Some(margin),
            scars: engine.memory().len(),
        };
        let edits = if control_tags::incomplete_control_hand(&pieces) {
            held_mouth.push_str(&shown);
            repl_tui::Edits::default()
        } else {
            let flush = if held_mouth.is_empty() {
                shown.clone()
            } else {
                let mut s = std::mem::take(&mut held_mouth);
                s.push_str(&shown);
                s
            };
            on_token(&flush, frame.clone())
        };

        // Fire hands after a complete tag is in the mouth. Spike never stops.
        if tags_on {
            let (stop, applied) =
                engine.apply_emitted_control(&pieces, &mut tags_seen, Some(&surface_hidden))?;
            if let Some(ref mut f) = collapse_log {
                use std::io::Write;
                let r = engine.hands_report();
                for tag in &applied {
                    let _ = writeln!(
                        f,
                        "{{\"event\":\"hand_fired\",\"action\":\"{}\",\"step\":{step},\"turn\":{turn_idx},\"physics_blend\":{},\"hand_beta\":{},\"kinetic_noise\":{},\"dynamic_repulsion\":{}}}",
                        tag.as_str(),
                        r["physics_blend"],
                        r["hand_beta"],
                        r["kinetic_noise"],
                        r["dynamic_repulsion"],
                    );
                }
            }
            if applied.iter().any(|t| t.is_physics_hand()) {
                keep_talking_until = step.saturating_add(64);
            }
            if stop && gemma4_lock_stops_turn(&pieces) {
                lock_stop = true;
            }
        }

        // Live control edits land here, between tokens, so the next steer sees them.
        for (name, value) in &edits.sets {
            if set_live_control(
                name,
                *value,
                engine,
                logit_chain,
                hook_controls,
                operator_temperature,
                operator_rep_penalty,
            ) {
                // Sampler knobs are per-turn copies; re-read the new baseline.
                live_temp = *operator_temperature;
                live_rep = *operator_rep_penalty;
            }
        }
        if edits.step_abort {
            break;
        }

        let token_tensor = Tensor::new(&[next], device)?.unsqueeze(0)?;
        let hook_direction = if residual_live {
            (&steer.steered - &raw_hidden)?
        } else {
            Tensor::zeros(raw_hidden.dims(), raw_hidden.dtype(), raw_hidden.device())?
        };
        let (next_logits, next_hidden, hook_report) = forward_decode_with_hook(
            model,
            &token_tensor,
            index_pos,
            &hook_direction,
            step + 1,
            hook_controls,
            hook_trace,
        )?;
        raw_logits = next_logits;
        raw_hidden = next_hidden;
        incoming_hook_report = hook_report;
        index_pos += 1;

        // Measured Internal monitor into the mouth. Forward each inject token
        // so KV/RoPE stay in sync (niodoo crash was append-without-forward).
        if tda_monitor_injection_ready(&pieces, pending_tda_monitor.is_some()) {
            let line = pending_tda_monitor
                .take()
                .expect("pending monitor checked immediately above");
            let inject = format!("\n{line}\n");
            pieces.push_str(&inject);
            let _ = on_token(&inject, frame.clone());
            if let Some(ref mut f) = collapse_log {
                use std::io::Write;
                let esc = line.replace('\\', "\\\\").replace('"', "\\\"");
                let _ = writeln!(
                    f,
                    "{{\"event\":\"internal_monitor\",\"turn\":{turn_idx},\"step\":{step},\"line\":\"{esc}\"}}"
                );
            }
            match tokenizer.encode(inject.as_str(), false) {
                Ok(enc) => {
                    let zeros =
                        Tensor::zeros(raw_hidden.dims(), raw_hidden.dtype(), raw_hidden.device())?;
                    let ids: Vec<u32> = enc
                        .get_ids()
                        .iter()
                        .copied()
                        .filter(|tid| *tid != 2 && !eos_token_ids.contains(tid))
                        .collect();
                    eprintln!(
                        "    [CHAT KV] internal_monitor tokens={} turn={turn_idx} step={step}",
                        ids.len()
                    );
                    for tid in ids {
                        if generated.len() >= max_tokens {
                            break;
                        }
                        generated.push(tid);
                        let tok_t = Tensor::new(&[tid], device)?.unsqueeze(0)?;
                        let (nl, nh, _) = forward_decode_with_hook(
                            model,
                            &tok_t,
                            index_pos,
                            &zeros,
                            step + 1,
                            hook_controls,
                            hook_trace,
                        )?;
                        raw_logits = nl;
                        raw_hidden = nh;
                        index_pos += 1;
                    }
                }
                Err(e) => eprintln!("    [CHAT MONITOR] encode failed: {e}"),
            }
        }
    }

    if engine.residual_enabled() {
        if let Some(ref pos) = last_will_pos {
            let alpha = cfg.generation.pleasure_alpha.abs().max(0.15);
            match deposit_chat_will(
                engine,
                pos,
                cfg.physics.splat_sigma,
                alpha,
                cfg.physics.min_splat_dist,
            ) {
                Ok(true) => {
                    engine.memory_mut().prune_to_limit(cfg.memory.max_splats);
                }
                Ok(false) | Err(_) => {}
            }
        }
        if turn_idx > 0 {
            eprintln!(
                "    [CHAT WILL] turn={turn_idx} wills={}",
                engine.memory().len()
            );
        }
    }

    // Mint after decode so this turn *reads* loaded scars first.
    // Previous order self-minted at the probe prefill and hijacked KEEP to a new ring.
    if mint_wills && engine.residual_enabled() && cfg.physics.prefill_bridge_scar {
        let n_br = mint_chat_prefill_bridge_at(
            engine,
            &goal_pos,
            cfg.physics.prefill_bridge_sigma,
            cfg.physics.prefill_bridge_alpha.abs(),
            cfg.physics.prefill_bridge_lambda,
            cfg.physics.prefill_bridge_offset_frac,
            prompt_fp,
        )?;
        let _ = engine
            .memory_mut()
            .enforce_max_prefill_bridges(cfg.memory.max_prefill_bridges);
        let g = chat_basin_query(engine, &goal_pos)?;
        basin_nearest = g.0;
        basin_sigma = g.1;
        basin_pot = g.2;
        basin_fs = g.3;
        eprintln!(
            "    [CHAT BASIN mint] turn={turn_idx} bridges={n_br} nearest={:.2} σ={:.2} pot={:.3} |F_s|={:.4}",
            basin_nearest, basin_sigma, basin_pot, basin_fs
        );
        match engine
            .memory_mut()
            .commit_decode_trail(prompt_fp, decode_trail, decode_trail_toks)?
        {
            memory::TrailCommit::Minted(n_tr) => {
                eprintln!(
                    "    [CHAT TRAIL mint] turn={turn_idx} n={n_tr} fp={:#x}",
                    prompt_fp
                );
            }
            memory::TrailCommit::Kept(n_tr) => {
                eprintln!(
                    "    [CHAT TRAIL keep] turn={turn_idx} n={n_tr} fp={:#x}",
                    prompt_fp
                );
            }
            memory::TrailCommit::Skipped => {}
        }
        if let Some(ref mut f) = collapse_log {
            use std::io::Write;
            let nearest_log = if basin_nearest.is_finite() {
                basin_nearest
            } else {
                -1.0
            };
            let _ = writeln!(
                f,
                "{{\"event\":\"basin_mint\",\"turn\":{turn_idx},\"nearest\":{nearest_log:.4},\"sigma\":{:.4},\"pot\":{:.6},\"splat_force\":{:.6},\"scars_active\":{},\"bridges\":{}}}",
                basin_sigma,
                basin_pot,
                basin_fs,
                engine.memory().len(),
                n_br
            );
        }
    }

    while control_tags::incomplete_control_hand(&pieces) {
        if let Some(i) = pieces.rfind('<') {
            pieces.truncate(i);
        } else {
            break;
        }
    }
    Ok(pieces)
}

/// Interactive multi-turn chat (stdin). Load once, talk many times.
/// Quit: empty line, `quit`, `exit`, or Ctrl-D. `reset` clears history.
/// Transcripts land in `private/chats/` (gitignored) — generation diagnostics,
/// not public logs.
/// Same-prefill tag ablation. Isolation + open seats. Path B proof is physics
/// (blend/β/σ/Δh), not T/rep. T and rep must stay at the seat baseline.
fn run_tag_ablation(
    model: &mut Model,
    tokenizer: &Tokenizer,
    device: &Device,
    cfg: &mut Config,
    engine: &mut NiodooEngine,
    logit_chain: &mut logit_physics::LogitChain,
    hook_controls: &mut hooks::HookControls,
    hook_trace: &mut Option<hooks::HookTrace>,
    scaler_receipt: &algo_scale::ScalerReceipt,
) -> Result<()> {
    use std::io::Write;
    let variant = model.variant_name();
    let eos = generation_eos_token_ids(variant, &cfg.generation.eos_token_ids);
    let prompt = format_multiturn_prompt_ex(
        &[(
            true,
            "Count from 1 to 20 using digits and spaces only.".to_string(),
        )],
        variant,
        true,
    );
    let out_path = Path::new("logs/tag_ablation.jsonl");
    if let Some(parent) = out_path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let mut out = std::fs::File::create(out_path)?;
    println!("=== TAG ABLATION ===");
    println!("    prompt: Count from 1 to 20 using digits and spaces only.");
    println!("    out: {}", out_path.display());
    println!("    wrap: god-tier system + one user turn (not a human tag menu)");

    let seats: [(&str, f64, f32, usize, Option<&str>); 2] = [
        ("isolation_T0_topk1", 0.0, 1.0, 1, None),
        ("open_T08_topk40_seed1", 0.8, 1.0, 40, Some("1")),
    ];
    let arms = [
        "none", "spike", "explore", "focus", "reset", "remember", "lock",
    ];

    for (seat, temp, rep, top_k, seed) in seats {
        cfg.generation.temperature = temp;
        cfg.generation.rep_penalty = rep;
        cfg.generation.top_k = top_k;
        cfg.generation.top_p = 1.0;
        match seed {
            Some(s) => std::env::set_var("HYDRO_SAMPLE_SEED", s),
            None => std::env::remove_var("HYDRO_SAMPLE_SEED"),
        }
        let mut baseline_ids: Option<Vec<char>> = None;
        for arm in arms {
            if arm == "none" {
                std::env::remove_var("HYDRO_INJECT_TAG");
            } else {
                std::env::set_var("HYDRO_INJECT_TAG", arm);
            }
            let mut t = temp;
            let mut r = rep;
            engine.restore_idle_hands();
            println!("--- seat={seat} arm={arm} ---");
            let text = generate_turn_ex(
                model,
                tokenizer,
                device,
                cfg,
                40,
                engine,
                logit_chain,
                hook_controls,
                hook_trace,
                &prompt,
                &prompt,
                &eos,
                true,
                &mut t,
                &mut r,
                1,
                0,
                false,
                scaler_receipt,
                &mut |_, _| repl_tui::Edits::default(),
            )?;
            let hands = engine.hands_report();
            let rec = serde_json::json!({
                "seat": seat,
                "arm": arm,
                "temperature": temp,
                "rep_penalty": rep,
                "t_after": t,
                "rep_after": r,
                "t_unchanged": (t - temp).abs() < 1e-12,
                "rep_unchanged": (r - rep).abs() < 1e-12,
                "top_k": top_k,
                "seed": seed,
                "text": text,
                "n_chars": text.chars().count(),
                "hands": hands,
            });
            writeln!(out, "{rec}")?;
            out.flush()?;
            println!(
                "    hands blend={} β={} σ={} Δh={} T/rep_unchanged={}/{}",
                hands["physics_blend"],
                hands["qsma_beta"],
                hands["kinetic_noise"],
                hands["delta_h_norm"],
                rec["t_unchanged"],
                rec["rep_unchanged"]
            );
            if arm == "none" {
                baseline_ids = Some(text.chars().collect());
                println!("    text={text:?}");
            } else if let Some(ref base) = baseline_ids {
                let cur: Vec<char> = text.chars().collect();
                let n = base.len().min(cur.len());
                let mut first_diff = None;
                for i in 0..n {
                    if base[i] != cur[i] {
                        first_diff = Some(i);
                        break;
                    }
                }
                if first_diff.is_none() && base.len() != cur.len() {
                    first_diff = Some(n);
                }
                let same = first_diff.is_none();
                println!(
                    "    vs_none={} first_diff_char={:?} text={text:?}",
                    if same { "IDENTICAL" } else { "CHANGED" },
                    first_diff
                );
            }
        }
    }
    std::env::remove_var("HYDRO_INJECT_TAG");
    std::env::remove_var("HYDRO_SAMPLE_SEED");
    println!("=== TAG ABLATION DONE ===");
    println!("    {}", out_path.display());
    Ok(())
}

/// Short Path B live proof: none vs spike, 24 tokens, physics must be ON.
/// T/rep stay at the seat baseline; the hand is blend/β/σ.
fn run_hands_smoke(
    model: &mut Model,
    tokenizer: &Tokenizer,
    device: &Device,
    cfg: &mut Config,
    engine: &mut NiodooEngine,
    logit_chain: &mut logit_physics::LogitChain,
    hook_controls: &mut hooks::HookControls,
    hook_trace: &mut Option<hooks::HookTrace>,
    scaler_receipt: &algo_scale::ScalerReceipt,
) -> Result<()> {
    use std::io::Write;
    if !engine.residual_enabled() {
        anyhow::bail!(
            "hands-smoke refuses physics-off (force_cap={})",
            engine.force_cap()
        );
    }
    let variant = model.variant_name();
    let eos = generation_eos_token_ids(variant, &cfg.generation.eos_token_ids);
    let prompt = format_multiturn_prompt_ex(
        &[(
            true,
            "Count from 1 to 20 using digits and spaces only.".to_string(),
        )],
        variant,
        true,
    );
    let out_path = Path::new("logs/hands_smoke.jsonl");
    if let Some(parent) = out_path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let mut out = std::fs::File::create(out_path)?;
    let temp = 0.8f64;
    let rep = 1.0f32;
    cfg.generation.temperature = temp;
    cfg.generation.rep_penalty = rep;
    cfg.generation.top_k = 40;
    cfg.generation.top_p = 1.0;
    std::env::set_var("HYDRO_SAMPLE_SEED", "1");
    println!("=== HANDS SMOKE (none vs spike, 24 tok, physics ON) ===");
    println!(
        "    force_cap={} residual_enabled={}",
        engine.force_cap(),
        engine.residual_enabled()
    );
    let mut none_text = String::new();
    for arm in ["none", "spike"] {
        if arm == "none" {
            std::env::remove_var("HYDRO_INJECT_TAG");
        } else {
            std::env::set_var("HYDRO_INJECT_TAG", arm);
        }
        let mut t = temp;
        let mut r = rep;
        engine.restore_idle_hands();
        println!("--- arm={arm} ---");
        let text = generate_turn_ex(
            model,
            tokenizer,
            device,
            cfg,
            24,
            engine,
            logit_chain,
            hook_controls,
            hook_trace,
            &prompt,
            &prompt,
            &eos,
            true,
            &mut t,
            &mut r,
            1,
            0,
            false,
            scaler_receipt,
            &mut |_, _| repl_tui::Edits::default(),
        )?;
        let hands = engine.hands_report();
        let rec = serde_json::json!({
            "arm": arm,
            "temperature": temp,
            "t_after": t,
            "rep_after": r,
            "t_unchanged": (t - temp).abs() < 1e-12,
            "rep_unchanged": (r - rep).abs() < 1e-12,
            "text": text,
            "hands": hands,
        });
        writeln!(out, "{rec}")?;
        out.flush()?;
        println!(
            "    blend={} β={} σ={} Δh={} T_unchanged={} text={text:?}",
            hands["physics_blend"],
            hands["qsma_beta"],
            hands["kinetic_noise"],
            hands["delta_h_norm"],
            rec["t_unchanged"]
        );
        if arm == "none" {
            none_text = text;
        } else {
            let changed = text != none_text;
            let blend_ok = hands["physics_blend"].as_f64().unwrap_or(0.0) > 6.0;
            println!(
                "    vs_none={} spike_blend_6.5={} qsma_beta={}",
                if changed { "CHANGED" } else { "IDENTICAL" },
                blend_ok,
                hands["qsma_beta"]
            );
        }
    }
    std::env::remove_var("HYDRO_INJECT_TAG");
    std::env::remove_var("HYDRO_SAMPLE_SEED");
    println!("=== HANDS SMOKE DONE {} ===", out_path.display());
    Ok(())
}

fn run_simple_chat(
    model: &mut Model,
    tokenizer: &tokenizers::Tokenizer,
    device: &Device,
    cfg: &Config,
    max_tokens: usize,
    engine: &mut NiodooEngine,
    logit_chain: &mut logit_physics::LogitChain,
    hook_controls: &mut hooks::HookControls,
    hook_trace: &mut Option<hooks::HookTrace>,
    hud: &mut hud::Hud,
    save_memory: bool,
    scaler_receipt: &algo_scale::ScalerReceipt,
    scaler_receipt_path: &Path,
) -> Result<()> {
    use std::io::Write;
    let variant = model.variant_name();
    let eos_token_ids = generation_eos_token_ids(variant, &cfg.generation.eos_token_ids);
    let mut operator_temperature = cfg.generation.temperature;
    let mut operator_rep_penalty = cfg.generation.rep_penalty;
    let top_k = cfg.generation.top_k;
    let top_p = cfg.generation.top_p;

    // Private transcript (never for public git)
    let chat_dir = std::path::Path::new("private/chats");
    let _ = std::fs::create_dir_all(chat_dir);
    let stamp = chrono_like_stamp();
    let transcript_path = chat_dir.join(format!("{stamp}_{variant}_chat.txt"));
    let mut transcript = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&transcript_path)
        .ok();
    if let Some(ref mut f) = transcript {
        let _ = writeln!(
            f,
            "# private chat — do not publish\n# variant={variant} force_cap={} T={} max_tokens={max_tokens} rep={} top_k={} top_p={}\n# scaler_receipt_id={} scaler_receipt={}\n",
            cfg.physics.force_cap,
            operator_temperature,
            operator_rep_penalty,
            top_k,
            top_p,
            scaler_receipt.receipt_id,
            scaler_receipt_path.display(),
        );
    }

    // Optional autonomic tags (tool-call style). Operator `reset` still clears history.
    let tags_on = true;
    eprintln!("\n=== Chat mode ({variant}) ===");
    eprintln!("Type messages. Empty line / quit / exit to stop.");
    eprintln!("  reset / clear  — wipe history (operator)");
    eprintln!("  /tui           — keyboard slider panel");
    eprintln!("  /phys          — print all live values");
    eprintln!("  quit / exit    — leave (no slash; /quit is a user message)");
    if tags_on {
        eprintln!("  tags           — model-emitted only; not a human menu");
    }
    eprintln!(
        "Physics: force_cap={} T={} max_tokens={} | gen: rep={} top_k={} top_p={}",
        cfg.physics.force_cap, operator_temperature, max_tokens, operator_rep_penalty, top_k, top_p
    );
    eprintln!(
        "Transcript (private/gitignored): {}\n",
        transcript_path.display()
    );

    let mut history: Vec<(bool, String)> = Vec::new();
    if let Ok(seed) = std::env::var("HYDRO_SEED_ASSISTANT") {
        let seed = seed.trim();
        if !seed.is_empty() {
            history.push((false, seed.to_string()));
            eprintln!(
                "    [SEED ASSISTANT] chars={} (visible history for unprimed-control arms)",
                seed.len()
            );
        }
    }
    let stdin = std::io::stdin();
    loop {
        if !official_pack_layout() {
            print!("you> ");
            let _ = std::io::stdout().flush();
        }
        let mut line = String::new();
        let n = stdin.read_line(&mut line)?;
        if n == 0 {
            println!();
            break;
        }
        let line = line.trim().to_string();
        if line.is_empty() || line.eq_ignore_ascii_case("quit") || line.eq_ignore_ascii_case("exit")
        {
            break;
        }
        if line.eq_ignore_ascii_case("reset") || line.eq_ignore_ascii_case("clear") {
            history.clear();
            println!("(history cleared)");
            if let Some(ref mut f) = transcript {
                let _ = writeln!(f, "# history reset");
            }
            continue;
        }
        // Live physics tuning — no rebuild, no restart.
        //   /tui                  keyboard slider panel
        //   /phys                 show sliders
        //   /set gov.brake 5      adjust (also accepts gov.brake=5)
        if line.eq_ignore_ascii_case("/tui") {
            let mut sliders = collect_live_sliders(
                engine,
                logit_chain,
                hook_controls,
                operator_temperature,
                operator_rep_penalty,
            );
            let result = tui::run_slider_tui(
                "HYDRODYNAMIC SWARM · LIVE PHYSICS",
                &mut sliders,
                |name, value| {
                    set_live_control(
                        name,
                        value,
                        engine,
                        logit_chain,
                        hook_controls,
                        &mut operator_temperature,
                        &mut operator_rep_penalty,
                    )
                },
            );
            match result {
                Ok(changes) if changes.is_empty() => println!("(no control changes)"),
                Ok(changes) => {
                    println!("(applied {} live control change(s))", changes.len());
                    if let Some(ref mut f) = transcript {
                        for (name, value) in changes {
                            let _ = writeln!(f, "# tui {name}={value}");
                        }
                    }
                }
                Err(error) => {
                    eprintln!("  TUI unavailable: {error}");
                    print!(
                        "{}",
                        render_live_controls(
                            engine,
                            logit_chain,
                            hook_controls,
                            operator_temperature,
                            operator_rep_penalty,
                        )
                    );
                }
            }
            let _ = std::io::stdout().flush();
            continue;
        }
        if line.eq_ignore_ascii_case("/phys") || line.eq_ignore_ascii_case("/sliders") {
            print!(
                "{}",
                render_live_controls(
                    engine,
                    logit_chain,
                    hook_controls,
                    operator_temperature,
                    operator_rep_penalty,
                )
            );
            let _ = std::io::stdout().flush();
            continue;
        }
        if let Some(rest) = line.strip_prefix("/set ") {
            match parse_set_arg(rest.trim()) {
                Some((name, value))
                    if set_live_control(
                        &name,
                        value,
                        engine,
                        logit_chain,
                        hook_controls,
                        &mut operator_temperature,
                        &mut operator_rep_penalty,
                    ) =>
                {
                    println!("  {name} = {value}");
                    print!(
                        "{}",
                        render_live_controls(
                            engine,
                            logit_chain,
                            hook_controls,
                            operator_temperature,
                            operator_rep_penalty,
                        )
                    );
                    if let Some(ref mut f) = transcript {
                        let _ = writeln!(f, "# set {name}={value}");
                    }
                }
                Some((name, _)) => {
                    println!("  unknown parameter '{name}'. /phys lists them.");
                }
                None => println!("  usage: /set <param> <value>   (e.g. /set gov.brake 5)"),
            }
            let _ = std::io::stdout().flush();
            continue;
        }

        if let Some(ref mut f) = transcript {
            let _ = writeln!(f, "you> {line}");
        }
        history.push((true, line.clone()));
        let prompt = format_multiturn_prompt_ex(&history, variant, tags_on);
        // Turn index = number of user turns so far (1-based in logs).
        let turn_idx = history.iter().filter(|(u, _)| *u).count();
        if variant == "gemma4" && tags_on && turn_idx == 1 {
            print_gemma4_control_channel_packing(&prompt);
        }
        if variant == "gemma4" && tags_on {
            print_prefill_see(&prompt, turn_idx);
        }
        let prev_assistant_len = history
            .iter()
            .rev()
            .find(|(u, _)| !*u)
            .map(|(_, t)| t.chars().count())
            .unwrap_or(0);
        let pack = official_pack_layout();
        if pack {
            print_official_turn_open(turn_idx, &line);
        } else {
            print!("{variant}> ");
            let _ = std::io::stdout().flush();
        }
        let t0 = Instant::now();
        hud.begin();
        let pieces = generate_turn_ex(
            model,
            tokenizer,
            device,
            cfg,
            max_tokens,
            engine,
            logit_chain,
            hook_controls,
            hook_trace,
            &prompt,
            history.last().map(|(_, t)| t.as_str()).unwrap_or(&prompt),
            &eos_token_ids,
            tags_on,
            &mut operator_temperature,
            &mut operator_rep_penalty,
            turn_idx,
            prev_assistant_len,
            save_memory,
            scaler_receipt,
            &mut |piece, frame| {
                hud.stream(piece).ok();
                hud.update(frame).ok();
                repl_tui::Edits::default()
            },
        )?;
        // Settle the footer before the next prompt so it does not sit in the
        // middle of the transcript.
        hud.finish().ok();
        println!();
        if official_pack_layout() {
            let label = official_prompt_label(turn_idx);
            println!(
                "[{} done  {:.1}s]",
                label.trim_end_matches('>'),
                t0.elapsed().as_secs_f64()
            );
        }
        let raw_reply = pieces.trim().to_string();
        // Transcript keeps FULL raw (channel thrash included) — research object.
        if let Some(ref mut f) = transcript {
            let _ = writeln!(f, "{variant}> {raw_reply}");
            let _ = f.flush();
        }
        // History keeps the raw mouth, including emitted tags. Niodoo leaves
        // tags in the stream so later attention can reaffirm the hand.
        let mut reply = raw_reply.clone();
        if variant == "gemma4" {
            reply = gemma4_history_clean(&reply);
        }
        if !reply.is_empty() {
            history.push((false, reply));
        }
    }
    eprintln!(
        "Chat ended ({} messages). Saved private: {}",
        history.len(),
        transcript_path.display()
    );
    let splat_file = Path::new("data/splat_memory.safetensors");
    if save_memory {
        persist_splat_store(engine, splat_file)?;
        eprintln!(
            "    [CHAT MEMORY] saved {} wills -> {}",
            engine.memory().len(),
            splat_file.display()
        );
    } else {
        eprintln!(
            "    [CHAT MEMORY] skip save ({} wills in RAM, --no-save-memory)",
            engine.memory().len()
        );
    }
    Ok(())
}

/// Local wall-clock stamp for private chat filenames (no extra crate).
fn chrono_like_stamp() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    // YYYYMMDD_HHMMSS approx via UTC epoch formatting is awkward without chrono;
    // use unix + local-ish: still unique and sort-friendly.
    format!("chat_{secs}")
}

/// `--d-run` seat policy. `main` applies this; tests drive the same function (no GPU).
#[derive(Debug, Clone, PartialEq)]
struct DRunSeatPolicy {
    d_run: bool,
    endocrine_enabled: bool,
    hooks_enabled: bool,
    max_tokens: usize,
    physics_required: bool,
    eos_masked: bool,
}

fn cli_flag(args: &[String], name: &str) -> bool {
    args.iter().any(|a| a == name)
}

fn cli_opt_usize(args: &[String], name: &str) -> Option<usize> {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1).and_then(|v| v.parse().ok()))
}

fn cli_opt_str(args: &[String], name: &str) -> Option<String> {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1).cloned())
}

fn hydro_tags_on() -> bool {
    match std::env::var("HYDRO_TAGS_ON") {
        Ok(v) => matches!(
            v.trim().to_ascii_lowercase().as_str(),
            "1" | "true" | "on" | "yes"
        ),
        Err(_) => true,
    }
}

/// Hidden-space direction add for John's c=0 / target / reverse / random arm.
/// Direct unembed row of `HYDRO_DIR_TOKEN` (or `HYDRO_DIR_FILE`), not a fitted J-lens.
struct DirSteer {
    mode: String,
    c: f32,
    token: String,
    vec: Tensor,
}

fn env_f32(name: &str, default: f32) -> f32 {
    std::env::var(name)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

fn l2_normalize_row(v: Tensor) -> Result<Tensor> {
    let n: f32 = v.sqr()?.sum_all()?.to_scalar::<f32>()?.sqrt();
    if n > 1e-8 {
        Ok(v.affine(1.0 / n as f64, 0.0)?)
    } else {
        Ok(v)
    }
}

fn encode_first_id(tokenizer: &Tokenizer, s: &str) -> Option<u32> {
    tokenizer
        .encode(s, false)
        .ok()
        .and_then(|e| e.get_ids().first().copied())
}

fn logit_rank(logits: &[f32], id: u32) -> Option<u32> {
    let i = id as usize;
    if i >= logits.len() {
        return None;
    }
    let t = logits[i];
    Some(1 + logits.iter().filter(|&&x| x > t).count() as u32)
}

fn load_dir_steer(
    model: &Model,
    tokenizer: &Tokenizer,
    device: &Device,
    dim: usize,
) -> Result<Option<DirSteer>> {
    let mode = std::env::var("HYDRO_DIR_MODE").unwrap_or_else(|_| "none".into());
    let mode = mode.trim().to_ascii_lowercase();
    if matches!(mode.as_str(), "" | "none" | "off" | "0") {
        return Ok(None);
    }
    let c = env_f32("HYDRO_DIR_C", 0.0);
    let token = std::env::var("HYDRO_DIR_TOKEN").unwrap_or_else(|_| "repetitive".into());
    let mut v = if let Ok(path) = std::env::var("HYDRO_DIR_FILE") {
        let raw = std::fs::read_to_string(&path)
            .map_err(|e| anyhow::anyhow!("HYDRO_DIR_FILE {path}: {e}"))?;
        let vals: Vec<f32> = serde_json::from_str(raw.trim()).map_err(|e| {
            anyhow::anyhow!("HYDRO_DIR_FILE {path} must be a JSON f32 array: {e}")
        })?;
        if vals.len() != dim {
            anyhow::bail!(
                "HYDRO_DIR_FILE len={} residual_d={dim}",
                vals.len()
            );
        }
        Tensor::from_vec(vals, (1, dim), device)?
    } else {
        let id = encode_first_id(tokenizer, &token).ok_or_else(|| {
            anyhow::anyhow!("HYDRO_DIR_TOKEN {token:?} produced no token id")
        })?;
        let emb = model.token_embeddings();
        let v_size = emb.dim(0)?;
        if (id as usize) >= v_size {
            anyhow::bail!("dir token id {id} >= vocab {v_size}");
        }
        let row = emb.narrow(0, id as usize, 1)?;
        let scale = model.embedding_input_scale();
        if (scale - 1.0).abs() > 1e-12 {
            row.affine(scale, 0.0)?
        } else {
            row
        }
    };
    v = l2_normalize_row(v)?;
    match mode.as_str() {
        "target" => {}
        "reverse" => {
            v = v.affine(-1.0, 0.0)?;
        }
        "random" => {
            let seed: u64 = std::env::var("HYDRO_DIR_SEED")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(1);
            let mut rng = StdRng::seed_from_u64(seed);
            let data: Vec<f32> = (0..dim).map(|_| rng.random::<f32>() * 2.0 - 1.0).collect();
            v = l2_normalize_row(Tensor::from_vec(data, (1, dim), device)?)?;
        }
        other => anyhow::bail!("HYDRO_DIR_MODE={other} (target|reverse|random|none)"),
    }
    eprintln!(
        "    [DIR STEER] mode={mode} c={c:.4} token={token:?} dim={dim} (direct unembed, not J-lens)"
    );
    Ok(Some(DirSteer {
        mode,
        c,
        token,
        vec: v,
    }))
}

/// Isolated Path B bench: one first-turn generate per JSONL row, no history,
/// hands restored to heartbeat after each item. Tags stay in the mouth.
fn run_eval_jsonl(
    model: &mut Model,
    tokenizer: &tokenizers::Tokenizer,
    device: &Device,
    cfg: &Config,
    max_tokens: usize,
    engine: &mut NiodooEngine,
    logit_chain: &mut logit_physics::LogitChain,
    hook_controls: &mut hooks::HookControls,
    hook_trace: &mut Option<hooks::HookTrace>,
    scaler_receipt: &algo_scale::ScalerReceipt,
    in_path: &Path,
    out_path: &Path,
) -> Result<()> {
    let variant = model.variant_name();
    let eos_token_ids = generation_eos_token_ids(variant, &cfg.generation.eos_token_ids);
    let tags_on = hydro_tags_on();
    let detect_only = engine.tags_detect_only
        || std::env::var("HYDRO_TAGS_DETECT_ONLY")
            .ok()
            .is_some_and(|v| {
                matches!(
                    v.trim().to_ascii_lowercase().as_str(),
                    "1" | "true" | "on" | "yes"
                )
            });
    let raw = std::fs::read_to_string(in_path)?;
    let mut items: Vec<(String, String)> = Vec::new();
    for (lineno, line) in raw.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let v: serde_json::Value = serde_json::from_str(line)
            .map_err(|e| anyhow::anyhow!("eval-jsonl {}:{}: {e}", in_path.display(), lineno + 1))?;
        let id = v
            .get("id")
            .and_then(|x| x.as_str())
            .map(|s| s.to_string())
            .unwrap_or_else(|| format!("item-{lineno}"));
        let prompt = v
            .get("prompt")
            .and_then(|x| x.as_str())
            .ok_or_else(|| anyhow::anyhow!("eval-jsonl line {} missing prompt", lineno + 1))?
            .to_string();
        items.push((id, prompt));
    }
    if let Some(parent) = out_path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let mut out = std::fs::OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(out_path)?;
    eprintln!(
        "[EVAL JSONL] n={} tags_on={tags_on} detect_only={detect_only} max_tokens={max_tokens} in={} out={}",
        items.len(),
        in_path.display(),
        out_path.display()
    );
    let mut operator_temperature = cfg.generation.temperature;
    let mut operator_rep_penalty = cfg.generation.rep_penalty;
    for (i, (id, prompt)) in items.iter().enumerate() {
        engine.reset_path_b_hands();
        let packed = format_multiturn_prompt_ex(&[(true, prompt.clone())], variant, tags_on);
        let t0 = Instant::now();
        let pieces = generate_turn_ex(
            model,
            tokenizer,
            device,
            cfg,
            max_tokens,
            engine,
            logit_chain,
            hook_controls,
            hook_trace,
            &packed,
            prompt,
            &eos_token_ids,
            tags_on,
            &mut operator_temperature,
            &mut operator_rep_penalty,
            1,
            0,
            false,
            scaler_receipt,
            &mut |_, _| repl_tui::Edits::default(),
        )?;
        let elapsed = t0.elapsed().as_secs_f64();
        let tags: Vec<String> = control_tags::scan_hits(&pieces)
            .into_iter()
            .map(|h| h.tag.as_str().to_string())
            .collect();
        let rec = serde_json::json!({
            "id": id,
            "prompt": prompt,
            "generation": pieces,
            "tags": tags,
            "physics_blend": engine.hands.physics_blend,
            "kinetic_noise": engine.hands.kinetic_noise,
            "beta": engine.hands.beta,
            "last_request": engine.hands.last_request,
            "detect_only": detect_only,
            "tags_on": tags_on,
            "elapsed_s": elapsed,
            "index": i,
        });
        writeln!(out, "{rec}")?;
        out.flush()?;
        eprintln!(
            "EVAL_ITEM i={}/{} id={id} tags={:?} blend={:.3} sigma={:.3} detect_only={detect_only} {:.1}s",
            i + 1,
            items.len(),
            tags,
            engine.hands.physics_blend,
            engine.hands.kinetic_noise,
            elapsed
        );
        engine.reset_path_b_hands();
    }
    eprintln!("[EVAL JSONL] done n={}", items.len());
    Ok(())
}

/// Shipped `--d-run` seat:
/// - endocrine forced **off** even without `--no-endocrine`
/// - residual physics **required**
/// - hooks **off**
/// - `--tokens N` is honored so a short diagnostic cannot become 131072
/// - bare `--d-run` still defaults to the 131k bar
fn d_run_seat_policy(
    args: &[String],
    default_max_tokens: usize,
    hooks_from_config: bool,
) -> DRunSeatPolicy {
    let d_run = cli_flag(args, "--d-run");
    let no_endocrine = cli_flag(args, "--no-endocrine");
    let require_physics_flag = cli_flag(args, "--require-physics");
    let tokens_explicit = cli_opt_usize(args, "--tokens");

    let mut max_tokens = tokens_explicit.unwrap_or(default_max_tokens).min(131_072);
    let mut endocrine_enabled = !no_endocrine;
    let mut hooks_enabled = hooks_from_config;
    let mut physics_required = require_physics_flag || max_tokens >= 131_072;
    let mut eos_masked = false;

    if d_run {
        endocrine_enabled = false;
        hooks_enabled = false;
        physics_required = true;
        eos_masked = true;
        if tokens_explicit.is_none() {
            max_tokens = 131_072;
        }
    }

    DRunSeatPolicy {
        d_run,
        endocrine_enabled,
        hooks_enabled,
        max_tokens,
        physics_required,
        eos_masked,
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    println!("=== SplatRAG v1 -- Hydrodynamic Swarm ===\n");

    // Parse CLI first so --config can choose the TOML before load.
    let args: Vec<String> = std::env::args().collect();
    let config_path = args
        .iter()
        .position(|a| a == "--config")
        .and_then(|i| args.get(i + 1).cloned())
        .unwrap_or_else(|| "config.toml".to_string());
    let mut cfg = Config::load(Path::new(&config_path))
        .map_err(|e| anyhow::anyhow!("configuration error in {config_path}: {e}"))?;
    println!("[*] Config: {}", config_path);

    let clear_memory = args.iter().any(|a| a == "--clear-memory");
    let no_save_memory = args.iter().any(|a| a == "--no-save-memory");
    // Ablation: skip IT chat wrap (raw user string only).
    let no_chat_template = args.iter().any(|a| a == "--no-chat-template");
    let cli_prompt = args
        .iter()
        .position(|a| a == "--prompt")
        .and_then(|i| args.get(i + 1).cloned());
    let cli_model = args
        .iter()
        .position(|a| a == "--model")
        .and_then(|i| args.get(i + 1).cloned());
    let cli_tokenizer = args
        .iter()
        .position(|a| a == "--tokenizer")
        .and_then(|i| args.get(i + 1).cloned());
    let seat = d_run_seat_policy(&args, cfg.generation.max_tokens, cfg.hooks.enabled);
    let d_run = seat.d_run;
    let max_tokens = seat.max_tokens;
    let d_binary_sha256 = if d_run {
        std::env::current_exe()
            .ok()
            .and_then(|p| {
                std::process::Command::new("sha256sum")
                    .arg(&p)
                    .output()
                    .ok()
                    .and_then(|o| {
                        String::from_utf8(o.stdout)
                            .ok()
                            .and_then(|s| s.split_whitespace().next().map(str::to_string))
                    })
            })
            .unwrap_or_else(|| "unknown".into())
    } else {
        String::new()
    };
    if d_run {
        println!(
            "[*] --d-run: tokens={}, EOS masked, physics required, endocrine forced OFF, hooks off, binary={}",
            max_tokens, d_binary_sha256
        );
    }
    let viz_enabled = args.iter().any(|a| a == "--viz");
    let chat_mode = args.iter().any(|a| a == "--chat");
    // Full-screen REPL: chat + live scalars + live sliders. --chat is unchanged.
    let tui_mode = args.iter().any(|a| a == "--tui");
    // 6-tab Ratatui frontend. Dry-run seat only this milestone (no GGUF load).
    if args
        .iter()
        .any(|a| a == "--ratatui" || a == "--tui-unified")
    {
        println!("[*] --ratatui: 6-tab frontend, dry-run Config/HookControls seat (no generate_turn_ex)");
        return frontend::run_ratatui_frontend(cfg, cli_model, true);
    }
    // Repeatable logit-physics override, applied after the chain is built:
    //   --set gov.brake=5 --set field.alpha=0.2
    // Sweeps and ablations use this instead of editing a TOML per arm; names match
    // the sliders printed at startup and the REPL's `/set`.
    let cli_set: Vec<String> = args
        .iter()
        .enumerate()
        .filter(|(_, a)| a.as_str() == "--set")
        .filter_map(|(i, _)| args.get(i + 1).cloned())
        .collect();
    // Live scalar HUD (sticky footer). ON by default on a TTY; --no-hud to skip.
    // `Hud::new` also disables itself when stdout is redirected, so piped runs
    // and ablation scripts are unaffected either way.
    let hud_enabled = !args.iter().any(|a| a == "--no-hud");
    let hud_every: usize = args
        .iter()
        .position(|a| a == "--hud-every")
        .and_then(|i| args.get(i + 1).and_then(|v| v.parse().ok()))
        .unwrap_or(1);
    // Model identity for the √-law readout: CLI > config > weights filename.
    let cli_model_params: Option<f32> = args
        .iter()
        .position(|a| a == "--model-params")
        .and_then(|i| args.get(i + 1).and_then(|v| v.parse().ok()));
    let cli_model_type = args
        .iter()
        .position(|a| a == "--model-type")
        .and_then(|i| args.get(i + 1).cloned());
    let cli_size_rule = args
        .iter()
        .position(|a| a == "--size-rule")
        .and_then(|i| args.get(i + 1).cloned());
    let cli_scaler_gain: Option<f32> = args
        .iter()
        .position(|a| a == "--scaler-gain")
        .and_then(|i| args.get(i + 1).and_then(|v| v.parse().ok()));
    let cli_scaler_apply = if args.iter().any(|a| a == "--apply-scaler") {
        Some(true)
    } else if args.iter().any(|a| a == "--no-apply-scaler") {
        Some(false)
    } else {
        None
    };
    // TermSplat live weather JSONL (FieldFrame). ON by default; --no-termsplat to skip.
    let termsplat_enabled = !args.iter().any(|a| a == "--no-termsplat");
    // Shep endocrine lane: ON by default for chat/oneshot. `--d-run` forces it OFF
    // (ablation receipt; IMMUTABLE_RUN_CONTRACT daily driver stays full-stack).
    let endocrine_enabled = seat.endocrine_enabled;
    // Optional: import a TCT-splat-lite store before the run (appends after safetensors load).
    let import_tct = args
        .iter()
        .position(|a| a == "--import-tct")
        .and_then(|i| args.get(i + 1).cloned());
    // Optional override path for TCT export (default: data/splat_memory.tct).
    let export_tct_path = args
        .iter()
        .position(|a| a == "--export-tct")
        .and_then(|i| args.get(i + 1).cloned());
    // SplatRAG pick → residual scar bridge (docs/BRIDGE_SPLATRAG_PICK.md).
    // Embeds pick **text** with this host; never injects semantics_64.
    let import_picks_path = args
        .iter()
        .position(|a| a == "--import-picks")
        .and_then(|i| args.get(i + 1).cloned());
    let picks_max_gain: f32 = args
        .iter()
        .position(|a| a == "--picks-max-gain")
        .and_then(|i| args.get(i + 1).and_then(|v| v.parse().ok()))
        .unwrap_or(picks::DEFAULT_PICKS_MAX_GAIN);
    let picks_dry_run = args.iter().any(|a| a == "--picks-dry-run");

    // Generation path requires NVIDIA CUDA (museum / unit tests do not).
    let device = Device::new_cuda(0).map_err(|e| {
        anyhow::anyhow!(
            "CUDA GPU required for generation (Device::new_cuda(0) failed: {e}).\n\
             \n\
             Fix:\n\
               • NVIDIA driver + CUDA toolkit 12+ on PATH (nvcc, libcuda)\n\
               • export CUDA_VISIBLE_DEVICES=0\n\
               • View-only path needs neither GPU nor model:  ./splat-lens museum\n\
             See SETUP.md"
        )
    })?;
    println!("[*] Using CUDA GPU (all tensors/physics on NVIDIA)");

    // =========================================================
    // Phase 1: Load GGUF (Gemma 4, Gemma 3, or Llama) + Tokenizer
    // =========================================================
    println!("\n--- Phase 1: Loading Model + Tokenizer ---");

    // Prefer --model, then local data/google links, then the ghost_team models desk.
    // Symlinks under data/google should point at /media/ruffianl/ghost_team/models/.
    let model_path = cli_model
        .filter(|path| Path::new(path).exists())
        .or_else(|| {
            find_existing_file(&[
                // Gemma 4 12B (lighter A0 / chat) — matches splat memory dim=3840
                "data/google/gemma-4-12b-it-Q4_K_M.gguf",
                "/media/ruffianl/ghost_team/models/gemma-4-12b-it-Q4_K_M.gguf",
                // Gemma 4 dense 31B (incident / A0 card-faithful class)
                "data/google/bart_google_gemma-4-31B-it-Q4_K_M.gguf",
                "/media/ruffianl/ghost_team/models/bart_google_gemma-4-31B-it-Q4_K_M.gguf",
                "data/google/unsloth_gemma-4-31B-it-Q4_K_M.gguf",
                "/media/ruffianl/ghost_team/models/unsloth_gemma-4-31B-it-Q4_K_M.gguf",
                // Gemma 3 4B (fast iteration + HUD)
                "data/google/gemma-3-4b-it-Q4_K_M.gguf",
                "/media/ruffianl/ghost_team/models/gemma-3-4b-it-q4_0.gguf",
                "data/google/gemma-3-27b-it-Q4_K_M.gguf",
                "data/google/gemma-3-27b-it-Q8_0.gguf",
                "data/gemma-3-27b-it-Q8_0.gguf",
                // gemma3n still needs its dedicated AltUp/Laurel loader
                "data/Meta-Llama-3.1-8B-Instruct-Q5_K_M.gguf",
                "data/google/Meta-Llama-3.1-8B-Instruct-Q5_K_M.gguf",
                "/media/ruffianl/ghost_team/models/Llama-3.1-8B-Q4_K_M.gguf",
            ])
        })
        .ok_or_else(|| {
            anyhow::anyhow!(
                "Required model file not found.\n\
                 Pass --model /path/to/model.gguf, or place a GGUF under data/google/,\n\
                 or use the models desk: /media/ruffianl/ghost_team/models/\n\
                 Target: data/google/bart_google_gemma-4-31B-it-Q4_K_M.gguf\n\
                 Download help: ./splat-lens check   (or see SETUP.md)\n\
                 View-only (no model): ./splat-lens museum"
            )
        })?;
    println!("    Model: {}", model_path);

    // Resolve equation inputs once. Applying the selected rule changes only
    // the residual force family; the receipt retains all historical formula
    // predictions and the unmodified TOML base profile.
    let (path_params, path_type) = algo_scale::infer_from_path(&model_path);
    let (params_b, params_source) = if let Some(p) = cli_model_params {
        (p, "cli:--model-params")
    } else if cfg.algo.params_b > 0.0 {
        (cfg.algo.params_b, "config:algo.params_b")
    } else if let Some(p) = path_params {
        (p, "model-filename")
    } else {
        anyhow::bail!(
            "model size unresolved for scaler receipt; set algo.params_b or --model-params <B>"
        );
    };
    let (model_type, type_source) = if let Some(raw) = cli_model_type.as_deref() {
        (
            algo_scale::ModelType::parse(raw)
                .ok_or_else(|| anyhow::anyhow!("invalid --model-type: {raw}"))?,
            "cli:--model-type",
        )
    } else if let Some(ty) = algo_scale::ModelType::parse(&cfg.algo.model_type) {
        (ty, "config:algo.model_type")
    } else {
        (path_type, "model-filename")
    };
    let env_rule = std::env::var("HYDRO_SIZE_RULE").ok();
    let (rule_raw, rule_source) = if let Some(raw) = cli_size_rule.as_deref() {
        (raw, "cli:--size-rule")
    } else if let Some(raw) = env_rule.as_deref() {
        (raw, "env:HYDRO_SIZE_RULE")
    } else {
        (cfg.algo.size_rule.as_str(), "config:algo.size_rule")
    };
    let size_rule = algo_scale::SizeRule::parse(rule_raw)
        .ok_or_else(|| anyhow::anyhow!("invalid scaler size rule: {rule_raw}"))?;
    let env_gain = std::env::var("HYDRO_SCALER_GAIN")
        .ok()
        .and_then(|v| v.parse::<f32>().ok());
    let (manual_gain, gain_source) = if let Some(gain) = cli_scaler_gain {
        (gain, "cli:--scaler-gain")
    } else if let Some(gain) = env_gain {
        (gain, "env:HYDRO_SCALER_GAIN")
    } else {
        (cfg.algo.gain, "config:algo.gain")
    };
    if !manual_gain.is_finite() || !(0.0..=4.0).contains(&manual_gain) {
        anyhow::bail!("scaler gain must be finite and in [0,4], got {manual_gain}");
    }
    let env_apply = std::env::var("HYDRO_SCALER_APPLY").ok().and_then(|v| {
        match v.trim().to_ascii_lowercase().as_str() {
            "1" | "true" | "on" | "yes" => Some(true),
            "0" | "false" | "off" | "no" => Some(false),
            _ => None,
        }
    });
    let (apply_scaler, apply_source) = if let Some(apply) = cli_scaler_apply {
        (apply, "cli:apply-scaler")
    } else if let Some(apply) = env_apply {
        (apply, "env:HYDRO_SCALER_APPLY")
    } else {
        (cfg.algo.apply, "config:algo.apply")
    };
    let scaler_base_profile = algo_scale::SeatProfile::from_config(&cfg);
    let selected_prediction = algo_scale::transform_prediction(
        params_b,
        model_type,
        size_rule,
        scaler_base_profile.temperature,
    );
    let scaler_cross_check = algo_scale::SizeRule::ALL
        .iter()
        .map(|rule| {
            algo_scale::transform_prediction(
                params_b,
                model_type,
                *rule,
                scaler_base_profile.temperature,
            )
        })
        .collect::<Vec<_>>();
    let effective_residual_gain = algo_scale::apply_to_hydro_profile(
        &mut cfg,
        &selected_prediction,
        manual_gain,
        apply_scaler,
    );
    let mut scaler_overrides = vec![
        algo_scale::ResolvedValue {
            name: "params_b".into(),
            value: params_b.to_string(),
            source: params_source.into(),
            applied: true,
        },
        algo_scale::ResolvedValue {
            name: "archetype".into(),
            value: model_type.as_str().into(),
            source: type_source.into(),
            applied: true,
        },
        algo_scale::ResolvedValue {
            name: "size_rule".into(),
            value: size_rule.as_str().into(),
            source: rule_source.into(),
            applied: true,
        },
        algo_scale::ResolvedValue {
            name: "manual_gain".into(),
            value: manual_gain.to_string(),
            source: gain_source.into(),
            applied: apply_scaler,
        },
        algo_scale::ResolvedValue {
            name: "apply".into(),
            value: apply_scaler.to_string(),
            source: apply_source.into(),
            applied: true,
        },
    ];
    println!(
        "    Scaler: ~{params_b:.0}B {} · rule={} size={:.3} archetype×{:.3} force={:.3} k={manual_gain:.3} applied_gain={effective_residual_gain:.3}",
        model_type.as_str(),
        size_rule.as_str(),
        selected_prediction.size_scale,
        selected_prediction.archetype_multiplier,
        selected_prediction.force_intensity,
    );
    println!(
        "            native σ={:.3} θ={:.3} β={:.1} repulsion={:.3} predicted_T={:.3}; Hydro T/ramp/logit/gov frozen",
        selected_prediction.sigma,
        selected_prediction.theta,
        selected_prediction.beta,
        selected_prediction.loop_repulsion,
        selected_prediction.predicted_temperature,
    );

    let loaded = loader::load_gguf(&model_path, cli_tokenizer, &device, true)?;
    let loader::Loaded {
        mut model,
        tokenizer,
        is_gemma4: load_gemma4,
        is_gemma3: load_gemma3,
        ..
    } = loaded;

    // John A0 live gate: static row-validity + finite first-token logits at SWA boundary.
    // Usage: … --a0-swa-check   (prefer Gemma 4 for window=1024; 4B still checks finite)
    if args.iter().any(|a| a == "--a0-swa-check") {
        return run_a0_swa_check(&mut model, &device, load_gemma4);
    }

    // =========================================================
    // Phase 2: Build live Diderot field from model embeddings
    // =========================================================
    // Steering map: field positions must sit on the same shell the stack
    // actually walks. Gemma multiplies tok_embeddings by √hidden_dim before
    // layer 0 (gemma.rs / gemma4.rs). Bloom mean-pool already applies that
    // scale; the field used to use raw rows (||μ||~O(1)) while residual
    // steering lives after √d (and the stack). Same dual-map bug as loading
    // a 26B@2816 universe onto 4B@2560 — wrong geometry, not "turn systems off."
    // Llama keeps scale=1 (raw rows are already pre-layer space).
    println!("\n--- Phase 2: Building Diderot Field ---");
    let emb_raw = model.token_embeddings();
    let emb_scale = model.embedding_input_scale();
    let emb_for_field = if (emb_scale - 1.0).abs() > 1e-12 {
        println!(
            "    Map shell: Gemma pre-layer scale √d = {:.4} (field matches forward, not raw table)",
            emb_scale
        );
        emb_raw.affine(emb_scale, 0.0)?
    } else {
        println!("    Map shell: raw tok_embeddings (scale=1, Llama-class)");
        emb_raw.clone()
    };
    let field = ContinuousField::from_embeddings(&emb_for_field, &device)?;
    let dim = field.dim;
    // Live residual width from GGUF-backed embeddings — single source of truth for asserts.
    let inventory_variant = if load_gemma4 {
        "gemma4"
    } else if load_gemma3 {
        "gemma3"
    } else {
        "llama"
    };
    dim_assert::log_startup_inventory(
        dim,
        inventory_variant,
        &model_path,
        &cfg.physics,
        &cfg.logit_physics,
        &cfg.hooks,
        &cfg.jacobian,
    );

    // =========================================================
    // Phase 3: Niodoo Engine + Shared Ocean (Lane C)
    // =========================================================
    println!("\n--- Phase 3: Niodoo Steering Engine ---");
    let mut memory = SplatMemory::new(device.clone());
    memory.set_residual_dim(dim);
    let backend = gpu::select_backend();
    let mut engine = NiodooEngine::new(
        field,
        memory,
        backend,
        cfg.physics.dt,
        cfg.physics.viscosity_scale,
        cfg.physics.force_cap,
    );
    if cfg.physics.gradient_topk > 0 {
        engine.set_gradient_topk(cfg.physics.gradient_topk);
    }
    engine.set_splat_force_limits(cfg.physics.splat_force_scale, cfg.physics.splat_force_max);
    engine.set_goal_force_limits(cfg.physics.goal_force_scale, cfg.physics.goal_force_max);
    engine.set_goal_late_attenuate(
        cfg.physics.goal_late_start,
        cfg.physics.goal_late_span,
        cfg.physics.goal_late_end,
    );
    let wake_mode = FieldWakeMode::parse(&cfg.physics.field_wake_mode);
    engine.set_field_wake(FieldWakeConfig {
        mode: wake_mode,
        k: cfg.physics.field_wake_k.max(1),
        scale: cfg.physics.field_wake_scale,
        max_mag: cfg.physics.field_wake_max,
        grad_blend: cfg.physics.field_grad_blend,
        dist_tau: cfg.physics.field_wake_dist_tau,
    });
    engine.set_force_ramp(cfg.physics.force_ramp_tokens, cfg.physics.force_ramp_start);
    engine.set_memory_warm_pot(cfg.physics.memory_warm_pot);
    engine.set_topic_mix(cfg.physics.topic_mix);
    let mem_pick = MemoryPickConfig {
        mode: MemoryForceMode::parse(&cfg.memory.memory_force_mode),
        k: cfg.memory.memory_pick_k.max(1),
        selective: cfg.memory.memory_pick_selective,
        entropy_min: cfg.memory.memory_pick_entropy_min,
        margin_max: cfg.memory.memory_pick_margin_max,
        residual_l2_min: cfg.memory.memory_pick_residual_l2_min,
        quality_weight: cfg.memory.memory_pick_quality_weight,
        fp_weight: cfg.memory.memory_pick_fp_weight,
    };
    engine.set_memory_pick(mem_pick);
    let remember_path = std::env::var("HYDRO_REMEMBER_STORE")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| {
            std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("logs/seat_remember.jsonl")
        });
    engine.open_remember_store(&remember_path);
    println!(
        "    Remember store: {} ({} keys)",
        remember_path.display(),
        engine.remember_len()
    );
    let require_physics = seat.physics_required;
    if require_physics && !engine.residual_enabled() {
        anyhow::bail!(
            "physics OFF (force_cap={}); refusing silent physics-off (need --require-physics / 131k with cap>0)",
            engine.force_cap()
        );
    }
    println!(
        "    Memory force: mode={} k={} selective={} (soft=legacy sum-all)",
        MemoryForceMode::parse(&cfg.memory.memory_force_mode).as_str(),
        cfg.memory.memory_pick_k.max(1),
        cfg.memory.memory_pick_selective,
    );

    // ── Logit-surface physics chain ────────────────────────────────────────
    // All three engines are registered unconditionally so they can be switched on
    // mid-session from the chat REPL (`/set`) without a rebuild. Each abstains when
    // its own strength is 0, so an all-zero config leaves the logits bit-identical
    // to a build without this module.
    let logit_engines: Vec<Box<dyn logit_physics::LogitEngine>> = vec![
        Box::new(logit_physics::FieldBias::new(cfg.logit_physics.field_alpha)),
        Box::new(logit_physics::SplatBias::new(
            cfg.logit_physics.splat_scale,
            cfg.logit_physics.splat_top_m,
            cfg.logit_physics.splat_top_k,
        )),
        Box::new(logit_physics::Governor::new(
            cfg.logit_physics.governor_enabled,
            cfg.logit_physics.governor_velocity_threshold,
            cfg.logit_physics.governor_brake,
            cfg.logit_physics.governor_window,
            cfg.logit_physics.governor_viscosity_threshold,
            cfg.logit_physics.governor_viscosity_gain,
            cfg.logit_physics.governor_max_bias,
        )),
        // Backslash penalty — breaks the `\` loop collapse mode observed at step 93.
        // Token ID 621 confirmed from log: step 24, token_id=621, token_text=' \\'.
        // Penalty=0.0 = off (no-op). Default in config is 2.0.
        Box::new(logit_physics::BackslashPenalty::new(
            cfg.logit_physics.backslash_penalty,
            621,
        )),
    ];
    let mut logit_chain = logit_physics::LogitChain::new(logit_engines);
    let hook_site = hooks::HookSite::parse(&cfg.hooks.site)
        .ok_or_else(|| anyhow::anyhow!("invalid hooks.site: {}", cfg.hooks.site))?;
    let mut hook_controls = hooks::HookControls::new(
        cfg.hooks.enabled,
        hook_site,
        cfg.hooks.start_frac,
        cfg.hooks.end_frac,
        cfg.hooks.norm_fraction,
    );
    let mut hook_trace = if cfg.hooks.trace_out.trim().is_empty() {
        None
    } else {
        let path = Path::new(&cfg.hooks.trace_out);
        println!("    Hook trace: {}", path.display());
        Some(hooks::HookTrace::open(path)?)
    };
    for kv in &cli_set {
        match parse_set_arg(kv) {
            Some((name, value))
                if engine.set_live_param(&name, value)
                    || logit_chain.set_param(&name, value)
                    || hook_controls.set_param(&name, value) =>
            {
                println!("    [--set] {name} = {value}");
                scaler_overrides.push(algo_scale::ResolvedValue {
                    name,
                    value: value.to_string(),
                    source: "cli:--set".into(),
                    applied: true,
                });
            }
            Some((name, value)) => {
                eprintln!("    [--set] unknown parameter '{name}' — ignored");
                scaler_overrides.push(algo_scale::ResolvedValue {
                    name,
                    value: value.to_string(),
                    source: "cli:--set".into(),
                    applied: false,
                });
            }
            None => {
                eprintln!("    [--set] expected name=value, got '{kv}' — ignored");
                scaler_overrides.push(algo_scale::ResolvedValue {
                    name: kv.clone(),
                    value: String::new(),
                    source: "cli:--set".into(),
                    applied: false,
                });
            }
        }
    }
    // Day 49: transformer hooks soup long decode; residual physics (force_cap) stays ON.
    // `--d-run` seat also forces hooks off after --set so sliders cannot re-enable them.
    if d_run {
        hook_controls.enabled = seat.hooks_enabled;
        println!("    [--d-run] hooks off (Day 49 soup prevention); residual/QSMA stay ON; endocrine OFF");
    }

    // Finalize exactly once, after startup overrides and before request 1.
    // Runtime `/set` changes are separate trajectory events and never mutate
    // this receipt.
    let scaler_final_profile = live_seat_profile(&cfg, &engine, &logit_chain);
    let binary_path = std::env::current_exe().unwrap_or_else(|_| PathBuf::from("unknown"));
    let binary_sha256 = if binary_path.as_os_str() == "unknown" {
        "unavailable".into()
    } else {
        sha256_file(&binary_path)
    };
    let model_sha256 = sha256_file(Path::new(&model_path));
    let created_unix_ms = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis();
    let receipt_id = format!(
        "scaler-{}-{}-k{:.3}-{}-{}-{}",
        size_rule.as_str(),
        params_b,
        manual_gain,
        &model_sha256[..model_sha256.len().min(12)],
        &binary_sha256[..binary_sha256.len().min(12)],
        created_unix_ms,
    );
    let model_layers = model.n_layers();
    let (hook_start, hook_end) = hook_controls.band.resolve(model_layers);
    let hook_applications =
        if !hook_controls.enabled || hook_controls.norm_fraction <= 0.0 || model_layers == 0 {
            0
        } else if hook_controls.site == hooks::HookSite::FinalNorm {
            usize::from(hook_controls.band.contains(model_layers - 1, model_layers))
        } else {
            hook_end.saturating_sub(hook_start) + 1
        };
    let prompt_input_path = std::env::var("PROMPTS_FILE").ok().map(PathBuf::from);
    let run_context = algo_scale::RunContext {
        config: file_snapshot("config", Some(Path::new(&config_path))),
        prompt_input: file_snapshot("prompt_input", prompt_input_path.as_deref()),
        memory_inputs: vec![
            file_snapshot(
                "splat_memory",
                Some(Path::new("data/splat_memory.safetensors")),
            ),
            file_snapshot("remember_store", Some(&remember_path)),
        ],
        memory_clear_requested: clear_memory,
        sample_seed: std::env::var("HYDRO_SAMPLE_SEED").unwrap_or_else(|_| "unset".into()),
        max_tokens,
        official_pack_layout: official_pack_layout(),
        chat_template: format!("{}-canonical-multiturn", model.variant_name()),
        control_tags_enabled: true,
        tda_monitor_enabled: tda_monitor::tda_monitor_enabled(),
        tda_window_tokens: 32,
        tda_stride_tokens: 8,
        tda_cooldown_tokens: 48,
    };
    let hook_profile = algo_scale::HookProfile {
        enabled: hook_controls.enabled,
        site: hook_controls.site.as_str().into(),
        start_frac: hook_controls.band.start_frac,
        end_frac: hook_controls.band.end_frac,
        norm_fraction: hook_controls.norm_fraction,
        model_layers,
        resolved_start_layer: hook_start,
        resolved_end_layer: hook_end,
        applications_per_decode: hook_applications,
    };
    let scaler_receipt = algo_scale::ScalerReceipt {
        schema: "hydro.scaler-receipt/v3".into(),
        receipt_id: receipt_id.clone(),
        created_unix_ms,
        model_path: model_path.clone(),
        model_sha256,
        binary_path: binary_path.display().to_string(),
        binary_sha256,
        params_b,
        archetype: model_type.as_str().into(),
        transform_id: size_rule,
        apply_to_residual_seat: apply_scaler,
        manual_gain,
        effective_residual_gain,
        selected_prediction: selected_prediction.clone(),
        cross_check: scaler_cross_check,
        adapter_id: "hydro-residual-profile-relative/v1".into(),
        adapter_notes: vec![
            "Multiply the TOML residual cap, field, splat, goal, and their ceilings by selected force_intensity × manual_gain.".into(),
            "Force ramp, sampling temperature, logit field/splat, and governor coefficients remain frozen for matched-panel isolation.".into(),
            "Formula-native sigma/theta/beta/repulsion are predictions, not aliases for Hydro residual.cap or Niodoo gravity/ghost telemetry.".into(),
        ],
        resolved_inputs_and_overrides: scaler_overrides,
        run_context,
        hook_profile,
        base_profile: scaler_base_profile,
        final_applied_coefficients: scaler_final_profile.clone(),
    };
    let scaler_receipt_path = write_scaler_receipt(&scaler_receipt)?;
    println!("    Scaler receipt: {}", scaler_receipt_path.display());
    println!("    Scaler receipt id: {}", scaler_receipt.receipt_id);
    let algo_view = Some(hud::AlgoView::from_applied(
        params_b,
        model_type,
        &selected_prediction,
        &scaler_final_profile,
    ));
    print!(
        "{}",
        render_live_controls(
            &engine,
            &logit_chain,
            &hook_controls,
            cfg.generation.temperature,
            cfg.generation.rep_penalty,
        )
    );

    // ── Shep endocrine: worker = text enzyme; geometry = native tok_embeddings ──
    let (_shared_universe, endocrine_tx, mut endocrine_rx) = if endocrine_enabled {
        let (universe, tx, rx) = endocrine::create_endocrine_system();
        println!("    Endocrine: ON (enzyme idle until signal · native embed · no TinyEmbed)");
        (Some(universe), Some(tx), Some(rx))
    } else if d_run {
        println!("    Endocrine: OFF (--d-run)");
        (None, None, None)
    } else {
        println!("    Endocrine: OFF (--no-endocrine)");
        (None, None, None)
    };
    let mut last_endocrine_signal_step: isize = -999;

    println!(
        "    Engine ready (backend: {}, Top-K: {}, F_s={}/{}, F_a={}/{})",
        engine.backend_name(),
        cfg.physics.gradient_topk,
        cfg.physics.splat_force_scale,
        cfg.physics.splat_force_max,
        cfg.physics.goal_force_scale,
        cfg.physics.goal_force_max
    );
    if cfg.physics.force_ramp_tokens > 0 {
        println!(
            "    Force ramp: first {} tokens from {:.2} → 1.0 (J-space respect)",
            cfg.physics.force_ramp_tokens, cfg.physics.force_ramp_start
        );
    }
    if cfg.physics.goal_late_start > 0 {
        println!(
            "    Late F_a attenuate: after step {} → ×{:.2} over {} tok (early goal intact)",
            cfg.physics.goal_late_start, cfg.physics.goal_late_end, cfg.physics.goal_late_span
        );
    }
    if cfg.physics.targeted_splat_only {
        println!("    Targeted splats: ON (high-δ or strong quality only)");
    }
    println!(
        "    Field wake: mode={} k={} scale={} max={} blend={} τ={}",
        wake_mode.as_str(),
        cfg.physics.field_wake_k,
        cfg.physics.field_wake_scale,
        cfg.physics.field_wake_max,
        cfg.physics.field_grad_blend,
        cfg.physics.field_wake_dist_tau
    );
    if cfg.logit_physics.field_alpha > 0.0 {
        println!(
            "    Field logit bias: α={}  (z += α·norm(E û_g) pre-softmax)",
            cfg.logit_physics.field_alpha
        );
    } else {
        println!("    Field logit bias: off");
    }

    // Shared ocean: multi-mind field packets. Full stack stays ON (IMMUTABLE_RUN_CONTRACT).
    // Single-host was soft-yanking (F_ocean ~15–24) and shredding text — keep enabled but
    // whisper-scale until multi-mind peers exist. force_cap=0 also mutes ocean force.
    let mut ocean_cfg = OceanConfig::default();
    if dim >= 4096 {
        // Large residual dim: still on, but even quieter (was hard-off; that violated full-stack).
        ocean_cfg.force_scale = 0.02;
        ocean_cfg.deposit_interval = 8;
    } else {
        // 4B-class: path on, force whisper (default 0.12 was destroying long-form).
        ocean_cfg.force_scale = 0.015;
        ocean_cfg.deposit_interval = 8;
    }
    if cfg.physics.force_cap < 1e-8 {
        ocean_cfg.force_scale = 0.0; // force-off control: no ocean shove; deposits may still log
    }
    let ocean = SharedOcean::new(dim, device.clone(), ocean_cfg.clone());
    engine.set_ocean(ocean);
    if ocean_cfg.enabled {
        println!(
            "    Shared Ocean online (dim={}, deposit every {}, force_scale={})",
            dim, ocean_cfg.deposit_interval, ocean_cfg.force_scale
        );
    } else {
        println!("    Shared Ocean: disabled in OceanConfig");
    }

    // --- TCT import BEFORE clear, so --import-tct files survive --clear-memory ---
    let mut tct_import_count = 0usize;
    if let Some(ref tct_in) = import_tct {
        match engine.memory_mut().import_tct(Path::new(tct_in)) {
            Ok(n) => {
                tct_import_count = n;
                println!(
                    "    Imported {} TCT records (total learned wills={})",
                    n,
                    engine.memory().len()
                );
            }
            Err(e) => eprintln!("    [TCT] import failed: {e}"),
        }
    }

    // Load persistent splat memory if it exists
    let splat_file = Path::new("data/splat_memory.safetensors");
    if clear_memory && splat_file.exists() {
        std::fs::remove_file(splat_file)?;
        // Also clear TCT companions so --clear-memory is a true process death for both formats.
        let _ = std::fs::remove_file("data/splat_memory.tct");
        let _ = std::fs::remove_file("data/splat_memory.tct.json");
        println!("    Cleared splat memory (--clear-memory)");
    }
    let loaded_count = engine.memory_mut().load(splat_file)?;
    if loaded_count == 0 && !clear_memory {
        println!("    No existing splat memory found (first run)");
    } else if loaded_count > 0 {
        println!(
            "    Loaded {} splats from {}",
            loaded_count,
            splat_file.display()
        );
    }
    // Continuity: drop legacy pain prefill-bridges (failed-gen deposits).
    let pain_dropped = engine.memory_mut().drop_pain_prefill_bridges();
    if pain_dropped > 0 {
        println!(
            "    [BRIDGE] dropped {} pain prefill-bridge(s) (pleasure-only continuity)",
            pain_dropped
        );
    }

    // --- SplatRAG pick import (text → residual scar) ---
    // Must run after model+tokenizer+memory load; before chat/generation so scars
    // are already in the field when prefill measures nearest_L2.
    if let Some(ref picks_path) = import_picks_path {
        let set = picks::load_pick_set(Path::new(picks_path))?;
        let replace_dist = (cfg.physics.prefill_bridge_sigma
            * (1.0 + cfg.physics.prefill_bridge_offset_frac.abs()))
        .max(cfg.memory.consolidation_dist);
        let opts = picks::ImportPicksOpts {
            max_gain: picks_max_gain,
            dry_run: picks_dry_run,
            gain_eps: 1e-6,
            sigma: cfg.physics.prefill_bridge_sigma,
            lambda: cfg.physics.prefill_bridge_lambda,
            offset_frac: cfg.physics.prefill_bridge_offset_frac,
            replace_dist,
        };
        let device_for_pick = device.clone();
        let report = picks::import_picks(
            &set,
            Path::new(picks_path),
            engine.memory_mut(),
            &opts,
            |text| {
                let encoded = tokenizer
                    .encode(text, true)
                    .map_err(|e| anyhow::anyhow!("pick tokenize: {e}"))?;
                let ids = encoded.get_ids();
                if ids.is_empty() {
                    anyhow::bail!("pick embed: empty token sequence");
                }
                let tokens = Tensor::new(ids, &device_for_pick)?.unsqueeze(0)?;
                // Fresh prefill of memory text → last-token residual (scar space).
                let (_logits, hidden) = model.forward_with_hidden(&tokens, 0)?;
                Ok(hidden.squeeze(0)?)
            },
        )?;
        picks::print_report(&report);
        if !picks_dry_run {
            let dropped = engine
                .memory_mut()
                .enforce_max_prefill_bridges(cfg.memory.max_prefill_bridges);
            if dropped > 0 {
                println!(
                    "    [PICKS] enforce_max_prefill_bridges dropped {dropped} (cap={})",
                    cfg.memory.max_prefill_bridges
                );
            }
            println!(
                "    [PICKS] learned wills now={} bridges={}",
                engine.memory().len(),
                engine.memory().count_prefill_bridges()
            );
        }
        println!(
            "    [PICKS] rule: semantics_64 is telemetry only; μ from host residual of pick.text"
        );
    }

    let scars_at_start = engine.memory().len();
    let memory_loaded = scars_at_start > 0;
    let n_prefill_bridges_start = engine.memory().count_prefill_bridges();

    // =========================================================
    // Chat mode (--chat): multi-turn stdin for Gemma 3/4 (and Llama via simple path)
    // =========================================================
    if tui_mode {
        if !repl_tui::App::fits() {
            eprintln!("    Terminal too small for --tui — falling back to --chat.");
        } else {
            return run_tui_chat(
                &mut model,
                &tokenizer,
                &device,
                &cfg,
                max_tokens,
                &mut engine,
                &mut logit_chain,
                &mut hook_controls,
                &mut hook_trace,
                algo_view,
                &scaler_receipt,
            );
        }
    }

    if args.iter().any(|a| a == "--tag-ablation") {
        return run_tag_ablation(
            &mut model,
            &tokenizer,
            &device,
            &mut cfg,
            &mut engine,
            &mut logit_chain,
            &mut hook_controls,
            &mut hook_trace,
            &scaler_receipt,
        );
    }
    if args.iter().any(|a| a == "--hands-smoke") {
        return run_hands_smoke(
            &mut model,
            &tokenizer,
            &device,
            &mut cfg,
            &mut engine,
            &mut logit_chain,
            &mut hook_controls,
            &mut hook_trace,
            &scaler_receipt,
        );
    }

    if let Some(eval_in) = cli_opt_str(&args, "--eval-jsonl") {
        let eval_out = cli_opt_str(&args, "--eval-out").unwrap_or_else(|| {
            let p = PathBuf::from(&eval_in);
            let stem = p.file_stem().and_then(|s| s.to_str()).unwrap_or("eval");
            format!("logs/{stem}.generations.jsonl")
        });
        return run_eval_jsonl(
            &mut model,
            &tokenizer,
            &device,
            &cfg,
            max_tokens,
            &mut engine,
            &mut logit_chain,
            &mut hook_controls,
            &mut hook_trace,
            &scaler_receipt,
            Path::new(&eval_in),
            Path::new(&eval_out),
        );
    }

    if chat_mode || tui_mode {
        // Prefer simple multi-turn for all variants so Jason can talk to Gemma now.
        // (Old Llama TUI still available if we re-special-case later.)
        println!(
            "    (/tui opens live sliders; /phys prints them; /set <param> <value> adjusts one)"
        );
        return run_simple_chat(
            &mut model,
            &tokenizer,
            &device,
            &cfg,
            max_tokens,
            &mut engine,
            &mut logit_chain,
            &mut hook_controls,
            &mut hook_trace,
            &mut hud::Hud::new(hud_enabled, hud_every, algo_view),
            !no_save_memory,
            &scaler_receipt,
            &scaler_receipt_path,
        );
    }

    let mut hud = hud::Hud::new(hud_enabled, hud_every, algo_view);

    // Initialize telemetry logger
    let model_variant = model.variant_name();
    let is_gemma = model.is_gemma();
    let raw_prompt = cli_prompt
        .as_deref()
        .unwrap_or(cfg.generation.default_prompt.as_str());
    let prompt = if no_chat_template {
        raw_prompt.to_string()
    } else {
        format_prompt_for_model(raw_prompt, model_variant)
    };
    let eos_token_ids = generation_eos_token_ids(model_variant, &cfg.generation.eos_token_ids);
    let test_label = format!(
        "{}_v3-forcecap{}_T{}_s{}_a{}_d{}",
        model_variant,
        cfg.physics.force_cap as i32,
        cfg.generation.temperature,
        cfg.physics.splat_sigma as i32,
        cfg.physics.splat_alpha as i32,
        cfg.physics.min_splat_dist as i32,
    );
    let mut logger = SessionLogger::new(&test_label, model_variant)?;
    let mut weather = if termsplat_enabled {
        match weather::WeatherPipe::open_beside_log(logger.path().as_path(), dim) {
            Ok(w) => Some(w),
            Err(e) => {
                eprintln!("    [weather] could not open TermSplat pipe: {e}");
                None
            }
        }
    } else {
        println!("    TermSplat weather: OFF (--no-termsplat)");
        None
    };
    let session_prompt_fp = tct::continuity_fp(raw_prompt);
    engine.set_prompt_fp(session_prompt_fp);
    let bridge_fps_start = engine.memory().list_bridge_prompt_fps();
    println!(
        "    Memory session: loaded={} wills_start={} bridges={} fps={:?} prompt_fp={:#x} (safetensors={} tct_import={}) clear={} ramp={}",
        memory_loaded,
        scars_at_start,
        n_prefill_bridges_start,
        bridge_fps_start
            .iter()
            .map(|f| format!("{:#x}", f))
            .collect::<Vec<_>>(),
        session_prompt_fp,
        loaded_count,
        tct_import_count,
        clear_memory,
        cfg.physics.force_ramp_tokens
    );

    // =========================================================
    // Phase 4: Real Prompt -> Physics-Steered Generation
    // =========================================================
    println!("\n--- Phase 4: Physics-Steered Generation ---");
    println!("    Prompt: \"{}\"", raw_prompt);
    if no_chat_template {
        println!("    Chat template: OFF (--no-chat-template, raw prompt)");
    } else {
        match model_variant {
            "gemma4" => {
                println!(
                    "    Chat template: Gemma 4 IT turns (<|turn>… from gemma4_assets) \
                     + god-tier control-channel system turn (available tags table)"
                );
                println!(
                    "    Prefill packing: {} chars → will encode to token ids next",
                    prompt.len()
                );
                print_gemma4_control_channel_packing(&prompt);
            }
            "gemma3" => println!("    Chat template: Gemma 3 IT turns (<start_of_turn>…)"),
            _ => {}
        }
    }

    // Encode prompt (no trailing EOS — see encode_prompt_no_trailing_eos)
    let prompt_ids: Vec<u32> =
        encode_prompt_no_trailing_eos(&tokenizer, prompt.as_str(), &eos_token_ids)?;
    println!(
        "    Prefill tokens: {} ids | first3={:?} last5={:?} | pos_ids=0..{}",
        prompt_ids.len(),
        &prompt_ids[..prompt_ids.len().min(3)],
        &prompt_ids[prompt_ids.len().saturating_sub(5)..],
        prompt_ids.len().saturating_sub(1)
    );
    println!("    Prompt tokens: {} IDs", prompt_ids.len());

    // Prefill
    let prompt_tensor = Tensor::new(prompt_ids.as_slice(), &device)?.unsqueeze(0)?;
    println!("    Prefilling {} prompt tokens...", prompt_ids.len());
    model.clear_kv_cache();

    // Use forward_with_hidden when steer_hidden is enabled
    let (prefill_logits, prefill_hidden) = if cfg.physics.steer_hidden {
        let (logits, hidden) = model.forward_with_hidden(&prompt_tensor, 0)?;
        dim_assert::assert_last_dim(&hidden, dim, "oneshot.prefill_hidden")?;
        (logits, Some(hidden))
    } else {
        let logits = model.forward(&prompt_tensor, 0)?;
        (logits, None)
    };
    let mut index_pos = prompt_ids.len();

    // Goal attractor: from hidden state (steer_hidden) or logit space (fallback)
    // This prefill hidden is the "J-space" / pre-verbal image of the prompt.
    let goal_pos = if let Some(ref hidden) = prefill_hidden {
        // Hidden state is already (1, D) -- squeeze to (D,)
        let h = hidden.squeeze(0)?;
        dim_assert::assert_last_dim(&h, dim, "oneshot.goal_pos")?;
        println!(
            "    Goal attractor (J-space): from prefill hidden (D={}, steer_hidden=true)",
            h.dim(0)?
        );
        h
    } else {
        let g = if prefill_logits.dim(1)? >= dim {
            prefill_logits.narrow(1, 0, dim)?.squeeze(0)?
        } else {
            prefill_logits.squeeze(0)?
        };
        println!("    Goal attractor: from logit space (steer_hidden=false)");
        g
    };
    let goal_norm: f32 = goal_pos.sqr()?.sum_all()?.to_scalar::<f32>()?.sqrt();
    println!("    Goal attractor norm: {:.4}", goal_norm);

    // Death→reload geometry: is the start residual near any scar? (LOCALITY cold diagnosis)
    let (nearest_scar_dist, nearest_scar_sigma, mean_scar_dist, scars_checked) =
        if engine.memory().len() > 0 {
            engine
                .memory()
                .nearest_scar_stats(&goal_pos, 64)
                .unwrap_or((f32::INFINITY, 0.0, 0.0, 0))
        } else {
            (f32::INFINITY, 0.0, 0.0, 0)
        };
    let nearest_scar_dist_log = if nearest_scar_dist.is_finite() {
        nearest_scar_dist
    } else {
        -1.0
    };
    let scar_potential_at_prefill = if engine.memory().len() > 0 {
        engine.memory().query_potential(&goal_pos).unwrap_or(0.0)
    } else {
        0.0
    };
    if scars_checked > 0 {
        let cold = nearest_scar_dist.is_finite()
            && nearest_scar_sigma > 1e-6
            && nearest_scar_dist > 3.0 * nearest_scar_sigma;
        println!(
            "    Learned-will geometry @ prefill: n={} nearest_L2={:.2} σ_near={:.2} mean_L2={:.2} pot={:.3}{}{}",
            scars_checked,
            nearest_scar_dist,
            nearest_scar_sigma,
            mean_scar_dist,
            scar_potential_at_prefill,
            if cold {
                "  [LOCALITY COLD: d > 3σ]"
            } else if nearest_scar_dist < 1.0 && scar_potential_at_prefill.abs() > 0.1 {
                "  [BASIN LIVE: pot high, F_s may be ~0 on-center]"
            } else {
                ""
            },
            if cfg.physics.force_ramp_tokens > 0 {
                format!(
                    "  ramp={}->1.0 over {} tok",
                    cfg.physics.force_ramp_start, cfg.physics.force_ramp_tokens
                )
            } else {
                "  ramp=OFF".into()
            }
        );
    } else {
        println!("    Learned-will geometry @ prefill: no learned wills loaded");
    }

    // Config log after prefill so scar geometry is real (not placeholders).
    // Read the live chain rather than the TOML because `--set` may have overridden it.
    let live_logit_params = logit_chain.params();
    let live_logit = |name: &str, fallback: f32| {
        live_logit_params
            .iter()
            .find(|(n, _, _, _)| *n == name)
            .map(|(_, value, _, _)| *value)
            .unwrap_or(fallback)
    };
    logger.log_config(SessionConfig {
        scaler_receipt_id: scaler_receipt.receipt_id.clone(),
        scaler_receipt: serde_json::to_value(&scaler_receipt)?,
        prompt: raw_prompt.to_string(),
        dt: cfg.physics.dt,
        viscosity: cfg.physics.viscosity_scale,
        kernel_sigma: engine.field_kernel_sigma(),
        embedding_dim: dim,
        field_points: engine.field_n_points(),
        model: model_path.clone(),
        model_variant: model_variant.to_string(),
        backend: engine.backend_name().to_string(),
        splat_sigma: cfg.physics.splat_sigma,
        splat_alpha: cfg.physics.splat_alpha,
        force_cap: cfg.physics.force_cap,
        temperature: cfg.generation.temperature as f32,
        min_splat_dist: cfg.physics.min_splat_dist,
        config_path: config_path.clone(),
        clear_memory,
        scars_loaded_safetensors: loaded_count,
        scars_imported_tct: tct_import_count,
        scars_at_start,
        memory_loaded,
        nearest_scar_dist: nearest_scar_dist_log,
        nearest_scar_sigma,
        mean_scar_dist,
        scars_checked,
        force_ramp_tokens: cfg.physics.force_ramp_tokens,
        scar_potential_at_prefill,
        n_prefill_bridges: n_prefill_bridges_start,
        memory_force_mode: cfg.memory.memory_force_mode.clone(),
        memory_pick_k: cfg.memory.memory_pick_k,
        memory_pick_selective: cfg.memory.memory_pick_selective,
        logit_field_alpha: live_logit("field.alpha", cfg.logit_physics.field_alpha),
        logit_splat_scale: live_logit("splat.scale", cfg.logit_physics.splat_scale),
        logit_governor_enabled: live_logit(
            "gov.on",
            if cfg.logit_physics.governor_enabled {
                1.0
            } else {
                0.0
            },
        ) >= 0.5,
        logit_governor_brake: live_logit("gov.brake", cfg.logit_physics.governor_brake),
        logit_governor_max_bias: live_logit("gov.max_bias", cfg.logit_physics.governor_max_bias),
        hook_enabled: hook_controls.enabled,
        hook_site: hook_controls.site.as_str().to_string(),
        hook_start_frac: hook_controls.band.start_frac,
        hook_end_frac: hook_controls.band.end_frac,
        hook_norm_fraction: hook_controls.norm_fraction,
    })?;

    // Optional reflective micro-dream on prefill (variant D) — work with J-space, not yank it
    let mut prefill_hidden = prefill_hidden;
    if cfg.physics.prefill_micro_dream {
        if let Some(ref h) = prefill_hidden {
            let r = micro_dream(&mut engine, h, &goal_pos, 0, 3, 0.08)?;
            println!(
                "    Prefill micro-dream (J-space): ||corr||={:.3} reflection={}",
                r.correction_norm, r.reflection_triggered
            );
            prefill_hidden = Some(r.consolidated);
        }
    }

    // Visualization collector (only when --viz is passed)
    let mut viz_collector: Option<VizCollector> = if viz_enabled {
        match VizCollector::new(engine.field_positions(), &goal_pos, raw_prompt, dim) {
            Ok(c) => Some(c),
            Err(e) => {
                eprintln!("    [VIZ] Failed to init collector: {}", e);
                None
            }
        }
    } else {
        None
    };

    // Now start generating from prefill
    let mut raw_logits = prefill_logits;
    let mut raw_hidden: Option<Tensor> = prefill_hidden;
    let mut hook_report = hooks::HookReport::default();

    // Collect generated tokens
    let mut generated_tokens: Vec<u32> = Vec::new();

    // Track last steered position for splat creation
    let mut last_steered_pos: Option<Tensor> = None;
    let mut last_online_splat_step: isize = -999;
    // Consecutive pain deposits this run — pleasure answers pain.
    let mut pain_deposit_streak: usize = 0;

    // Full generation trajectory (real hidden states for dream replay)
    // trajectory_masses: per-token weight (1 - prob) — surprise = high mass
    let mut generation_trajectory: Vec<Tensor> = Vec::new();
    let mut trajectory_masses: Vec<f32> = Vec::new();

    println!(
        "\n    === Generation ({} tokens, physics-steered) ===\n",
        max_tokens
    );

    // Live stream file: tokens + physics lines for `tail -f logs/live.txt`
    // (stdout alone is easy to miss when runs are backgrounded / piped).
    use std::io::Write;
    let live_path = std::path::Path::new("logs/live.txt");
    let mut live_file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(live_path)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = std::fs::set_permissions(live_path, std::fs::Permissions::from_mode(0o664));
    }
    writeln!(
        live_file,
        "\n=== [{}] \"{}\" ===\njsonl={}  latest=logs/latest.jsonl",
        model_variant,
        raw_prompt,
        logger.path().display()
    )?;
    live_file.flush()?;
    println!(
        "    Live log: {}  (tail -f logs/live.txt)",
        live_path.display()
    );

    // Dual-write: keep stdout and live.txt in sync for force / splat / bloom lines.
    // Per-token diagnostics. Always logged; printed only when no HUD is already
    // showing the same scalars — otherwise they scroll straight through the
    // reading you opened the HUD for.
    macro_rules! live_println {
        ($($arg:tt)*) => {{
            let line = format!($($arg)*);
            hud::hud_quiet_println!("{}", line);
            let _ = writeln!(live_file, "{}", line);
            let _ = live_file.flush();
        }};
    }

    /// Run events worth interrupting the stream for, HUD or not.
    macro_rules! live_event {
        ($($arg:tt)*) => {{
            let line = format!($($arg)*);
            println!("{}", line);
            let _ = writeln!(live_file, "{}", line);
            let _ = live_file.flush();
        }};
    }

    let mut d_phys_on_all = true;
    let mut d_blend_max = 0.0f64;
    let mut d_beta_max = 0.0f64;
    let mut d_force_max = 0.0f64;
    let mut d_dh_max = 0.0f64;
    if d_run {
        let _ = std::fs::create_dir_all("logs");
        let header = serde_json::json!({
            "event": "d_start",
            "model": model_path,
            "config": config_path,
            "binary_sha256": d_binary_sha256,
            "tokens_target": max_tokens,
            "hooks_enabled": hook_controls.enabled,
            "endocrine_enabled": endocrine_enabled,
            "force_cap": engine.force_cap(),
            "residual_enabled": engine.residual_enabled(),
        });
        let _ = std::fs::write("logs/D_gemma.jsonl", format!("{header}\n"));
    }

    hud.begin();
    for step in 0..max_tokens {
        // Steer: hidden state (steer_hidden=true) or logit slice (fallback)
        let (steer_input, is_hidden_steer) = if cfg.physics.steer_hidden {
            if let Some(ref h) = raw_hidden {
                (h.clone(), true) // already (1, D) from forward_with_hidden
            } else {
                // Fallback if hidden state unavailable
                let s = if raw_logits.dim(1)? >= dim {
                    raw_logits.narrow(1, 0, dim)?
                } else {
                    raw_logits.clone()
                };
                (s, false)
            }
        } else {
            let s = if raw_logits.dim(1)? >= dim {
                raw_logits.narrow(1, 0, dim)?
            } else {
                raw_logits.clone()
            };
            (s, false)
        };

        // Endocrine: drain blooms (text) → embed with **native** tok_embeddings → eureka.
        if let Some(ref mut rx) = endocrine_rx {
            while let Ok(bloom) = rx.try_recv() {
                let (pos4, native_opt) =
                    match native_embed_mean(&model, &tokenizer, &bloom.raw_text) {
                        Ok(t) => {
                            let flat = t
                                .flatten_all()
                                .and_then(|x| x.to_vec1::<f32>())
                                .unwrap_or_default();
                            let p4 = endocrine::project_native_to_4d(&flat);
                            (p4, Some(t))
                        }
                        Err(e) => {
                            eprintln!("    [ENDOCRINE] native embed failed: {e}");
                            ([0.0; 4], None)
                        }
                    };
                let mono = endocrine::Monolith {
                    pos: pos4,
                    mass: 750.0,
                    repulsion: 1.0,
                };
                engine.apply_monolith_native(&mono, native_opt);
                let fact_line = bloom.raw_text.chars().take(120).collect::<String>();
                live_println!("    [BLOOM native] {}", fact_line.replace('\n', " "));
            }
        }
        engine.tick_endocrine();
        engine.tick_hands();

        // Kept whole (not destructured) so the residual/hidden telemetry added in
        // `SteerResult` survives to the HUD below. Tensor clones are Arc bumps.
        if is_hidden_steer {
            dim_assert::assert_last_dim(&steer_input, dim, "oneshot.steer_input")?;
        }
        let steer = engine.steer(&steer_input, &goal_pos, step)?;
        let grad_mag = steer.grad_mag;
        let splat_mag = steer.splat_mag;
        let goal_mag = steer.goal_mag;
        let ocean_mag = steer.ocean_mag;
        let memory_ranked = steer.memory_ranked;
        let mut steered_slice = steer.steered.clone();
        if is_hidden_steer {
            dim_assert::assert_last_dim(&steered_slice, dim, "oneshot.steered_slice")?;
        }
        let residual_live = engine.residual_enabled();

        // Ocean deposit moved to *after* token quality scoring (quality-gated).
        // Depositing every 4 steps without quality was crystallizing late garbage.

        // Manifold safety: blend steered state back toward baseline each step
        // Prevents cumulative drift off the model manifold
        if residual_live && cfg.physics.manifold_pullback > 0.0 {
            let pb = cfg.physics.manifold_pullback as f64;
            steered_slice =
                (&steered_slice.affine(1.0 - pb, 0.0)? + &steer_input.affine(pb, 0.0)?)?;
            if is_hidden_steer {
                dim_assert::assert_last_dim(&steered_slice, dim, "oneshot.manifold_pullback")?;
            }
        }

        // Bundle stress: light K-NN scar pull AFTER main steer.
        // NOTE: this path intentionally bypasses niodoo force_cap/ramp (applied
        // inside engine.steer). Keep the 0.01 scale tiny; do not raise without
        // folding bundle into the capped total_force sum in niodoo.rs.
        // Skip entirely when splat force is disabled — otherwise scars still
        // "nuke" the residual via this uncapped side path (Jason/27B LOL).
        if residual_live && cfg.physics.splat_force_scale > 1e-8 && engine.memory().len() > 3 {
            let pos = steered_slice.squeeze(0)?;
            let bundle = engine.memory().query_bundle_force(&pos, 8)?;
            dim_assert::assert_last_dim(&bundle, dim, "oneshot.bundle_force")?;
            let bundle_2d = bundle.unsqueeze(0)?;
            steered_slice = (&steered_slice + &bundle_2d.affine(0.01, 0.0)?)?;
            dim_assert::assert_last_dim(&steered_slice, dim, "oneshot.bundle_residual_add")?;
        }

        last_steered_pos = Some(steered_slice.clone());

        // Reconstruct full logits for sampling
        let mut steered_logits = if !residual_live {
            raw_logits.clone()
        } else if is_hidden_steer {
            // Project steered hidden state through lm_head to get full vocab logits
            model.project_to_logits(&steered_slice)?
        } else {
            // Logit-space steering: cat steered slice with remaining logits
            if raw_logits.dim(1)? > dim {
                let rest = raw_logits.narrow(1, dim, raw_logits.dim(1)? - dim)?;
                Tensor::cat(&[&steered_slice, &rest], 1)?
            } else {
                // Clone (cheap Arc bump) — the logit chain below still needs the
                // steered hidden state as context.
                steered_slice.clone()
            }
        };

        // ── Logit-surface physics ──────────────────────────────────────────
        // Runs after lm_head and before the sampling controls. See src/logit_physics.rs.
        //   field    — z += α · normalize(E û_g), the original surface bridge
        //   splat    — per-scar token-targeted bias (cosine; residual↔emb scales differ)
        //   governor — entropy brake + viscosity + minority report on the top-5
        let logit_report = {
            let steer_snapshot = steer.with_residual(steered_slice.clone());
            let ctx = logit_physics::StepCtx {
                step,
                steered_hidden: Some(&steered_slice),
                steer: Some(&steer_snapshot),
                token_embeddings: model.token_embeddings(),
                field: Some(engine.field()),
                memory: Some(engine.memory()),
                memory_pick: Some(engine.memory_pick()),
                prompt_fp: engine.prompt_fp(),
            };
            steered_logits = logit_chain.apply(&steered_logits, &ctx)?;
            logit_chain.last_report().clone()
        };

        // Repetition penalty: once per unique token id already seen, not once
        // per occurrence (see generate_turn for why per-occurrence compounds
        // toward zero on common glue tokens across a long context).
        let rep_penalty = cfg.generation.rep_penalty;
        let steered_logits = {
            let mut logits_vec: Vec<f32> = steered_logits.squeeze(0)?.to_vec1()?;
            let seen_ids: std::collections::HashSet<u32> = prompt_ids
                .iter()
                .chain(generated_tokens.iter())
                .copied()
                .collect();
            for tid in seen_ids {
                if (tid as usize) < logits_vec.len() {
                    let l = &mut logits_vec[tid as usize];
                    if *l > 0.0 {
                        *l /= rep_penalty;
                    } else {
                        *l *= rep_penalty;
                    }
                }
            }
            Tensor::from_vec(logits_vec, steered_logits.dim(1)?, steered_logits.device())?
                .unsqueeze(0)?
        };

        let temperature: f64 = cfg.generation.temperature;
        let mut logits_vec: Vec<f32> = steered_logits.squeeze(0)?.to_vec1()?;
        if seat.eos_masked {
            for &id in &eos_token_ids {
                if (id as usize) < logits_vec.len() {
                    logits_vec[id as usize] = f32::NEG_INFINITY;
                }
            }
        }
        let qsma_pick = engine.apply_qsma_logits(&mut logits_vec, &generated_tokens, step);
        let next_token = qsma_pick.index as u32;
        engine.observe_token(next_token, 1.0);
        // Probs for quality scoring (approx: softmax over full vec at T, or one-hot if greedy)
        let probs_vec: Vec<f32> = if temperature < 1e-5 {
            let mut p = vec![0.0f32; logits_vec.len()];
            if (next_token as usize) < p.len() {
                p[next_token as usize] = 1.0;
            }
            p
        } else {
            let t = Tensor::from_vec(
                logits_vec.clone(),
                logits_vec.len(),
                steered_logits.device(),
            )?;
            let scaled = (&t / temperature)?;
            let probs = candle_nn::ops::softmax(&scaled, 0)?;
            probs.to_vec1()?
        };

        // Steering delta (telemetry / multi-scale only — NOT the definition of "good")
        let delta = (&steered_logits - &raw_logits)?;
        let delta_norm: f32 = delta.sqr()?.sum_all()?.to_scalar::<f32>()?.sqrt();

        // Decode first so quality scoring can see the surface form
        let decoded = tokenizer
            .decode(&[next_token], false)
            .unwrap_or_else(|_| format!("[{}]", next_token));

        // ── Semantic splat: "good" = confident non-spam, "bad" = surprise/loop ──
        // Quantified from P(token), top-k entropy, recent repeats — not δ.
        let q_thr = QualityThresholds::default();
        let quality = score_token(&probs_vec, next_token, &decoded, &generated_tokens, &q_thr);
        // Selective memory pick gate: high entropy / low margin → hard Top-K next step.
        // Margin = p1 − p2 over full posterior (geometric-native, no side embedder).
        let mut top2 = [0.0f32; 2];
        for &p in &probs_vec {
            if p > top2[0] {
                top2[1] = top2[0];
                top2[0] = p;
            } else if p > top2[1] {
                top2[1] = p;
            }
        }
        let conf_margin = (top2[0] - top2[1]).clamp(0.0, 1.0);
        engine.set_pick_context(quality.topk_entropy, conf_margin);
        let kind = classify(&quality, &q_thr);
        let interval = cfg.physics.online_splat_interval.max(1);
        let rate_ok = step > 4 && (step as isize - last_online_splat_step) >= interval as isize;
        // High-signal: steering event OR pain OR strong pleasure (original Niodoo targeting)
        let high_delta = delta_norm > cfg.physics.splat_delta_threshold;
        let high_signal = high_delta
            || kind == SplatKind::Pain
            || (kind == SplatKind::Pleasure && quality.p_chosen >= 0.25);
        let splat_ok = if cfg.physics.targeted_splat_only {
            rate_ok && kind != SplatKind::Skip && high_signal
        } else {
            rate_ok && kind != SplatKind::Skip
        };
        if splat_ok {
            if let Some(ref pos) = last_steered_pos {
                let current_pos = pos.squeeze(0)?;
                let too_close = engine
                    .memory()
                    .has_nearby(&current_pos, cfg.physics.min_splat_dist)?;
                if !too_close {
                    let splat_alpha = alpha_for(
                        kind,
                        &quality,
                        cfg.generation.pleasure_alpha,
                        cfg.generation.pain_alpha,
                    );
                    // Hierarchical width relative to deposit threshold (not absolute 20/30).
                    // flux = p_chosen → quality history for ranked picker (not bridge marker).
                    let mut trail = Splat::with_scale_ref_lambda(
                        current_pos,
                        cfg.physics.splat_sigma,
                        splat_alpha,
                        delta_norm,
                        cfg.physics.splat_delta_threshold,
                        cfg.physics.splat_lambda_default,
                    );
                    trail.flux = quality.p_chosen.clamp(0.0, 1.0);
                    engine.memory_mut().add_splat(trail);
                    // Cap during generation — prune_to_limit used to run only in Phase 5
                    // after the full loop, so 1000-tok runs could grow memory unbounded
                    // and F_s latched even with 1/√n damp + force caps.
                    engine.memory_mut().prune_to_limit(cfg.memory.max_splats);
                    // Pain snowball brake: scars can log; they must not own the residual.
                    if splat_alpha < 0.0
                        || cfg.memory.max_pain_splats > 0
                        || cfg.memory.max_pain_mass > 0.0
                    {
                        engine.memory_mut().enforce_pain_budget(
                            cfg.memory.max_pain_splats,
                            cfg.memory.max_pain_mass,
                        );
                    }

                    // Heart: pleasure answers pain — soft +α near goal after a pain streak.
                    if kind == SplatKind::Pain {
                        pain_deposit_streak = pain_deposit_streak.saturating_add(1);
                    } else if kind == SplatKind::Pleasure {
                        pain_deposit_streak = 0;
                    }
                    let answer_after = cfg.memory.pleasure_answer_after;
                    if answer_after > 0
                        && kind == SplatKind::Pain
                        && pain_deposit_streak >= answer_after
                    {
                        let ans_alpha = cfg.memory.pleasure_answer_alpha.abs().max(0.15);
                        let ans_sigma = (cfg.physics.splat_sigma
                            * cfg.memory.pleasure_answer_sigma_scale.max(0.5))
                        .max(1.0);
                        // goal_pos is (D,) residual attractor — pleasure at the home basin
                        engine.memory_mut().add_splat(Splat::new(
                            goal_pos.clone(),
                            ans_sigma,
                            ans_alpha,
                        ));
                        pain_deposit_streak = 0;
                        live_println!(
                            "    [WILL + answer] α=+{:.2} σ={:.1} near goal (−wills were stacking)",
                            ans_alpha,
                            ans_sigma
                        );
                    }

                    last_online_splat_step = step as isize;
                    let will_tag = match kind {
                        SplatKind::Pleasure => "+will",
                        SplatKind::Pain => "−will",
                        _ => "will",
                    };
                    let log_every = cfg.physics.will_log_every.max(1);
                    let log_will = step % log_every == 0
                        || high_delta
                        || (cfg.physics.will_log_neg_always && kind == SplatKind::Pain);
                    if log_will {
                        live_println!(
                            "    [{will_tag}] p={:.3} H≈{:.2} δ={:.1} α={:.2} «{}»",
                            quality.p_chosen,
                            quality.topk_entropy,
                            delta_norm,
                            splat_alpha,
                            decoded.replace('\n', "⏎")
                        );
                    }
                    // Endocrine stays ON — cooldown/threshold from config (tune, don't amputate).
                    if let Some(ref tx) = endocrine_tx {
                        let cool = cfg.physics.endocrine_cooldown_steps.max(1) as isize;
                        let cooldown_ok = (step as isize - last_endocrine_signal_step) >= cool;
                        let h_min = cfg.physics.endocrine_entropy_min;
                        if cooldown_ok
                            && (kind == SplatKind::Pain
                                || (high_delta && quality.topk_entropy > h_min))
                        {
                            let intent = format!(
                                "stabilize generation after {will_tag} token «{}»",
                                decoded
                                    .replace('\n', " ")
                                    .chars()
                                    .take(40)
                                    .collect::<String>()
                            );
                            let context = format!(
                                "prompt_prefix={} step={} p={:.3} H={:.2} delta={:.1}",
                                raw_prompt.chars().take(80).collect::<String>(),
                                step,
                                quality.p_chosen,
                                quality.topk_entropy,
                                delta_norm
                            );
                            match tx.try_send(endocrine::EndocrineSignal::ExecuteTool {
                                intent,
                                context,
                            }) {
                                Ok(()) => {
                                    last_endocrine_signal_step = step as isize;
                                    live_println!(
                                        "    [ENDOCRINE] signal step {} ({will_tag}/high-δ)",
                                        step
                                    );
                                }
                                Err(_) => {}
                            }
                        }
                    }
                }
            }
        }

        // Lane C ocean: quality-gated deposits (original: not every token)
        if let Some(ocean) = engine.ocean_mut() {
            if step > 0 && step % ocean.config.deposit_interval == 0 {
                let host_vec = steer_input.squeeze(0)?;
                let mind = if is_gemma {
                    MindId::Gemma
                } else {
                    MindId::Host
                };
                match kind {
                    SplatKind::Pleasure if high_signal || !cfg.physics.targeted_splat_only => {
                        let w = quality.p_chosen.clamp(0.3, 1.0);
                        let noise = (0.55 - 0.3 * quality.p_chosen).clamp(0.15, 0.55);
                        ocean.deposit(mind, &host_vec, w, noise)?;
                    }
                    SplatKind::Pain => {
                        if cfg.physics.pain_recovery_ocean {
                            // Variant E: recovery anchor — stronger corrective packet
                            ocean.deposit(mind, &host_vec, 0.85, 0.35)?;
                            if step % 10 == 0 {
                                live_println!(
                                    "    [OCEAN recovery] pain packet p={:.3} δ={:.1}",
                                    quality.p_chosen,
                                    delta_norm
                                );
                            }
                        } else {
                            ocean.deposit(mind, &host_vec, 0.15, 0.92)?;
                        }
                    }
                    _ => {}
                }
            }
        }

        // Mid-run F_s control: per-token scar alpha decay (not wall-clock decay_step).
        if cfg.memory.online_decay_rate < 1.0 && engine.memory().len() > 0 {
            engine
                .memory_mut()
                .decay_per_token(cfg.memory.online_decay_rate, cfg.physics.pain_decay_factor);
            if step > 0 && step % 25 == 0 {
                let _ = engine.memory_mut().cull(cfg.memory.prune_threshold);
            }
        }

        generated_tokens.push(next_token);
        if d_run && (step % 256 == 0 || generated_tokens.len() == max_tokens) {
            use std::io::Write;
            if !engine.residual_enabled() {
                anyhow::bail!(
                    "physics dropped mid --d-run at n={} force_cap={}",
                    generated_tokens.len(),
                    engine.force_cap()
                );
            }
            let hands = engine.hands_report();
            d_phys_on_all &= hands["residual_enabled"].as_bool().unwrap_or(false);
            d_blend_max = d_blend_max.max(hands["physics_blend"].as_f64().unwrap_or(0.0).abs());
            d_beta_max = d_beta_max.max(hands["qsma_beta"].as_f64().unwrap_or(0.0).abs());
            d_force_max = d_force_max.max(hands["force_cap"].as_f64().unwrap_or(0.0).abs());
            d_dh_max = d_dh_max.max(hands["delta_h_norm"].as_f64().unwrap_or(0.0).abs());
            let rec = serde_json::json!({
                "event": "d_ckpt",
                "step": step,
                "n": generated_tokens.len(),
                "token": decoded,
                "hands": hands,
            });
            if let Ok(mut f) = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open("logs/D_gemma.jsonl")
            {
                let _ = writeln!(f, "{rec}");
            }
            live_event!(
                "    [D] n={} blend={} β={} Δh={} cap={} on={}",
                generated_tokens.len(),
                rec["hands"]["physics_blend"],
                rec["hands"]["qsma_beta"],
                rec["hands"]["delta_h_norm"],
                rec["hands"]["force_cap"],
                rec["hands"]["residual_enabled"]
            );
        }

        // Viz snapshot with nearest token attractors (zero cost when --viz not passed)
        if let Some(ref mut collector) = viz_collector {
            // Find top-5 highest probability tokens every 5 steps as attractors
            let neighbors = if step % 5 == 0 {
                // Use softmax probs to find what the model is attracted to
                // Partial sort: only find top-5 without fully sorting 128K items
                let mut prob_indices: Vec<(u32, f32)> = probs_vec
                    .iter()
                    .enumerate()
                    .map(|(i, &p)| (i as u32, p))
                    .collect();
                if prob_indices.len() > 5 {
                    prob_indices.select_nth_unstable_by(4, |a, b| {
                        b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
                    });
                    prob_indices.truncate(5);
                }
                prob_indices
                    .iter()
                    .take(5)
                    .map(|&(tid, prob)| {
                        let text = tokenizer
                            .decode(&[tid], false)
                            .unwrap_or_else(|_| format!("[{}]", tid));
                        (tid, text, prob)
                    })
                    .collect()
            } else {
                Vec::new()
            };
            let _ = collector.snapshot(
                step,
                next_token,
                &decoded,
                &steered_logits,
                delta_norm,
                neighbors,
            );
        }

        // Stream tokens live -- print without newline for flowing text.
        // Routed through the HUD so the sticky footer stays pinned below.
        hud.stream(&decoded).ok();

        // Write to live stream file (for tail -f in separate terminal)
        write!(live_file, "{}", decoded).ok();
        live_file.flush().ok();

        // Milestone markers every 50 steps
        if step > 0 && step % 50 == 0 {
            let ocean_info = engine
                .ocean()
                .map(|o| {
                    format!(
                        " ocean_n={} noise={:.2} F_ocean={:.2}",
                        o.len(),
                        o.mean_noise(),
                        ocean_mag
                    )
                })
                .unwrap_or_default();
            live_println!(
                "  [{}/{}] δ={:.1} F_g={:.1} F_s={:.1}{} F_a={:.1}{}",
                step,
                max_tokens,
                delta_norm,
                grad_mag,
                splat_mag,
                if memory_ranked { "ᵣ" } else { "" },
                goal_mag,
                ocean_info
            );
        }

        // Log every step to JSONL
        let residual_norm: f32 = steered_logits.sqr()?.sum_all()?.to_scalar::<f32>()?.sqrt();
        logger.log_step(StepEntry {
            scaler_receipt_id: scaler_receipt.receipt_id.clone(),
            step,
            token_id: next_token,
            token_text: decoded.clone(),
            steering_delta: delta_norm,
            residual_norm,
            grad_force_mag: grad_mag,
            splat_force_mag: splat_mag,
            goal_force_mag: goal_mag,
            scars_active: engine.memory().len(),
            memory_ranked,
            logit_field_mag: logit_report.field_mag,
            logit_splat_mag: logit_report.splat_mag,
            logit_governor_mag: logit_report.governor_mag,
            logit_velocity: logit_report.velocity,
            logit_viscosity: logit_report.viscosity,
            logit_engines_fired: logit_report.engines_fired,
            hook_applications: hook_report.applications,
            hook_delta_mean: hook_report.delta_mean,
            hook_delta_max: hook_report.delta_max,
        })?;

        // Live scalar HUD: algo knobs + residual/hidden state under the stream.
        if hud.is_enabled() {
            let cos_drift = hud.cos_drift(&steered_slice);
            hud.update(hud::HudFrame {
                step,
                max_tokens,
                force_cap: engine.force_cap(),
                goal_force_scale: engine.goal_force_scale(),
                temperature: temperature as f32,
                force_ramp_start: cfg.physics.force_ramp_start,
                force_ramp_tokens: cfg.physics.force_ramp_tokens,
                field_grad_blend: cfg.physics.field_grad_blend,
                baseline_norm: steer.baseline_norm,
                steered_norm: steer.steered_norm,
                pullback: steer.pullback,
                delta_h_norm: steer.delta_h_norm,
                clip_frac: steer.clip_frac,
                ramp: steer.ramp,
                eureka_boost: steer.eureka_boost,
                cos_drift,
                grad_mag,
                splat_mag,
                goal_mag,
                ocean_mag,
                memory_ranked,
                field_wake_max: cfg.physics.field_wake_max,
                splat_force_max: cfg.physics.splat_force_max,
                goal_force_max: cfg.physics.goal_force_max,
                logit_delta: Some(delta_norm),
                logit_velocity: logit_report.velocity,
                logit_viscosity: logit_report.viscosity,
                hook_delta_mean: Some(hook_report.delta_mean),
                hook_applications: Some(hook_report.applications),
                p_chosen: Some(quality.p_chosen),
                entropy: Some(quality.topk_entropy),
                margin: Some(conf_margin),
                scars: engine.memory().len(),
            })
            .ok();
        }

        // TermSplat live FieldFrame (entropy weather) — same contract as termsplat frame.rs
        if let Some(ref mut w) = weather {
            if let Err(e) = w.emit_step(
                step,
                quality.topk_entropy,
                &decoded,
                delta_norm,
                engine.memory(),
                &logit_report,
                &hook_report,
            ) {
                eprintln!("    [weather] emit fail step {step}: {e}");
            }
        }

        // Stop on EOS tokens (D-run masks EOS in logits and must not stop early)
        if eos_token_ids.contains(&next_token) {
            if seat.eos_masked {
                live_event!("    → EOS skipped at step {} (--d-run)", step);
            } else {
                live_event!("    → EOS at step {}", step);
                break;
            }
        }

        // Feed next token
        let next_input = Tensor::new(&[next_token], &device)?.unsqueeze(0)?;
        let hook_direction = if is_hidden_steer && residual_live {
            (&steered_slice - &steer_input)?
        } else {
            Tensor::zeros(
                steer_input.dims(),
                steer_input.dtype(),
                steer_input.device(),
            )?
        };
        let (logits, hidden, report) = forward_decode_with_hook(
            &mut model,
            &next_input,
            index_pos,
            &hook_direction,
            step + 1,
            &hook_controls,
            &mut hook_trace,
        )?;
        raw_logits = logits;
        raw_hidden = Some(hidden);
        hook_report = report;
        index_pos += 1;

        // Collect hidden state for dream replay — AFTER forward pass so
        // trajectory[N] = state that produced token[N] (correct alignment)
        // Token mass: weight by surprise (low prob = high mass = stronger splat)
        if let Some(ref h) = raw_hidden {
            let mass = 1.0_f32 - probs_vec[next_token as usize].min(1.0);
            generation_trajectory.push(h.squeeze(0)?);
            trajectory_masses.push(mass);
        }
    }

    // Settle the sticky footer: leave the last frame on screen as a final
    // readout and put the cursor below it before the summary prints.
    hud.finish().ok();

    // =========================================================
    // Decode full output
    // =========================================================
    println!("\n    === Full Decoded Output ===\n");
    let full_text = tokenizer
        .decode(&generated_tokens, true)
        .unwrap_or_else(|_| "[decode error]".to_string());
    println!("    {}", full_text);

    if d_run {
        let n = generated_tokens.len();
        let dash = full_text.chars().filter(|c| *c == '-').count();
        let bs = full_text.matches('\\').count();
        let channel =
            full_text.matches("<|channel>").count() + full_text.matches("<channel|>").count();
        let chars = full_text.chars().count().max(1);
        let physics_on = d_phys_on_all
            && engine.residual_enabled()
            && d_force_max > 1e-8
            && (d_blend_max > 1e-8 || d_beta_max > 1e-8);
        let deg = n < 131_072
            || !physics_on
            || (dash as f64 / chars as f64) > 0.25
            || bs > 40
            || channel > 8;
        let hands = engine.hands_report();
        let card = serde_json::json!({
            "model": "gemma-4-12b-it-Q4_K_M",
            "model_path": model_path,
            "config": config_path,
            "binary_sha256": d_binary_sha256,
            "tokens_reached": n,
            "tokens_target": max_tokens,
            "physics_on": physics_on,
            "endocrine_enabled": endocrine_enabled,
            "physics_on_proof": {
                "residual_enabled_all_ckpts": d_phys_on_all,
                "blend_max": d_blend_max,
                "qsma_beta_max": d_beta_max,
                "force_cap_max": d_force_max,
                "delta_h_norm_max": d_dh_max,
                "hooks_enabled": hook_controls.enabled,
            },
            "hands": hands,
            "dash_count": dash,
            "backslash_count": bs,
            "channel_count": channel,
            "degraded": deg,
            "reached_131k": n >= 131_072,
        });
        let _ = std::fs::write(
            "logs/D_gemma_card.json",
            serde_json::to_string_pretty(&card).unwrap_or_else(|_| "{}".into()),
        );
        println!(
            "    [D] card logs/D_gemma_card.json n={n} physics_on={physics_on} degraded={deg} sha256={d_binary_sha256}"
        );
    }

    // =========================================================
    // Populate real splats from this generation
    // =========================================================
    println!("\n--- Phase 5: Learned wills ---");
    if let Some(final_pos) = last_steered_pos {
        let pos_1d = final_pos.squeeze(0)?;
        if generated_tokens.len() > cfg.generation.min_success_tokens {
            engine.memory_mut().add_splat(Splat::new(
                pos_1d,
                cfg.physics.splat_sigma,
                cfg.generation.pleasure_alpha,
            ));
            println!(
                "    + Added +will (generation succeeded: {} tokens)",
                generated_tokens.len()
            );
        } else {
            engine.memory_mut().add_splat(Splat::new(
                pos_1d,
                cfg.physics.splat_sigma,
                cfg.generation.pain_alpha,
            ));
            println!(
                "    x Added PAIN splat (generation too short: {} tokens)",
                generated_tokens.len()
            );
        }
        println!("    Splats in memory: {}", engine.memory().len());
    }

    // Evaporation: time-based decay + cull dead splats
    engine.memory_mut().decay_step(cfg.memory.decay_rate);
    let culled = engine.memory_mut().cull(cfg.memory.prune_threshold);
    if culled > 0 {
        println!("    [EVAPORATE] Culled {} dead splats", culled);
    }

    // Consolidate and cap splat memory before saving
    let _ = engine
        .memory_mut()
        .consolidate(cfg.memory.consolidation_dist);
    engine.memory_mut().prune_to_limit(cfg.memory.max_splats);

    println!(
        "    Splats in memory: {} (will persist unless --no-save-memory)",
        engine.memory().len()
    );

    // =========================================================
    // Phase 6: Dream Replay (REAL — replays actual generation trajectory)
    // =========================================================
    println!("\n--- Phase 6: Dream Replay ---");
    let splat_count_before = engine.memory().len();
    if !generation_trajectory.is_empty() {
        let traj_refs: Vec<&Tensor> = generation_trajectory.iter().collect();
        let traj_stack = Tensor::stack(&traj_refs, 0)?;
        let noise = Tensor::randn(0.0f32, 0.05, traj_stack.dims(), &device)?;
        let noisy_traj = (&traj_stack + &noise)?;
        let replay_bonus = 1.25_f32;
        let masses_ref = if trajectory_masses.is_empty() {
            None
        } else {
            Some(trajectory_masses.as_slice())
        };
        let replay_count = engine.memory_mut().consolidate_trajectory(
            &noisy_traj,
            cfg.physics.splat_sigma,
            replay_bonus,
            cfg.physics.min_splat_dist,
            masses_ref,
        )?;
        let avg_mass = if trajectory_masses.is_empty() {
            1.0
        } else {
            trajectory_masses.iter().sum::<f32>() / trajectory_masses.len() as f32
        };
        println!(
            "    Dream replay: {} points -> {} splats (avg mass {:.3}, bonus {:.2})",
            generation_trajectory.len(),
            replay_count,
            avg_mass,
            replay_bonus,
        );
    } else {
        println!("    No hidden trajectory collected (steer_hidden disabled?)");
    }
    engine.memory_mut().decay_step(cfg.memory.decay_rate);
    println!(
        "    Applied decay ({:.3}). Splats remaining: {}",
        cfg.memory.decay_rate,
        engine.memory().len(),
    );

    // Prefill-bridge scar: land memory in the next-run start basin (LOCALITY cold fix).
    // After final decay so it is not wiped the same session.
    // Pleasure-only: pain bridges (failed gen) pollute multi-bridge weight tables and
    // false "continuity" on garbage short runs. Skip deposit when success gate fails.
    if cfg.physics.prefill_bridge_scar && !no_save_memory {
        let success = generated_tokens.len() > cfg.generation.min_success_tokens;
        if !success {
            println!(
                "    [BRIDGE] skip prefill-bridge (gen tokens {} ≤ min_success {}) — no pain deposit",
                generated_tokens.len(),
                cfg.generation.min_success_tokens
            );
        }
        let bridge_alpha = cfg.physics.prefill_bridge_alpha.abs();
        let replace_dist = (cfg.physics.prefill_bridge_sigma
            * (1.0 + cfg.physics.prefill_bridge_offset_frac.abs()))
        .max(cfg.memory.consolidation_dist);
        let prompt_fp = tct::continuity_fp(raw_prompt);
        if success {
            match engine.memory_mut().deposit_prefill_bridge(
                &goal_pos,
                cfg.physics.prefill_bridge_sigma,
                bridge_alpha,
                cfg.physics.prefill_bridge_lambda,
                replace_dist,
                cfg.physics.prefill_bridge_offset_frac,
                prompt_fp,
            ) {
                Ok(removed) => {
                    let dropped = engine
                        .memory_mut()
                        .enforce_max_prefill_bridges(cfg.memory.max_prefill_bridges);
                    let fps = engine.memory().list_bridge_prompt_fps();
                    if let Err(e) = tct::upsert_bridge_prompt_registry(
                        &tct::bridge_prompts_path_default(),
                        prompt_fp,
                        raw_prompt,
                    ) {
                        eprintln!("    [BRIDGE] prompt registry update failed: {e}");
                    }
                    println!(
                    "    + Prefill-bridge learned will (σ={:.1} α={:.2} λ={:.4} offset={:.2}σ fp={:#x}) replaced={} bridges_now={} cap_drop={} fps={:?} total={}",
                    cfg.physics.prefill_bridge_sigma,
                    bridge_alpha,
                    cfg.physics.prefill_bridge_lambda,
                    cfg.physics.prefill_bridge_offset_frac,
                    prompt_fp,
                    removed,
                    engine.memory().count_prefill_bridges(),
                    dropped,
                    fps.iter().map(|f| format!("{:#x}", f)).collect::<Vec<_>>(),
                    engine.memory().len()
                );
                }
                Err(e) => eprintln!("    [BRIDGE] prefill learned-will failed: {e}"),
            }
        } // success
    }

    // =========================================================
    // Persist memory (safetensors + TCT-splat-lite)
    // Continuity goal: scars must survive process death.
    // =========================================================
    if !no_save_memory && engine.memory().len() > 0 {
        println!("\n--- Phase 6b: Persist Memory ---");
        if let Some(parent) = splat_file.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        match engine.memory().save(splat_file) {
            Ok(()) => {}
            Err(e) => eprintln!("    [MEMORY] safetensors save failed: {e}"),
        }
        let tct_out = export_tct_path
            .as_deref()
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("data/splat_memory.tct"));
        let fp = tct::model_fp_from_path(&model_path);
        let reg = tct::bridge_prompts_path_default();
        match engine.memory().export_tct_with_registry(
            &tct_out,
            dim,
            fp,
            /* json sidecar */ true,
            Some(&reg),
        ) {
            Ok(()) => {}
            Err(e) => eprintln!("    [TCT] export failed: {e}"),
        }
    } else if no_save_memory {
        println!("    Skipping memory save (--no-save-memory)");
    }

    // =========================================================
    // Summary
    // =========================================================
    let splat_type = if generated_tokens.len() > cfg.generation.min_success_tokens {
        "pleasure"
    } else {
        "pain"
    };
    let splat_count_after = engine.memory().len();
    logger.log_summary(SessionSummary {
        prompt: raw_prompt.to_string(),
        prompt_token_count: prompt_ids.len(),
        generated_token_count: generated_tokens.len(),
        goal_attractor_norm: goal_norm,
        splat_count_before,
        splat_count_after,
        splat_type_added: splat_type.to_string(),
        decoded_output: full_text.clone(),
        delta_min: 0.0, // filled by log_summary
        delta_max: 0.0,
        delta_mean: 0.0,
    })?;

    let ocean_summary = engine
        .ocean()
        .map(|o| {
            format!(
                "  Ocean:    {} packets | deposits={} | mean_noise={:.3}",
                o.len(),
                o.total_deposits(),
                o.mean_noise()
            )
        })
        .unwrap_or_else(|| "  Ocean:    offline".into());

    println!("\n========================================");
    println!("  SplatRAG v1.1 -- OPERATIONAL");
    println!("========================================");
    println!("  Model:    {}", model_path);
    println!("  Variant:  {}", model_variant);
    println!("  Prompt:   \"{}\"", raw_prompt);
    println!("  Tokens:   {} generated", generated_tokens.len());
    println!("{}", ocean_summary);
    println!("  Log:      {}", logger.path().display());
    println!("  TACO:     {}", logger.taco_stats());
    println!(
        "  Backend:  {} + Niodoo physics + Shared Ocean",
        engine.backend_name()
    );
    println!("========================================");

    // Append to human-readable log
    {
        use std::io::Write;
        let readable_path = Path::new("logs/readable.txt");
        let mut f = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(readable_path)?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let _ = std::fs::set_permissions(readable_path, std::fs::Permissions::from_mode(0o664));
        }
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        let days = now / 86400;
        let day_secs = now % 86400;
        let hours = day_secs / 3600;
        let minutes = (day_secs % 3600) / 60;
        let (y, m, d) = logger::days_to_date(days);
        writeln!(
            f,
            "=== Run: {}-{:02}-{:02} {:02}:{:02} UTC ===",
            y, m, d, hours, minutes
        )?;
        writeln!(
            f,
            "Model: {} | Tokens: {} | Splats: {} | jsonl: {}",
            model_variant,
            generated_tokens.len(),
            engine.memory().len(),
            logger.path().display()
        )?;
        writeln!(f, "Prompt: \"{}\"", raw_prompt)?;
        writeln!(f)?;
        writeln!(f, "{}", full_text)?;
        writeln!(f)?;
        writeln!(f, "---")?;
        writeln!(f)?;
    }

    // =========================================================
    // Visualization export (JSON only — HTML viewer removed)
    // =========================================================
    if let Some(mut collector) = viz_collector {
        // Load real splat scar data from engine memory
        collector.load_splats(engine.memory());

        // Export JSON snapshot data
        let viz_path = logger.path().with_extension("viz.json");
        let _ = collector.export_json(&viz_path);
    }

    Ok(())
}

/// Mean token embedding of `text` from the **live** model — pre-layer space.
///
/// Gemma (3/4 family): `tok_embeddings * √hidden_dim` before the first layer
/// (see `gemma.rs` run_layers). Without that scale, blooms sit on the raw
/// matrix shell and miss the geometry the stack actually eats.
/// Llama: raw rows (scale 1).
fn native_embed_mean(model: &Model, tokenizer: &Tokenizer, text: &str) -> anyhow::Result<Tensor> {
    let emb = model.token_embeddings(); // (V, D) raw weight matrix
    let vocab = emb.dim(0)?;
    let dim = emb.dim(1)?;
    let encoded = tokenizer
        .encode(text, true)
        .map_err(|e| anyhow::anyhow!("tokenize bloom: {e}"))?;
    let ids: Vec<u32> = encoded.get_ids().to_vec();
    if ids.is_empty() {
        return Ok(Tensor::zeros(dim, candle_core::DType::F32, emb.device())?);
    }

    let mut sum: Option<Tensor> = None;
    let mut n = 0usize;
    for &id in ids.iter().take(96) {
        let i = id as usize;
        if i >= vocab {
            continue;
        }
        let row = emb.get(i)?.to_dtype(candle_core::DType::F32)?;
        sum = Some(match sum {
            None => row,
            Some(s) => (&s + &row)?,
        });
        n += 1;
    }
    if n == 0 {
        return Ok(Tensor::zeros(dim, candle_core::DType::F32, emb.device())?);
    }
    let mean = sum.unwrap().affine(1.0 / n as f64, 0.0)?;
    // Pre-layer scale (Gemma √d) — matches forward, not TinyEmbed hash.
    let scale = model.embedding_input_scale();
    if (scale - 1.0).abs() > 1e-12 {
        Ok(mean.affine(scale, 0.0)?)
    } else {
        Ok(mean)
    }
}

/// Measure Jacobian sensitivity at a single decode step.
///
/// Finite-difference perturbation of hidden state → measure logit changes.
/// Returns a JacobianReport with global sensitivity and top dimensions/tokens.
fn measure_jacobian_step(
    hidden: &Tensor,
    model: &Model,
    epsilon: f32,
    top_k: usize,
    max_dims: usize,
    step: usize,
    residual_d: usize,
) -> Result<jacobian::JacobianReport> {
    // Read-only lens — still assert the hidden it measures is residual-width.
    dim_assert::assert_last_dim(hidden, residual_d, "jacobian.measure_hidden")?;
    let sites = vec![hooks::HookSite::FinalNorm];
    let mut lens = jacobian::JacobianLens::new(epsilon, sites, top_k, max_dims);
    let project = |h: &Tensor| {
        dim_assert::assert_last_dim(h, residual_d, "jacobian.project_to_logits_in")?;
        model.project_to_logits(h)
    };
    lens.measure(hidden, project)
        .map_err(|e| anyhow::anyhow!("jacobian step {}: {}", step, e))
}

/// Capture one phase-edge key into the turn's `MultiKeyAddress`.
///
/// Fires at most once per phase per turn (`already_captured` is the latch), so a turn
/// yields at most one answer / revise / settle key. The signature is a *proxy* built from
/// the local finite-difference probe — see `research_logs/2026-08-02_jacobian_lens_repo_vs_hydro_fd.md`
/// and `docs/jlens-gguf/`: this is not the fitted jlens transport, and anything that reports
/// it must say "proxy".
///
/// The key carries a hash of the tail of the generated text, never a raw residual vector —
/// the bridge rule is that a pick carries text and the host re-embeds in *its* residual dim.
#[allow(clippy::too_many_arguments)]
fn capture_phase_edge_key(
    phase: &str,
    step: usize,
    turn_idx: usize,
    surface_hidden: &Tensor,
    pieces: &str,
    model: &Model,
    epsilon: f32,
    top_k: usize,
    max_dims: usize,
    residual_d: usize,
    multi_keys: &mut jacobian::MultiKeyAddress,
    collapse_log: &mut Option<std::fs::File>,
    already_captured: &mut bool,
) {
    if *already_captured {
        return;
    }
    let Some(key_phase) = jacobian::KeyPhase::from_str_lossy(phase) else {
        return;
    };

    let report = match measure_jacobian_step(
        surface_hidden,
        model,
        epsilon,
        top_k,
        max_dims,
        step,
        residual_d,
    ) {
        Ok(report) => report,
        Err(e) => {
            if let Some(ref mut f) = collapse_log {
                use std::io::Write;
                let _ = writeln!(
                    f,
                    "{{\"event\":\"phase_key_error\",\"phase\":\"{phase}\",\"turn\":{turn_idx},\"step\":{step},\"error\":{:?}}}",
                    e.to_string()
                );
            }
            return;
        }
    };

    // Text bridge: tail of what the model has actually said this turn. Short and hashed,
    // so the address stays a permutation address rather than smuggling an embedding.
    let bridge = pieces.trim();
    let bridge_tail: String = if bridge.chars().count() > 120 {
        bridge.chars().skip(bridge.chars().count() - 120).collect()
    } else {
        bridge.to_string()
    };

    let key = jacobian::JacobianKey::from_report(&report, key_phase, step, residual_d, top_k)
        .with_turn(turn_idx)
        .with_text_bridge_hash(jacobian::text_bridge_hash(&bridge_tail));

    if key.signature.is_empty() {
        // An empty signature means the probe found no positive sensitivity anywhere —
        // worth logging rather than pushing a key that addresses nothing.
        if let Some(ref mut f) = collapse_log {
            use std::io::Write;
            let _ = writeln!(
                f,
                "{{\"event\":\"phase_key_empty\",\"phase\":\"{phase}\",\"turn\":{turn_idx},\"step\":{step},\"global_sensitivity\":{:.6}}}",
                report.global_sensitivity
            );
        }
        return;
    }

    if let Some(ref mut f) = collapse_log {
        use std::io::Write;
        let dims: Vec<usize> = key.signature.dims.iter().map(|(d, _)| *d).collect();
        let _ = writeln!(
            f,
            "{{\"event\":\"phase_key\",\"phase\":\"{phase}\",\"turn\":{turn_idx},\"step\":{step},\"residual_d\":{residual_d},\"dims\":{dims:?},\"sensitivity_norm\":{:.6},\"bridge_hash\":{},\"source\":\"fd_proxy\"}}",
            report.global_sensitivity,
            key.text_bridge_hash.unwrap_or(0)
        );
    }

    multi_keys.push(key);
    *already_captured = true;
}

#[cfg(test)]
mod generation_tests {
    use super::*;

    #[test]
    fn gemma4_chat_packing_includes_tag_table_not_one_tag_forbid() {
        let p = format_multiturn_prompt_ex(&[(true, "Say hi.".into())], "gemma4", true);
        assert!(
            p.contains("<|turn>system"),
            "packed --chat must include a system turn"
        );
        assert!(
            p.contains("Available tags and what they do") && p.contains("<spike>"),
            "system turn must list available tags; packed snippet={}",
            p.chars().take(280).collect::<String>()
        );
        let lower = p.to_ascii_lowercase();
        assert!(!lower.contains("exactly one"));
        assert!(!lower.contains("at most one"));
        assert!(!lower.contains("do not emit"));
        assert!(!lower.contains("do not narrate"));
        assert!(
            !p.contains("DO NOT emit your tags"),
            "stale forbid-emit string must not be the packing key"
        );
        assert_eq!(gemma4_control_channel_status(&p), "PRESENT");
        // The old diagnostic: system turn + stale string → would have printed ABSENT.
        assert!(
            !(p.contains("<|turn>system") && p.contains("DO NOT emit your tags")),
            "old PRESENT predicate is false on a real packed prompt"
        );
        let off = format_multiturn_prompt_ex(&[(true, "Say hi.".into())], "gemma4", false);
        assert_eq!(gemma4_control_channel_status(&off), "ABSENT");
    }

    #[test]
    fn next_prefill_keeps_emitted_tag_for_attention() {
        let turns = [
            (true, "Say hi.".into()),
            (false, "hello\n<spike>\nmore".into()),
            (true, "again".into()),
        ];
        let p = format_multiturn_prompt_ex(&turns, "gemma4", true);
        assert!(
            p.contains("<spike>"),
            "next prefill must keep the hand so she can attend; packed={}",
            p.chars().take(400).collect::<String>()
        );
        let cleaned = gemma4_history_clean("hello\n<focus>\nmore");
        assert!(
            cleaned.contains("<focus>"),
            "history clean must not mask the hand; cleaned={cleaned:?}"
        );
    }

    #[test]
    fn thought_channel_is_a_live_stream_not_a_settle_stop() {
        assert!(!gemma4_should_settle_channel("<|channel>thought", 100));
        assert!(!gemma4_should_settle_channel("answer text", 100));
        assert!(!gemma4_should_settle_channel("answer text", 101));
        assert!(gemma4_in_open_thought("<|channel>thought\nplan <focus>"));
        assert!(!gemma4_in_open_thought(
            "<|channel>thought\nplan\n<channel|>\nfinal"
        ));
        assert!(!gemma4_lock_stops_turn(
            "<|channel>thought\nplan\n<lock>\nmore"
        ));
        assert!(gemma4_lock_stops_turn(
            "<|channel>thought\nplan\n<channel|>\nfinal\n<lock>"
        ));
        let kept =
            gemma4_history_clean("<|channel>thought\nplan <focus>\n<channel|>\nfinal answer");
        assert!(
            kept.contains("<|channel>thought") && kept.contains("final answer"),
            "next prefill must keep the thought trace; cleaned={kept:?}"
        );
    }

    #[test]
    fn tda_monitor_waits_for_complete_control_hand() {
        assert!(!tda_monitor_injection_ready(
            "answer\n<remember>key=value",
            true
        ));
        assert!(tda_monitor_injection_ready(
            "answer\n<remember>key=value</remember>",
            true
        ));
        assert!(!tda_monitor_injection_ready("answer", false));
    }

    #[test]
    fn apply_emitted_control_writes_on_raw_pieces() {
        use candle_core::{Device, Tensor};
        use gpu::CpuBackend;

        let raw = "count 1\n<spike>\n2";
        assert_eq!(
            control_tags::scan(raw),
            vec![control_tags::ControlTag::Spike]
        );
        assert_eq!(control_tags::strip(raw), raw);

        let device = Device::Cpu;
        let field = ContinuousField::load_dummy(4, 8, &device).unwrap();
        let memory = SplatMemory::new(device.clone());
        let mut live =
            NiodooEngine::new(field, memory, Box::new(CpuBackend::new()), 0.035, 0.25, 5.0);
        let pos = Tensor::new(&[0.5f32, 0.0, 0.0, 0.0], &device).unwrap();
        let mut tags_seen: Vec<control_tags::TagHit> = Vec::new();
        let (stop, applied) = live
            .apply_emitted_control(raw, &mut tags_seen, Some(&pos))
            .unwrap();
        assert!(!stop);
        assert_eq!(applied, vec![control_tags::ControlTag::Spike]);
        let at = live.memory().query_potential(&pos).unwrap_or(0.0);
        assert!(at > 1e-3, "raw pieces must write a scar, pot={at}");
    }

    #[test]
    fn trailing_short_cycle_lock_catches_esese_not_prose() {
        assert!(trailing_short_cycle_lock(
            "the rest of the networkmateseshesiesesesesesesesesesesesese"
        ));
        assert!(trailing_short_cycle_lock(&"es".repeat(10)));
        assert!(!trailing_short_cycle_lock(
            "In transformer architectures, the residual stream acts as the primary highway."
        ));
        assert!(!trailing_short_cycle_lock(
            "The operator codeword lumina-basin-7 refers to residual scar memory that steers later tokens."
        ));
    }

    #[test]
    fn line_repeat_detects_confident_math_thrash() {
        // Turn-8 class: identical long lines, low entropy — old short-only thrash missed this.
        let s = "17 × 10 = 170\n17 × 10 = 170\n17 × 10 = 170\n17 × 10 = 170\n";
        assert!(line_repeat_at_least(s, 2, 6));
        assert!(line_repeat_at_least(s, 4, 6));
        assert!(!line_repeat_at_least(s, 5, 6));
        let (n, len) = trailing_identical_line_run(s);
        assert_eq!(n, 4);
        assert!(len >= 6);
    }

    #[test]
    fn line_repeat_ignores_short_acks() {
        let s = "ok\nok\nok\nok\n";
        assert!(!line_repeat_at_least(s, 2, 6));
        assert!(line_repeat_at_least(s, 2, 2)); // if min_chars lowered
    }

    #[test]
    fn line_repeat_breaks_on_different_line() {
        let s = "red\nred\nblue\n";
        let (n, _) = trailing_identical_line_run(s);
        assert_eq!(n, 1);
        assert!(!line_repeat_at_least(s, 2, 3));
    }

    #[test]
    fn wait_loop_counts_spell_cat_blocks() {
        let s = "C-A-T.\nWait, that's wrong. Let me try again.\n\nC-A-T.\nWait, that's wrong. Let me try again.\n\nC-A-T.\nWait, that's wrong. Let me try again.\n";
        assert!(wait_loop_count(s) >= 3);
        assert_eq!(wait_loop_count("just fine"), 0);
    }

    #[test]
    fn phrase_repeat_detects_same_line_math_thrash() {
        let unit = "No, the question is 17 x 17? ";
        let s = unit.repeat(4);
        assert_eq!(unit.len() * 4, s.len());
        assert!(
            phrase_repeat_at_least(&s, 2, 12, 48),
            "need=2 failed on exact thrash len={}",
            s.len()
        );
        assert!(
            phrase_repeat_at_least(&s, 4, 12, 48),
            "need=4 failed on exact thrash unit_len={}",
            unit.len()
        );
        // Truncated last copy still matches via offset search.
        let trunc = format!("{}{}", unit.repeat(3), &unit[..10]);
        assert!(phrase_repeat_at_least(&trunc, 3, 12, 48));
        assert!(!phrase_repeat_at_least("unique answer once", 2, 12, 48));
    }

    #[test]
    fn gemma4_uses_its_complete_eos_set() {
        assert_eq!(generation_eos_token_ids("gemma4", &[999]), vec![1, 106, 50]);
        assert_eq!(generation_eos_token_ids("gemma3", &[999]), vec![1, 106]);
        assert_eq!(
            generation_eos_token_ids("llama3.1", &[128009, 128001]),
            vec![128009, 128001]
        );
    }

    /// The defect this guards: Gemma's tokenizer post-processor is
    /// `<bos> A <eos>`, so a prompt encoded with `add_special_tokens = true`
    /// ends with `<eos>`. Prefilling that tells the model the sequence is over
    /// and then asks it to continue — degraded text on every turn, with the
    /// physics stack completely innocent.
    ///
    /// Skips (rather than fails) when the tokenizer asset is absent, so the
    /// suite still runs on a machine without the model files.
    #[test]
    fn gemma_prompt_never_ends_in_eos() {
        let path = std::path::Path::new("data/google/tokenizer.json");
        let Ok(tokenizer) = tokenizers::Tokenizer::from_file(path) else {
            eprintln!("skipped: {} not present", path.display());
            return;
        };
        let eos = generation_eos_token_ids("gemma3", &[]);
        let prompt = format_multiturn_prompt(&[(true, "hi".to_string())], "gemma3");

        // Establish the defect is real for this asset, so this test fails
        // loudly if the tokenizer is ever swapped for one that behaves
        // differently — otherwise the assertion below would pass vacuously.
        let raw = tokenizer.encode(prompt.as_str(), true).expect("encode");
        assert_eq!(
            raw.get_ids().last(),
            Some(&1u32),
            "expected the raw encode to append <eos>; asset may have changed"
        );

        let fixed = encode_prompt_no_trailing_eos(&tokenizer, &prompt, &eos).expect("encode");
        assert!(
            !eos.contains(fixed.last().expect("non-empty")),
            "prompt still ends with an EOS token: {:?}",
            fixed.last()
        );
        // BOS is the post-processor's job and must survive untouched.
        assert_eq!(fixed.first(), Some(&2u32), "leading <bos> was lost");
    }

    /// Families whose tokenizer appends no EOS must come through byte-identical
    /// — the fix has to be surgical, since Llama and Qwen already read clean.
    #[test]
    fn encode_is_a_no_op_when_no_trailing_eos() {
        let path = std::path::Path::new("data/google/tokenizer.json");
        let Ok(tokenizer) = tokenizers::Tokenizer::from_file(path) else {
            return;
        };
        // Empty EOS set stands in for "post-processor appends nothing".
        let raw = tokenizer.encode("hello there", true).expect("encode");
        let fixed = encode_prompt_no_trailing_eos(&tokenizer, "hello there", &[]).expect("encode");
        assert_eq!(fixed, raw.get_ids().to_vec());
    }

    fn args(parts: &[&str]) -> Vec<String> {
        std::iter::once("hydrodynamic-swarm".to_string())
            .chain(parts.iter().map(|s| s.to_string()))
            .collect()
    }

    /// `--d-run` without `--no-endocrine` still disables endocrine; physics required; hooks off.
    #[test]
    fn d_run_forces_endocrine_off_physics_required_hooks_off() {
        let seat = d_run_seat_policy(&args(&["--d-run"]), 512, true);
        assert!(seat.d_run);
        assert!(
            !seat.endocrine_enabled,
            "endocrine must be off on --d-run even without --no-endocrine"
        );
        assert!(seat.physics_required);
        assert!(!seat.hooks_enabled);
        assert!(seat.eos_masked);
        assert_eq!(seat.max_tokens, 131_072);
    }

    #[test]
    fn d_run_honors_short_token_budget() {
        let seat = d_run_seat_policy(&args(&["--d-run", "--tokens", "200"]), 512, true);
        assert!(!seat.endocrine_enabled);
        assert!(seat.physics_required);
        assert!(!seat.hooks_enabled);
        assert_eq!(seat.max_tokens, 200);
    }

    #[test]
    fn chat_without_d_run_keeps_endocrine_on() {
        let seat = d_run_seat_policy(&args(&["--chat"]), 512, true);
        assert!(!seat.d_run);
        assert!(seat.endocrine_enabled);
        assert!(seat.hooks_enabled);
        assert!(!seat.physics_required);
        assert_eq!(seat.max_tokens, 512);
    }

    #[test]
    fn no_endocrine_flag_still_works_off_d_run() {
        let seat = d_run_seat_policy(&args(&["--no-endocrine"]), 64, false);
        assert!(!seat.d_run);
        assert!(!seat.endocrine_enabled);
        assert!(!seat.hooks_enabled);
        assert!(!seat.physics_required);
    }

    #[test]
    fn hydro_inject_synth_spike_is_angle_tag() {
        assert_eq!(hydro_inject_synth("spike").as_deref(), Some("<spike>"));
        assert_eq!(hydro_inject_synth("none"), None);
        assert_eq!(
            hydro_inject_synth("remember").as_deref(),
            Some("<remember>k=v</remember>")
        );
        let hits = control_tags::scan_hits(&hydro_inject_synth("spike").unwrap());
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].tag, control_tags::ControlTag::Spike);
    }

    /// Shipped inject path generate_turn_ex uses: synth → scan → fire_tag.
    /// SPIKE blend 6.5 must move later residual Δh and QSMA β (not T/rep).
    #[test]
    fn hydro_inject_spike_scan_fire_moves_later_residual_and_qsma() {
        use candle_core::{Device, Tensor};
        use gpu::CpuBackend;

        let device = Device::Cpu;
        let field = ContinuousField::load_dummy(4, 8, &device).unwrap();
        let memory = SplatMemory::new(device.clone());
        let mut live =
            NiodooEngine::new(field, memory, Box::new(CpuBackend::new()), 0.035, 0.25, 5.0);
        let baseline = Tensor::new(&[[1.25f32, -2.5, 3.75, -4.0]], &device).unwrap();
        let goal = Tensor::zeros(4, candle_core::DType::F32, &device).unwrap();

        let pre = live.steer(&baseline, &goal, 4).unwrap();
        assert!((pre.physics_blend - 1.0).abs() < 1e-6);
        let scheduled_beta = live.qsma_beta(4);
        live.hands.kinetic_noise = 0.0;
        let mut logits_pre = vec![0.2f32, 1.1, 0.8, 0.4];
        live.apply_qsma_logits(&mut logits_pre, &[], 4);

        let applied = apply_hydro_inject(&mut live, "spike");
        assert_eq!(applied.len(), 1);
        assert_eq!(applied[0].tag, control_tags::ControlTag::Spike);
        assert!((live.hands.physics_blend - 6.5).abs() < 1e-6);
        assert!((live.qsma_beta(4) - 1.5).abs() < 1e-12);
        assert!((live.qsma_beta(4) - scheduled_beta).abs() > 1e-6);

        let post = live.steer(&baseline, &goal, 4).unwrap();
        assert!(
            (post.delta_h_norm - pre.delta_h_norm).abs() > 1e-6,
            "later residual must move after inject; pre Δh={} post Δh={}",
            pre.delta_h_norm,
            post.delta_h_norm
        );
        assert!((post.physics_blend - 6.5).abs() < 1e-6);
        assert!((live.hands.kinetic_noise - 1.5).abs() < 1e-6);

        live.hands.kinetic_noise = 0.0;
        let mut logits_post = vec![0.2f32, 1.1, 0.8, 0.4];
        live.apply_qsma_logits(&mut logits_post, &[], 4);
        assert!(
            logits_pre
                .iter()
                .zip(logits_post.iter())
                .any(|(a, b)| (a - b).abs() > 1e-5),
            "later QSMA logits must perturb; pre={logits_pre:?} post={logits_post:?}"
        );
    }

    #[test]
    fn take_hydro_inject_tag_consumes_once() {
        std::env::set_var("HYDRO_INJECT_TAG", "spike");
        assert_eq!(take_hydro_inject_tag().as_deref(), Some("spike"));
        assert_eq!(take_hydro_inject_tag(), None);
        assert!(std::env::var("HYDRO_INJECT_TAG").is_err());
    }

    /// Shipped chat deposit → save → load → query_force. Empty store is distinct.
    #[test]
    fn chat_will_deposit_save_load_query_force_roundtrip() {
        use gpu::CpuBackend;

        let device = Device::Cpu;
        let field = ContinuousField::load_dummy(4, 8, &device).unwrap();
        let memory = SplatMemory::new(device.clone());
        let mut engine =
            NiodooEngine::new(field, memory, Box::new(CpuBackend::new()), 0.035, 0.25, 5.0);
        let pos = Tensor::zeros(&[4], candle_core::DType::F32, &device).unwrap();
        assert!(
            deposit_chat_will(&mut engine, &pos, 1.0, 5.0, 0.0).unwrap(),
            "first chat will must land"
        );
        assert!(engine.memory().len() >= 1);

        let dir = std::env::temp_dir().join(format!(
            "hydro_chat_will_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("splat_memory.safetensors");
        persist_splat_store(&engine, &path).unwrap();
        drop(engine);

        let field = ContinuousField::load_dummy(4, 8, &device).unwrap();
        let memory = SplatMemory::new(device.clone());
        let mut revived =
            NiodooEngine::new(field, memory, Box::new(CpuBackend::new()), 0.035, 0.25, 5.0);
        let n = load_splat_store(&mut revived, &path).unwrap();
        assert!(n >= 1, "reload must restore at least one will, got {n}");

        let probe = Tensor::new(&[1.0f32, 0.0, 0.0, 0.0], &device).unwrap();
        let force = revived.memory().query_force(&probe).unwrap();
        let fv: Vec<f32> = force.to_vec1().unwrap();
        let mag = fv.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            mag > 1e-6,
            "reloaded store must steer; |F|={mag} force={fv:?}"
        );

        let empty = SplatMemory::new(device.clone());
        let z = empty.query_force(&probe).unwrap();
        let zv: Vec<f32> = z.to_vec1().unwrap();
        assert!(
            zv.iter().all(|x| x.abs() < 1e-6),
            "cleared/empty store must be ~0, got {zv:?}"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Chat write→query at the prefill basin vs a far/empty store.
    #[test]
    fn chat_prefill_bridge_query_near_not_far_or_empty() {
        use gpu::CpuBackend;

        let device = Device::Cpu;
        let field = ContinuousField::load_dummy(4, 8, &device).unwrap();
        let memory = SplatMemory::new(device.clone());
        let mut engine =
            NiodooEngine::new(field, memory, Box::new(CpuBackend::new()), 0.035, 0.25, 5.0);
        let goal = Tensor::zeros(&[4], candle_core::DType::F32, &device).unwrap();
        let empty = chat_basin_query(&engine, &goal).unwrap();
        assert_eq!(empty.3, 0.0, "empty |F_s| must be 0, got {}", empty.3);
        assert!(empty.2.abs() < 1e-8, "empty pot must be 0, got {}", empty.2);

        let n = mint_chat_prefill_bridge_at(&mut engine, &goal, 90.0, 0.75, 0.005, 0.35, 0xabcdu32)
            .unwrap();
        assert!(n >= 1);

        let near = chat_basin_query(&engine, &goal).unwrap();
        assert!(
            near.3 > 1e-4,
            "basin |F_s| must be non-zero after offset bridge, got {}",
            near.3
        );
        assert!(
            near.2.abs() > 1e-4,
            "basin pot must be non-zero, got {}",
            near.2
        );
        assert!(
            near.0.is_finite() && near.0 < 3.0 * near.1.max(1.0),
            "nearest should sit in the offset ring, nearest={} σ={}",
            near.0,
            near.1
        );

        let far = Tensor::new(&[10_000.0f32, 10_000.0, 10_000.0, 10_000.0], &device).unwrap();
        let cold = chat_basin_query(&engine, &far).unwrap();
        assert!(
            cold.2.abs() < near.2.abs() * 0.1,
            "novel/far pot must be much smaller than basin: far={} near={}",
            cold.2,
            near.2
        );
        assert!(
            cold.3 < near.3 * 0.1,
            "novel/far |F_s| must be much smaller than basin: far={} near={}",
            cold.3,
            near.3
        );
    }

    #[test]
    fn blend_topic_logits_raises_scar_token_and_identity_at_zero() {
        let device = Device::Cpu;
        // probe peaked at idx 1; scar peaked at idx 2 (content the probe does not prefer).
        let probe = Tensor::new(&[[0.0f32, 5.0, 0.2, 1.0]], &device).unwrap();
        let scar = Tensor::new(&[[0.1f32, 0.0, 8.0, 0.3]], &device).unwrap();

        let ident = blend_topic_logits(&probe, &scar, 0.0).unwrap();
        let iv: Vec<f32> = ident.flatten_all().unwrap().to_vec1().unwrap();
        let pv: Vec<f32> = probe.flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(iv, pv, "mix=0 must be the probe");

        let mixed = blend_topic_logits(&probe, &scar, 0.55).unwrap();
        let mv: Vec<f32> = mixed.flatten_all().unwrap().to_vec1().unwrap();
        assert!(
            mv[2] > pv[2] + 3.0,
            "scar token logit must rise under mix; probe={pv:?} mixed={mv:?}"
        );
        let argmax = mv
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap();
        assert_eq!(
            argmax, 2,
            "shipped blend must make the scar's token win; mixed={mv:?}"
        );
    }

    /// Chat write→save→load of a decode trail: matching fp step 1 is the
    /// minted residual; a novel fp is empty; keep does not replace; two fps stay distinct.
    #[test]
    fn chat_decode_trail_write_save_load_step_mu() {
        use gpu::CpuBackend;
        use memory::TrailCommit;

        let device = Device::Cpu;
        let mut memory = SplatMemory::new(device.clone());
        let goal_a = Tensor::zeros(&[4], candle_core::DType::F32, &device).unwrap();
        let goal_b = Tensor::new(&[0.0f32, 1.0, 0.0, 0.0], &device).unwrap();
        memory
            .deposit_prefill_bridge(&goal_a, 90.0, 0.75, 0.005, 90.0, 0.35, 0xabcdu32)
            .unwrap();
        memory
            .deposit_prefill_bridge(&goal_b, 90.0, 0.75, 0.005, 90.0, 0.35, 0x1111u32)
            .unwrap();
        let mu0 = Tensor::new(&[4.0f32, 0.0, 0.0, 0.0], &device).unwrap();
        let mu1 = Tensor::new(&[0.0f32, 7.0, 0.0, 0.0], &device).unwrap();
        assert_eq!(
            memory
                .commit_decode_trail(0xabcdu32, vec![mu0, mu1], vec![10, 20])
                .unwrap(),
            TrailCommit::Minted(2)
        );
        let mu_fail = Tensor::new(&[9.0f32, 9.0, 9.0, 9.0], &device).unwrap();
        assert_eq!(
            memory
                .commit_decode_trail(0xabcd, vec![mu_fail], vec![99])
                .unwrap(),
            TrailCommit::Kept(2)
        );
        let mu_b = Tensor::new(&[0.0f32, 0.0, 3.0, 0.0], &device).unwrap();
        assert_eq!(
            memory
                .commit_decode_trail(0x1111, vec![mu_b], vec![30])
                .unwrap(),
            TrailCommit::Minted(1)
        );

        let dir = std::env::temp_dir().join(format!(
            "hydro_chat_trail_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("splat_memory.safetensors");
        let field = ContinuousField::load_dummy(4, 8, &device).unwrap();
        let engine =
            NiodooEngine::new(field, memory, Box::new(CpuBackend::new()), 0.035, 0.25, 5.0);
        persist_splat_store(&engine, &path).unwrap();

        let mut loaded = SplatMemory::new(device.clone());
        loaded.load(&path).unwrap();
        assert_eq!(loaded.decode_trail_len(0xabcd), 2);
        assert_eq!(loaded.decode_trail_len(0x1111), 1);
        assert_eq!(loaded.matched_trail_token(0xabcd, 1), Some(20));
        assert_eq!(loaded.matched_trail_token(0x1111, 0), Some(30));
        let step1: Vec<f32> = loaded
            .matched_trail_mu(0xabcd, 1)
            .unwrap()
            .unwrap()
            .to_vec1()
            .unwrap();
        assert!(
            (step1[1] - 7.0).abs() < 1e-5,
            "matching reload must read minted step-1 residual, got {step1:?}"
        );
        assert!(loaded.matched_trail_mu(0x2222, 0).unwrap().is_none());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
