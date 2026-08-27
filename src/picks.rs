//! SplatRAG `pick` → hydro scar bridge.
//!
//! Spec: SplatRAG `docs/BRIDGE_SPLATRAG_PICK.md`.
//!
//! Core rule: embed the pick **text** with this host's residual path. Never inject
//! `semantics_64` into residual space (64-d unit Qwen vectors vs un-normalized
//! Gemma residual at ‖μ‖ ~ O(100)).
//!
//! Gain = meaning / steering α. Mass < 0 = repel on the force side. Do not glue them.

#![allow(dead_code)]

use crate::memory::SplatMemory;
use crate::tct;
use anyhow::{bail, Context, Result};
use candle_core::Tensor;
use serde::Deserialize;
use std::path::Path;

/// Collapse onset used by SplatRAG picker docs (~0.40). Local ceiling defaults under that.
pub const DEFAULT_PICKS_MAX_GAIN: f32 = 0.35;

/// Wire types — subset of SplatRAG `MemoryPickSet` / `Pick`. Extra JSON fields ignored.
#[derive(Debug, Clone, Deserialize)]
pub struct MemoryPickSet {
    pub version: u32,
    pub prompt: String,
    pub source_embedder: String,
    pub source_dim: usize,
    #[serde(default)]
    pub confidence: f32,
    #[serde(default)]
    pub separation: f32,
    #[serde(default)]
    pub total_suggested_gain: f32,
    pub picks: Vec<Pick>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Pick {
    /// UUID string on the wire (no uuid crate required here).
    pub memory_id: String,
    /// **The payload.** Host embeds this.
    pub text: String,
    #[serde(default)]
    pub text_truncated: bool,
    #[serde(default)]
    pub injection: String,
    #[serde(default)]
    pub score: f32,
    #[serde(default)]
    pub cosine: f32,
    /// Telemetry only — never write into residual.
    #[serde(default)]
    pub semantics_64: Vec<f32>,
    /// Authoritative steered α. 0.0 = unsteered.
    #[serde(default)]
    pub gain: f32,
    /// Picker proposal (budget share × confidence). Ignore freely.
    #[serde(default)]
    pub suggested_gain: f32,
    #[serde(default)]
    pub budget_share: f32,
    /// Negative = repel in field / force law.
    #[serde(default = "default_mass")]
    pub mass: f32,
    pub basin_id: Option<String>,
    pub basin_label: Option<String>,
    #[serde(default)]
    pub domain: String,
    #[serde(default)]
    pub source: String,
}

fn default_mass() -> f32 {
    1.0
}

#[derive(Debug, Clone)]
pub struct ImportPicksOpts {
    pub max_gain: f32,
    pub dry_run: bool,
    /// Prefer recorded `gain` when |gain| > this; else use `suggested_gain`.
    pub gain_eps: f32,
    pub sigma: f32,
    pub lambda: f32,
    pub offset_frac: f32,
    pub replace_dist: f32,
}

impl Default for ImportPicksOpts {
    fn default() -> Self {
        Self {
            max_gain: DEFAULT_PICKS_MAX_GAIN,
            dry_run: false,
            gain_eps: 1e-6,
            sigma: 90.0,
            lambda: 0.005,
            offset_frac: 0.35,
            replace_dist: 90.0 * 1.35,
        }
    }
}

#[derive(Debug, Clone)]
pub struct PickDepositReport {
    pub memory_id: String,
    pub text_chars: usize,
    pub text_truncated: bool,
    pub mu_l2: f32,
    pub alpha: f32,
    pub mass: f32,
    pub used_suggested: bool,
    pub deposited: bool,
    pub note: String,
}

#[derive(Debug, Clone)]
pub struct ImportPicksReport {
    pub path: String,
    pub prompt: String,
    pub confidence: f32,
    pub separation: f32,
    pub total_suggested_gain: f32,
    pub total_applied_gain: f32,
    pub dry_run: bool,
    pub deposits: Vec<PickDepositReport>,
}

/// Load and validate a pick JSON file.
pub fn load_pick_set(path: &Path) -> Result<MemoryPickSet> {
    let raw = std::fs::read_to_string(path)
        .with_context(|| format!("read pick file {}", path.display()))?;
    let set: MemoryPickSet = serde_json::from_str(&raw)
        .with_context(|| format!("parse pick JSON {}", path.display()))?;
    validate_provenance(&set)?;
    Ok(set)
}

/// Refuse unknown provenance so 64-d floats are never treated as residual centers.
pub fn validate_provenance(set: &MemoryPickSet) -> Result<()> {
    if set.source_dim != 64 {
        bail!(
            "unknown pick provenance: source_dim={} (expected 64); refuse rather than guess",
            set.source_dim
        );
    }
    if !set.source_embedder.starts_with("Qwen3-Embedding")
        && !set.source_embedder.to_ascii_lowercase().contains("qwen3")
    {
        bail!(
            "unknown pick provenance: embedder {:?} @ dim {}; expected Qwen3-Embedding*",
            set.source_embedder,
            set.source_dim
        );
    }
    for (i, p) in set.picks.iter().enumerate() {
        if !p.injection.is_empty() && p.injection != "text" {
            bail!(
                "pick[{i}] injection={:?}: only text injection is supported (never semantics_64)",
                p.injection
            );
        }
        if p.text.trim().is_empty() {
            bail!("pick[{i}] has empty text (memory_id={})", p.memory_id);
        }
    }
    Ok(())
}

/// Resolve deposit α: authoritative `gain` if set, else picker `suggested_gain`.
/// `mass < 0` forces repel (negative α) without inventing a second force law.
pub fn resolve_alpha(pick: &Pick, max_gain: f32, gain_eps: f32) -> (f32, bool) {
    let (raw, used_suggested) = if pick.gain.abs() > gain_eps {
        (pick.gain, false)
    } else {
        (pick.suggested_gain, true)
    };
    let mut alpha = raw.clamp(-max_gain, max_gain);
    if pick.mass < 0.0 {
        // Repel: keep magnitude, force pain/push sign.
        alpha = -alpha.abs();
    }
    (alpha, used_suggested)
}

fn tensor_l2(t: &Tensor) -> Result<f32> {
    Ok(t.sqr()?.sum_all()?.to_scalar::<f32>()?.sqrt())
}

/// Import every pick as a prefill-bridge-style scar (or dry-run log only).
///
/// `embed_text` must return a residual-space center in the **live generation**
/// model (last-token hidden after prefill of `text`). Never pass L2-normalized
/// embedding-shell vectors here.
pub fn import_picks<F>(
    set: &MemoryPickSet,
    path: &Path,
    memory: &mut SplatMemory,
    opts: &ImportPicksOpts,
    mut embed_text: F,
) -> Result<ImportPicksReport>
where
    F: FnMut(&str) -> Result<Tensor>,
{
    let mut deposits = Vec::with_capacity(set.picks.len());
    let mut total_applied = 0.0f32;

    for pick in &set.picks {
        let (alpha, used_suggested) = resolve_alpha(pick, opts.max_gain, opts.gain_eps);
        let mu = embed_text(&pick.text)
            .with_context(|| format!("residual embed for pick {}", pick.memory_id))?;
        let mu_l2 = tensor_l2(&mu)?;

        let mut note = String::new();
        if pick.mass < 0.0 {
            note.push_str("mass<0→repel; ");
        }
        if used_suggested {
            note.push_str("α from suggested_gain; ");
        } else {
            note.push_str("α from gain; ");
        }
        if pick.text_truncated {
            note.push_str("text_truncated; ");
        }
        // Norm family check: existing scars ~ O(100). Flag order-of-magnitude misses.
        if mu_l2 < 10.0 || mu_l2 > 2000.0 {
            note.push_str(&format!(
                "WARN mu_l2={mu_l2:.2} outside scar family (~100–200); check encoder/pooling; "
            ));
        }

        let mut deposited = false;
        if !opts.dry_run && alpha.abs() > 1e-8 {
            let fp = tct::prompt_fp(&format!("pick:{}", pick.memory_id));
            memory.deposit_prefill_bridge(
                &mu,
                opts.sigma,
                alpha,
                opts.lambda,
                opts.replace_dist,
                opts.offset_frac,
                fp,
            )?;
            deposited = true;
            total_applied += alpha.abs();
        } else if opts.dry_run {
            note.push_str("dry-run (no deposit); ");
            total_applied += alpha.abs();
        } else {
            note.push_str("α≈0 skip deposit; ");
        }

        deposits.push(PickDepositReport {
            memory_id: pick.memory_id.clone(),
            text_chars: pick.text.chars().count(),
            text_truncated: pick.text_truncated,
            mu_l2,
            alpha,
            mass: pick.mass,
            used_suggested,
            deposited,
            note: note.trim().to_string(),
        });
    }

    Ok(ImportPicksReport {
        path: path.display().to_string(),
        prompt: set.prompt.clone(),
        confidence: set.confidence,
        separation: set.separation,
        total_suggested_gain: set.total_suggested_gain,
        total_applied_gain: total_applied,
        dry_run: opts.dry_run,
        deposits,
    })
}

/// Human-readable summary for the session banner.
pub fn print_report(report: &ImportPicksReport) {
    println!(
        "    [PICKS] {}  dry_run={}  n={}  confidence={:.3}  separation={:.3}  total_suggested_α={:.4}  total_applied|α|={:.4}",
        report.path,
        report.dry_run,
        report.deposits.len(),
        report.confidence,
        report.separation,
        report.total_suggested_gain,
        report.total_applied_gain
    );
    println!(
        "    [PICKS] prompt: {}",
        truncate_chars(&report.prompt, 120)
    );
    for (i, d) in report.deposits.iter().enumerate() {
        println!(
            "    [PICKS]   #{i} id={}  ‖μ‖={:.2}  α={:+.4}  mass={:.3}  chars={}  deposited={}  {}",
            short_id(&d.memory_id),
            d.mu_l2,
            d.alpha,
            d.mass,
            d.text_chars,
            d.deposited,
            d.note
        );
    }
    if report
        .deposits
        .iter()
        .any(|d| d.mu_l2 < 10.0 || d.mu_l2 > 2000.0)
    {
        println!(
            "    [PICKS] WARN: at least one ‖μ‖ outside scar family — abort generation if this is unexpected"
        );
    }
}

fn short_id(id: &str) -> &str {
    if id.len() > 8 {
        &id[..8]
    } else {
        id
    }
}

fn truncate_chars(s: &str, max: usize) -> String {
    let mut out: String = s.chars().take(max).collect();
    if s.chars().count() > max {
        out.push('…');
    }
    out
}

/// Unit tests that do not need a GPU model.
#[cfg(test)]
mod tests {
    use super::*;

    fn sample_pick(gain: f32, suggested: f32, mass: f32) -> Pick {
        Pick {
            memory_id: "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee".into(),
            text: "Friendship is a shared residual basin.".into(),
            text_truncated: false,
            injection: "text".into(),
            score: 1.0,
            cosine: 0.5,
            semantics_64: vec![0.0; 64],
            gain,
            suggested_gain: suggested,
            budget_share: 1.0,
            mass,
            basin_id: None,
            basin_label: None,
            domain: "chat".into(),
            source: "test".into(),
        }
    }

    fn sample_set(picks: Vec<Pick>) -> MemoryPickSet {
        MemoryPickSet {
            version: 1,
            prompt: "physics of friendship".into(),
            source_embedder: "Qwen3-Embedding-8B".into(),
            source_dim: 64,
            confidence: 0.3,
            separation: 0.05,
            total_suggested_gain: picks.iter().map(|p| p.suggested_gain).sum(),
            picks,
        }
    }

    #[test]
    fn provenance_accepts_qwen3() {
        let set = sample_set(vec![sample_pick(0.0, 0.12, 1.0)]);
        assert!(validate_provenance(&set).is_ok());
    }

    #[test]
    fn provenance_rejects_wrong_dim() {
        let mut set = sample_set(vec![sample_pick(0.0, 0.12, 1.0)]);
        set.source_dim = 2560;
        assert!(validate_provenance(&set).is_err());
    }

    #[test]
    fn provenance_rejects_vector_injection() {
        let mut p = sample_pick(0.0, 0.12, 1.0);
        p.injection = "semantics_64".into();
        let set = sample_set(vec![p]);
        assert!(validate_provenance(&set).is_err());
    }

    #[test]
    fn alpha_prefers_gain_over_suggested() {
        let (a, used) = resolve_alpha(&sample_pick(0.2, 0.05, 1.0), 0.35, 1e-6);
        assert!((a - 0.2).abs() < 1e-5);
        assert!(!used);
    }

    #[test]
    fn alpha_falls_back_to_suggested() {
        let (a, used) = resolve_alpha(&sample_pick(0.0, 0.12, 1.0), 0.35, 1e-6);
        assert!((a - 0.12).abs() < 1e-5);
        assert!(used);
    }

    #[test]
    fn mass_negative_forces_repel() {
        let (a, _) = resolve_alpha(&sample_pick(0.2, 0.0, -1.0), 0.35, 1e-6);
        assert!((a + 0.2).abs() < 1e-5);
    }

    #[test]
    fn alpha_clamped_to_max() {
        let (a, _) = resolve_alpha(&sample_pick(0.9, 0.0, 1.0), 0.35, 1e-6);
        assert!((a - 0.35).abs() < 1e-5);
    }

    #[test]
    fn parse_minimal_json() {
        let j = r#"{
          "version": 1,
          "prompt": "hi",
          "source_embedder": "Qwen3-Embedding-8B",
          "source_dim": 64,
          "confidence": 0.2,
          "separation": 0.1,
          "total_suggested_gain": 0.1,
          "picks": [{
            "memory_id": "00000000-0000-0000-0000-000000000001",
            "text": "hello residual",
            "injection": "text",
            "gain": 0.0,
            "suggested_gain": 0.1,
            "mass": 1.0
          }]
        }"#;
        let set: MemoryPickSet = serde_json::from_str(j).unwrap();
        validate_provenance(&set).unwrap();
        assert_eq!(set.picks.len(), 1);
    }
}
