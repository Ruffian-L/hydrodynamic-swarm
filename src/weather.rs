//! TermSplat live pipe — FieldFrame-shaped JSONL (stateless contract).
//!
//! Same shape as termsplat `frame.rs`. Swarm emits; TermSplat paints.
//! No shared crate — one JSON object per line. Stateless on purpose.

use crate::memory::SplatMemory;
use crate::{hooks::HookReport, logit_physics::ChainReport};
use serde::Serialize;
use std::fs::OpenOptions;
use std::io::Write;
use std::path::Path;

/// Match termsplat ChaosTier::from_entropy thresholds.
fn tier_name(entropy: f32) -> &'static str {
    if entropy < 1.5 {
        "zen"
    } else if entropy < 3.0 {
        "busy"
    } else if entropy < 4.0 {
        "rotting"
    } else if entropy < 5.5 {
        "json_scream"
    } else {
        "collapse"
    }
}

#[derive(Serialize, Clone)]
struct SplatPointOut {
    x: f32,
    y: f32,
    z: f32,
    sigma: f32,
    alpha: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    label: Option<String>,
}

#[derive(Serialize)]
struct FieldFrameOut {
    tick: u64,
    entropy: f32,
    tier: String,
    splats: Vec<SplatPointOut>,
    #[serde(skip_serializing_if = "Option::is_none")]
    token: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    note: Option<String>,
}

/// Append-only TermSplat JSONL writer (one FieldFrame per generation step).
pub struct WeatherPipe {
    file: std::fs::File,
    /// Fixed 3-col projection of high-D scar μ → viz cube (seed 7).
    proj: Vec<f32>,
    dim: usize,
}

impl WeatherPipe {
    /// Open next to session log: `foo.jsonl` → `foo.termsplat.jsonl`, plus `logs/latest.termsplat.jsonl`.
    pub fn open_beside_log(log_path: &Path, dim: usize) -> std::io::Result<Self> {
        let stem = log_path.with_extension("termsplat.jsonl");
        let file = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(&stem)?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let _ = std::fs::set_permissions(&stem, std::fs::Permissions::from_mode(0o664));
        }
        // latest pointer for `termsplat pipe logs/latest.termsplat.jsonl --follow`
        if let Some(dir) = log_path.parent() {
            let latest = dir.join("latest.termsplat.jsonl");
            let _ = std::fs::remove_file(&latest);
            #[cfg(unix)]
            {
                if let Some(name) = stem.file_name() {
                    let _ = std::os::unix::fs::symlink(name, &latest);
                }
            }
            #[cfg(not(unix))]
            {
                let _ = std::fs::write(&latest, format!("{}\n", stem.display()));
            }
        }
        println!("    TermSplat weather: {}", stem.display());
        println!("    TermSplat latest:  logs/latest.termsplat.jsonl");
        Ok(Self {
            file,
            proj: build_projection(dim, 7),
            dim,
        })
    }

    /// Emit one frame: entropy = model top-k H, splats = projected scars + head crumb.
    pub fn emit_step(
        &mut self,
        step: usize,
        entropy: f32,
        token: &str,
        delta_norm: f32,
        memory: &SplatMemory,
        logit: &ChainReport,
        hook: &HookReport,
    ) -> anyhow::Result<()> {
        let mut splats = project_scars(memory, &self.proj, self.dim, 48)?;
        // Trajectory head: put current token at origin-ish with entropy-sized blob
        let head_r = (0.15 + entropy * 0.04).min(0.55);
        let ang = (step as f32) * 0.17;
        splats.push(SplatPointOut {
            x: (ang.cos() * head_r * 0.4).clamp(-1.2, 1.2),
            y: (ang.sin() * head_r * 0.4).clamp(-1.2, 1.2),
            z: (delta_norm / 200.0).clamp(-1.0, 1.0),
            sigma: (0.08 + entropy * 0.03).min(0.4),
            alpha: 0.95,
            label: token
                .chars()
                .find(|c| !c.is_whitespace())
                .map(|c| c.to_string()),
        });

        let note = format!(
            "δ={:.1} scars={} logit={:.3}/{:.3}/{:.3} hook={}:{:.4}",
            delta_norm,
            memory.len(),
            logit.field_mag,
            logit.splat_mag,
            logit.governor_mag,
            hook.applications,
            hook.delta_max,
        );
        let frame = FieldFrameOut {
            tick: step as u64,
            entropy,
            tier: tier_name(entropy).to_string(),
            splats,
            token: Some(token.trim().to_string()),
            note: Some(note),
        };
        let line = serde_json::to_string(&frame)?;
        writeln!(self.file, "{}", line)?;
        self.file.flush()?;
        Ok(())
    }
}

fn build_projection(dim: usize, seed: u64) -> Vec<f32> {
    let mut rng = seed;
    let scale = 1.0 / (dim as f32).sqrt();
    let mut p = vec![0.0f32; dim * 3];
    for v in p.iter_mut() {
        rng = rng
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let u = (rng >> 33) as f32 / (1u64 << 31) as f32;
        rng = rng
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let w = (rng >> 33) as f32 / (1u64 << 31) as f32;
        *v = (u + w - 1.0) * scale;
    }
    p
}

fn project_vec(v: &[f32], proj: &[f32], dim: usize) -> [f32; 3] {
    let mut out = [0.0f32; 3];
    let n = dim.min(v.len());
    for j in 0..3 {
        let mut s = 0.0f32;
        for i in 0..n {
            s += v[i] * proj[i * 3 + j];
        }
        out[j] = s;
    }
    out
}

fn project_scars(
    memory: &SplatMemory,
    proj: &[f32],
    dim: usize,
    max_n: usize,
) -> anyhow::Result<Vec<SplatPointOut>> {
    let scars = memory.splats_ref();
    let n = scars.len().min(max_n);
    let start = scars.len().saturating_sub(n);
    let mut raw: Vec<([f32; 3], f32, f32)> = Vec::with_capacity(n);
    for s in &scars[start..] {
        let flat: Vec<f32> = s.mu.flatten_all()?.to_vec1()?;
        raw.push((project_vec(&flat, proj, dim), s.sigma, s.alpha));
    }
    // Normalize projected points into ~[-1.1, 1.1]
    if raw.is_empty() {
        return Ok(Vec::new());
    }
    let mut min = [f32::INFINITY; 3];
    let mut max = [f32::NEG_INFINITY; 3];
    for (p, _, _) in &raw {
        for i in 0..3 {
            min[i] = min[i].min(p[i]);
            max[i] = max[i].max(p[i]);
        }
    }
    let mut out = Vec::with_capacity(raw.len());
    for (p, sigma, alpha) in raw {
        let mut q = [0.0f32; 3];
        for i in 0..3 {
            let span = (max[i] - min[i]).max(1e-6);
            q[i] = (((p[i] - min[i]) / span) * 2.2 - 1.1).clamp(-1.2, 1.2);
        }
        // sigma in residual space is huge; compress for term density map
        let sig = (sigma.abs() * 0.002 + 0.06).clamp(0.05, 0.45);
        out.push(SplatPointOut {
            x: q[0],
            y: q[1],
            z: q[2],
            sigma: sig,
            alpha: alpha.clamp(-1.0, 1.0),
            label: if alpha < 0.0 {
                Some("P".into())
            } else if alpha > 0.3 {
                Some("+".into())
            } else {
                None
            },
        });
    }
    Ok(out)
}
