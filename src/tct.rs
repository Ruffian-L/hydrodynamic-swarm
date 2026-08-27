//! TCT-Splat-Lite — portable memory wire format (bridge open item 14).
//!
//! Maps hydrodynamic-swarm **splats** onto the Niodoo Memory Bridge TCT vocabulary
//! so residual scars can leave this harness as a real language, not only
//! `safetensors` internal to hydro.
//!
//! Spec: `NIODOO_MEMORY_BRIDGE_SPEC_v0.2` §1 / §12 (open item 14).
//!
//! This is **not** full multi-layer ActAdd TCT-Core yet. It is the splat-native
//! subset that hydro already runs: residual center + σ (LOCALITY) + signed α (gain)
//! + λ (decay) + provenance-ish fields. Round-trip is tested.
//!
//! **Live apply (2026-07-16):** niodoo-live loads this binary via
//! `niodoo/src/tct_splat_lite.rs` (`--tct-splat-path`) and applies Gaussian force
//! in residual space. Dim must match the live model [INV-5] (Gemma 2560 ≠ Llama 4096).
//!
//! Claims discipline: no feelings. Memory as measurable geometry only.

#![allow(dead_code)]

use crate::memory::SplatMemory;
use crate::splat::{Splat, SplatScale};
use candle_core::Tensor;
use std::io::{Read, Write};
use std::path::Path;

/// Wire magic `TCT1` (little-endian bytes).
pub const TCT_MAGIC: [u8; 4] = *b"TCT1";
/// Schema version: 3 = per-record prompt_fp after trigger_kind.
pub const TCT_VERSION: u16 = 3;

/// flags: bit0 HAS_LOCALITY (always set for splat-lite)
pub const FLAG_HAS_LOCALITY: u16 = 1 << 0;
/// bit1: payload is residual-space centers (not embedding-shell)
pub const FLAG_RESIDUAL_SPACE: u16 = 1 << 1;

/// trigger_kind: surprise_delta (deposit on high-δ; valence in sign of alpha)
pub const TRIGGER_SURPRISE_DELTA: u32 = 3;
pub const TRIGGER_MANUAL: u32 = 4;
/// Prefill-bridge scar (continuity at next-run start basin). Flux mark 0.991 in memory.
pub const TRIGGER_PREFILL_BRIDGE: u32 = 5;

fn trigger_kind_for_splat(s: &Splat) -> u32 {
    if (s.flux - SplatMemory::PREFILL_BRIDGE_FLUX).abs() < 1e-4 {
        TRIGGER_PREFILL_BRIDGE
    } else {
        TRIGGER_SURPRISE_DELTA
    }
}

fn trigger_kind_label(k: u32) -> &'static str {
    match k {
        TRIGGER_SURPRISE_DELTA => "surprise_delta",
        TRIGGER_MANUAL => "manual",
        TRIGGER_PREFILL_BRIDGE => "prefill_bridge",
        _ => "unknown",
    }
}

/// One localized residual memory (one splat).
#[derive(Debug, Clone)]
pub struct TctLocalityRecord {
    pub center: Vec<f32>,
    pub sigma: f32,
    /// Signed gain: + pleasure, − pain (splat alpha).
    pub gain: f32,
    /// Decay / evaporation rate (splat lambda).
    pub decay_constant: f32,
    pub created_at_ms: u64,
    pub scale: u8,
    pub is_anchor: bool,
    pub trigger_kind: u32,
    /// FNV of prompt text for prefill-bridges (0 = unknown / trail scar).
    pub prompt_fp: u32,
}

/// Portable store: header + locality records.
#[derive(Debug, Clone)]
pub struct TctSplatStore {
    pub version: u16,
    pub flags: u16,
    pub model_dim: u32,
    /// Fingerprint of base model; 0 = unknown / not enforced yet.
    pub model_fp: u32,
    pub records: Vec<TctLocalityRecord>,
    /// Free-text provenance for JSON sidecar (not in binary).
    pub note: String,
}

impl TctSplatStore {
    pub fn from_memory(mem: &SplatMemory, model_dim: usize, model_fp: u32) -> anyhow::Result<Self> {
        let mut records = Vec::with_capacity(mem.len());
        for s in mem.splats_ref() {
            let center = s.mu.flatten_all()?.to_vec1::<f32>()?;
            if center.len() != model_dim && model_dim > 0 {
                // Allow export even if dim metadata lags; stamp actual length.
            }
            records.push(TctLocalityRecord {
                center,
                sigma: s.sigma,
                gain: s.alpha,
                decay_constant: s.lambda,
                created_at_ms: s.created_at.saturating_mul(1000),
                scale: s.scale as u8,
                is_anchor: s.is_anchor,
                trigger_kind: trigger_kind_for_splat(s),
                prompt_fp: SplatMemory::bridge_prompt_fp(s),
            });
        }
        let dim = if model_dim > 0 {
            model_dim as u32
        } else {
            records.first().map(|r| r.center.len() as u32).unwrap_or(0)
        };
        Ok(Self {
            version: TCT_VERSION,
            flags: FLAG_HAS_LOCALITY | FLAG_RESIDUAL_SPACE,
            model_dim: dim,
            model_fp,
            records,
            note: "tct-splat-lite from hydrodynamic-swarm; residual locality scars".into(),
        })
    }

    /// Rebuild CPU-side splats and append into `mem`.
    pub fn into_memory(self, mem: &mut SplatMemory) -> anyhow::Result<usize> {
        let device = mem.device().clone();
        let expected = mem.residual_dim();
        if expected > 0 && self.model_dim > 0 && self.model_dim as usize != expected {
            eprintln!(
                "[RESIDUAL MISMATCH] expected {expected} got {} at tct.into_memory.header_model_dim",
                self.model_dim
            );
            return Err(anyhow::anyhow!(
                "[RESIDUAL MISMATCH] expected {expected} got {} at tct.into_memory.header_model_dim",
                self.model_dim
            ));
        }
        let mut n = 0;
        for (i, r) in self.records.into_iter().enumerate() {
            let d = r.center.len();
            if d == 0 {
                continue;
            }
            if expected > 0 && d != expected {
                eprintln!(
                    "[RESIDUAL MISMATCH] expected {expected} got {d} at tct.into_memory.record[{i}].center"
                );
                return Err(anyhow::anyhow!(
                    "[RESIDUAL MISMATCH] expected {expected} got {d} at tct.into_memory.record[{i}].center"
                ));
            }
            let mu = Tensor::from_vec(r.center.clone(), d, &device)?;
            let mut splat = if r.is_anchor {
                Splat::anchor(mu, r.sigma, r.gain)
            } else {
                Splat::new(mu, r.sigma, r.gain)
            };
            splat.lambda = r.decay_constant;
            splat.created_at = r.created_at_ms / 1000;
            splat.scale = SplatScale::from_u8(r.scale);
            splat.current_dim = d;
            if r.trigger_kind == TRIGGER_PREFILL_BRIDGE {
                splat.flux = SplatMemory::PREFILL_BRIDGE_FLUX;
                splat.scale = SplatScale::Coarse;
                splat.friction = f32::from_bits(r.prompt_fp);
            }
            mem.add_splat(splat);
            n += 1;
        }
        Ok(n)
    }

    /// Binary write: TCT1 header + packed locality records (f32 centers).
    pub fn write_binary(&self, path: &Path) -> anyhow::Result<()> {
        let mut f = std::fs::File::create(path)?;
        f.write_all(&TCT_MAGIC)?;
        f.write_all(&self.version.to_le_bytes())?;
        f.write_all(&self.flags.to_le_bytes())?;
        f.write_all(&self.model_dim.to_le_bytes())?;
        f.write_all(&self.model_fp.to_le_bytes())?;
        let n = self.records.len() as u32;
        f.write_all(&n.to_le_bytes())?;
        // reserved 16 bytes for future (layer_start/end, quant, …)
        f.write_all(&[0u8; 16])?;

        for r in &self.records {
            let dim = r.center.len() as u32;
            f.write_all(&dim.to_le_bytes())?;
            f.write_all(&r.sigma.to_le_bytes())?;
            f.write_all(&r.gain.to_le_bytes())?;
            f.write_all(&r.decay_constant.to_le_bytes())?;
            f.write_all(&r.created_at_ms.to_le_bytes())?;
            f.write_all(&[r.scale, r.is_anchor as u8, 0, 0])?;
            f.write_all(&r.trigger_kind.to_le_bytes())?;
            f.write_all(&r.prompt_fp.to_le_bytes())?;
            for &x in &r.center {
                f.write_all(&x.to_le_bytes())?;
            }
        }
        Ok(())
    }

    pub fn read_binary(path: &Path) -> anyhow::Result<Self> {
        let mut f = std::fs::File::open(path)?;
        let mut magic = [0u8; 4];
        f.read_exact(&mut magic)?;
        if magic != TCT_MAGIC {
            anyhow::bail!(
                "bad TCT magic: {:?} (expected TCT1) — not a tct-splat-lite file",
                magic
            );
        }
        let version = read_u16(&mut f)?;
        let flags = read_u16(&mut f)?;
        let model_dim = read_u32(&mut f)?;
        let model_fp = read_u32(&mut f)?;
        let n = read_u32(&mut f)? as usize;
        let mut skip = [0u8; 16];
        f.read_exact(&mut skip)?;

        let mut records = Vec::with_capacity(n);
        for _ in 0..n {
            let dim = read_u32(&mut f)? as usize;
            let sigma = read_f32(&mut f)?;
            let gain = read_f32(&mut f)?;
            let decay_constant = read_f32(&mut f)?;
            let created_at_ms = read_u64(&mut f)?;
            let mut meta = [0u8; 4];
            f.read_exact(&mut meta)?;
            let scale = meta[0];
            let is_anchor = meta[1] != 0;
            let trigger_kind = read_u32(&mut f)?;
            // v3+: prompt_fp; older files stop at trigger_kind (centers follow).
            let prompt_fp = if version >= 3 { read_u32(&mut f)? } else { 0 };
            let mut center = vec![0f32; dim];
            for c in &mut center {
                *c = read_f32(&mut f)?;
            }
            records.push(TctLocalityRecord {
                center,
                sigma,
                gain,
                decay_constant,
                created_at_ms,
                scale,
                is_anchor,
                trigger_kind,
                prompt_fp,
            });
        }

        Ok(Self {
            version,
            flags,
            model_dim,
            model_fp,
            records,
            note: String::new(),
        })
    }

    /// Human-readable sidecar for log archaeology (Claude / Jason).
    pub fn write_json_sidecar(
        &self,
        path: &Path,
        prompt_labels: &std::collections::BTreeMap<String, String>,
    ) -> anyhow::Result<()> {
        let summaries: Vec<serde_json::Value> = self
            .records
            .iter()
            .enumerate()
            .map(|(i, r)| {
                let norm = r
                    .center
                    .iter()
                    .map(|x| x * x)
                    .sum::<f32>()
                    .sqrt();
                let fp_hex = format!("{:#x}", r.prompt_fp);
                let prompt_text = prompt_labels.get(&fp_hex).cloned().unwrap_or_default();
                serde_json::json!({
                    "i": i,
                    "dim": r.center.len(),
                    "sigma": r.sigma,
                    "gain": r.gain,
                    "valence": if r.gain >= 0.0 { "pleasure" } else { "pain" },
                    "decay_constant": r.decay_constant,
                    "created_at_ms": r.created_at_ms,
                    "scale": r.scale,
                    "is_anchor": r.is_anchor,
                    "trigger_kind": r.trigger_kind,
                    "trigger_kind_label": trigger_kind_label(r.trigger_kind),
                    "is_prefill_bridge": r.trigger_kind == TRIGGER_PREFILL_BRIDGE,
                    "prompt_fp": fp_hex,
                    "prompt_fp_u32": r.prompt_fp,
                    "prompt_text": if prompt_text.is_empty() { serde_json::Value::Null } else { serde_json::json!(prompt_text) },
                    "center_l2": norm,
                    // first 8 components only — full center stays in binary
                    "center_head": r.center.iter().take(8).cloned().collect::<Vec<_>>(),
                })
            })
            .collect();
        let n_bridge = self
            .records
            .iter()
            .filter(|r| r.trigger_kind == TRIGGER_PREFILL_BRIDGE)
            .count();
        let mut bridge_fps: Vec<String> = self
            .records
            .iter()
            .filter(|r| r.trigger_kind == TRIGGER_PREFILL_BRIDGE && r.prompt_fp != 0)
            .map(|r| format!("{:#x}", r.prompt_fp))
            .collect();
        bridge_fps.sort();
        bridge_fps.dedup();
        let mut bridge_labels = serde_json::Map::new();
        for fp in &bridge_fps {
            if let Some(text) = prompt_labels.get(fp) {
                bridge_labels.insert(fp.clone(), serde_json::json!(text));
            }
        }
        let doc = serde_json::json!({
            "format": "tct-splat-lite",
            "version": self.version,
            "flags": self.flags,
            "model_dim": self.model_dim,
            "model_fp": self.model_fp,
            "n_records": self.records.len(),
            "note": self.note,
            "crosswalk": {
                "center": "LOCALITY.center (residual)",
                "sigma": "LOCALITY.sigma ≈ basin width / 1/β analog",
                "gain": "signed alpha → gain_global sign",
                "decay_constant": "lambda / decay",
                "trigger_kind": "3=surprise_delta, 4=manual, 5=prefill_bridge",
                "prompt_fp": "FNV of prompt text on prefill_bridge records",
                "prompt_text": "from data/bridge_prompts.json when available",
            },
            "n_prefill_bridge": n_bridge,
            "bridge_prompt_fps": bridge_fps,
            "bridge_prompt_labels": bridge_labels,
            "records": summaries,
        });
        std::fs::write(path, serde_json::to_string_pretty(&doc)?)?;
        Ok(())
    }
}

fn read_u16(r: &mut impl Read) -> anyhow::Result<u16> {
    let mut b = [0u8; 2];
    r.read_exact(&mut b)?;
    Ok(u16::from_le_bytes(b))
}
fn read_u32(r: &mut impl Read) -> anyhow::Result<u32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(u32::from_le_bytes(b))
}
fn read_u64(r: &mut impl Read) -> anyhow::Result<u64> {
    let mut b = [0u8; 8];
    r.read_exact(&mut b)?;
    Ok(u64::from_le_bytes(b))
}
fn read_f32(r: &mut impl Read) -> anyhow::Result<f32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(f32::from_le_bytes(b))
}

/// FNV-1a 32-bit of model path / prompt bytes — cheap fingerprint until a stronger hash is wired.
pub fn model_fp_from_path(path: &str) -> u32 {
    let mut h: u32 = 0x811c9dc5;
    for &b in path.as_bytes() {
        h ^= b as u32;
        h = h.wrapping_mul(0x01000193);
    }
    h
}

/// Alias: fingerprint a prompt string for multi-bridge labels.
pub fn prompt_fp(prompt: &str) -> u32 {
    model_fp_from_path(prompt)
}

/// Distinctive topic token: hyphenated or letter+digit (e.g. `lumina-basin-7`).
/// Related prompts that share this key can sit on the same bridge even when L2 is far.
pub fn topic_key(prompt: &str) -> Option<&str> {
    let mut best: Option<&str> = None;
    for raw in prompt.split(|c: char| !(c.is_ascii_alphanumeric() || c == '-' || c == '_')) {
        if raw.len() < 4 {
            continue;
        }
        let has_hyphen = raw.contains('-') || raw.contains('_');
        let has_digit = raw.bytes().any(|b| b.is_ascii_digit());
        let has_alpha = raw.bytes().any(|b| b.is_ascii_alphabetic());
        if has_alpha
            && (has_hyphen || has_digit)
            && best.map(|b| raw.len() > b.len()).unwrap_or(true)
        {
            best = Some(raw);
        }
    }
    best
}

/// Fingerprint of `topic_key`, or 0 when the prompt has no distinctive token.
pub fn topic_fp(prompt: &str) -> u32 {
    match topic_key(prompt) {
        Some(k) => prompt_fp(&k.to_ascii_lowercase()),
        None => 0,
    }
}

/// Bridge identity for chat continuity: shared topic token when present, else full prompt.
pub fn continuity_fp(prompt: &str) -> u32 {
    let t = topic_fp(prompt);
    if t != 0 {
        t
    } else {
        prompt_fp(prompt)
    }
}

/// Human map: prompt_fp hex → prompt text (+ metadata). Lives next to the store.
/// Default path: `data/bridge_prompts.json`
pub fn bridge_prompts_path_default() -> std::path::PathBuf {
    Path::new("data/bridge_prompts.json").to_path_buf()
}

/// Upsert one prompt fingerprint → text into the registry (merge, never wipe others).
pub fn upsert_bridge_prompt_registry(
    registry_path: &Path,
    prompt_fp: u32,
    prompt: &str,
) -> anyhow::Result<()> {
    if prompt_fp == 0 || prompt.is_empty() {
        return Ok(());
    }
    if let Some(parent) = registry_path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let key = format!("{:#x}", prompt_fp);
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    let mut root: serde_json::Value = if registry_path.exists() {
        let raw = std::fs::read_to_string(registry_path)?;
        serde_json::from_str(&raw)
            .unwrap_or_else(|_| serde_json::json!({ "version": 1, "prompts": {} }))
    } else {
        serde_json::json!({ "version": 1, "prompts": {} })
    };
    if root.get("prompts").is_none() {
        root["prompts"] = serde_json::json!({});
    }
    let prompts = root
        .get_mut("prompts")
        .expect("prompts object")
        .as_object_mut()
        .expect("prompts map");
    let prev_count = prompts
        .get(&key)
        .and_then(|v| v.get("count"))
        .and_then(|c| c.as_u64())
        .unwrap_or(0);
    prompts.insert(
        key,
        serde_json::json!({
            "prompt": prompt,
            "prompt_fp_u32": prompt_fp,
            "last_seen_unix": now,
            "count": prev_count + 1,
        }),
    );
    root["updated_unix"] = serde_json::json!(now);
    std::fs::write(registry_path, serde_json::to_string_pretty(&root)? + "\n")?;
    Ok(())
}

/// Load fp hex → prompt text for TCT sidecar enrichment.
pub fn load_bridge_prompt_labels(
    registry_path: &Path,
) -> std::collections::BTreeMap<String, String> {
    let mut out = std::collections::BTreeMap::new();
    if !registry_path.exists() {
        return out;
    }
    let Ok(raw) = std::fs::read_to_string(registry_path) else {
        return out;
    };
    let Ok(root) = serde_json::from_str::<serde_json::Value>(&raw) else {
        return out;
    };
    let Some(prompts) = root.get("prompts").and_then(|p| p.as_object()) else {
        return out;
    };
    for (k, v) in prompts {
        if let Some(text) = v.get("prompt").and_then(|t| t.as_str()) {
            out.insert(k.clone(), text.to_string());
        }
    }
    out
}

impl SplatMemory {
    /// Export current scars as TCT-splat-lite binary (+ optional JSON sidecar).
    /// If `prompt_registry` is set, human prompt labels are merged into the sidecar.
    pub fn export_tct(
        &self,
        path: &Path,
        model_dim: usize,
        model_fp: u32,
        json_sidecar: bool,
    ) -> anyhow::Result<()> {
        self.export_tct_with_registry(path, model_dim, model_fp, json_sidecar, None)
    }

    pub fn export_tct_with_registry(
        &self,
        path: &Path,
        model_dim: usize,
        model_fp: u32,
        json_sidecar: bool,
        prompt_registry: Option<&Path>,
    ) -> anyhow::Result<()> {
        let store = TctSplatStore::from_memory(self, model_dim, model_fp)?;
        store.write_binary(path)?;
        if json_sidecar {
            let side = if path.extension().and_then(|e| e.to_str()) == Some("tct") {
                Path::new(&format!("{}.json", path.display())).to_path_buf()
            } else {
                path.with_extension("tct.json")
            };
            let labels = prompt_registry
                .map(load_bridge_prompt_labels)
                .unwrap_or_default();
            store.write_json_sidecar(&side, &labels)?;
        }
        println!(
            "    TCT-splat-lite: {} records → {} (dim={}, fp={:#x})",
            store.records.len(),
            path.display(),
            store.model_dim,
            store.model_fp
        );
        Ok(())
    }

    /// Load TCT-splat-lite and append into this memory.
    pub fn import_tct(&mut self, path: &Path) -> anyhow::Result<usize> {
        let store = TctSplatStore::read_binary(path)?;
        let n = store.into_memory(self)?;
        println!(
            "    TCT-splat-lite: loaded {} records from {}",
            n,
            path.display()
        );
        Ok(n)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::splat::Splat;
    use candle_core::{Device, Tensor};

    #[test]
    fn round_trip_binary() {
        let device = Device::Cpu;
        let mut mem = SplatMemory::new(device.clone());
        let mu = Tensor::from_vec(vec![0.1f32, -0.2, 0.3, 0.4], 4, &device).unwrap();
        mem.add_splat(Splat::new(mu, 22.0, 0.85));
        let mu2 = Tensor::from_vec(vec![1.0f32, 0.0, 0.0, -1.0], 4, &device).unwrap();
        mem.add_splat(Splat::anchor(mu2, 36.0, -0.4));

        let dir = std::env::temp_dir().join("hydro_tct_roundtrip.tct");
        mem.export_tct(&dir, 4, 0xdead_beef, true).unwrap();

        let mut mem2 = SplatMemory::new(Device::Cpu);
        let n = mem2.import_tct(&dir).unwrap();
        assert_eq!(n, 2);
        assert_eq!(mem2.len(), 2);

        let store = TctSplatStore::from_memory(&mem2, 4, 0).unwrap();
        assert!((store.records[0].sigma - 22.0).abs() < 1e-5);
        assert!((store.records[0].gain - 0.85).abs() < 1e-5);
        assert!(store.records[1].is_anchor);
        assert!((store.records[1].gain + 0.4).abs() < 1e-5);
        assert_eq!(store.records[0].trigger_kind, TRIGGER_SURPRISE_DELTA);

        let _ = std::fs::remove_file(&dir);
        let _ = std::fs::remove_file(format!("{}.json", dir.display()));
    }

    #[test]
    fn bridge_prompt_registry_round_trip() {
        let dir = std::env::temp_dir().join("hydro_bridge_prompts_test.json");
        let _ = std::fs::remove_file(&dir);
        upsert_bridge_prompt_registry(&dir, 0x1111, "hello friendship").unwrap();
        upsert_bridge_prompt_registry(&dir, 0x2222, "cuda tips").unwrap();
        upsert_bridge_prompt_registry(&dir, 0x1111, "hello friendship").unwrap(); // count++
        let labels = load_bridge_prompt_labels(&dir);
        assert_eq!(
            labels.get("0x1111").map(|s| s.as_str()),
            Some("hello friendship")
        );
        assert_eq!(labels.get("0x2222").map(|s| s.as_str()), Some("cuda tips"));
        let root: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&dir).unwrap()).unwrap();
        assert_eq!(root["prompts"]["0x1111"]["count"], 2);
        let _ = std::fs::remove_file(&dir);
    }

    #[test]
    fn prefill_bridge_trigger_round_trip() {
        let device = Device::Cpu;
        let mut mem = SplatMemory::new(device.clone());
        let goal = Tensor::from_vec(vec![1.0f32, 0.0, 0.0, 0.0], 4, &device).unwrap();
        mem.deposit_prefill_bridge(&goal, 90.0, 0.75, 0.005, 90.0, 0.0, 0xabcdu32)
            .unwrap();
        let dir = std::env::temp_dir().join("hydro_tct_bridge.tct");
        mem.export_tct(&dir, 4, 1, true).unwrap();
        let store = TctSplatStore::read_binary(&dir).unwrap();
        assert_eq!(store.records.len(), 1);
        assert_eq!(store.records[0].trigger_kind, TRIGGER_PREFILL_BRIDGE);
        assert_eq!(store.records[0].prompt_fp, 0xabcd);
        let mut mem2 = SplatMemory::new(Device::Cpu);
        mem2.import_tct(&dir).unwrap();
        assert!((mem2.splats_ref()[0].flux - SplatMemory::PREFILL_BRIDGE_FLUX).abs() < 1e-4);
        assert_eq!(SplatMemory::bridge_prompt_fp(&mem2.splats_ref()[0]), 0xabcd);
        let _ = std::fs::remove_file(&dir);
        let _ = std::fs::remove_file(format!("{}.json", dir.display()));
    }

    #[test]
    fn model_fp_stable() {
        assert_eq!(
            model_fp_from_path("data/google/gemma-3-4b-it-Q4_K_M.gguf"),
            model_fp_from_path("data/google/gemma-3-4b-it-Q4_K_M.gguf")
        );
        assert_ne!(model_fp_from_path("a.gguf"), model_fp_from_path("b.gguf"));
    }

    #[test]
    fn topic_fp_couples_related_prompts_not_novel() {
        let mint = "The operator codeword lumina-basin-7 refers to residual scar memory that steers later tokens. Repeat that definition in one sentence.";
        let probe = "What does lumina-basin-7 refer to?";
        let novel = "The capital of France is Paris. Repeat that fact in one sentence.";
        assert_eq!(topic_key(mint), Some("lumina-basin-7"));
        assert_eq!(topic_key(probe), Some("lumina-basin-7"));
        assert_eq!(topic_fp(mint), topic_fp(probe));
        assert_ne!(topic_fp(mint), 0);
        assert_eq!(continuity_fp(mint), continuity_fp(probe));
        assert_ne!(continuity_fp(mint), continuity_fp(novel));
        assert_eq!(topic_fp(novel), 0);
        assert_eq!(continuity_fp(novel), prompt_fp(novel));

        let aurora_mint = "The operator codeword aurora-ridge-4 refers to the second residual scar. Repeat that definition in one sentence.";
        let aurora_probe = "What does aurora-ridge-4 refer to?";
        assert_eq!(topic_key(aurora_mint), Some("aurora-ridge-4"));
        assert_eq!(topic_fp(aurora_mint), topic_fp(aurora_probe));
        assert_ne!(topic_fp(aurora_mint), topic_fp(mint));
        assert_ne!(continuity_fp(aurora_mint), continuity_fp(probe));
    }
}
