//! ORG-H5 sidecar: dormant why-geometry on `<remember>` close.
//!
//! Not a RememberLine column. Not a splat force. Not `<spike>`. Not KV-drop.
//! Dual-site ring (S_res / S_logit) × offsets, dumped to JSONL + `.f32` bins.
//! Ranking of offsets is a later offline pass (C2 shuffle / C3 sign-flip live
//! here as helpers; they are not written on the generate path).

use std::collections::VecDeque;
use std::path::{Path, PathBuf};

const RING: usize = 16;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CaptureSite {
    SRes,
    SLogit,
}

impl CaptureSite {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::SRes => "S_res",
            Self::SLogit => "S_logit",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ProbePhase {
    Idle,
    Open,
    ClosedThisMint,
}

#[derive(Debug, Clone)]
struct Sample {
    step: usize,
    res: Option<Vec<f32>>,
    logit: Option<Vec<f32>>,
}

/// Offsets relative to t* = the decode step where remember closes.
pub const OFFSET_NAMES: [&str; 9] = [
    "t_star",
    "t_star_minus_1",
    "t_star_minus_2",
    "t_star_minus_3",
    "predict",
    "mean_3",
    "mean_5",
    "maxnorm_3",
    "maxnorm_5",
];

#[derive(Debug, Clone)]
pub struct RememberOffsetProbe {
    ring: VecDeque<Sample>,
    predict_step: Option<usize>,
    phase: ProbePhase,
    sidecar_path: PathBuf,
    mint_count: usize,
    last_rows: usize,
    model: String,
    build: String,
}

impl Default for RememberOffsetProbe {
    fn default() -> Self {
        Self {
            ring: VecDeque::with_capacity(RING),
            predict_step: None,
            phase: ProbePhase::Idle,
            sidecar_path: PathBuf::new(),
            mint_count: 0,
            last_rows: 0,
            model: String::new(),
            build: "ORG-H5".to_string(),
        }
    }
}

impl RememberOffsetProbe {
    pub fn set_sidecar_path(&mut self, path: PathBuf) {
        self.sidecar_path = path;
    }

    pub fn set_sidecar_from_remember_path(&mut self, remember: &Path) {
        self.sidecar_path = sidecar_path_for(remember);
    }

    pub fn sidecar_path(&self) -> &Path {
        &self.sidecar_path
    }

    #[allow(dead_code)]
    pub fn set_model(&mut self, model: impl Into<String>) {
        self.model = model.into();
    }

    pub fn mint_count(&self) -> usize {
        self.mint_count
    }

    #[allow(dead_code)]
    pub fn last_rows(&self) -> usize {
        self.last_rows
    }

    pub fn push(&mut self, step: usize, s_res: Option<&[f32]>, s_logit: Option<&[f32]>) {
        if s_res.is_none() && s_logit.is_none() {
            return;
        }
        if let Some(last) = self.ring.back_mut() {
            if last.step == step {
                if s_res.is_some() {
                    last.res = s_res.map(Vec::from);
                }
                if s_logit.is_some() {
                    last.logit = s_logit.map(Vec::from);
                }
                return;
            }
        }
        self.ring.push_back(Sample {
            step,
            res: s_res.map(Vec::from),
            logit: s_logit.map(Vec::from),
        });
        while self.ring.len() > RING {
            self.ring.pop_front();
        }
    }

    /// Arm predict on the first remember sighting since the last mint.
    pub fn note_pieces(&mut self, pieces: &str) {
        if !remember_present(pieces) {
            if self.phase == ProbePhase::ClosedThisMint {
                // closed tag still in the buffer is remember_present; this
                // branch is the gap after a wipe / new turn with no tag.
                self.phase = ProbePhase::Idle;
            }
            return;
        }
        let unclosed = remember_unclosed(pieces);
        match self.phase {
            ProbePhase::Idle => {
                self.predict_step = self.sample_before_last();
                self.phase = if unclosed {
                    ProbePhase::Open
                } else {
                    ProbePhase::ClosedThisMint
                };
            }
            ProbePhase::Open => {
                if !unclosed {
                    self.phase = ProbePhase::ClosedThisMint;
                }
            }
            ProbePhase::ClosedThisMint => {}
        }
    }

    /// KV drop is not a mint. Ring may go; mint_count does not.
    pub fn on_kv_drop(&mut self) {
        self.ring.clear();
        self.predict_step = None;
        self.phase = ProbePhase::Idle;
    }

    /// Dump all formable offsets. No-op (0 rows) when the ring has no t*.
    /// Never injects. Never writes RememberLine.
    pub fn dump_on_remember_close(
        &mut self,
        key: &str,
        value: &str,
    ) -> std::io::Result<usize> {
        self.last_rows = 0;
        if self.sidecar_path.as_os_str().is_empty() {
            self.reset_mint_phase();
            return Ok(0);
        }
        let t_star = match self.ring.back().map(|s| s.step) {
            Some(s) => s,
            None => {
                self.reset_mint_phase();
                return Ok(0);
            }
        };
        let mut rows = 0usize;
        for site in [CaptureSite::SRes, CaptureSite::SLogit] {
            rows += self.dump_site(site, t_star, key, value)?;
        }
        if rows > 0 {
            self.mint_count = self.mint_count.saturating_add(1);
        }
        self.last_rows = rows;
        self.reset_mint_phase();
        Ok(rows)
    }

    fn reset_mint_phase(&mut self) {
        self.predict_step = None;
        self.phase = ProbePhase::Idle;
    }

    fn dump_site(
        &mut self,
        site: CaptureSite,
        t_star: usize,
        key: &str,
        value: &str,
    ) -> std::io::Result<usize> {
        let mut n = 0usize;
        for name in OFFSET_NAMES {
            if let Some((step, vec)) = self.offset_vec(site, t_star, name) {
                self.write_row(site, name, step, key, value, &vec)?;
                n += 1;
            }
        }
        Ok(n)
    }

    fn offset_vec(
        &self,
        site: CaptureSite,
        t_star: usize,
        name: &str,
    ) -> Option<(usize, Vec<f32>)> {
        match name {
            "t_star" => self.vec_at_step(site, t_star).map(|v| (t_star, v)),
            "t_star_minus_1" => self.vec_n_back(site, t_star, 1),
            "t_star_minus_2" => self.vec_n_back(site, t_star, 2),
            "t_star_minus_3" => self.vec_n_back(site, t_star, 3),
            "predict" => {
                let p = self.predict_step?;
                self.vec_at_step(site, p).map(|v| (p, v))
            }
            "mean_3" => self.window_mean(site, t_star, 3).map(|v| (t_star, v)),
            "mean_5" => self.window_mean(site, t_star, 5).map(|v| (t_star, v)),
            "maxnorm_3" => self.window_maxnorm(site, t_star, 3),
            "maxnorm_5" => self.window_maxnorm(site, t_star, 5),
            _ => None,
        }
    }

    fn vec_at_step(&self, site: CaptureSite, step: usize) -> Option<Vec<f32>> {
        let s = self.ring.iter().find(|s| s.step == step)?;
        match site {
            CaptureSite::SRes => s.res.clone(),
            CaptureSite::SLogit => s.logit.clone(),
        }
    }

    fn vec_n_back(
        &self,
        site: CaptureSite,
        t_star: usize,
        n: usize,
    ) -> Option<(usize, Vec<f32>)> {
        let step = t_star.checked_sub(n)?;
        self.vec_at_step(site, step).map(|v| (step, v))
    }

    /// Mean over [t*−N, t*−1]. t* itself is excluded.
    fn window_mean(&self, site: CaptureSite, t_star: usize, n: usize) -> Option<Vec<f32>> {
        let window = self.window(site, t_star, n)?;
        if window.len() != n {
            return None;
        }
        let dim = window[0].1.len();
        if dim == 0 || window.iter().any(|(_, v)| v.len() != dim) {
            return None;
        }
        let mut acc = vec![0.0f32; dim];
        for (_, v) in &window {
            for (i, x) in v.iter().enumerate() {
                acc[i] += *x;
            }
        }
        let inv = 1.0 / n as f32;
        for x in &mut acc {
            *x *= inv;
        }
        Some(acc)
    }

    fn window_maxnorm(
        &self,
        site: CaptureSite,
        t_star: usize,
        n: usize,
    ) -> Option<(usize, Vec<f32>)> {
        let window = self.window(site, t_star, n)?;
        if window.len() != n {
            return None;
        }
        window.into_iter().max_by(|a, b| {
            l2(&a.1)
                .partial_cmp(&l2(&b.1))
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    }

    fn window(&self, site: CaptureSite, t_star: usize, n: usize) -> Option<Vec<(usize, Vec<f32>)>> {
        if n == 0 || t_star < n {
            return None;
        }
        let mut out = Vec::with_capacity(n);
        for k in (1..=n).rev() {
            let step = t_star - k;
            let v = self.vec_at_step(site, step)?;
            out.push((step, v));
        }
        Some(out)
    }

    fn sample_before_last(&self) -> Option<usize> {
        if self.ring.len() < 2 {
            return None;
        }
        self.ring.iter().rev().nth(1).map(|s| s.step)
    }

    fn write_row(
        &self,
        site: CaptureSite,
        offset: &str,
        step: usize,
        key: &str,
        value: &str,
        vec: &[f32],
    ) -> std::io::Result<()> {
        let sidecar = &self.sidecar_path;
        if let Some(parent) = sidecar.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)?;
            }
        }
        let stem = sidecar
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| "remember_offset_probe".into());
        let dir = sidecar.parent().unwrap_or_else(|| Path::new("."));
        let vec_file = dir.join(format!(
            "{stem}.m{:04}.{}.{}.f32",
            self.mint_count,
            site.as_str(),
            offset
        ));
        write_f32le(&vec_file, vec)?;
        let row = serde_json::json!({
            "event": "remember_offset_probe",
            "kind": "C1",
            "key": key,
            "value_fnv64": format!("{:016x}", fnv1a64(value.as_bytes())),
            "site": site.as_str(),
            "offset": offset,
            "step": step,
            "dim": vec.len(),
            "norm": l2(vec),
            "vec_fnv64": format!("{:016x}", fnv1a64_f32(vec)),
            "vec_file": vec_file.file_name().map(|s| s.to_string_lossy()).unwrap_or_default(),
            "inject": false,
            "model": self.model,
            "build": self.build,
        });
        append_jsonl(sidecar, &row)
    }
}

pub fn sign_flip(v: &[f32]) -> Vec<f32> {
    v.iter().map(|x| -x).collect()
}

/// Same-dim permutation. Seeded LCG. C2 chance floor — not written live.
pub fn shuffle_same_dim(v: &[f32], seed: u64) -> Vec<f32> {
    let mut out = v.to_vec();
    let n = out.len();
    if n < 2 {
        return out;
    }
    let mut state = seed | 1;
    for i in (1..n).rev() {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1);
        let j = (state as usize) % (i + 1);
        out.swap(i, j);
    }
    out
}

pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let mut dot = 0.0f32;
    let mut na = 0.0f32;
    let mut nb = 0.0f32;
    for (x, y) in a.iter().zip(b) {
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    let d = na.sqrt() * nb.sqrt();
    if d < 1e-12 {
        0.0
    } else {
        dot / d
    }
}

fn remember_present(text: &str) -> bool {
    let l = text.to_ascii_lowercase();
    l.contains("<remember") || l.contains("<request:remember") || l.contains("[remember")
}

fn remember_unclosed(text: &str) -> bool {
    let l = text.to_ascii_lowercase();
    for (open, close) in [
        ("<remember", "</remember>"),
        ("<request:remember", "</request:remember>"),
    ] {
        if let Some(i) = l.rfind(open) {
            if !l[i..].contains(close) {
                return true;
            }
        }
    }
    false
}

fn sidecar_path_for(remember: &Path) -> PathBuf {
    if let Ok(v) = std::env::var("HYDRO_REMEMBER_SIDECAR") {
        let t = v.trim();
        if t == "0" || t.eq_ignore_ascii_case("off") || t.eq_ignore_ascii_case("false") {
            return PathBuf::new();
        }
        if !t.is_empty() && t != "1" && !t.eq_ignore_ascii_case("true") && !t.eq_ignore_ascii_case("on")
        {
            return PathBuf::from(t);
        }
    }
    if remember.as_os_str().is_empty() {
        return PathBuf::new();
    }
    let name = remember
        .file_name()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "remember.jsonl".into());
    let stem = name.strip_suffix(".jsonl").unwrap_or(&name);
    remember.with_file_name(format!("{stem}.offset_probe.jsonl"))
}

fn l2(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut h = 0xcbf29ce484222325u64;
    for b in bytes {
        h ^= *b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

fn fnv1a64_f32(v: &[f32]) -> u64 {
    let mut buf = Vec::with_capacity(v.len() * 4);
    for x in v {
        buf.extend_from_slice(&x.to_le_bytes());
    }
    fnv1a64(&buf)
}

fn write_f32le(path: &Path, v: &[f32]) -> std::io::Result<()> {
    let mut buf = Vec::with_capacity(v.len() * 4);
    for x in v {
        buf.extend_from_slice(&x.to_le_bytes());
    }
    std::fs::write(path, buf)
}

fn append_jsonl(path: &Path, v: &serde_json::Value) -> std::io::Result<()> {
    use std::io::Write;
    let mut f = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)?;
    writeln!(f, "{v}")?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp_dir(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "hydro_{tag}_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn fill_ring(p: &mut RememberOffsetProbe, n: usize) {
        for i in 0..n {
            let res: Vec<f32> = (0..4).map(|k| (i as f32 + 1.0) * 0.1 + k as f32).collect();
            let logit: Vec<f32> = (0..4)
                .map(|k| 10.0 + (i as f32) * 0.2 + k as f32 * 0.01)
                .collect();
            p.push(i, Some(&res), Some(&logit));
        }
    }

    #[test]
    fn closed_remember_dumps_all_offsets_both_sites() {
        let dir = tmp_dir("h5_dump");
        let path = dir.join("probe.jsonl");
        let mut p = RememberOffsetProbe::default();
        p.set_sidecar_path(path.clone());
        fill_ring(&mut p, 8);
        p.note_pieces("ok <remember>lumina=minted-why</remember>");
        let rows = p
            .dump_on_remember_close("lumina", "minted-why")
            .unwrap();
        assert_eq!(rows, 18, "9 offsets × 2 sites, got {rows}");
        assert_eq!(p.mint_count(), 1);
        let body = std::fs::read_to_string(&path).unwrap();
        let lines: Vec<&str> = body.lines().filter(|l| !l.is_empty()).collect();
        assert_eq!(lines.len(), 18);
        let mut saw_res = 0;
        let mut saw_logit = 0;
        for line in &lines {
            assert!(line.contains("\"event\":\"remember_offset_probe\""), "{line}");
            assert!(line.contains("\"inject\":false"), "{line}");
            assert!(line.contains("\"kind\":\"C1\""), "{line}");
            assert!(!line.contains("\"value\":"), "payload must not leak into sidecar");
            if line.contains("\"site\":\"S_res\"") {
                saw_res += 1;
            }
            if line.contains("\"site\":\"S_logit\"") {
                saw_logit += 1;
            }
        }
        assert_eq!(saw_res, 9);
        assert_eq!(saw_logit, 9);
        for name in OFFSET_NAMES {
            assert!(
                body.contains(&format!("\"offset\":\"{name}\"")),
                "missing {name}"
            );
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn spike_and_incomplete_remember_do_not_dump() {
        let dir = tmp_dir("h5_nodump");
        let mut p = RememberOffsetProbe::default();
        p.set_sidecar_path(dir.join("spike.jsonl"));
        fill_ring(&mut p, 6);
        p.note_pieces("hello <spike> still thinking");
        assert_eq!(p.phase, ProbePhase::Idle, "spike must not arm a remember mint");
        assert_eq!(p.mint_count(), 0);
        assert!(!dir.join("spike.jsonl").exists());

        let mut q = RememberOffsetProbe::default();
        q.set_sidecar_path(dir.join("incomplete.jsonl"));
        fill_ring(&mut q, 6);
        q.note_pieces("<remember>");
        assert_eq!(q.phase, ProbePhase::Open);
        assert_eq!(q.mint_count(), 0);
        assert!(!dir.join("incomplete.jsonl").exists());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn kv_drop_does_not_mint() {
        let dir = tmp_dir("h5_kv");
        let path = dir.join("probe.jsonl");
        let mut p = RememberOffsetProbe::default();
        p.set_sidecar_path(path.clone());
        fill_ring(&mut p, 8);
        p.note_pieces("<remember>k=v</remember>");
        assert_eq!(p.dump_on_remember_close("k", "v").unwrap(), 18);
        assert_eq!(p.mint_count(), 1);
        let bytes = std::fs::metadata(&path).unwrap().len();
        p.on_kv_drop();
        assert_eq!(p.mint_count(), 1);
        assert_eq!(std::fs::metadata(&path).unwrap().len(), bytes);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn sign_flip_is_minus_one_shuffle_is_not_identity() {
        let v: Vec<f32> = (0..32).map(|i| (i as f32) * 0.3 - 4.0).collect();
        let flipped = sign_flip(&v);
        assert!((cosine(&v, &flipped) + 1.0).abs() < 1e-5);
        let shuffled = shuffle_same_dim(&v, 42);
        assert_ne!(shuffled, v);
        assert_eq!(shuffled.len(), v.len());
        let mut a = v.clone();
        let mut b = shuffled.clone();
        a.sort_by(|x, y| x.partial_cmp(y).unwrap());
        b.sort_by(|x, y| x.partial_cmp(y).unwrap());
        assert_eq!(a, b, "shuffle must be a permutation");
    }

    #[test]
    fn sites_are_not_mixed_on_one_row() {
        let dir = tmp_dir("h5_sites");
        let path = dir.join("probe.jsonl");
        let mut p = RememberOffsetProbe::default();
        p.set_sidecar_path(path.clone());
        fill_ring(&mut p, 6);
        p.note_pieces("<remember>k=v</remember>");
        p.dump_on_remember_close("k", "v").unwrap();
        for line in std::fs::read_to_string(&path).unwrap().lines() {
            let res = line.contains("\"site\":\"S_res\"");
            let logit = line.contains("\"site\":\"S_logit\"");
            assert!(res ^ logit, "mixed site row: {line}");
        }
        let _ = std::fs::remove_dir_all(&dir);
    }
}
