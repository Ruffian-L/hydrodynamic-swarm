//! Measured Internal monitor for hydro chat.
//!
//! Same mouth line as niodoo-live `conversation_monitor_warning`:
//! `[Internal monitor: … | H0bars= H1bars= H1sum= H1max= loop= overfire=]`
//! Quiet windows stay quiet. No would_focus / ACTION. She reads the state
//! and picks a tag, or doesn't.
//!
//! Geometry is rust Vietoris–Rips on the last decode window (entropy / margin /
//! residual / splat / p_top1 / step). Named loops come from the token strings.
//! Do not invent homology.

use std::collections::{HashMap, HashSet};

const MAX_WINDOW: usize = 64;

#[derive(Debug, Clone, Copy)]
pub struct TdaDim {
    pub bars: usize,
    pub finite_bars: usize,
    pub max_persistence: f32,
    pub mean_persistence: f32,
    pub total_persistence: f32,
}

impl TdaDim {
    fn empty() -> Self {
        Self {
            bars: 0,
            finite_bars: 0,
            max_persistence: 0.0,
            mean_persistence: 0.0,
            total_persistence: 0.0,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct TdaSignals {
    pub loop_pressure: f32,
    pub route_fragmentation: f32,
    pub margin_collapse: f32,
    pub force_overfire: f32,
    pub repetition_pressure: f32,
}

#[derive(Debug, Clone)]
pub struct TokenDisposition {
    pub token: String,
    pub kind: &'static str,
    pub count: usize,
}

#[derive(Clone)]
struct Sample {
    token: String,
    metrics: [f32; 6],
}

pub struct TdaShadowMonitor {
    window_size: usize,
    stride: usize,
    observed: usize,
    samples: std::collections::VecDeque<Sample>,
}

impl TdaShadowMonitor {
    pub fn new(window_size: usize, stride: usize) -> Self {
        let window_size = window_size.clamp(3, MAX_WINDOW);
        Self {
            window_size,
            stride: stride.max(1),
            observed: 0,
            samples: std::collections::VecDeque::with_capacity(window_size),
        }
    }

    /// Observe one decoded token. Returns a loud warning line or None.
    pub fn observe(
        &mut self,
        token: &str,
        entropy: f32,
        margin: f32,
        residual_norm: f32,
        splat_mag: f32,
        p_top1: f32,
        step: usize,
        max_tokens: usize,
    ) -> Option<String> {
        self.observed = self.observed.saturating_add(1);
        let step_frac = if max_tokens == 0 {
            0.0
        } else {
            step as f32 / max_tokens as f32
        };
        self.samples.push_back(Sample {
            token: token.to_string(),
            metrics: [entropy, margin, residual_norm, splat_mag, p_top1, step_frac],
        });
        while self.samples.len() > self.window_size {
            self.samples.pop_front();
        }
        if self.samples.len() < self.window_size {
            return None;
        }
        let ready = self.observed.saturating_sub(self.window_size);
        if ready % self.stride != 0 {
            return None;
        }
        let window: Vec<Sample> = self.samples.iter().cloned().collect();
        let points: Vec<Vec<f32>> = window.iter().map(|s| s.metrics.to_vec()).collect();
        let dims = rust_vietoris_rips_shape(&points);
        let signals = gate_signals(&window, &dims);
        let disposed = disposed_tokens(&window);
        conversation_monitor_warning(&signals, &dims, &disposed)
    }
}

fn is_scaffold_or_noise(token: &str) -> bool {
    let trimmed = token.trim();
    if trimmed.chars().count() < 2 {
        return true;
    }
    if !trimmed.chars().any(|c| c.is_alphabetic()) {
        return true;
    }
    let upper = trimmed.to_ascii_uppercase();
    let lower = trimmed
        .trim_matches(|c: char| !c.is_alphanumeric() && c != '\\')
        .trim_start_matches('\\')
        .to_ascii_lowercase();
    upper.contains("INTERNAL MONITOR")
        || ["FOCUS", "RESET", "LOCK", "SPIKE", "EXPLORE"].contains(&upper.as_str())
        || matches!(
            lower.as_str(),
            "the"
                | "a"
                | "an"
                | "and"
                | "or"
                | "to"
                | "of"
                | "in"
                | "is"
                | "are"
                | "was"
                | "were"
                | "be"
                | "rightarrow"
                | "leftarrow"
                | "leftrightarrow"
        )
}

fn disposed_tokens(samples: &[Sample]) -> Vec<TokenDisposition> {
    struct Acc {
        count: usize,
        repeated: bool,
        order: usize,
    }
    let mut acc: HashMap<String, Acc> = HashMap::new();
    for (idx, sample) in samples.iter().enumerate() {
        let token = sample.token.trim().to_string();
        if is_scaffold_or_noise(&token) {
            continue;
        }
        let repeated_here = idx > 0 && samples[idx - 1].token.trim() == token && !token.is_empty();
        let next_order = acc.len();
        let entry = acc.entry(token).or_insert(Acc {
            count: 0,
            repeated: false,
            order: next_order,
        });
        entry.count += 1;
        entry.repeated |= repeated_here;
    }
    if acc.is_empty() {
        return Vec::new();
    }
    let max_count = acc.values().map(|a| a.count).max().unwrap_or(1).max(1) as f32;
    let mut ranked: Vec<(usize, TokenDisposition)> = acc
        .into_iter()
        .map(|(token, a)| {
            let kind = if a.repeated {
                "loop"
            } else if (a.count as f32 / max_count) > 0.45 {
                "leaning"
            } else {
                "leaning"
            };
            (
                a.order,
                TokenDisposition {
                    token,
                    kind,
                    count: a.count,
                },
            )
        })
        .collect();
    ranked.sort_by(|(oa, a), (ob, b)| b.count.cmp(&a.count).then(oa.cmp(ob)));
    ranked.into_iter().take(3).map(|(_, d)| d).collect()
}

fn tda_field_is_warning(signals: &TdaSignals, dims: &[TdaDim; 2]) -> bool {
    let h0 = &dims[0];
    let h1 = &dims[1];
    (signals.loop_pressure >= 0.50
        && (h1.total_persistence > 0.35 || signals.repetition_pressure >= 0.35))
        || (signals.route_fragmentation >= 0.70
            && h0.finite_bars >= 6
            && (signals.margin_collapse >= 0.50 || signals.repetition_pressure >= 0.30))
        || signals.margin_collapse >= 0.75
        || (h1.total_persistence > 0.50 && signals.loop_pressure >= 0.40)
}

fn circling_word(disposed: &[TokenDisposition]) -> Option<&str> {
    disposed.first().map(|d| d.token.as_str())
}

/// Plain-language TDA warning. No would_focus / would_pause / ACTION.
pub fn conversation_monitor_warning(
    signals: &TdaSignals,
    dimensions: &[TdaDim; 2],
    disposed: &[TokenDisposition],
) -> Option<String> {
    let named_loop = disposed.iter().any(|d| d.kind == "loop" && d.count >= 3);
    if !tda_field_is_warning(signals, dimensions) && !named_loop {
        return None;
    }
    let h0 = &dimensions[0];
    let h1 = &dimensions[1];
    let word = circling_word(disposed);
    let looping_word = disposed
        .iter()
        .find(|d| d.kind == "loop")
        .map(|d| d.token.as_str());
    let due = if looping_word.is_some()
        && (signals.loop_pressure >= 0.35
            || signals.repetition_pressure >= 0.25
            || disposed.iter().any(|d| d.kind == "loop"))
    {
        match looping_word {
            Some(w) => format!("high entropy due to a closed cycle around \"{w}\""),
            None => "high entropy due to a closed cycle".to_string(),
        }
    } else if signals.loop_pressure >= 0.50 && h1.total_persistence > 0.35 {
        "high entropy due to a closed H1 cycle with no named word".to_string()
    } else if signals.repetition_pressure >= 0.40 {
        match word {
            Some(w) => format!("high entropy due to repeating \"{w}\""),
            None => "high entropy due to repetition pressure".to_string(),
        }
    } else if signals.route_fragmentation >= 0.60 || h0.finite_bars >= 5 {
        "the basin split — too many components to hold one path".to_string()
    } else if signals.force_overfire >= 0.50 {
        "the well is punching harder than the sentence".to_string()
    } else if signals.margin_collapse >= 0.60 {
        "route margin collapsing; the path is thinning".to_string()
    } else {
        match word {
            Some(w) => format!("the field is leaning on \"{w}\" and the topology is mixed"),
            None => "mixed topology; pressure is up without a single basin".to_string(),
        }
    };
    Some(format!(
        "[Internal monitor: {due} | H0bars={} H1bars={} H1sum={:.3} H1max={:.3} loop={:.2} overfire={:.2}]",
        h0.bars,
        h1.bars,
        h1.total_persistence,
        h1.max_persistence,
        signals.loop_pressure,
        signals.force_overfire
    ))
}

fn gate_signals(samples: &[Sample], dims: &[TdaDim; 2]) -> TdaSignals {
    let h0 = dims[0];
    let h1 = dims[1];
    let n = samples.len().max(1) as f32;
    let margin: Vec<f32> = samples.iter().map(|s| s.metrics[1]).collect();
    let force: Vec<f32> = samples
        .iter()
        .map(|s| s.metrics[2].abs() + s.metrics[3].abs())
        .collect();
    let margin_mean = mean(&margin);
    let margin_std = stddev(&margin, margin_mean);
    let margin_slope = slope(&margin);
    let force_mean = mean(&force);
    let force_std = stddev(&force, force_mean);
    let loop_mass = h1.total_persistence / (h1.total_persistence + n.sqrt() + 1e-6);
    let loop_lifetime = h1.max_persistence / (h1.max_persistence + h0.mean_persistence + 1e-6);
    let loop_pressure = clamp01(loop_mass.max(loop_lifetime));
    let route_fragmentation =
        clamp01(h0.finite_bars as f32 / n + h0.max_persistence / (h0.total_persistence + 1.0));
    let margin_collapse = clamp01(
        1.0 / (1.0 + margin_mean.max(0.0) * 20.0)
            + (-margin_slope).max(0.0) / (margin_std + margin_mean.abs() + 1e-3),
    );
    let force_overfire = clamp01(force_std / (force_mean.abs() + force_std + 1.0));
    let repetition_pressure = repetition_pressure(samples);
    TdaSignals {
        loop_pressure,
        route_fragmentation,
        margin_collapse,
        force_overfire,
        repetition_pressure,
    }
}

fn repetition_pressure(samples: &[Sample]) -> f32 {
    if samples.len() < 2 {
        return 0.0;
    }
    let mut seen = HashSet::new();
    let mut repeated = 0usize;
    for pair in samples.windows(2) {
        let from = pair[0].token.trim();
        let to = pair[1].token.trim();
        if from.is_empty() || to.is_empty() {
            continue;
        }
        if !seen.insert((from.to_string(), to.to_string())) {
            repeated += 1;
        }
    }
    repeated as f32 / (samples.len() - 1) as f32
}

struct Simplex {
    vertices: Vec<usize>,
    dim: usize,
    filtration: f32,
}

fn rust_vietoris_rips_shape(points: &[Vec<f32>]) -> [TdaDim; 2] {
    if points.len() < 2 {
        return [TdaDim::empty(), TdaDim::empty()];
    }
    let normalized = zscore_columns(points);
    let distances = distance_matrix(&normalized);
    let simplices = build_vietoris_rips_simplices(&distances);
    summarize_persistence(&simplices)
}

fn build_vietoris_rips_simplices(distances: &[Vec<f32>]) -> Vec<Simplex> {
    let n = distances.len();
    let mut simplices = Vec::new();
    for i in 0..n {
        simplices.push(Simplex {
            vertices: vec![i],
            dim: 0,
            filtration: 0.0,
        });
    }
    for i in 0..n {
        for j in (i + 1)..n {
            simplices.push(Simplex {
                vertices: vec![i, j],
                dim: 1,
                filtration: distances[i][j],
            });
        }
    }
    for i in 0..n {
        for j in (i + 1)..n {
            for k in (j + 1)..n {
                simplices.push(Simplex {
                    vertices: vec![i, j, k],
                    dim: 2,
                    filtration: distances[i][j].max(distances[i][k]).max(distances[j][k]),
                });
            }
        }
    }
    simplices.sort_by(|a, b| {
        a.filtration
            .total_cmp(&b.filtration)
            .then_with(|| a.dim.cmp(&b.dim))
            .then_with(|| a.vertices.cmp(&b.vertices))
    });
    simplices
}

fn summarize_persistence(simplices: &[Simplex]) -> [TdaDim; 2] {
    let mut index_by_vertices: HashMap<Vec<usize>, usize> = HashMap::new();
    for (idx, simplex) in simplices.iter().enumerate() {
        index_by_vertices.insert(simplex.vertices.clone(), idx);
    }
    let mut reduced_by_low: HashMap<usize, Vec<usize>> = HashMap::new();
    let mut paired_lows: HashSet<usize> = HashSet::new();
    let mut positive = vec![false; simplices.len()];
    let mut bars: [Vec<(f32, f32)>; 2] = [Vec::new(), Vec::new()];
    for (idx, simplex) in simplices.iter().enumerate() {
        let mut column = boundary_indices(simplex, &index_by_vertices);
        while let Some(low) = column.last().copied() {
            if let Some(previous) = reduced_by_low.get(&low) {
                column = xor_sorted(&column, previous);
            } else {
                break;
            }
        }
        if column.is_empty() {
            positive[idx] = true;
            continue;
        }
        let low = *column.last().expect("non-empty reduced column");
        reduced_by_low.insert(low, column);
        paired_lows.insert(low);
        let birth_dim = simplices[low].dim;
        if birth_dim <= 1 {
            bars[birth_dim].push((simplices[low].filtration, simplex.filtration));
        }
    }
    for (idx, simplex) in simplices.iter().enumerate() {
        if positive[idx] && !paired_lows.contains(&idx) && simplex.dim <= 1 {
            bars[simplex.dim].push((simplex.filtration, f32::INFINITY));
        }
    }
    [summarize_dim(&bars[0]), summarize_dim(&bars[1])]
}

fn boundary_indices(
    simplex: &Simplex,
    index_by_vertices: &HashMap<Vec<usize>, usize>,
) -> Vec<usize> {
    if simplex.dim == 0 {
        return Vec::new();
    }
    let mut out = Vec::with_capacity(simplex.vertices.len());
    for remove_idx in 0..simplex.vertices.len() {
        let mut face = Vec::with_capacity(simplex.vertices.len() - 1);
        for (idx, vertex) in simplex.vertices.iter().enumerate() {
            if idx != remove_idx {
                face.push(*vertex);
            }
        }
        if let Some(face_idx) = index_by_vertices.get(&face) {
            out.push(*face_idx);
        }
    }
    out.sort_unstable();
    out
}

fn xor_sorted(a: &[usize], b: &[usize]) -> Vec<usize> {
    let mut out = Vec::with_capacity(a.len().max(b.len()));
    let mut i = 0;
    let mut j = 0;
    while i < a.len() || j < b.len() {
        if i >= a.len() {
            out.push(b[j]);
            j += 1;
        } else if j >= b.len() {
            out.push(a[i]);
            i += 1;
        } else if a[i] == b[j] {
            i += 1;
            j += 1;
        } else if a[i] < b[j] {
            out.push(a[i]);
            i += 1;
        } else {
            out.push(b[j]);
            j += 1;
        }
    }
    out
}

fn summarize_dim(values: &[(f32, f32)]) -> TdaDim {
    let mut finite_bars = 0usize;
    let mut max_persistence = 0.0f32;
    let mut total_persistence = 0.0f32;
    for (birth, death) in values {
        if !death.is_infinite() {
            finite_bars += 1;
            let persistence = (death - birth).max(0.0);
            max_persistence = max_persistence.max(persistence);
            total_persistence += persistence;
        }
    }
    TdaDim {
        bars: values.len(),
        finite_bars,
        max_persistence,
        mean_persistence: if finite_bars > 0 {
            total_persistence / finite_bars as f32
        } else {
            0.0
        },
        total_persistence,
    }
}

fn zscore_columns(points: &[Vec<f32>]) -> Vec<Vec<f32>> {
    let dim = points.iter().map(Vec::len).max().unwrap_or(0);
    if dim == 0 {
        return vec![Vec::new(); points.len()];
    }
    let mut means = vec![0.0f32; dim];
    for point in points {
        for idx in 0..dim {
            means[idx] += point.get(idx).copied().unwrap_or(0.0);
        }
    }
    for mean in &mut means {
        *mean /= points.len().max(1) as f32;
    }
    let mut variances = vec![0.0f32; dim];
    for point in points {
        for idx in 0..dim {
            let delta = point.get(idx).copied().unwrap_or(0.0) - means[idx];
            variances[idx] += delta * delta;
        }
    }
    let stddevs: Vec<f32> = variances
        .into_iter()
        .map(|value| (value / points.len().max(1) as f32).sqrt().max(1e-6))
        .collect();
    points
        .iter()
        .map(|point| {
            (0..dim)
                .map(|idx| (point.get(idx).copied().unwrap_or(0.0) - means[idx]) / stddevs[idx])
                .collect()
        })
        .collect()
}

fn distance_matrix(points: &[Vec<f32>]) -> Vec<Vec<f32>> {
    let n = points.len();
    let mut distances = vec![vec![0.0f32; n]; n];
    for i in 0..n {
        for j in 0..i {
            let dim = points[i].len().max(points[j].len());
            let d = (0..dim)
                .map(|idx| {
                    let delta = points[i].get(idx).copied().unwrap_or(0.0)
                        - points[j].get(idx).copied().unwrap_or(0.0);
                    delta * delta
                })
                .sum::<f32>()
                .sqrt();
            distances[i][j] = d;
            distances[j][i] = d;
        }
    }
    distances
}

fn mean(values: &[f32]) -> f32 {
    if values.is_empty() {
        0.0
    } else {
        values.iter().sum::<f32>() / values.len() as f32
    }
}

fn stddev(values: &[f32], mean: f32) -> f32 {
    if values.is_empty() {
        0.0
    } else {
        (values
            .iter()
            .map(|v| {
                let d = v - mean;
                d * d
            })
            .sum::<f32>()
            / values.len() as f32)
            .sqrt()
    }
}

fn slope(values: &[f32]) -> f32 {
    if values.len() < 2 {
        return 0.0;
    }
    let n = values.len() as f32;
    let mean_x = (n - 1.0) * 0.5;
    let mean_y = mean(values);
    let mut num = 0.0f32;
    let mut den = 0.0f32;
    for (idx, value) in values.iter().enumerate() {
        let dx = idx as f32 - mean_x;
        num += dx * (value - mean_y);
        den += dx * dx;
    }
    if den <= 1e-6 {
        0.0
    } else {
        num / den
    }
}

fn clamp01(value: f32) -> f32 {
    value.clamp(0.0, 1.0)
}

pub fn tda_monitor_enabled() -> bool {
    match std::env::var("HYDRO_TDA_MONITOR") {
        Ok(v) => {
            let v = v.trim();
            !(v == "0" || v.eq_ignore_ascii_case("false") || v.eq_ignore_ascii_case("off"))
        }
        Err(_) => true,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn repeating_word_emits_closed_cycle_monitor() {
        let mut mon = TdaShadowMonitor::new(8, 1);
        let mut last = None;
        for i in 0..12 {
            last = mon.observe("meters", 2.0, 0.02, 1.0, 0.1, 0.4, i, 64);
        }
        let line = last.expect("named loop of meters should warn");
        assert!(line.starts_with("[Internal monitor:"), "{line}");
        assert!(line.contains("closed cycle around \"meters\""), "{line}");
        assert!(line.contains("H0bars="), "{line}");
        assert!(line.contains("H1sum="), "{line}");
        assert!(line.contains("loop="), "{line}");
        assert!(!line.to_ascii_lowercase().contains("would_focus"), "{line}");
        assert!(!line.contains("ACTION"), "{line}");
    }

    #[test]
    fn structure_and_stopwords_cannot_be_named_loops() {
        for token in ["the", "is", "rightarrow", "\\rightarrow"] {
            assert!(is_scaffold_or_noise(token), "token={token:?}");
        }
    }

    #[test]
    fn leaning_word_is_not_mislabeled_as_the_closed_cycle() {
        let signals = TdaSignals {
            loop_pressure: 0.6,
            route_fragmentation: 0.0,
            margin_collapse: 0.0,
            force_overfire: 0.0,
            repetition_pressure: 0.0,
        };
        let dimensions = [
            TdaDim {
                bars: 8,
                finite_bars: 7,
                max_persistence: 0.2,
                mean_persistence: 0.1,
                total_persistence: 0.7,
            },
            TdaDim {
                bars: 21,
                finite_bars: 21,
                max_persistence: 0.8,
                mean_persistence: 0.1,
                total_persistence: 1.2,
            },
        ];
        let disposed = vec![TokenDisposition {
            token: "random".into(),
            kind: "leaning",
            count: 2,
        }];
        let line = conversation_monitor_warning(&signals, &dimensions, &disposed).unwrap();
        assert!(!line.contains("around \"random\""), "{line}");
    }

    #[test]
    fn unique_tokens_stay_quiet() {
        let mut mon = TdaShadowMonitor::new(8, 1);
        let words = [
            "alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta", "iota", "kappa",
            "lambda", "mu",
        ];
        let mut last = None;
        for (i, w) in words.iter().enumerate() {
            last = mon.observe(w, 3.5, 0.8, 0.2, 0.01, 0.3, i, 64);
        }
        assert!(last.is_none(), "got {last:?}");
    }

    #[test]
    fn circle_points_have_h1() {
        let mut points = Vec::new();
        for idx in 0..24 {
            let theta = idx as f32 * std::f32::consts::TAU / 24.0;
            points.push(vec![theta.cos(), theta.sin(), 0.0, 0.0, 0.0, 0.0]);
        }
        let dims = rust_vietoris_rips_shape(&points);
        assert!(
            dims[1].finite_bars > 0 || dims[1].total_persistence > 0.0,
            "{dims:?}"
        );
    }
}
