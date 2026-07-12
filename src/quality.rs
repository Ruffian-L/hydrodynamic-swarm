//! Token quality scoring for semantic splat deposits.
//!
//! We do **not** use "high steering delta" as good — that marks *stress*, not quality.
//! Good / bad is quantified from the model's own posterior at sample time.

/// What we can measure each step without an external judge.
#[derive(Debug, Clone)]
pub struct TokenQuality {
    /// P(chosen token) under steered sampling distribution.
    pub p_chosen: f32,
    /// Approximate entropy from top-k mass (nats). Low = confident.
    pub topk_entropy: f32,
    /// True if this token appears in the last `repeat_window` generated tokens.
    pub is_recent_repeat: bool,
    /// How many times it appears in that window (spam detector).
    pub recent_count: usize,
    /// Decoded text (for logging).
    pub token_text: String,
}

/// Deposit decision for scar memory.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SplatKind {
    /// Attract: confident, non-spam token — crystallize "this was ok".
    Pleasure,
    /// Repel: model stumbled or looped — scar "don't linger here".
    Pain,
    /// Do not deposit.
    Skip,
}

/// Thresholds for good / bad (editable via config later).
#[derive(Debug, Clone)]
pub struct QualityThresholds {
    /// Min P(token) to count as pleasure.
    pub good_p_min: f32,
    /// Max top-k entropy (nats) for pleasure.
    pub good_entropy_max: f32,
    /// Max P(token) for pain (model was surprised).
    pub bad_p_max: f32,
    /// If token appears this many times in the window → pain.
    pub repeat_pain_count: usize,
    /// Lookback window for repeats.
    pub repeat_window: usize,
    /// Top-k for entropy estimate.
    pub entropy_topk: usize,
}

impl Default for QualityThresholds {
    fn default() -> Self {
        Self {
            good_p_min: 0.12,
            good_entropy_max: 3.5,
            bad_p_max: 0.03,
            repeat_pain_count: 3,
            repeat_window: 24,
            entropy_topk: 64,
        }
    }
}

/// Score the chosen token from the full probability vector.
pub fn score_token(
    probs: &[f32],
    token_id: u32,
    token_text: &str,
    recent: &[u32],
    thr: &QualityThresholds,
) -> TokenQuality {
    let idx = token_id as usize;
    let p_chosen = if idx < probs.len() { probs[idx] } else { 0.0 };

    let topk = thr.entropy_topk.min(probs.len());
    let mut pairs: Vec<(usize, f32)> = probs
        .iter()
        .enumerate()
        .map(|(i, &p)| (i, p))
        .collect();
    if topk > 0 && pairs.len() > topk {
        pairs.select_nth_unstable_by(topk - 1, |a, b| {
            b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
        });
        pairs.truncate(topk);
    }
    let mass: f32 = pairs.iter().map(|(_, p)| p).sum::<f32>().max(1e-12);
    let topk_entropy: f32 = pairs
        .iter()
        .map(|(_, p)| {
            let q = (p / mass).max(1e-12);
            -q * q.ln()
        })
        .sum();

    let window = thr.repeat_window;
    let start = recent.len().saturating_sub(window);
    let recent_count = recent[start..].iter().filter(|&&t| t == token_id).count();
    // +1 would count current; we score before push, so count is prior occurrences
    let is_recent_repeat = recent_count >= 1;

    TokenQuality {
        p_chosen,
        topk_entropy,
        is_recent_repeat,
        recent_count,
        token_text: token_text.to_string(),
    }
}

/// Map quality → pleasure / pain / skip.
///
/// **Good (pleasure):** model is confident, not looping.
/// **Bad (pain):** model is guessing hard, or repeating spam.
/// **Skip:** everything else (don't fill memory with noise).
pub fn classify(q: &TokenQuality, thr: &QualityThresholds) -> SplatKind {
    if q.recent_count + 1 >= thr.repeat_pain_count {
        return SplatKind::Pain;
    }
    if q.p_chosen <= thr.bad_p_max {
        return SplatKind::Pain;
    }
    if q.p_chosen >= thr.good_p_min
        && q.topk_entropy <= thr.good_entropy_max
        && !q.is_recent_repeat
    {
        return SplatKind::Pleasure;
    }
    SplatKind::Skip
}

/// Alpha magnitude for a classified deposit.
pub fn alpha_for(kind: SplatKind, q: &TokenQuality, pleasure_base: f32, pain_base: f32) -> f32 {
    match kind {
        SplatKind::Pleasure => {
            // Stronger pleasure for higher confidence
            let boost = (q.p_chosen / 0.3).clamp(0.5, 2.0);
            (pleasure_base.abs() * boost).clamp(0.3, 2.5)
        }
        SplatKind::Pain => {
            // Stronger pain for lower p / more repeats
            let surprise = (0.05 / q.p_chosen.max(1e-4)).ln().abs().clamp(0.5, 2.5);
            let rep = 1.0 + 0.25 * q.recent_count as f32;
            -(pain_base.abs() * surprise * rep).clamp(0.3, 3.0)
        }
        SplatKind::Skip => 0.0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn confident_token_is_pleasure() {
        let mut probs = vec![1e-6f32; 100];
        probs[7] = 0.4;
        let q = score_token(&probs, 7, "friend", &[], &QualityThresholds::default());
        assert!(matches!(classify(&q, &QualityThresholds::default()), SplatKind::Pleasure));
    }

    #[test]
    fn low_p_is_pain() {
        let mut probs = vec![1e-6f32; 100];
        probs[3] = 0.01;
        let q = score_token(&probs, 3, "???", &[], &QualityThresholds::default());
        assert!(matches!(classify(&q, &QualityThresholds::default()), SplatKind::Pain));
    }

    #[test]
    fn spam_repeat_is_pain() {
        let mut probs = vec![1e-6f32; 100];
        probs[1] = 0.5;
        let recent = vec![1u32, 1, 1, 2];
        let thr = QualityThresholds::default();
        let q = score_token(&probs, 1, "Two", &recent, &thr);
        assert!(matches!(classify(&q, &thr), SplatKind::Pain));
    }
}
