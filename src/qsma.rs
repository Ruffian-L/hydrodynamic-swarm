//! Q-SMA inside hydro decode: π = argmax[Q + ease(F)×β + C]; hands set β/σ.
//!
//! Source: niodoo-adaptive-agency `src/qsma.rs`. Hands override scheduled β.

use std::collections::HashMap;

pub fn beta(t: u32) -> f64 {
    (1.5 * 0.995_f64.powi(t as i32)).max(0.1)
}

pub fn ease(flux: f64) -> f64 {
    1.0 / (1.0 + (-flux).exp())
}

#[derive(Debug, Clone)]
pub struct QsmaAction {
    pub index: usize,
    pub score: f64,
    pub beta: f64,
}

/// Hybrid policy. `q`, `flux`, and `curiosity` must be the same length.
/// Scheduled β = max(0.1, 1.5×0.995^t).
pub fn qsma_select(q: &[f64], flux: &[f64], curiosity: &[f64], t: u32) -> QsmaAction {
    qsma_select_beta(q, flux, curiosity, beta(t))
}

/// Same policy with an explicit β (hands: SPIKE 1.5, FOCUS 0.5, EXPLORE 2.0, RESET 0.0).
pub fn qsma_select_beta(q: &[f64], flux: &[f64], curiosity: &[f64], beta: f64) -> QsmaAction {
    let n = q.len().min(flux.len()).min(curiosity.len());
    assert!(n > 0, "qsma_select requires at least one action");
    let mut best_i = 0usize;
    let mut best = f64::NEG_INFINITY;
    for i in 0..n {
        let score = q[i] + ease(flux[i]) * beta + curiosity[i];
        if score > best {
            best = score;
            best_i = i;
        }
    }
    QsmaAction {
        index: best_i,
        score: best,
        beta,
    }
}

/// Log-scale flux: high energy deepens the groove; low energy writes pain.
pub fn update_flux(flux: f64, energy: f64) -> f64 {
    if energy > 0.1 {
        flux + (1.0 + energy * 100.0).ln()
    } else {
        let missing = (0.1 - energy).max(0.0);
        flux - (1.0 + missing * 50.0).ln()
    }
}

/// Additive QSMA term on the leading logits: Q stays the model logit.
/// score_i = Q_i + ease(F_i)×β + C_i + σ·ξ. Only the top-`cap` logits are touched.
pub fn bias_top_k(
    logits: &mut [f32],
    cap: usize,
    flux: &HashMap<u32, f64>,
    generated: &[u32],
    beta: f64,
    kinetic_noise: f64,
) -> QsmaAction {
    let n = logits.len();
    assert!(n > 0, "bias_top_k requires logits");
    let k = cap.clamp(1, n);
    let mut idxs: Vec<usize> = (0..n).collect();
    if k < n {
        idxs.select_nth_unstable_by(k - 1, |&a, &b| {
            logits[b]
                .partial_cmp(&logits[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        idxs.truncate(k);
    }

    let mut q = Vec::with_capacity(idxs.len());
    let mut f = Vec::with_capacity(idxs.len());
    let mut c = Vec::with_capacity(idxs.len());
    for &i in &idxs {
        q.push(logits[i] as f64);
        f.push(flux.get(&(i as u32)).copied().unwrap_or(0.0));
        let visits = generated.iter().filter(|&&t| t == i as u32).count();
        c.push(0.1 / (1.0 + visits as f64));
    }

    let mut best = QsmaAction {
        index: idxs[0],
        score: f64::NEG_INFINITY,
        beta,
    };
    for (j, &i) in idxs.iter().enumerate() {
        let mut extra = ease(f[j]) * beta + c[j];
        if kinetic_noise > 1e-9 {
            use rand::RngExt;
            let xi: f64 = rand::rng().random::<f64>() * 2.0 - 1.0;
            extra += kinetic_noise * xi;
        }
        logits[i] = (q[j] + extra) as f32;
        let score = q[j] + extra;
        if score > best.score {
            best.index = i;
            best.score = score;
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn high_flux_plus_beta_picks_the_groove() {
        let q = [1.0, 1.0];
        let flux = [8.0, 0.0];
        let curiosity = [0.0, 0.0];
        let a = qsma_select_beta(&q, &flux, &curiosity, 1.5);
        assert_eq!(a.index, 0);
        assert!((a.beta - 1.5).abs() < 1e-12);
    }

    #[test]
    fn curiosity_can_beat_a_dead_groove_when_beta_is_zero() {
        let q = [0.0, 0.0];
        let flux = [8.0, 0.0];
        let curiosity = [0.0, 1.0];
        let a = qsma_select_beta(&q, &flux, &curiosity, 0.0);
        assert_eq!(a.index, 1);
    }

    #[test]
    fn bias_top_k_adds_ease_beta_to_leading_logit() {
        let mut logits = vec![0.0f32, 2.0, 1.0];
        let mut flux = HashMap::new();
        flux.insert(1, 8.0);
        let a = bias_top_k(&mut logits, 3, &flux, &[], 1.5, 0.0);
        assert_eq!(a.index, 1);
        assert!(logits[1] > 2.0);
        assert!(
            logits[1] > logits[0],
            "grooved token must outrank idle after QSMA"
        );
        assert!(logits[0] > 0.0, "idle token still gets ease(0)×β + C");
    }

    #[test]
    fn scheduled_select_uses_beta_of_t() {
        let q = [1.0];
        let flux = [0.0];
        let curiosity = [0.0];
        let a = qsma_select(&q, &flux, &curiosity, 0);
        assert!((a.beta - 1.5).abs() < 1e-12);
        assert_eq!(a.index, 0);
    }

    #[test]
    fn scheduled_beta_floors_at_point_one() {
        assert!((beta(0) - 1.5).abs() < 1e-12);
        assert!(beta(10_000) >= 0.1 - 1e-12);
    }
}
