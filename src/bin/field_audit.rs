//! Diderot field geometry + vector-field divergence audit.
//!
//! ```bash
//! cargo run --release --bin field_audit -- \
//!   --model data/google/gemma-3-27b-it-Q8_0.gguf [--sigma 11] [--points 4096]
//! ```

use anyhow::{Context, Result};
use candle_core::quantized::gguf_file;
use candle_core::{DType, Device, Tensor};
use std::io::BufReader;
use std::path::Path;

/// Matches `src/field.rs` density / gradient math (CPU audit copy).
struct Field {
    /// Row-major (N, D)
    flat: Vec<f32>,
    sigma: f32,
    dim: usize,
    n: usize,
}

impl Field {
    fn from_positions(positions: Tensor, sigma: f32) -> Result<Self> {
        let positions = positions.to_dtype(DType::F32)?.to_device(&Device::Cpu)?;
        let n = positions.dim(0)?;
        let dim = positions.dim(1)?;
        let flat = positions.flatten_all()?.to_vec1::<f32>()?;
        Ok(Self {
            flat,
            sigma,
            dim,
            n,
        })
    }

    fn sample_indices(&self, max_points: usize) -> Vec<usize> {
        let max_points = max_points.min(self.n).max(1);
        let step = (self.n / max_points).max(1);
        let mut out = Vec::with_capacity(max_points);
        let mut i = 0;
        while i < self.n && out.len() < max_points {
            out.push(i);
            i += step;
        }
        out
    }

    fn mass_scale(&self, used: usize) -> f64 {
        self.n as f64 / used.max(1) as f64
    }

    /// ρ(x) = Σ_i exp(-||μ_i - x||² / σ²)
    fn density(&self, x: &[f32], max_points: usize) -> f32 {
        let idxs = self.sample_indices(max_points);
        let inv_s2 = 1.0 / (self.sigma as f64 * self.sigma as f64);
        let mut rho = 0.0f64;
        for &i in &idxs {
            let base = i * self.dim;
            let mut r2 = 0.0f64;
            for d in 0..self.dim {
                let diff = self.flat[base + d] as f64 - x[d] as f64;
                r2 += diff * diff;
            }
            rho += (-r2 * inv_s2).exp();
        }
        (rho * self.mass_scale(idxs.len())) as f32
    }

    /// F = ∇ρ = (2/σ²) Σ_i G_i (μ_i - x)   [same as field.rs]
    fn force(&self, x: &[f32], max_points: usize) -> Vec<f32> {
        let idxs = self.sample_indices(max_points);
        let inv_s2 = 1.0 / (self.sigma as f64 * self.sigma as f64);
        let scale = 2.0 * inv_s2;
        let mut f = vec![0.0f64; self.dim];
        for &i in &idxs {
            let base = i * self.dim;
            let mut r2 = 0.0f64;
            for d in 0..self.dim {
                let diff = self.flat[base + d] as f64 - x[d] as f64;
                r2 += diff * diff;
            }
            let k = (-r2 * inv_s2).exp();
            for d in 0..self.dim {
                let diff = self.flat[base + d] as f64 - x[d] as f64;
                f[d] += scale * k * diff;
            }
        }
        let m = self.mass_scale(idxs.len());
        f.into_iter().map(|v| (v * m) as f32).collect()
    }

    /// Analytic div F = ∇²ρ for G_i = exp(-r_i²/σ²):
    /// ∇²G = (2G/σ²)( -D + 2 r²/σ² )
    fn analytic_divergence(&self, x: &[f32], max_points: usize) -> f32 {
        let idxs = self.sample_indices(max_points);
        let s2 = self.sigma as f64 * self.sigma as f64;
        let inv_s2 = 1.0 / s2;
        let d = self.dim as f64;
        let mut lap = 0.0f64;
        for &i in &idxs {
            let base = i * self.dim;
            let mut r2 = 0.0f64;
            for j in 0..self.dim {
                let diff = self.flat[base + j] as f64 - x[j] as f64;
                r2 += diff * diff;
            }
            let g = (-r2 * inv_s2).exp();
            lap += (2.0 * g * inv_s2) * (-d + 2.0 * r2 * inv_s2);
        }
        (lap * self.mass_scale(idxs.len())) as f32
    }

    fn row(&self, idx: usize) -> Vec<f32> {
        let base = idx * self.dim;
        self.flat[base..base + self.dim].to_vec()
    }

    fn pairwise_stats(&self, pairs: usize) -> (f32, f32, f32) {
        let mut rng = 42u64;
        let mut dists = Vec::with_capacity(pairs);
        for _ in 0..pairs {
            rng = rng
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let i = (rng >> 33) as usize % self.n;
            rng = rng
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let mut j = (rng >> 33) as usize % (self.n - 1);
            if j >= i {
                j += 1;
            }
            let bi = i * self.dim;
            let bj = j * self.dim;
            let mut r2 = 0.0f32;
            for d in 0..self.dim {
                let diff = self.flat[bi + d] - self.flat[bj + d];
                r2 += diff * diff;
            }
            dists.push(r2.sqrt());
        }
        dists.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mean = dists.iter().sum::<f32>() / dists.len() as f32;
        (mean, dists[dists.len() / 10], dists[dists.len() * 9 / 10])
    }

    fn norm_stats(&self, samples: usize) -> (f32, f32, f32) {
        let step = (self.n / samples.max(1)).max(1);
        let mut norms = Vec::new();
        let mut i = 0;
        while i < self.n && norms.len() < samples {
            let v = self.row(i);
            norms.push(mag(&v));
            i += step;
        }
        norms.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mean = norms.iter().sum::<f32>() / norms.len() as f32;
        (mean, norms[0], *norms.last().unwrap())
    }

    fn min_dist_to_cloud(&self, x: &[f32], samples: usize) -> f32 {
        let step = (self.n / samples.max(1)).max(1);
        let mut min_d = f32::MAX;
        let mut i = 0;
        while i < self.n {
            let base = i * self.dim;
            let mut r2 = 0.0f32;
            for d in 0..self.dim {
                let diff = self.flat[base + d] - x[d];
                r2 += diff * diff;
            }
            min_d = min_d.min(r2.sqrt());
            i += step;
        }
        min_d
    }
}

fn mag(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

fn scale_to_norm(v: &[f32], target: f32) -> Vec<f32> {
    let n = mag(v).max(1e-6);
    v.iter().map(|x| x * target / n).collect()
}

fn main() -> Result<()> {
    println!("=== Diderot Field Geometry + Divergence Audit ===\n");

    let args: Vec<String> = std::env::args().collect();
    let model = args
        .iter()
        .position(|a| a == "--model")
        .and_then(|i| args.get(i + 1))
        .map(|s| s.as_str())
        .unwrap_or("data/google/gemma-3-27b-it-Q8_0.gguf");
    let sigma_override = args
        .iter()
        .position(|a| a == "--sigma")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse::<f32>().ok());
    let max_points: usize = args
        .iter()
        .position(|a| a == "--points")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(4096);

    if !Path::new(model).exists() {
        anyhow::bail!("model not found: {model}");
    }

    println!("--- Load token embeddings from GGUF ---");
    println!("    model: {model}");
    let device = Device::Cpu;
    let mut file = std::fs::File::open(model)?;
    let mut reader = BufReader::new(&mut file);
    let ct = gguf_file::Content::read(&mut reader)?;
    let arch = ct
        .metadata
        .get("general.architecture")
        .and_then(|v| v.to_string().ok())
        .map(|s| s.as_str().to_string())
        .unwrap_or_else(|| "?".into());
    println!("    arch: {arch}");

    println!("    dequantizing token_embd.weight (slow on first load)...");
    let emb_q = ct
        .tensor(&mut reader, "token_embd.weight", &device)
        .context("token_embd.weight")?;
    let emb = emb_q.dequantize(&device)?;
    let n = emb.dim(0)?;
    let dim = emb.dim(1)?;
    println!("    embeddings: {n} × {dim}");

    // temp field for pairwise with unit sigma
    let tmp = Field::from_positions(emb.clone(), 1.0)?;
    let (mean_dist, p10, p90) = tmp.pairwise_stats(200);
    let floor = (dim as f32).sqrt() * 0.15;
    let auto_sigma = (mean_dist * 4.0).max(floor);
    let sigma = sigma_override.unwrap_or(auto_sigma);

    println!("\n--- Geometry of embedding cloud ---");
    println!("    pairwise L2  mean={mean_dist:.4}  p10={p10:.4}  p90={p90:.4}");
    println!("    auto sigma (from_embeddings) = {auto_sigma:.4}  floor={floor:.2}");
    println!("    using sigma                  = {sigma:.4}");

    let field = Field::from_positions(emb, sigma)?;
    let (nmean, nmin, nmax) = field.norm_stats(500);
    println!("    ||emb||      mean={nmean:.3}  min={nmin:.3}  max={nmax:.3}");

    let sink_r = (dim as f32 * sigma * sigma / 2.0).sqrt();
    println!("\n--- Analytic structure ---");
    println!("    ρ(x)   = Σ_i exp(-||μ_i - x||² / σ²)");
    println!("    F(x)   = ∇ρ = (2/σ²) Σ_i G_i (μ_i - x)   → toward peaks");
    println!("    div F  = ∇·F = ∇²ρ");
    println!("    single-Gaussian zero-Laplacian radius r* = σ√(D/2) = {sink_r:.2}");
    println!("    r < r* ⇒ div F < 0 (sink / converging)");
    println!("    r > r* ⇒ div F > 0 (expanding shell)");

    let emb0 = field.row(0);
    let near = {
        let mut v = emb0.clone();
        for d in 0..8.min(dim) {
            v[d] += 0.5;
        }
        v
    };
    let residual_like = scale_to_norm(&emb0, 450.0);
    let random_far = {
        let mut v = vec![0.0f32; dim];
        let mut rng = 7u64;
        for x in &mut v {
            rng = rng
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            *x = ((rng >> 33) as f32 / u32::MAX as f32) * 2.0 - 1.0;
        }
        scale_to_norm(&v, 450.0)
    };
    let midway = scale_to_norm(&emb0, nmean * 2.0);

    let probes: [(&str, Vec<f32>); 5] = [
        ("at emb[0] (density peak)", emb0.clone()),
        ("near emb[0] (+0.5 on 8 axes)", near),
        ("||x||=2·||emb|| along emb[0]", midway),
        ("residual-like ||x||=450 // emb[0]", residual_like),
        ("random sphere ||x||=450", random_far),
    ];

    println!("\n--- Probe table (≈{max_points} kernels, mass-scaled to N={n}) ---");
    println!(
        "{:<40} {:>8} {:>11} {:>11} {:>12} {:>10}",
        "probe", "||x||", "ρ", "||F||", "div F", "d_min"
    );

    for (name, x) in &probes {
        let rho = field.density(x, max_points);
        let f = field.force(x, max_points);
        let div = field.analytic_divergence(x, max_points);
        let dmin = field.min_dist_to_cloud(x, 3000);
        println!(
            "{:<40} {:>8.2} {:>11.3e} {:>11.3e} {:>12.3e} {:>10.2}",
            name,
            mag(x),
            rho,
            mag(&f),
            div,
            dmin
        );
    }

    println!("\n--- FD divergence cross-check (16 random axes, extrapolated ×D) ---");
    let x = field.row(42);
    let eps = (sigma * 0.05).max(1e-3);
    let mut div_fd = 0.0f64;
    let mut rng = 99u64;
    let n_axes = 16usize;
    let fd_points = max_points.min(2048);
    for _ in 0..n_axes {
        rng = rng
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let axis = (rng >> 33) as usize % dim;
        let mut xp = x.clone();
        let mut xm = x.clone();
        xp[axis] += eps;
        xm[axis] -= eps;
        let fp = field.force(&xp, fd_points);
        let fm = field.force(&xm, fd_points);
        div_fd += ((fp[axis] - fm[axis]) / (2.0 * eps)) as f64;
    }
    div_fd *= dim as f64 / n_axes as f64;
    let div_an = field.analytic_divergence(&x, fd_points);
    println!("    probe emb[42], eps={eps:.4}");
    println!("    analytic div F = {div_an:.6e}");
    println!("    FD approx      = {div_fd:.6e}");
    if div_an.abs() > 1e-20 {
        println!("    FD/analytic    = {:.3}", div_fd / div_an as f64);
    }

    println!("\n--- Residual space vs emb cloud (why F_g died) ---");
    let x_res = scale_to_norm(&field.row(100), 450.0);
    let min_d = field.min_dist_to_cloud(&x_res, 4000);
    let g = (-(min_d * min_d) / (sigma * sigma)).exp();
    println!("    ||emb|| mean ≈ {nmean:.2}");
    println!("    typical residual ||h|| ≈ 400–450 (goal attractor logs)");
    println!("    min dist residual→emb sample ≈ {min_d:.2}");
    println!("    kernel G=exp(-d²/σ²) ≈ {g:.3e}");
    if g < 1e-20 {
        println!("    → kernels UNDERFLOW → field.rs returns F = 0 (fast path)");
    } else if g < 1e-6 {
        println!("    → kernels nearly dead → ||F|| numerically tiny");
    } else {
        println!("    → some support remains; check Top-K / viscosity scale");
    }
    println!("    nearest-emb pull in niodoo is the correct F_g wake off-manifold");

    println!("\n--- Flow topology (what div F means for steering) ---");
    println!("    • F = +∇ρ  = gradient ASCENT  → flow into token-embedding sinks");
    println!("    • div F < 0 near μ_i         → volume contracts (attractors)");
    println!("    • div F > 0 far shells       → expanding / unstable");
    println!("    • High-D: r* = σ√(D/2) = {sink_r:.1}");
    println!("      emb pairwise mean ≈ {mean_dist:.2}  <<  r*");
    println!("      → ON the emb cloud you are almost always deep in sink basins");
    println!("      → OFF cloud (residual) ρ≈0 and div F≈0 — dead field");

    // Write machine-readable summary
    let out = "logs/field_audit_summary.txt";
    std::fs::create_dir_all("logs").ok();
    std::fs::write(
        out,
        format!(
            "n={n} dim={dim} sigma={sigma:.6} mean_pair={mean_dist:.6} \
             emb_norm_mean={nmean:.6} sink_radius={sink_r:.6} \
             residual_min_dist={min_d:.6} residual_kernel={g:.6e}\n"
        ),
    )?;
    println!("\nWrote {out}");
    println!("=== Audit complete ===");
    Ok(())
}
