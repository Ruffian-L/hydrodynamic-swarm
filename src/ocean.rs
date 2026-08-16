//! Shared Ocean — multi-mind field packets in one thermodynamic space.
//!
//! Lane C foundation: minds deposit dense vectors into a shared ocean.
//! Packets are refined by iterative "denoising" (noise residual decays as
//! representations agree). Steering force pulls the host residual toward
//! crystallized consensus — not token votes.
//!
//! This is the Architect's Bridge / diffusion-style pass without requiring
//! a full DiffusionGemma stack: block-level field exchange, not AR voting.

use candle_core::{Device, Result, Tensor};

/// Which mind deposited a packet.
#[derive(Debug, Clone, PartialEq, Eq)]
#[allow(dead_code)] // Llama/Qwen/Other reserved for multi-mind ocean deposits
pub enum MindId {
    Host,
    Gemma,
    Llama,
    Qwen,
    Other(String),
}

impl MindId {
    #[allow(dead_code)]
    pub fn as_str(&self) -> &str {
        match self {
            MindId::Host => "host",
            MindId::Gemma => "gemma",
            MindId::Llama => "llama",
            MindId::Qwen => "qwen",
            MindId::Other(s) => s.as_str(),
        }
    }
}

/// One mind's contribution to the shared ocean.
#[derive(Debug, Clone)]
pub struct FieldPacket {
    #[allow(dead_code)]
    pub source: MindId,
    /// Position in host residual dim (D,).
    pub mu: Tensor,
    pub weight: f32,
    /// 1.0 = raw / noisy; 0.0 = fully crystallized.
    pub residual_noise: f32,
}

/// Config for ocean physics (also wired from TOML later).
#[derive(Debug, Clone)]
pub struct OceanConfig {
    pub enabled: bool,
    /// Deposit host hidden state every N generated tokens.
    pub deposit_interval: usize,
    /// Run refine (denoise) every N deposits.
    pub refine_every: usize,
    /// Scale on ocean force before it joins the Niodoo sum.
    pub force_scale: f32,
    pub max_packets: usize,
    /// Per refine step: noise *= this (0.85 = 15% crystallization).
    pub noise_decay: f32,
    /// Blend factor toward consensus mean during refine.
    pub consensus_blend: f32,
}

impl Default for OceanConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            deposit_interval: 4,
            refine_every: 2,
            // Lower pull so contaminated consensus can't dominate late-run
            force_scale: 0.12,
            max_packets: 48,
            // Slower crystallization — less “lock in junk” by step 100
            noise_decay: 0.93,
            consensus_blend: 0.08,
        }
    }
}

/// Shared multi-mind latent ocean.
pub struct SharedOcean {
    pub dim: usize,
    pub device: Device,
    pub config: OceanConfig,
    packets: Vec<FieldPacket>,
    deposits_since_refine: usize,
    total_deposits: usize,
    /// Last measured force magnitude (telemetry).
    pub last_force_mag: f32,
}

impl SharedOcean {
    pub fn new(dim: usize, device: Device, config: OceanConfig) -> Self {
        Self {
            dim,
            device,
            config,
            packets: Vec::new(),
            deposits_since_refine: 0,
            total_deposits: 0,
            last_force_mag: 0.0,
        }
    }

    pub fn len(&self) -> usize {
        self.packets.len()
    }

    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool {
        self.packets.is_empty()
    }

    pub fn total_deposits(&self) -> usize {
        self.total_deposits
    }

    /// Project an arbitrary-length vector into host dim (pad / truncate).
    /// Temporary bridge until learned projection matrices land.
    pub fn project_to_host(&self, vec: &Tensor) -> Result<Tensor> {
        let flat = if vec.dims().len() == 1 {
            vec.clone()
        } else {
            vec.flatten_all()?
        };
        let n = flat.dim(0)?;
        if n == self.dim {
            return Ok(flat
                .to_device(&self.device)?
                .to_dtype(candle_core::DType::F32)?);
        }
        let data: Vec<f32> = flat
            .to_dtype(candle_core::DType::F32)?
            .to_device(&Device::Cpu)?
            .to_vec1()?;
        let mut out = vec![0.0f32; self.dim];
        let copy_n = n.min(self.dim);
        out[..copy_n].copy_from_slice(&data[..copy_n]);
        // If source is smaller, residual dims stay 0 — intentional stub for learned proj.
        Tensor::from_vec(out, self.dim, &self.device)
    }

    /// Deposit a mind packet into the ocean.
    pub fn deposit(
        &mut self,
        source: MindId,
        vector: &Tensor,
        weight: f32,
        initial_noise: f32,
    ) -> Result<()> {
        if !self.config.enabled {
            return Ok(());
        }
        let mu = self.project_to_host(vector)?;
        self.packets.push(FieldPacket {
            source,
            mu,
            weight: weight.max(0.0),
            residual_noise: initial_noise.clamp(0.0, 1.0),
        });
        self.total_deposits += 1;
        self.deposits_since_refine += 1;

        if self.packets.len() > self.config.max_packets {
            // Drop highest-noise packets first (keep crystallized structure).
            self.packets.sort_by(|a, b| {
                a.residual_noise
                    .partial_cmp(&b.residual_noise)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            self.packets.truncate(self.config.max_packets);
        }

        if self.deposits_since_refine >= self.config.refine_every {
            self.refine_step()?;
            self.deposits_since_refine = 0;
        }
        Ok(())
    }

    /// Diffusion-style refine: pull packets toward mass-weighted consensus,
    /// decay residual noise (crystallization).
    pub fn refine_step(&mut self) -> Result<()> {
        if self.packets.len() < 2 {
            for p in &mut self.packets {
                p.residual_noise *= self.config.noise_decay;
            }
            return Ok(());
        }

        // Mass-weighted mean (weight * crystallization).
        let mut acc = Tensor::zeros(self.dim, candle_core::DType::F32, &self.device)?;
        let mut mass = 0.0f32;
        for p in &self.packets {
            let m = p.weight * (1.0 - p.residual_noise).max(0.05);
            acc = (&acc + p.mu.affine(m as f64, 0.0)?)?;
            mass += m;
        }
        if mass <= 1e-8 {
            return Ok(());
        }
        let consensus = acc.affine(1.0 / mass as f64, 0.0)?;
        let blend = self.config.consensus_blend as f64;

        for p in &mut self.packets {
            // mu <- (1-blend)*mu + blend*consensus
            p.mu = (&p.mu.affine(1.0 - blend, 0.0)? + consensus.affine(blend, 0.0)?)?;
            p.residual_noise = (p.residual_noise * self.config.noise_decay).clamp(0.0, 1.0);
        }
        Ok(())
    }

    /// Force at position: pull toward crystallized packets.
    /// F = scale * sum_i w_i * (1 - noise_i) * (mu_i - pos)
    pub fn query_force(&mut self, pos: &Tensor) -> Result<Tensor> {
        if !self.config.enabled || self.packets.is_empty() {
            self.last_force_mag = 0.0;
            return Tensor::zeros(self.dim, candle_core::DType::F32, &self.device);
        }

        let pos = if pos.dims().len() == 1 {
            pos.clone()
        } else {
            pos.flatten_all()?
        };

        let mut force = Tensor::zeros(self.dim, candle_core::DType::F32, &self.device)?;
        let mut total_w = 0.0f32;

        for p in &self.packets {
            let crystal = (1.0 - p.residual_noise).max(0.0);
            let w = p.weight * crystal;
            if w < 1e-8 {
                continue;
            }
            let delta = (&p.mu - &pos)?;
            force = (&force + delta.affine(w as f64, 0.0)?)?;
            total_w += w;
        }

        if total_w > 1e-8 {
            force = force.affine((self.config.force_scale / total_w) as f64, 0.0)?;
        } else {
            force = force.affine(self.config.force_scale as f64, 0.0)?;
        }

        self.last_force_mag = force.sqr()?.sum_all()?.to_scalar::<f32>()?.sqrt();
        Ok(force)
    }

    /// Mean residual noise across packets (1 = chaos, 0 = fully crystallized).
    pub fn mean_noise(&self) -> f32 {
        if self.packets.is_empty() {
            return 1.0;
        }
        self.packets.iter().map(|p| p.residual_noise).sum::<f32>() / self.packets.len() as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deposit_and_force_pulls_toward_packet() {
        let device = Device::Cpu;
        let dim = 8;
        let mut ocean = SharedOcean::new(dim, device.clone(), OceanConfig::default());
        let target = Tensor::from_vec(vec![1.0f32; dim], dim, &device).unwrap();
        ocean.deposit(MindId::Gemma, &target, 1.0, 0.0).unwrap();
        let pos = Tensor::zeros(dim, candle_core::DType::F32, &device).unwrap();
        let f = ocean.query_force(&pos).unwrap();
        let fv: Vec<f32> = f.to_vec1().unwrap();
        // Force should point positive (toward ones)
        assert!(fv.iter().sum::<f32>() > 0.0);
    }

    #[test]
    fn refine_reduces_noise() {
        let device = Device::Cpu;
        let dim = 4;
        let mut cfg = OceanConfig::default();
        cfg.refine_every = 100; // manual refine
        let mut ocean = SharedOcean::new(dim, device.clone(), cfg);
        let a = Tensor::from_vec(vec![1.0f32, 0.0, 0.0, 0.0], dim, &device).unwrap();
        let b = Tensor::from_vec(vec![0.0f32, 1.0, 0.0, 0.0], dim, &device).unwrap();
        ocean.deposit(MindId::Gemma, &a, 1.0, 1.0).unwrap();
        ocean.deposit(MindId::Llama, &b, 1.0, 1.0).unwrap();
        let before = ocean.mean_noise();
        ocean.refine_step().unwrap();
        let after = ocean.mean_noise();
        assert!(after < before);
    }

    #[test]
    fn project_pads_short_vectors() {
        let device = Device::Cpu;
        let ocean = SharedOcean::new(8, device.clone(), OceanConfig::default());
        let short = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], 3, &device).unwrap();
        let p = ocean.project_to_host(&short).unwrap();
        assert_eq!(p.dim(0).unwrap(), 8);
        let v: Vec<f32> = p.to_vec1().unwrap();
        assert!((v[0] - 1.0).abs() < 1e-5);
        assert!((v[7] - 0.0).abs() < 1e-5);
    }
}
