//! Qwen3.5 Hybrid Model -- Gated DeltaNet + Gated Attention
//!
//! Ported from: transformers/models/qwen3_5/modeling_qwen3_5.py
//! and llama.cpp Qwen3.5 GGUF support.
//!
//! Architecture: 32 blocks, 2560D hidden
//!   - DeltaNet layers (0,1,2, 4,5,6, ...): Gated DeltaNet linear attention
//!   - Full Attention layers (3,7,11,...every 4th): Gated self-attention w/ RoPE
//!   - Each block: input_norm -> attn -> residual -> post_norm -> MLP -> residual
//!
//! RMSNorm uses (1+weight) convention (differs from Llama).



use candle_core::quantized::QTensor;
use candle_core::quantized::gguf_file;
use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{Embedding, Module};

pub const MAX_SEQ_LEN: usize = 4096;

// ─── QMatMul wrapper ──────────────────────────────────────────────────
#[derive(Debug, Clone)]
enum QMatMulInner {
    Quantized(candle_core::quantized::QMatMul),
    Dequantized(Tensor), // (out_dim, in_dim) stored as F32
}

#[derive(Debug, Clone)]
struct QMatMul {
    inner: QMatMulInner,
}

impl QMatMul {
    fn from_qtensor(qtensor: QTensor) -> Result<Self> {
        let inner = candle_core::quantized::QMatMul::from_qtensor(qtensor)?;
        Ok(Self { inner: QMatMulInner::Quantized(inner) })
    }

    /// Dequantize at load time -- uses plain F32 matmul (bypasses Q5_K bugs).
    fn from_qtensor_dequantized(qtensor: QTensor, device: &Device) -> Result<Self> {
        let weight = qtensor.dequantize(device)?; // (out_dim, in_dim)
        Ok(Self { inner: QMatMulInner::Dequantized(weight) })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        match &self.inner {
            QMatMulInner::Quantized(qmm) => {
                // Metal quantized matmul requires contiguous 2D F32 input.
                let shape = xs.shape().clone();
                let dims = shape.dims();
                if dims.len() <= 2 {
                    return qmm.forward(&xs.to_dtype(DType::F32)?.contiguous()?);
                }
                let last = dims[dims.len() - 1];
                let batch: usize = dims[..dims.len() - 1].iter().product();
                let flat = xs.to_dtype(DType::F32)?.contiguous()?.reshape((batch, last))?;
                let out = qmm.forward(&flat)?;
                let out_dim = out.dim(1)?;
                let mut out_shape: Vec<usize> = dims[..dims.len() - 1].to_vec();
                out_shape.push(out_dim);
                out.reshape(out_shape)
            }
            QMatMulInner::Dequantized(weight) => {
                // Plain matmul: xs @ weight^T, flattened to 2D to avoid broadcast issues
                let xs_f32 = xs.to_dtype(DType::F32)?.contiguous()?;
                let shape = xs_f32.shape().clone();
                let dims = shape.dims();
                let last = dims[dims.len() - 1];
                let batch: usize = dims[..dims.len() - 1].iter().product();
                let flat = xs_f32.reshape((batch, last))?;
                let out = flat.matmul(&weight.t()?)?;
                let out_dim = out.dim(1)?;
                let mut out_shape: Vec<usize> = dims[..dims.len() - 1].to_vec();
                out_shape.push(out_dim);
                out.reshape(out_shape)
            }
        }
    }
}

// ─── RMSNorm (1+weight convention) ───────────────────────────────────
#[derive(Debug, Clone)]
struct Qwen35RmsNorm {
    weight: Tensor, // (dim,). Norm = x * rsqrt(mean(x^2) + eps) * (1 + weight)
    eps: f64,
}

impl Qwen35RmsNorm {
    fn from_qtensor(qtensor: QTensor, eps: f64, device: &Device) -> Result<Self> {
        let weight = qtensor.dequantize(device)?;
        Ok(Self { weight, eps })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs_f32 = xs.to_dtype(DType::F32)?;
        let variance = xs_f32.sqr()?.mean_keepdim(candle_core::D::Minus1)?;
        let rsqrt = (variance + self.eps)?.recip()?.sqrt()?;
        let normed = xs_f32.broadcast_mul(&rsqrt)?;
        // (1 + weight) convention
        let ones_plus_w = (&self.weight + 1.0f64)?;
        let result = normed.broadcast_mul(&ones_plus_w)?;
        result.to_dtype(xs.dtype())
    }
}

// ─── Gated RMSNorm (for DeltaNet output) ─────────────────────────────
/// RMSNorm with SiLU gating: output = rms_norm(x) * silu(z)
fn gated_rms_norm(x: &Tensor, z: &Tensor, weight: &Tensor, eps: f64) -> Result<Tensor> {
    let x_f32 = x.to_dtype(DType::F32)?;
    let variance = x_f32.sqr()?.mean_keepdim(candle_core::D::Minus1)?;
    let rsqrt = (variance + eps)?.recip()?.sqrt()?;
    let normed = x_f32.broadcast_mul(&rsqrt)?;
    let ones_plus_w = (weight + 1.0f64)?;
    let normed = normed.broadcast_mul(&ones_plus_w)?;
    // SiLU gate
    let z_f32 = z.to_dtype(DType::F32)?;
    let gate = candle_nn::ops::silu(&z_f32)?;
    let result = (&normed * &gate)?;
    result.to_dtype(x.dtype())
}

// ─── MLP (SiLU gated) ────────────────────────────────────────────────
#[derive(Debug, Clone)]
struct Mlp {
    gate: QMatMul,
    up: QMatMul,
    down: QMatMul,
}

impl Mlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let gate = candle_nn::ops::silu(&self.gate.forward(xs)?)?;
        let up = self.up.forward(xs)?;
        self.down.forward(&(gate * up)?)
    }
}

// ─── L2 Norm (for DeltaNet Q/K) ──────────────────────────────────────
fn l2norm(x: &Tensor) -> Result<Tensor> {
    let x_f32 = x.to_dtype(DType::F32)?;
    let sq_sum = x_f32.sqr()?.sum_keepdim(candle_core::D::Minus1)?;
    let inv_norm = (sq_sum + 1e-6f64)?.recip()?.sqrt()?;
    let result = x_f32.broadcast_mul(&inv_norm)?;
    result.to_dtype(x.dtype())
}

// ─── Softplus ─────────────────────────────────────────────────────────
fn softplus(x: &Tensor) -> Result<Tensor> {
    // softplus(x) = log(1 + exp(x)), numerically stable:
    // For large x (>20), exp(x) overflows, but softplus(x) ~ x.
    // Use: where(x > 20, x, log(1 + exp(x)))
    let threshold = (x.ge(&Tensor::new(20.0f32, x.device())?.broadcast_as(x.shape())?))?;
    let safe_exp = x.clamp(-50.0f32, 20.0f32)?.exp()?;
    let log_branch = (safe_exp + 1.0f64)?.log()?;
    // threshold is 1 where x>20 (use x directly), 0 where x<=20 (use log_branch)
    let result = (x.to_dtype(DType::F32)?.broadcast_mul(&threshold.to_dtype(DType::F32)?)?
        + log_branch.to_dtype(DType::F32)?.broadcast_mul(&(1.0f64 - threshold.to_dtype(DType::F32)?)?)?)?;
    result.to_dtype(x.dtype())
}

// ─── DeltaNet Layer ──────────────────────────────────────────────────
#[derive(Debug, Clone)]
struct DeltaNetLayer {
    input_norm: Qwen35RmsNorm,
    post_norm: Qwen35RmsNorm,
    // QKV fused projection: (hidden, inner*2)
    in_proj_qkv: QMatMul,
    // Gate projection: (hidden, inner)
    in_proj_z: QMatMul,
    // Beta projection: (n_heads, hidden) -- dequantized (too small for Metal Q kernel)
    in_proj_b: Tensor,
    // Alpha projection: (n_heads, hidden) -- dequantized
    in_proj_a: Tensor,
    // Conv1d weight: (inner*2, kernel_size)
    conv1d_weight: Tensor,
    // SSM parameters
    a_log: Tensor,      // (n_heads,) -- log of decay base
    dt_bias: Tensor,    // (n_heads,) -- time step bias
    ssm_norm: Tensor,   // (head_v_dim,) -- RMSNormGated weight
    // Output projection
    out_proj: QMatMul,
    // Dimensions
    n_v_heads: usize,
    n_k_heads: usize,
    head_k_dim: usize,
    head_v_dim: usize,
    key_dim: usize,
    value_dim: usize,
    conv_kernel: usize,
    rms_eps: f64,
    // MLP
    mlp: Mlp,
    // State
    conv_state: Option<Tensor>,
    recurrent_state: Option<Tensor>,
    // Debug: layer index (printed once during prefill)
    layer_idx: usize,
}

impl DeltaNetLayer {
    fn forward(&mut self, xs: &Tensor, _index_pos: usize) -> Result<Tensor> {
        let (b_sz, seq_len, _hidden) = xs.dims3()?;
        let residual = xs;

        // Pre-norm
        let h = self.input_norm.forward(xs)?;


        // QKV projection
        let mixed_qkv = self.in_proj_qkv.forward(&h)?; // (b, seq, key_dim*2 + value_dim)
        // Z gate
        let z = self.in_proj_z.forward(&h)?; // (b, seq, value_dim)
        // Beta (update gate) -- dequantized matmul: h @ in_proj_b^T
        let b = h.broadcast_matmul(&self.in_proj_b.t()?)?; // (b, seq, n_v_heads)
        // Alpha (decay gate)
        let a = h.broadcast_matmul(&self.in_proj_a.t()?)?; // (b, seq, n_v_heads)

        // Causal Conv1d
        let conv_dim = self.key_dim * 2 + self.value_dim;
        let mixed_qkv = if seq_len == 1 {
            // Single token: use cached conv state
            self.causal_conv1d_update(&mixed_qkv, conv_dim)?
        } else {
            // Prefill: full conv
            self.causal_conv1d_prefill(&mixed_qkv, conv_dim, seq_len)?
        };

        // Split into Q, K, V
        let query = mixed_qkv.narrow(2, 0, self.key_dim)?;
        let key = mixed_qkv.narrow(2, self.key_dim, self.key_dim)?;
        let value = mixed_qkv.narrow(2, self.key_dim * 2, self.value_dim)?;

        // Reshape to heads
        let query = query.reshape((b_sz, seq_len, self.n_k_heads, self.head_k_dim))?;
        let key = key.reshape((b_sz, seq_len, self.n_k_heads, self.head_k_dim))?;
        let value = value.reshape((b_sz, seq_len, self.n_v_heads, self.head_v_dim))?;

        // Beta = sigmoid(b)
        let beta = candle_nn::ops::sigmoid(&b)?; // (b, seq, n_v_heads)

        // g = -exp(A_log) * softplus(a + dt_bias)
        // a: [b, seq, n_heads], dt_bias: [n_heads] -> unsqueeze to [1, 1, n_heads]
        let a_exp = self.a_log.to_dtype(DType::F32)?.exp()?.neg()?;
        let a_exp = a_exp.unsqueeze(0)?.unsqueeze(0)?; // [1, 1, n_heads]
        let dt_bias_3d = self.dt_bias.unsqueeze(0)?.unsqueeze(0)?; // [1, 1, n_heads]
        let a_plus_dt = a.to_dtype(DType::F32)?.broadcast_add(&dt_bias_3d)?;
        let sp = softplus(&a_plus_dt)?;
        let g = a_exp.broadcast_mul(&sp)?; // (b, seq, n_v_heads)

        // DEBUG: print layer-0 stats during prefill to diagnose garbage output
        if self.layer_idx == 0 && seq_len > 1 {
            let g_vec: Vec<f32> = g.flatten_all()?.to_vec1().unwrap_or_default();
            let g_exp_vals: Vec<f32> = g_vec.iter().map(|x| x.exp()).collect();
            let beta_vec: Vec<f32> = beta.flatten_all()?.to_vec1().unwrap_or_default();
            let qkv_vec: Vec<f32> = mixed_qkv.flatten_all()?.to_dtype(DType::F32)?.to_vec1().unwrap_or_default();
            eprintln!("[DBG L0] g range: [{:.4},{:.4}]  exp(g) range: [{:.4},{:.4}]  beta range: [{:.4},{:.4}]  conv_out rms: {:.4}",
                g_vec.iter().cloned().fold(f32::INFINITY, f32::min),
                g_vec.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
                g_exp_vals.iter().cloned().fold(f32::INFINITY, f32::min),
                g_exp_vals.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
                beta_vec.iter().cloned().fold(f32::INFINITY, f32::min),
                beta_vec.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
                (qkv_vec.iter().map(|x| x*x).sum::<f32>() / qkv_vec.len() as f32).sqrt(),
            );
        }

        // Repeat Q/K heads if n_v_heads > n_k_heads (repeat_interleave)
        let (query, key) = if self.n_v_heads > self.n_k_heads {
            let rep = self.n_v_heads / self.n_k_heads;
            let (b, s, nk, hk) = query.dims4()?;
            let q = query.unsqueeze(3)?.expand((b, s, nk, rep, hk))?.reshape((b, s, nk * rep, hk))?;
            let k = key.unsqueeze(3)?.expand((b, s, nk, rep, hk))?.reshape((b, s, nk * rep, hk))?;
            (q, k)
        } else {
            (query, key)
        };

        // DeltaNet recurrence (single token mode for generation)
        let attn_out = if seq_len == 1 {
            self.recurrent_deltanet(&query, &key, &value, &g, &beta)?
        } else {
            self.prefill_deltanet(&query, &key, &value, &g, &beta, seq_len)?
        };

        // Gated RMSNorm: norm(attn_out) * silu(z)
        let attn_flat = attn_out.reshape((b_sz * seq_len, self.n_v_heads * self.head_v_dim))?;
        let z_flat = z.reshape((b_sz * seq_len, self.n_v_heads * self.head_v_dim))?;
        // Per-head norm: reshape to (..., head_v_dim), norm, then reshape back
        let attn_heads = attn_flat.reshape((b_sz * seq_len * self.n_v_heads, self.head_v_dim))?;
        let z_heads = z_flat.reshape((b_sz * seq_len * self.n_v_heads, self.head_v_dim))?;
        let normed = gated_rms_norm(&attn_heads, &z_heads, &self.ssm_norm, self.rms_eps)?;
        let normed = normed.reshape((b_sz, seq_len, self.value_dim))?;

        // Output projection
        let output = self.out_proj.forward(&normed)?;

        // Residual
        let hidden = (residual + &output)?;

        // MLP block
        let residual2 = &hidden;
        let h2 = self.post_norm.forward(&hidden)?;
        let mlp_out = self.mlp.forward(&h2)?;
        let output = (residual2 + &mlp_out)?;

        Ok(output)
    }

    fn causal_conv1d_update(&mut self, x: &Tensor, conv_dim: usize) -> Result<Tensor> {
        // x: (b, 1, conv_dim) -- single new token
        let x_squeezed = x.to_dtype(DType::F32)?.squeeze(1)?; // (b, conv_dim)
        let b_sz = x_squeezed.dim(0)?;

        // Maintain a sliding window of size `conv_kernel`: (b, conv_dim, conv_kernel)
        let window = match &self.conv_state {
            Some(prev) => {
                // prev: (b, conv_dim, conv_kernel), drop oldest, append new
                let kept = prev.narrow(2, 1, self.conv_kernel - 1)?; // (b, conv_dim, kernel-1)
                let new_col = x_squeezed.unsqueeze(2)?; // (b, conv_dim, 1)
                Tensor::cat(&[&kept, &new_col], 2)? // (b, conv_dim, kernel)
            }
            None => {
                // First call: zero-pad left, current token at rightmost position
                let zeros = Tensor::zeros(&[b_sz, conv_dim, self.conv_kernel - 1], DType::F32, x.device())?;
                let new_col = x_squeezed.unsqueeze(2)?;
                Tensor::cat(&[&zeros, &new_col], 2)?
            }
        };
        self.conv_state = Some(window.clone());

        // Depthwise conv1d: each channel gets dot product with its weight.
        // window: (b, conv_dim, kernel)
        // conv1d_weight may be stored as (kernel, conv_dim) in GGUF — always
        // normalise to (conv_dim, kernel) here so the shapes are unambiguous.
        let w_raw = self.conv1d_weight.to_dtype(DType::F32)?;
        let w = if w_raw.dim(0)? == self.conv_kernel && w_raw.dim(1)? != self.conv_kernel {
            // shape is (kernel, conv_dim) — transpose to (conv_dim, kernel)
            w_raw.t()?.contiguous()?
        } else {
            // already (conv_dim, kernel)
            w_raw.contiguous()?
        };
        let out = window.broadcast_mul(&w.unsqueeze(0)?)?.sum(2)?; // (b, conv_dim)
        let out = candle_nn::ops::silu(&out)?;
        out.unsqueeze(1)?.to_dtype(x.dtype()) // (b, 1, conv_dim)
    }

    fn causal_conv1d_prefill(&mut self, x: &Tensor, conv_dim: usize, seq_len: usize) -> Result<Tensor> {
        // x: (b, seq_len, conv_dim)
        let x_t = x.to_dtype(DType::F32)?.transpose(1, 2)?; // (b, conv_dim, seq_len)
        let b_sz = x_t.dim(0)?;

        // Pad left with kernel-1 zeros for causal
        let pad = Tensor::zeros(&[b_sz, conv_dim, self.conv_kernel - 1], DType::F32, x.device())?;
        let padded = Tensor::cat(&[&pad, &x_t], 2)?; // (b, conv_dim, seq_len + kernel - 1)

        // Depthwise conv: for each position, dot product with weight.
        // Normalise weight shape to (conv_dim, kernel) same as update path.
        let w_raw = self.conv1d_weight.to_dtype(DType::F32)?; // (conv_dim OR kernel, other)
        let w = if w_raw.dim(0)? == self.conv_kernel && w_raw.dim(1)? != self.conv_kernel {
            w_raw.t()?.contiguous()?
        } else {
            w_raw.contiguous()?
        };
        let mut outputs = Vec::with_capacity(seq_len);
        for t in 0..seq_len {
            let window = padded.narrow(2, t, self.conv_kernel)?; // (b, conv_dim, kernel)
            let out = window.broadcast_mul(&w.unsqueeze(0)?)?.sum(2)?; // (b, conv_dim)
            outputs.push(out.unsqueeze(1)?); // (b, 1, conv_dim)
        }
        let result = Tensor::cat(&outputs, 1)?; // (b, seq_len, conv_dim)
        let result = candle_nn::ops::silu(&result)?;

        // Cache last kernel tokens as the sliding window for subsequent generation
        // padded has shape (b, conv_dim, seq_len + kernel - 1), last window is at end
        let window_start = padded.dim(2)? - self.conv_kernel;
        self.conv_state = Some(padded.narrow(2, window_start, self.conv_kernel)?);

        result.to_dtype(x.dtype())
    }

    fn recurrent_deltanet(
        &mut self,
        query: &Tensor, // (b, 1, n_heads, head_k_dim)
        key: &Tensor,
        value: &Tensor,
        g: &Tensor,     // (b, 1, n_heads) -- already computed as -exp(A)*softplus(...)
        beta: &Tensor,  // (b, 1, n_heads)
    ) -> Result<Tensor> {
        let b_sz = query.dim(0)?;

        // L2-normalize Q and K
        let q = l2norm(&query.squeeze(1)?)?; // (b, n_heads, head_k_dim)
        let k = l2norm(&key.squeeze(1)?)?;
        let v = value.squeeze(1)?; // (b, n_heads, head_v_dim)
        let g_t = g.squeeze(1)?.to_dtype(DType::F32)?; // (b, n_heads)
        let beta_t = beta.squeeze(1)?.to_dtype(DType::F32)?; // (b, n_heads)

        let q = q.to_dtype(DType::F32)?;
        let k = k.to_dtype(DType::F32)?;
        let v = v.to_dtype(DType::F32)?;

        // Scale query
        let scale = 1.0 / (self.head_k_dim as f64).sqrt();
        let q = (q * scale)?;

        // Initialize or get recurrent state: (b, n_heads, head_k_dim, head_v_dim)
        let mut state = match &self.recurrent_state {
            Some(s) => s.clone(),
            None => Tensor::zeros(
                &[b_sz, self.n_v_heads, self.head_k_dim, self.head_v_dim],
                DType::F32,
                query.device(),
            )?,
        };

        // g_t: decay factor. state *= exp(g_t) per head
        let g_exp = g_t.exp()?.unsqueeze(2)?.unsqueeze(3)?; // (b, n_heads, 1, 1)
        state = state.broadcast_mul(&g_exp)?;

        // delta = (v - S^T @ k) * beta
        // kv_mem = sum over head_k_dim: state[h,k,:] * k[h,k] = (state * k_expanded).sum(k_dim)
        let k_exp = k.unsqueeze(3)?; // (b, n_heads, head_k_dim, 1)
        let kv_mem = state.broadcast_mul(&k_exp)?.sum(2)?; // (b, n_heads, head_v_dim)
        let delta = (&v - &kv_mem)?.broadcast_mul(&beta_t.unsqueeze(2)?)?; // (b, n_heads, head_v_dim)

        // state += k * delta^T (outer product per head)
        let k_outer = k.unsqueeze(3)?;       // (b, n_heads, head_k_dim, 1)
        let delta_outer = delta.unsqueeze(2)?; // (b, n_heads, 1, head_v_dim)
        state = (state + k_outer.broadcast_mul(&delta_outer)?)?;

        // output = S^T @ q = sum over head_k_dim: state[h,k,:] * q[h,k]
        let q_exp = q.unsqueeze(3)?; // (b, n_heads, head_k_dim, 1)
        let out = state.broadcast_mul(&q_exp)?.sum(2)?; // (b, n_heads, head_v_dim)

        self.recurrent_state = Some(state);

        // Reshape: (b, n_heads, head_v_dim) -> (b, 1, n_heads * head_v_dim)
        out.reshape((b_sz, 1, self.n_v_heads * self.head_v_dim))
    }

    fn prefill_deltanet(
        &mut self,
        query: &Tensor, // (b, seq, n_heads, head_k_dim)
        key: &Tensor,
        value: &Tensor,
        g: &Tensor,     // (b, seq, n_heads)
        beta: &Tensor,  // (b, seq, n_heads)
        seq_len: usize,
    ) -> Result<Tensor> {
        let b_sz = query.dim(0)?;

        let q = l2norm(query)?.to_dtype(DType::F32)?;
        let k = l2norm(key)?.to_dtype(DType::F32)?;
        let v = value.to_dtype(DType::F32)?;
        let g = g.to_dtype(DType::F32)?;
        let beta = beta.to_dtype(DType::F32)?;

        let scale = 1.0 / (self.head_k_dim as f64).sqrt();
        let q = (q * scale)?;

        let mut state = Tensor::zeros(
            &[b_sz, self.n_v_heads, self.head_k_dim, self.head_v_dim],
            DType::F32,
            query.device(),
        )?;

        let mut outputs = Vec::with_capacity(seq_len);
        for t in 0..seq_len {
            let q_t = q.narrow(1, t, 1)?.squeeze(1)?;       // (b, n_heads, hk)
            let k_t = k.narrow(1, t, 1)?.squeeze(1)?;       // (b, n_heads, hk)
            let v_t = v.narrow(1, t, 1)?.squeeze(1)?;       // (b, n_heads, hv)
            let g_t = g.narrow(1, t, 1)?.squeeze(1)?;       // (b, n_heads)
            let beta_t = beta.narrow(1, t, 1)?.squeeze(1)?; // (b, n_heads)

            // Decay
            let g_exp = g_t.exp()?.unsqueeze(2)?.unsqueeze(3)?;
            state = state.broadcast_mul(&g_exp)?;

            // Delta update
            let k_exp = k_t.unsqueeze(3)?;
            let kv_mem = state.broadcast_mul(&k_exp)?.sum(2)?;
            let delta = (&v_t - &kv_mem)?.broadcast_mul(&beta_t.unsqueeze(2)?)?;
            let delta_outer = delta.unsqueeze(2)?;
            state = (state + k_exp.broadcast_mul(&delta_outer)?)?;

            // Output
            let q_exp = q_t.unsqueeze(3)?;
            let out_t = state.broadcast_mul(&q_exp)?.sum(2)?; // (b, n_heads, hv)
            outputs.push(out_t.unsqueeze(1)?);
        }

        self.recurrent_state = Some(state);
        let result = Tensor::cat(&outputs, 1)?; // (b, seq, n_heads, hv)
        result.reshape((b_sz, seq_len, self.value_dim))
    }
}

// ─── Full Attention Layer ─────────────────────────────────────────────
#[derive(Debug, Clone)]
struct AttentionLayer {
    input_norm: Qwen35RmsNorm,
    post_norm: Qwen35RmsNorm,
    // Q proj includes gate: (hidden, n_heads * head_dim * 2)
    wq: QMatMul,
    wk: QMatMul,
    wv: QMatMul,
    wo: QMatMul,
    q_norm: Tensor,  // (head_dim,)
    k_norm: Tensor,  // (head_dim,)
    n_head: usize,
    n_kv_head: usize,
    head_dim: usize,
    rope_dim: usize,
    cos: Tensor,
    sin: Tensor,
    neg_inf: Tensor,
    kv_cache: Option<(Tensor, Tensor)>,
    rms_eps: f64,
    mlp: Mlp,
}

impl AttentionLayer {
    fn apply_rotary_emb(&self, x: &Tensor, index_pos: usize) -> Result<Tensor> {
        let (_b_sz, _n_head, seq_len, _head_dim) = x.dims4()?;
        let cos = self.cos.narrow(0, index_pos, seq_len)?; // (seq, rope_dim/2)
        let sin = self.sin.narrow(0, index_pos, seq_len)?;
        let rd2 = cos.dim(1)?; // rope_dim / 2

        // Only apply RoPE to first rope_dim dimensions
        let x_rot = if self.rope_dim < self.head_dim {
            x.narrow(3, 0, self.rope_dim)?
        } else {
            x.clone()
        };
        let x_pass = if self.rope_dim < self.head_dim {
            Some(x.narrow(3, self.rope_dim, self.head_dim - self.rope_dim)?)
        } else {
            None
        };

        // Manual half-rotation RoPE: split into two halves, apply rotation
        // x_rot: (b, n_heads, seq, rope_dim)
        let x1 = x_rot.narrow(3, 0, rd2)?;    // first half
        let x2 = x_rot.narrow(3, rd2, rd2)?;   // second half

        // cos/sin: (seq, rd2) -> (1, 1, seq, rd2) for broadcasting
        let cos = cos.unsqueeze(0)?.unsqueeze(0)?;
        let sin = sin.unsqueeze(0)?.unsqueeze(0)?;

        // rotated = [x1*cos - x2*sin, x1*sin + x2*cos]
        let r1 = x1.broadcast_mul(&cos)?.broadcast_sub(&x2.broadcast_mul(&sin)?)?;
        let r2 = x1.broadcast_mul(&sin)?.broadcast_add(&x2.broadcast_mul(&cos)?)?;
        let rotated = Tensor::cat(&[&r1, &r2], 3)?;

        match x_pass {
            Some(pass) => Tensor::cat(&[&rotated, &pass], 3),
            None => Ok(rotated),
        }
    }

    fn rms_norm_head(&self, x: &Tensor, weight: &Tensor) -> Result<Tensor> {
        // Per-head RMSNorm with (1+weight) convention
        // x: (b, seq, n_heads, head_dim)
        let x_f32 = x.to_dtype(DType::F32)?;
        let variance = x_f32.sqr()?.mean_keepdim(candle_core::D::Minus1)?;
        let rsqrt = (variance + self.rms_eps)?.recip()?.sqrt()?;
        let normed = x_f32.broadcast_mul(&rsqrt)?;
        let scale = (weight + 1.0f64)?;
        normed.broadcast_mul(&scale)?.to_dtype(x.dtype())
    }

    fn forward(&mut self, xs: &Tensor, index_pos: usize) -> Result<Tensor> {
        let (b_sz, seq_len, _) = xs.dims3()?;
        let residual = xs;

        let h = self.input_norm.forward(xs)?;

        // Q projection (includes gate): split into query + gate
        let qg = self.wq.forward(&h)?;
        let qg = qg.reshape((b_sz, seq_len, self.n_head, self.head_dim * 2))?;

        let query = qg.narrow(3, 0, self.head_dim)?;
        let gate = qg.narrow(3, self.head_dim, self.head_dim)?;
        let gate = gate.reshape((b_sz, seq_len, self.n_head * self.head_dim))?;


        // K, V
        let k = self.wk.forward(&h)?;

        let v = self.wv.forward(&h)?;

        let k = k.reshape((b_sz, seq_len, self.n_kv_head, self.head_dim))?;
        let v = v.reshape((b_sz, seq_len, self.n_kv_head, self.head_dim))?;


        // Per-head Q/K norms
        let query = self.rms_norm_head(&query, &self.q_norm)?;

        let k = self.rms_norm_head(&k, &self.k_norm)?;



        // Transpose to (b, n_heads, seq, head_dim)
        let query = query.transpose(1, 2)?;

        let k = k.transpose(1, 2)?;

        let v = v.transpose(1, 2)?.contiguous()?;


        // RoPE
        let query = self.apply_rotary_emb(&query, index_pos)?;


        let k = self.apply_rotary_emb(&k, index_pos)?;


        // KV cache
        let (k, v) = match &self.kv_cache {
            None => (k, v),
            Some((k_cache, v_cache)) => {
                if index_pos == 0 {
                    (k, v)
                } else {
                    let k = Tensor::cat(&[k_cache, &k], 2)?;
                    let v = Tensor::cat(&[v_cache, &v], 2)?;
                    (k, v)
                }
            }
        };
        self.kv_cache = Some((k.clone(), v.clone()));


        // Attention: repeat KV heads for GQA (expand along head dim)
        let n_rep = self.n_head / self.n_kv_head;
        let (k, v) = if n_rep > 1 {
            let (b, nkv, s, hd) = k.dims4()?;
            let k = k.unsqueeze(2)?.expand((b, nkv, n_rep, s, hd))?.reshape((b, nkv * n_rep, s, hd))?;
            let (b, nkv, s, hd) = v.dims4()?;
            let v = v.unsqueeze(2)?.expand((b, nkv, n_rep, s, hd))?.reshape((b, nkv * n_rep, s, hd))?;
            (k, v)
        } else { (k, v) };


        // Force materialization before attention matmul (breaks lazy eval chain)
        let query = query.contiguous()?;
        let k = k.contiguous()?;
        let v = v.contiguous()?;

        let y = if query.device().is_metal() && seq_len == 1 {
            candle_nn::ops::sdpa(&query, &k, &v, 1. / (self.head_dim as f32).sqrt(), 1.)?
        } else {
            let att = (query.matmul(&k.t()?)? / (self.head_dim as f64).sqrt())?;
            let att = if seq_len > 1 {
                let mask: Vec<_> = (0..seq_len)
                    .flat_map(|i| (0..seq_len).map(move |j| u8::from(j > i)))
                    .collect();
                let mask = Tensor::from_slice(&mask, (seq_len, seq_len), xs.device())?;
                let mask = mask.broadcast_as(att.shape())?;
                let on_true = self.neg_inf.broadcast_as(att.shape())?;
                mask.where_cond(&on_true, &att)?
            } else {
                att
            };
            let att = candle_nn::ops::softmax_last_dim(&att)?;
            att.matmul(&v.contiguous()?)?
        };

        // (b, n_heads, seq, head_dim) -> (b, seq, n_heads * head_dim)
        let y = y.transpose(1, 2)?.reshape(&[b_sz, seq_len, self.n_head * self.head_dim])?;

        // Apply gate: output * sigmoid(gate)
        let y = (y * candle_nn::ops::sigmoid(&gate)?)?;

        // Output projection
        let y = self.wo.forward(&y)?;

        // Residual
        let hidden = (residual + &y)?;

        // MLP
        let residual2 = &hidden;
        let h2 = self.post_norm.forward(&hidden)?;
        let mlp_out = self.mlp.forward(&h2)?;
        let output = (residual2 + &mlp_out)?;

        Ok(output)
    }
}

// ─── Layer enum ───────────────────────────────────────────────────────
#[derive(Debug, Clone)]
enum Qwen35Layer {
    DeltaNet(DeltaNetLayer),
    Attention(AttentionLayer),
}

// ─── Model ────────────────────────────────────────────────────────────
#[derive(Debug, Clone)]
pub struct Qwen35Model {
    tok_embeddings: Embedding,
    layers: Vec<Qwen35Layer>,
    norm: Qwen35RmsNorm,
    output: QMatMul,
}

impl Qwen35Model {
    pub fn from_gguf<R: std::io::Seek + std::io::Read>(
        ct: gguf_file::Content,
        reader: &mut R,
        device: &Device,
    ) -> Result<Self> {
        let md_get = |s: &str| match ct.metadata.get(s) {
            None => candle_core::bail!("cannot find {s} in metadata"),
            Some(v) => Ok(v),
        };

        // Architecture metadata (qwen35.* prefix)
        let n_head = md_get("qwen35.attention.head_count")?.to_u32()? as usize;
        let n_kv_head = md_get("qwen35.attention.head_count_kv")?.to_u32()? as usize;
        let block_count = md_get("qwen35.block_count")?.to_u32()? as usize;
        let embedding_length = md_get("qwen35.embedding_length")?.to_u32()? as usize;
        let head_dim = md_get("qwen35.attention.key_length")?.to_u32()? as usize;
        let rope_dim = md_get("qwen35.rope.dimension_count")?.to_u32()? as usize;
        let rms_eps = md_get("qwen35.attention.layer_norm_rms_epsilon")?.to_f32()? as f64;
        let rope_freq_base = md_get("qwen35.rope.freq_base")
            .and_then(|m| m.to_f32())
            .unwrap_or(10000000f32);
        let full_attn_interval = md_get("qwen35.full_attention_interval")?.to_u32()? as usize;

        // SSM params
        let ssm_inner = md_get("qwen35.ssm.inner_size")?.to_u32()? as usize;
        let ssm_state = md_get("qwen35.ssm.state_size")?.to_u32()? as usize;
        let ssm_conv_kernel = md_get("qwen35.ssm.conv_kernel")?.to_u32()? as usize;
        let ssm_n_heads = md_get("qwen35.ssm.time_step_rank")?.to_u32()? as usize;
        let ssm_n_groups = md_get("qwen35.ssm.group_count")?.to_u32()? as usize;

        // DeltaNet dimensions
        let n_v_heads = ssm_n_heads; // 32
        let n_k_heads = ssm_n_groups; // 16
        let head_v_dim = ssm_inner / n_v_heads; // 128
        let head_k_dim = ssm_state; // 128 (confirmed by ssm_norm weight shape)
        let key_dim = head_k_dim * n_k_heads; // 2048
        let value_dim = head_v_dim * n_v_heads; // 4096 = ssm_inner

        println!("    Qwen3.5 architecture:");
        println!("      blocks={}, hidden={}, head_dim={}", block_count, embedding_length, head_dim);
        println!("      attn: heads={}, kv_heads={}, rope_dim={}", n_head, n_kv_head, rope_dim);
        println!("      deltanet: v_heads={}, k_heads={}, hk={}, hv={}", n_v_heads, n_k_heads, head_k_dim, head_v_dim);
        println!("      full_attn_interval={}", full_attn_interval);

        // Precompute RoPE
        let (cos, sin) = precompute_freqs_cis(rope_dim, rope_freq_base, device)?;
        let neg_inf = Tensor::new(f32::NEG_INFINITY, device)?;

        // Token embeddings
        let tok_embeddings_q = ct.tensor(reader, "token_embd.weight", device)?;
        let tok_embeddings = tok_embeddings_q.dequantize(device)?;

        // Output norm
        let norm = Qwen35RmsNorm::from_qtensor(
            ct.tensor(reader, "output_norm.weight", device)?,
            rms_eps,
            device,
        )?;

        // Output projection (may be tied to token_embd)
        let output = match ct.tensor(reader, "output.weight", device) {
            Ok(tensor) => tensor,
            Err(_) => tok_embeddings_q,
        };

        // Build layers
        let mut layers = Vec::with_capacity(block_count);
        for i in 0..block_count {
            let prefix = format!("blk.{i}");
            let is_full_attn = (i + 1) % full_attn_interval == 0;

            // Common: input norm, post norm, MLP
            let input_norm = Qwen35RmsNorm::from_qtensor(
                ct.tensor(reader, &format!("{prefix}.attn_norm.weight"), device)?,
                rms_eps,
                device,
            )?;
            let post_norm = Qwen35RmsNorm::from_qtensor(
                ct.tensor(reader, &format!("{prefix}.post_attention_norm.weight"), device)?,
                rms_eps,
                device,
            )?;
            let mlp = Mlp {
                gate: QMatMul::from_qtensor(ct.tensor(reader, &format!("{prefix}.ffn_gate.weight"), device)?)?,
                up: QMatMul::from_qtensor(ct.tensor(reader, &format!("{prefix}.ffn_up.weight"), device)?)?,
                down: QMatMul::from_qtensor(ct.tensor(reader, &format!("{prefix}.ffn_down.weight"), device)?)?,
            };

            if is_full_attn {
                // Full attention layer -- dequantize weights to bypass Q5K matmul issues
                let wq = QMatMul::from_qtensor_dequantized(ct.tensor(reader, &format!("{prefix}.attn_q.weight"), device)?, device)?;
                let wk = QMatMul::from_qtensor_dequantized(ct.tensor(reader, &format!("{prefix}.attn_k.weight"), device)?, device)?;
                let wv = QMatMul::from_qtensor_dequantized(ct.tensor(reader, &format!("{prefix}.attn_v.weight"), device)?, device)?;
                let wo = QMatMul::from_qtensor_dequantized(ct.tensor(reader, &format!("{prefix}.attn_output.weight"), device)?, device)?;
                let q_norm = ct.tensor(reader, &format!("{prefix}.attn_q_norm.weight"), device)?.dequantize(device)?;
                let k_norm = ct.tensor(reader, &format!("{prefix}.attn_k_norm.weight"), device)?.dequantize(device)?;

                layers.push(Qwen35Layer::Attention(AttentionLayer {
                    input_norm,
                    post_norm,
                    wq, wk, wv, wo,
                    q_norm, k_norm,
                    n_head,
                    n_kv_head,
                    head_dim,
                    rope_dim,
                    cos: cos.clone(),
                    sin: sin.clone(),
                    neg_inf: neg_inf.clone(),
                    kv_cache: None,
                    rms_eps,
                    mlp,
                }));
            } else {
                // DeltaNet layer
                let in_proj_qkv = QMatMul::from_qtensor(ct.tensor(reader, &format!("{prefix}.attn_qkv.weight"), device)?)?;
                let in_proj_z = QMatMul::from_qtensor(ct.tensor(reader, &format!("{prefix}.attn_gate.weight"), device)?)?;
                let in_proj_b = ct.tensor(reader, &format!("{prefix}.ssm_beta.weight"), device)?.dequantize(device)?;
                let in_proj_a = ct.tensor(reader, &format!("{prefix}.ssm_alpha.weight"), device)?.dequantize(device)?;
                // conv1d_weight: GGUF stores as (kernel=4, conv_dim=8192).
                // We need (conv_dim, kernel) for the broadcast mul in causal_conv1d_update.
                // Use to_vec + Tensor::from_vec to force a fresh (conv_dim, kernel) allocation
                // because candle's .t().contiguous() on Metal doesn't always materialize the transpose.
                let conv1d_raw = ct.tensor(reader, &format!("{prefix}.ssm_conv1d.weight"), device)?
                    .dequantize(device)?; // (kernel=4, conv_dim=8192)
                let (raw_k, raw_c) = (conv1d_raw.dim(0)?, conv1d_raw.dim(1)?); // 4, 8192
                let conv1d_weight = {
                    // Read as flat vec then repack in (conv_dim, kernel) order
                    let flat: Vec<f32> = conv1d_raw.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;
                    // flat is row-major (kernel, conv_dim): flat[k * conv_dim + c]
                    // want (conv_dim, kernel): new[c * kernel + k] = flat[k * conv_dim + c]
                    let mut transposed = vec![0f32; raw_k * raw_c];
                    for k in 0..raw_k {
                        for c in 0..raw_c {
                            transposed[c * raw_k + k] = flat[k * raw_c + c];
                        }
                    }
                    Tensor::from_vec(transposed, (raw_c, raw_k), conv1d_raw.device())?
                }; // shape: (conv_dim=8192, kernel=4)
                let a_log = ct.tensor(reader, &format!("{prefix}.ssm_a"), device)?.dequantize(device)?;
                let dt_bias = ct.tensor(reader, &format!("{prefix}.ssm_dt.bias"), device)?.dequantize(device)?;
                let ssm_norm_w = ct.tensor(reader, &format!("{prefix}.ssm_norm.weight"), device)?.dequantize(device)?;
                let out_proj = QMatMul::from_qtensor(ct.tensor(reader, &format!("{prefix}.ssm_out.weight"), device)?)?;

                layers.push(Qwen35Layer::DeltaNet(DeltaNetLayer {
                    input_norm,
                    post_norm,
                    in_proj_qkv,
                    in_proj_z,
                    in_proj_b,
                    in_proj_a,
                    conv1d_weight,
                    a_log,
                    dt_bias,
                    ssm_norm: ssm_norm_w,
                    out_proj,
                    n_v_heads,
                    n_k_heads,
                    head_k_dim,
                    head_v_dim,
                    key_dim,
                    value_dim,
                    conv_kernel: ssm_conv_kernel,
                    rms_eps,
                    mlp,
                    conv_state: None,
                    recurrent_state: None,
                    layer_idx: i,
                }));
            }
        }

        let embed_dim = embedding_length;
        Ok(Self {
            tok_embeddings: Embedding::new(tok_embeddings, embed_dim),
            layers,
            norm,
            output: QMatMul::from_qtensor(output)?,
        })
    }

    fn run_layers(&mut self, x: &Tensor, index_pos: usize) -> Result<Tensor> {
        let (_b_sz, seq_len) = x.dims2()?;
        let mut layer_in = self.tok_embeddings.forward(x)?;

        let do_debug = seq_len > 1 && index_pos == 0; // prefill only
        if do_debug {
            let v: Vec<f32> = layer_in.flatten_all()?.to_dtype(DType::F32)?.to_vec1()?;
            let rms = (v.iter().map(|x| x*x).sum::<f32>() / v.len() as f32).sqrt();
            eprintln!("[RMS] embed: {:.4}", rms);
        }

        for (i, layer) in self.layers.iter_mut().enumerate() {
            let kind = match layer { Qwen35Layer::DeltaNet(_) => "DN", Qwen35Layer::Attention(_) => "ATT" };
            layer_in = match layer {
                Qwen35Layer::DeltaNet(l) => l.forward(&layer_in, index_pos)?,
                Qwen35Layer::Attention(l) => l.forward(&layer_in, index_pos)?,
            };
            if do_debug && i < 6 {
                let v: Vec<f32> = layer_in.flatten_all()?.to_dtype(DType::F32)?.to_vec1().unwrap_or_default();
                let rms = (v.iter().map(|x| x*x).sum::<f32>() / v.len() as f32).sqrt();
                eprintln!("[RMS] blk.{} ({}): {:.4}", i, kind, rms);
            }
        }

        let x = self.norm.forward(&layer_in)?;
        // Take last position, squeeze seq dim
        x.narrow(1, seq_len - 1, 1)?.squeeze(1)
    }

    /// Standard forward: returns logits (vocab-sized).
    pub fn forward(&mut self, x: &Tensor, index_pos: usize) -> Result<Tensor> {
        let hidden = self.run_layers(x, index_pos)?;
        self.output.forward(&hidden)
    }

    /// Return both logits AND hidden state in one pass.
    pub fn forward_with_hidden(
        &mut self,
        x: &Tensor,
        index_pos: usize,
    ) -> Result<(Tensor, Tensor)> {
        let hidden = self.run_layers(x, index_pos)?;
        let logits = self.output.forward(&hidden)?;
        Ok((logits, hidden))
    }

    /// Project a hidden state through the lm_head.
    pub fn project_to_logits(&self, hidden: &Tensor) -> Result<Tensor> {
        self.output.forward(hidden)
    }

    /// Access the raw token embedding matrix.
    pub fn token_embeddings(&self) -> &Tensor {
        self.tok_embeddings.embeddings()
    }
}

fn precompute_freqs_cis(
    head_dim: usize,
    freq_base: f32,
    device: &Device,
) -> Result<(Tensor, Tensor)> {
    let theta: Vec<_> = (0..head_dim)
        .step_by(2)
        .map(|i| 1f32 / freq_base.powf(i as f32 / head_dim as f32))
        .collect();
    let theta = Tensor::new(theta.as_slice(), device)?;
    let idx_theta = Tensor::arange(0, MAX_SEQ_LEN as u32, device)?
        .to_dtype(DType::F32)?
        .reshape((MAX_SEQ_LEN, 1))?
        .matmul(&theta.reshape((1, theta.elem_count()))?)?;
    let cos = idx_theta.cos()?;
    let sin = idx_theta.sin()?;
    Ok((cos, sin))
}
