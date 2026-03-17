//! Vendored quantized Gemma 3 model — GGUF native reader
//!
//! Ported from the hydrodynamic-swarm llama.rs pattern and extended with
//! Gemma 3 architecture specifics discovered from GGUF inspection:
//!
//!   - Per-head QK RMS norms (attn_q_norm, attn_k_norm)
//!   - Post-attention and post-FFW scaling norms (Gemma 3 pre-residual)
//!   - GeGLU activation in the MLP (gate × up, then down)
//!   - Grouped query attention (n_head != n_kv_head)
//!   - Token embeddings shared with output projection (tied weights)
//!
//! Tensor layout (from `cargo run --bin inspect_gguf`):
//!   token_embd.weight          [vocab_size, hidden_dim]   Q8_0
//!   output_norm.weight         [hidden_dim]               F32
//!   blk.{i}.attn_norm.weight   [hidden_dim]               F32
//!   blk.{i}.attn_q.weight      [hidden_dim, hidden_dim]   Q8_0
//!   blk.{i}.attn_q_norm.weight [head_dim]                 F32
//!   blk.{i}.attn_k.weight      [kv_dim, hidden_dim]       Q8_0
//!   blk.{i}.attn_k_norm.weight [head_dim]                 F32
//!   blk.{i}.attn_v.weight      [kv_dim, hidden_dim]       Q8_0
//!   blk.{i}.attn_output.weight [hidden_dim, hidden_dim]   Q8_0
//!   blk.{i}.post_attention_norm.weight [hidden_dim]       F32
//!   blk.{i}.ffn_norm.weight    [hidden_dim]               F32
//!   blk.{i}.ffn_gate.weight    [ffn_dim, hidden_dim]      Q8_0
//!   blk.{i}.ffn_up.weight      [ffn_dim, hidden_dim]      Q8_0
//!   blk.{i}.ffn_down.weight    [hidden_dim, ffn_dim]      Q8_0
//!   blk.{i}.post_ffw_norm.weight [hidden_dim]             F32

use candle_core::quantized::{gguf_file, QTensor};
use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::Embedding;
use candle_transformers::quantized_nn::RmsNorm;
use std::collections::HashMap;

pub const MAX_SEQ_LEN: usize = 8192;

// ── QMatMul ────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct QMatMul {
    inner: candle_core::quantized::QMatMul,
}

impl QMatMul {
    fn from_qtensor(qt: QTensor) -> Result<Self> {
        Ok(Self {
            inner: candle_core::quantized::QMatMul::from_qtensor(qt)?,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.inner.forward(xs)
    }
}

// ── RoPE precomputation ───────────────────────────────────────────────────────

fn precompute_freqs_cis(head_dim: usize, freq_base: f32, device: &Device) -> Result<(Tensor, Tensor)> {
    let theta: Vec<_> = (0..head_dim)
        .step_by(2)
        .map(|i| 1f32 / freq_base.powf(i as f32 / head_dim as f32))
        .collect();
    let theta = Tensor::new(theta.as_slice(), device)?;
    let idx = Tensor::arange(0, MAX_SEQ_LEN as u32, device)?
        .to_dtype(DType::F32)?
        .reshape((MAX_SEQ_LEN, 1))?
        .matmul(&theta.reshape((1, theta.elem_count()))?)?;
    Ok((idx.cos()?, idx.sin()?))
}

// ── GQA repeat_kv ─────────────────────────────────────────────────────────────

fn repeat_kv(x: Tensor, n_rep: usize) -> Result<Tensor> {
    if n_rep == 1 {
        return Ok(x);
    }
    let (b, n_kv, seq, d) = x.dims4()?;
    // [b, n_kv, seq, d] -> [b, n_kv*n_rep, seq, d]
    x.unsqueeze(2)?
        .expand((b, n_kv, n_rep, seq, d))?
        .reshape((b, n_kv * n_rep, seq, d))
}

// ── Causal mask ───────────────────────────────────────────────────────────────

fn make_causal_mask(seq_len: usize, device: &Device) -> Result<Tensor> {
    let mask: Vec<u8> = (0..seq_len)
        .flat_map(|i| (0..seq_len).map(move |j| u8::from(j > i)))
        .collect();
    Tensor::from_slice(&mask, (seq_len, seq_len), device)
}

// ── Layer ─────────────────────────────────────────────────────────────────────

struct Layer {
    // Pre-attention norm
    attn_norm: RmsNorm,
    // Attention projections
    attn_q: QMatMul,
    attn_q_norm: RmsNorm, // Gemma 3: per-head QK norm
    attn_k: QMatMul,
    attn_k_norm: RmsNorm, // Gemma 3: per-head QK norm
    attn_v: QMatMul,
    attn_out: QMatMul,
    // Post-attention scaling (Gemma 3 pre-residual norm)
    post_attn_norm: RmsNorm,
    // FFN
    ffn_norm: RmsNorm,
    ffn_gate: QMatMul, // GeGLU: gate branch
    ffn_up: QMatMul,   // GeGLU: up branch
    ffn_down: QMatMul,
    post_ffn_norm: RmsNorm, // Gemma 3 pre-residual norm
    // Config
    n_head: usize,
    n_kv_head: usize,
    head_dim: usize,
    cos: Tensor,
    sin: Tensor,
    // KV cache
    kv_cache: Option<(Tensor, Tensor)>,
}

impl Layer {
    fn apply_rope(&self, x: &Tensor, index_pos: usize) -> Result<Tensor> {
        let (_b, _h, seq, _d) = x.dims4()?;
        let cos = self.cos.narrow(0, index_pos, seq)?;
        let sin = self.sin.narrow(0, index_pos, seq)?;
        candle_nn::rotary_emb::rope_i(&x.contiguous()?, &cos, &sin)
    }

    fn forward(&mut self, xs: &Tensor, mask: Option<&Tensor>, index_pos: usize) -> Result<Tensor> {
        let (b, seq, _) = xs.dims3()?;

        // ── Pre-attention norm ───────────────────────────────────────────────
        let residual = xs;
        let h = self.attn_norm.forward(xs)?;

        // ── Attention projections ────────────────────────────────────────────
        let q = self.attn_q.forward(&h)?; // [b, seq, n_head*head_dim]
        let k = self.attn_k.forward(&h)?; // [b, seq, n_kv*head_dim]
        let v = self.attn_v.forward(&h)?; // [b, seq, n_kv*head_dim]

        // Reshape to multi-head layout
        let q = q
            .reshape((b, seq, self.n_head, self.head_dim))?
            .transpose(1, 2)?; // [b, n_head, seq, head_dim]
        let k = k
            .reshape((b, seq, self.n_kv_head, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((b, seq, self.n_kv_head, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;

        // Gemma 3: per-head QK RMS norms (applied before RoPE)
        // RmsNorm expects [..., d] — transpose back, norm, transpose again
        let q = self
            .attn_q_norm
            .forward(&q.transpose(1, 2)?.contiguous()?)?
            .transpose(1, 2)?;
        let k = self
            .attn_k_norm
            .forward(&k.transpose(1, 2)?.contiguous()?)?
            .transpose(1, 2)?;

        // RoPE
        let q = self.apply_rope(&q, index_pos)?;
        let k = self.apply_rope(&k, index_pos)?;

        // KV cache
        let (k, v) = match &self.kv_cache {
            None => (k, v),
            Some((kc, vc)) => {
                if index_pos == 0 {
                    (k, v)
                } else {
                    let k = Tensor::cat(&[kc, &k], 2)?;
                    let v = Tensor::cat(&[vc, &v], 2)?;
                    (k, v)
                }
            }
        };
        self.kv_cache = Some((k.clone(), v.clone()));

        // GQA expand
        let k = repeat_kv(k, self.n_head / self.n_kv_head)?;
        let v = repeat_kv(v, self.n_head / self.n_kv_head)?;

        // Scaled dot-product
        let scale = (self.head_dim as f64).sqrt();
        let att = (q.matmul(&k.t()?)? / scale)?;
        let att = match mask {
            None => att,
            Some(m) => {
                let neg_inf = Tensor::new(f32::NEG_INFINITY, att.device())?;
                let m = m.broadcast_as(att.shape())?;
                m.where_cond(&neg_inf.broadcast_as(att.shape())?, &att)?
            }
        };
        let att = candle_nn::ops::softmax_last_dim(&att)?;
        let y = att.matmul(&v.contiguous()?)?; // [b, n_head, seq, head_dim]

        let y = y
            .transpose(1, 2)?
            .contiguous()?
            .reshape((b, seq, self.n_head * self.head_dim))?;
        let y = self.attn_out.forward(&y)?;

        // Post-attention pre-residual norm (Gemma 3)
        let y = self.post_attn_norm.forward(&y)?;
        let h = (y + residual)?;

        // ── FFN (GeGLU) ──────────────────────────────────────────────────────
        let residual_ffn = &h;
        let h_normed = self.ffn_norm.forward(&h)?;
        let gate = candle_nn::ops::silu(&self.ffn_gate.forward(&h_normed)?)?;
        let up = self.ffn_up.forward(&h_normed)?;
        let ffn_out = self.ffn_down.forward(&(gate * up)?)?;
        // Post-FFN pre-residual norm (Gemma 3)
        let ffn_out = self.post_ffn_norm.forward(&ffn_out)?;
        let h = (ffn_out + residual_ffn)?;

        Ok(h)
    }

    fn clear_kv_cache(&mut self) {
        self.kv_cache = None;
    }
}

// ── ModelWeights ──────────────────────────────────────────────────────────────

pub struct ModelWeights {
    tok_embeddings: Embedding,
    layers: Vec<Layer>,
    norm: RmsNorm,
    // Output projection — for Gemma 300M embedding model, output.weight
    // is NOT in the GGUF (tied to tok_embeddings). None = tied.
    output: Option<QMatMul>,
    masks: HashMap<usize, Tensor>,
    device: Device,
    /// Hidden dimension (for embedding size reporting)
    pub hidden_dim: usize,
}

impl ModelWeights {
    /// Load a quantized Gemma 3 model directly from a GGUF file.
    pub fn from_gguf<R: std::io::Seek + std::io::Read>(
        ct: gguf_file::Content,
        reader: &mut R,
        device: &Device,
    ) -> Result<Self> {
        // ── Read architecture metadata ─────────────────────────────────────


        // Try both "gemma3.*" and "llama.*" namespaces — Unsloth uses llama
        let get_u32 = |k1: &str, k2: &str| -> Result<u32> {
            ct.metadata
                .get(k1)
                .or_else(|| ct.metadata.get(k2))
                .ok_or_else(|| candle_core::Error::Msg(format!("missing key {k1} or {k2}")))?
                .to_u32()
        };

        let get_f32 = |k1: &str, k2: &str, default: f32| -> f32 {
            ct.metadata
                .get(k1)
                .or_else(|| ct.metadata.get(k2))
                .and_then(|v| v.to_f32().ok())
                .unwrap_or(default)
        };

        let n_head = get_u32("gemma3.attention.head_count", "llama.attention.head_count")? as usize;
        let n_kv_head = get_u32("gemma3.attention.head_count_kv", "llama.attention.head_count_kv")? as usize;
        let block_count = get_u32("gemma3.block_count", "llama.block_count")? as usize;
        let hidden_dim = get_u32("gemma3.embedding_length", "llama.embedding_length")? as usize;
        let rms_eps = get_f32(
            "gemma3.attention.layer_norm_rms_epsilon",
            "llama.attention.layer_norm_rms_epsilon",
            1e-6,
        ) as f64;
        let rope_base = get_f32("gemma3.rope.freq_base", "llama.rope.freq_base", 10000.0);

        // head_dim from the GGUF Q weight shape: [n_head*head_dim, hidden_dim]
        // We can also compute: hidden_dim / n_head (for query heads)
        // But Q tensor shape [768, 768] means n_head*head_dim = hidden_dim
        // head_dim = hidden_dim / n_head = 768 / 12 = 64
        // For KV: [256, 768] means n_kv_head * head_dim = 256 => head_dim = 256/4 = 64 ✓
        let head_dim = hidden_dim / n_head;

        let (cos, sin) = precompute_freqs_cis(head_dim, rope_base, device)?;

        // ── Token embeddings ──────────────────────────────────────────────────
        let tok_embd_q = ct.tensor(reader, "token_embd.weight", device)?;
        let tok_embd = tok_embd_q.dequantize(device)?;

        // ── Output norm ───────────────────────────────────────────────────────
        let norm = RmsNorm::from_qtensor(ct.tensor(reader, "output_norm.weight", device)?, rms_eps)?;

        // ── Output projection (may be tied) ───────────────────────────────────
        let (_output_tied, output) = match ct.tensor(reader, "output.weight", device) {
            Ok(t) => (false, Some(QMatMul::from_qtensor(t)?)),
            Err(_) => (true, None), // Embedding-tied (300M model)
        };

        // ── Transformer layers ────────────────────────────────────────────────
        let mut layers = Vec::with_capacity(block_count);
        for i in 0..block_count {
            let p = format!("blk.{i}");
            layers.push(Layer {
                attn_norm: RmsNorm::from_qtensor(
                    ct.tensor(reader, &format!("{p}.attn_norm.weight"), device)?,
                    rms_eps,
                )?,
                attn_q: QMatMul::from_qtensor(ct.tensor(reader, &format!("{p}.attn_q.weight"), device)?)?,
                attn_q_norm: RmsNorm::from_qtensor(
                    ct.tensor(reader, &format!("{p}.attn_q_norm.weight"), device)?,
                    rms_eps,
                )?,
                attn_k: QMatMul::from_qtensor(ct.tensor(reader, &format!("{p}.attn_k.weight"), device)?)?,
                attn_k_norm: RmsNorm::from_qtensor(
                    ct.tensor(reader, &format!("{p}.attn_k_norm.weight"), device)?,
                    rms_eps,
                )?,
                attn_v: QMatMul::from_qtensor(ct.tensor(reader, &format!("{p}.attn_v.weight"), device)?)?,
                attn_out: QMatMul::from_qtensor(
                    ct.tensor(reader, &format!("{p}.attn_output.weight"), device)?,
                )?,
                post_attn_norm: RmsNorm::from_qtensor(
                    ct.tensor(reader, &format!("{p}.post_attention_norm.weight"), device)?,
                    rms_eps,
                )?,
                ffn_norm: RmsNorm::from_qtensor(
                    ct.tensor(reader, &format!("{p}.ffn_norm.weight"), device)?,
                    rms_eps,
                )?,
                ffn_gate: QMatMul::from_qtensor(ct.tensor(reader, &format!("{p}.ffn_gate.weight"), device)?)?,
                ffn_up: QMatMul::from_qtensor(ct.tensor(reader, &format!("{p}.ffn_up.weight"), device)?)?,
                ffn_down: QMatMul::from_qtensor(
                    ct.tensor(reader, &format!("{p}.ffn_down.weight"), device)?,
                )?,
                post_ffn_norm: RmsNorm::from_qtensor(
                    ct.tensor(reader, &format!("{p}.post_ffw_norm.weight"), device)?,
                    rms_eps,
                )?,
                n_head,
                n_kv_head,
                head_dim,
                cos: cos.clone(),
                sin: sin.clone(),
                kv_cache: None,
            });
        }

        Ok(Self {
            tok_embeddings: Embedding::new(tok_embd, hidden_dim),
            layers,
            norm,
            output,
            masks: HashMap::new(),
            device: device.clone(),
            hidden_dim,
        })
    }

    // ── Internal ─────────────────────────────────────────────────────────────

    fn cached_mask(&mut self, seq_len: usize) -> Result<Tensor> {
        if let Some(m) = self.masks.get(&seq_len) {
            return Ok(m.clone());
        }
        let m = make_causal_mask(seq_len, &self.device)?;
        self.masks.insert(seq_len, m.clone());
        Ok(m)
    }

    /// Run all transformer layers; returns post-norm hidden state.
    fn run_layers(&mut self, tokens: &Tensor, index_pos: usize) -> Result<Tensor> {
        let (_b, seq) = tokens.dims2()?;
        let mask = if seq == 1 {
            None
        } else {
            Some(self.cached_mask(seq)?)
        };

        let mut h = self.tok_embeddings.forward(tokens)?;
        // Gemma embedding scaling: multiply by sqrt(hidden_dim)
        let scale = (self.hidden_dim as f64).sqrt();
        h = (h * scale)?;

        for layer in &mut self.layers {
            h = layer.forward(&h, mask.as_ref(), index_pos)?;
        }
        // Final output norm
        let h = self.norm.forward(&h)?;
        // Return last position: [b, hidden_dim]
        h.narrow(1, seq - 1, 1)?.squeeze(1)
    }

    // ── Public API ────────────────────────────────────────────────────────────

    /// Standard forward pass: returns logits [b, vocab_size].
    pub fn forward(&mut self, tokens: &Tensor, index_pos: usize) -> Result<Tensor> {
        let hidden = self.run_layers(tokens, index_pos)?;
        match &self.output {
            Some(out) => out.forward(&hidden),
            None => {
                // Tied weights: project through tok_embeddings matrix
                // hidden: [b, hidden_dim], emb: [vocab, hidden_dim]
                // logits = hidden @ emb.T
                let emb = self.tok_embeddings.embeddings();
                hidden.matmul(&emb.t()?)
            }
        }
    }

    /// Returns pre-lm_head hidden state [b, hidden_dim] — for embedding.
    pub fn forward_hidden(&mut self, tokens: &Tensor, index_pos: usize) -> Result<Tensor> {
        self.run_layers(tokens, index_pos)
    }

    /// Clear all KV caches (call before each new sequence).
    pub fn clear_kv_cache(&mut self) {
        for layer in &mut self.layers {
            layer.clear_kv_cache();
        }
    }
}
