//! Gemma model integration for EmbedSwarm
//!
//! Wraps the vendored quantized Gemma 3 GGUF reader (`quantized_gemma.rs`)
//! in a clean API for embedding generation and text completion.
//!
//! Supports all four model sizes:
//!   - 0.3B  EmbeddingGemma 300M    (Embed Nodes, Q8_0)
//!   - 1.0B  Gemma 3 1B Instruct    (Function Nodes, Q4_K_M)
//!   - 4.0B  Gemma 3 4B Instruct    (Function Nodes, Q4_K_M)
//!   - 12.0B Gemma 3 12B Instruct   (Governor, Q4_K_M)

use std::sync::Arc;
use crate::concourse::types::{Node, NodeClass};
use crate::concourse::{SwarmError, SwarmResult};

#[cfg(feature = "with-candle")]
use candle_core::{DType, Device, Tensor};
#[cfg(feature = "with-candle")]
use candle_core::quantized::gguf_file;
#[cfg(feature = "with-candle")]
use candle_transformers::generation::LogitsProcessor;
#[cfg(feature = "with-candle")]
use hf_hub::api::sync::Api;
#[cfg(feature = "with-candle")]
use tokenizers::Tokenizer;
#[cfg(feature = "with-candle")]
use tracing::info;
#[cfg(feature = "with-candle")]
use std::io::BufReader;

#[cfg(feature = "with-candle")]
use super::quantized_gemma::ModelWeights;

#[cfg(feature = "with-candle")]
use crate::concourse::cache::CacheManager;

/// Configuration for Gemma model loading
#[derive(Debug, Clone)]
pub struct GemmaConfig {
    /// Path to GGUF model file
    pub model_path: String,
    /// Model size in billions of parameters (0.3, 1.0, 4.0, 12.0)
    pub size_b: f32,
    /// Quantization type (e.g. "Q4_K_M", "Q8_0")
    pub quantization: String,
    /// Device: "cpu", "metal", "cuda"
    pub device: String,
    /// Optional local path to tokenizer.json (skips HF Hub fetch)
    pub tokenizer_path: Option<String>,
}

// ── Real implementation (with-candle) ─────────────────────────────────────────

/// Gemma model instance (candle-backed)
#[cfg(feature = "with-candle")]
pub struct GemmaModel {
    device: Device,
    model: std::sync::Mutex<ModelWeights>,
    tokenizer: Tokenizer,
}

#[cfg(feature = "with-candle")]
impl GemmaModel {
    /// Load a Gemma model from a GGUF file.
    pub fn load(config: GemmaConfig) -> SwarmResult<Self> {
        let device = match config.device.as_str() {
            "cuda"  => Device::new_cuda(0)
                .map_err(|e| SwarmError::Embedding(format!("CUDA: {e}")))?,
            "metal" => Device::new_metal(0)
                .map_err(|e| SwarmError::Embedding(format!("Metal: {e}")))?,
            _ => Device::Cpu,
        };

        info!("Loading GGUF: {} ({}B, {})", config.model_path, config.size_b, config.quantization);

        let mut file = std::fs::File::open(&config.model_path)
            .map_err(|e| SwarmError::Io(e))?;
        let mut reader = BufReader::new(&mut file);
        let ct = gguf_file::Content::read(&mut reader)
            .map_err(|e| SwarmError::Embedding(format!("GGUF read error: {e}")))?;

        let model = ModelWeights::from_gguf(ct, &mut reader, &device)
            .map_err(|e| SwarmError::Candle(e))?;

        info!("Model loaded. Hidden dim: {}", model.hidden_dim);

        let tokenizer = Self::load_tokenizer(&config)?;

        Ok(Self {
            device,
            model: std::sync::Mutex::new(model),
            tokenizer,
        })
    }

    /// Load tokenizer: local file → models/tokenizer.json → HF Hub.
    /// Fails loudly if vocab size ≠ 262144.
    fn load_tokenizer(config: &GemmaConfig) -> SwarmResult<Tokenizer> {
        // 1. User-provided explicit path
        if let Some(path) = &config.tokenizer_path {
            info!("Loading tokenizer from: {path}");
            let tok = Tokenizer::from_file(path).map_err(|e| {
                SwarmError::Candle(candle_core::Error::Msg(format!(
                    "Failed to load tokenizer {path}: {e}"
                )))
            })?;
            return Self::validate_tokenizer(tok);
        }

        // 2. Local models/tokenizer.json (offline first)
        let local_tok = std::path::Path::new("models/tokenizer.json");
        if local_tok.exists() {
            info!("Loading local tokenizer: models/tokenizer.json");
            let tok = Tokenizer::from_file(local_tok).map_err(|e| {
                SwarmError::Candle(candle_core::Error::Msg(format!(
                    "Failed to load local tokenizer: {e}"
                )))
            })?;
            return Self::validate_tokenizer(tok);
        }

        // 3. Fetch from HF Hub (requires HF_TOKEN in environment)
        info!("Fetching Gemma tokenizer from HF Hub (unsloth/embeddinggemma-300m-GGUF)...");
        let api = Api::new().map_err(|e| {
            SwarmError::Candle(candle_core::Error::Msg(format!("HF API init: {e}")))
        })?;
        let repo = api.model("unsloth/embeddinggemma-300m-GGUF".to_string());
        let path = repo.get("tokenizer.json").map_err(|e| {
            SwarmError::Candle(candle_core::Error::Msg(format!(
                "CRITICAL: HF Hub tokenizer fetch failed: {e}\n\
                 → Put tokenizer.json in models/tokenizer.json for offline use."
            )))
        })?;
        let tok = Tokenizer::from_file(&path).map_err(|e| {
            SwarmError::Candle(candle_core::Error::Msg(format!(
                "Failed to parse downloaded tokenizer: {e}"
            )))
        })?;
        Self::validate_tokenizer(tok)
    }

    fn validate_tokenizer(tok: Tokenizer) -> SwarmResult<Tokenizer> {
        let vsz = tok.get_vocab_size(true);
        info!("Tokenizer vocab size: {vsz}");
        if vsz != 262144 {
            return Err(SwarmError::Candle(candle_core::Error::Msg(format!(
                "Tokenizer vocab size {vsz} ≠ 262144. \
                 Wrong tokenizer — must use EmbeddingGemma tokenizer."
            ))));
        }
        Ok(tok)
    }

    /// Generate embeddings: tokenize → hidden states → mean pool → L2 norm.
    pub fn embed(&self, text: &str) -> SwarmResult<Vec<f32>> {
        let enc = self.tokenizer.encode(text, true)
            .map_err(|e| SwarmError::Embedding(format!("Tokenize: {e}")))?;
        let ids = enc.get_ids();

        let tokens = Tensor::new(ids, &self.device)
            .map_err(|e| SwarmError::Candle(e))?
            .unsqueeze(0)
            .map_err(|e| SwarmError::Candle(e))?;

        let mut model = self.model.lock().unwrap();
        model.clear_kv_cache();

        // Process all tokens in one pass to get full sequence hidden states
        // For embedding we want full-sequence mean, not just the last position.
        // So process token by token and collect all hiddens.
        let hidden = Self::embed_mean_pool(&mut model, &tokens, &self.device)?;

        // L2 normalize for cosine similarity
        let normed = Self::l2_norm(&hidden)?;
        normed.to_vec1::<f32>().map_err(|e| SwarmError::Candle(e))
    }

    /// Forward all tokens, collect per-token hiddens, return mean-pooled [hidden_dim].
    fn embed_mean_pool(
        model: &mut ModelWeights,
        tokens: &Tensor,    // [1, seq_len]
        device: &Device,
    ) -> SwarmResult<Tensor> {
        let (_b, seq_len) = tokens.dims2()
            .map_err(|e| SwarmError::Candle(e))?;

        let mut hiddens: Vec<Tensor> = Vec::with_capacity(seq_len);

        for i in 0..seq_len {
            let tok = tokens.narrow(1, i, 1)
                .map_err(|e| SwarmError::Candle(e))?;
            // forward_hidden returns [1, hidden_dim] (last/only position)
            let h = model.forward_hidden(&tok, i)
                .map_err(|e| SwarmError::Candle(e))?;
            hiddens.push(h);
        }

        // Stack to [seq_len, hidden_dim] then mean over seq dim
        let stacked = Tensor::cat(&hiddens, 0)
            .map_err(|e| SwarmError::Candle(e))?;
        stacked.mean(0).map_err(|e| SwarmError::Candle(e))
    }

    /// L2 normalize: x / ||x||₂
    fn l2_norm(x: &Tensor) -> SwarmResult<Tensor> {
        let x_f32 = x.to_dtype(DType::F32)
            .map_err(|e| SwarmError::Candle(e))?;
        let norm = x_f32.sqr()
            .map_err(|e| SwarmError::Candle(e))?
            .sum_all()
            .map_err(|e| SwarmError::Candle(e))?
            .sqrt()
            .map_err(|e| SwarmError::Candle(e))?;
        let normed = x_f32.broadcast_div(&norm)
            .map_err(|e| SwarmError::Candle(e))?;
        Ok(normed)
    }

    /// Autoregressive text completion with temperature 0.7.
    pub fn complete(&self, prompt: &str, max_tokens: usize) -> SwarmResult<String> {
        let enc = self.tokenizer.encode(prompt, true)
            .map_err(|e| SwarmError::Embedding(format!("Tokenize: {e}")))?;
        let prompt_ids = enc.get_ids().to_vec();

        let eos = self.tokenizer.token_to_id("<eos>")
            .or_else(|| self.tokenizer.token_to_id("</s>"))
            .unwrap_or(1);

        let mut model = self.model.lock().unwrap();
        model.clear_kv_cache();

        let mut logits_proc = LogitsProcessor::new(299792458, Some(0.7), None);
        let mut offset = 0usize;

        // Feed prompt tokens into KV cache
        for &tok in &prompt_ids {
            let t = Tensor::new(&[tok], &self.device)
                .map_err(|e| SwarmError::Candle(e))?
                .unsqueeze(0)
                .map_err(|e| SwarmError::Candle(e))?;
            model.forward(&t, offset)
                .map_err(|e| SwarmError::Candle(e))?;
            offset += 1;
        }

        let mut generated = Vec::new();
        let mut next = *prompt_ids.last().unwrap_or(&0);

        for _ in 0..max_tokens {
            let t = Tensor::new(&[next], &self.device)
                .map_err(|e| SwarmError::Candle(e))?
                .unsqueeze(0)
                .map_err(|e| SwarmError::Candle(e))?;
            let logits = model.forward(&t, offset)
                .map_err(|e| SwarmError::Candle(e))?;

            let token_id = logits_proc.sample(&logits.squeeze(0)
                .map_err(|e| SwarmError::Candle(e))?)
                .map_err(|e| SwarmError::Embedding(format!("Sample: {e}")))?;

            generated.push(token_id);
            offset += 1;
            if token_id == eos { break; }
            next = token_id;
        }

        let all_ids = [&prompt_ids[..], &generated[..]].concat();
        self.tokenizer.decode(&all_ids, true)
            .map_err(|e| SwarmError::Embedding(format!("Decode: {e}")))
    }

    pub fn tokenizer_vocab_size(&self) -> usize {
        self.tokenizer.get_vocab_size(true)
    }

    pub fn clear_kv_cache(&self) {
        self.model.lock().unwrap().clear_kv_cache();
    }
}

// ── Stub implementation (no candle) ───────────────────────────────────────────

#[cfg(not(feature = "with-candle"))]
pub struct GemmaModel {
    _config: GemmaConfig,
}

#[cfg(not(feature = "with-candle"))]
impl GemmaModel {
    pub fn load(config: GemmaConfig) -> SwarmResult<Self> {
        Ok(Self { _config: config })
    }
    pub fn embed(&self, _text: &str) -> SwarmResult<Vec<f32>> {
        Ok(vec![0.0; 768])
    }
    pub fn complete(&self, prompt: &str, _max_tokens: usize) -> SwarmResult<String> {
        Ok(format!("[STUB: {}]", prompt))
    }
}

// ── EmbeddingGemma300M wrapper ─────────────────────────────────────────────────

#[cfg(feature = "with-candle")]
pub struct EmbeddingGemma300M {
    pub model: Arc<GemmaModel>,
}

#[cfg(feature = "with-candle")]
impl EmbeddingGemma300M {
    pub fn new(model_path: &str) -> SwarmResult<Self> {
        let config = GemmaConfig {
            model_path: model_path.to_string(),
            size_b: 0.3,
            quantization: "Q8_0".to_string(),
            device: "cpu".to_string(),
            tokenizer_path: None,
        };
        Ok(Self { model: Arc::new(GemmaModel::load(config)?) })
    }
    pub fn embed(&self, text: &str) -> SwarmResult<Vec<f32>> {
        self.model.embed(text)
    }
}

#[cfg(not(feature = "with-candle"))]
pub struct EmbeddingGemma300M {
    pub model: Arc<GemmaModel>,
}

#[cfg(not(feature = "with-candle"))]
impl EmbeddingGemma300M {
    pub fn new(_model_path: &str) -> SwarmResult<Self> {
        let config = GemmaConfig {
            model_path: "".to_string(),
            size_b: 0.3,
            quantization: "Q8_0".to_string(),
            device: "cpu".to_string(),
            tokenizer_path: None,
        };
        Ok(Self { model: Arc::new(GemmaModel::load(config)?) })
    }
    pub fn embed(&self, text: &str) -> SwarmResult<Vec<f32>> {
        self.model.embed(text)
    }
}

// ── EmbedGemma adapter ────────────────────────────────────────────────────────

use super::EmbedGemma;

pub struct GemmaEmbedder {
    model: Arc<GemmaModel>,
    class: NodeClass,
    #[cfg(feature = "with-candle")]
    cache: Option<Arc<CacheManager>>,
}

impl GemmaEmbedder {
    pub fn new(model: Arc<GemmaModel>, class: NodeClass) -> Self {
        Self {
            model,
            class,
            #[cfg(feature = "with-candle")]
            cache: None,
        }
    }

    #[cfg(feature = "with-candle")]
    pub fn new_with_cache(model: Arc<GemmaModel>, class: NodeClass, cache: Arc<CacheManager>) -> Self {
        Self { model, class, cache: Some(cache) }
    }
}

impl EmbedGemma for GemmaEmbedder {
    fn compress(&self, raw_text: &str) -> SwarmResult<Node> {
        #[cfg(feature = "with-candle")]
        let embedding = {
            if let Some(cache) = &self.cache {
                if let Some(cached) = cache.get_embedding(raw_text)
                    .map_err(|e| SwarmError::Embedding(format!("Cache read: {e}")))?
                {
                    cached
                } else {
                    let emb = self.model.embed(raw_text)?;
                    cache.cache_embedding(raw_text, &emb)
                        .map_err(|e| SwarmError::Embedding(format!("Cache write: {e}")))?;
                    emb
                }
            } else {
                self.model.embed(raw_text)?
            }
        };
        #[cfg(not(feature = "with-candle"))]
        let embedding = self.model.embed(raw_text)?;

        let semantic_hash = if raw_text.len() > 100 {
            format!("{}...", &raw_text[..100])
        } else {
            raw_text.to_string()
        }.to_uppercase();

        use std::time::{SystemTime, UNIX_EPOCH};
        let id = format!(
            "N_{}",
            SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos()
        );

        Ok(Node { id, class: self.class.clone(), semantic_hash, embedding: Some(embedding) })
    }

    fn specialization(&self) -> NodeClass {
        self.class.clone()
    }

    fn embedding_dimension(&self) -> usize {
        768
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(not(feature = "with-candle"))]
    #[test]
    fn test_stub_embedder() {
        let config = GemmaConfig {
            model_path: "test.gguf".to_string(),
            size_b: 0.3,
            quantization: "Q8_0".to_string(),
            device: "cpu".to_string(),
            tokenizer_path: None,
        };
        let model = GemmaModel::load(config).unwrap();
        let embedder = GemmaEmbedder::new(Arc::new(model), NodeClass::Axiom);
        let node = embedder.compress("test text").unwrap();
        assert_eq!(node.class, NodeClass::Axiom);
        assert!(!node.id.is_empty());
    }

    #[cfg(feature = "with-candle")]
    #[test]
    fn test_gemma_model_loading() {
        use std::path::Path;
        let _ = tracing_subscriber::fmt::try_init();
        let path = "models/embeddinggemma-300M-Q8_0.gguf";
        if !Path::new(path).exists() {
            println!("Model not found, skipping");
            return;
        }
        let config = GemmaConfig {
            model_path: path.to_string(),
            size_b: 0.3,
            quantization: "Q8_0".to_string(),
            device: "cpu".to_string(),
            tokenizer_path: None,
        };
        let model = GemmaModel::load(config).unwrap();
        let emb = model.embed("Hello, world!").unwrap();
        assert_eq!(emb.len(), 768, "Embedding dimension should be 768");
        let norm: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 0.01, "Embedding should be L2-normalized (norm={norm:.4})");
        println!("✓ Embedding norm: {norm:.6}");
    }

    #[cfg(feature = "with-candle")]
    #[test]
    fn test_cosine_similarity() {
        use std::path::Path;
        let _ = tracing_subscriber::fmt::try_init();
        let path = "models/embeddinggemma-300M-Q8_0.gguf";
        if !Path::new(path).exists() {
            println!("Model not found, skipping");
            return;
        }
        let config = GemmaConfig {
            model_path: path.to_string(),
            size_b: 0.3,
            quantization: "Q8_0".to_string(),
            device: "cpu".to_string(),
            tokenizer_path: None,
        };
        let model = GemmaModel::load(config).unwrap();
        let a = model.embed("The dog ran through the park").unwrap();
        let b = model.embed("A puppy sprinted across the garden").unwrap();
        let c = model.embed("Quantum entanglement in superconductors").unwrap();
        let sim_ab: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let sim_ac: f32 = a.iter().zip(c.iter()).map(|(x, y)| x * y).sum();
        println!("Cosine (dog/puppy): {sim_ab:.4}");
        println!("Cosine (dog/quantum): {sim_ac:.4}");
        assert!(sim_ab > sim_ac, "Semantically similar texts should have higher cosine similarity");
    }

    #[cfg(feature = "with-candle")]
    #[test]
    fn test_tokenizer_vocab_size() {
        use std::path::Path;
        let _ = tracing_subscriber::fmt::try_init();
        let path = "models/embeddinggemma-300M-Q8_0.gguf";
        if !Path::new(path).exists() {
            println!("Model not found, skipping");
            return;
        }
        let config = GemmaConfig {
            model_path: path.to_string(),
            size_b: 0.3,
            quantization: "Q8_0".to_string(),
            device: "cpu".to_string(),
            tokenizer_path: None,
        };
        let model = GemmaModel::load(config).unwrap();
        let vsz = model.tokenizer_vocab_size();
        assert_eq!(vsz, 262144, "Vocab size must be 262144 for Gemma 3");
        println!("✓ Tokenizer vocab size: {vsz}");
    }
}
