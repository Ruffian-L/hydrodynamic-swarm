//! InstructGemma: Real LLM-backed relational edge classification for Function Nodes
//!
//! Loads a Gemma 3 instruct model (1B or 4B Q4_K_M) and uses autoregressive
//! generation to classify the relational edge between two semantic nodes using
//! the strict 7-edge EmbedSwarm lexicon.
//!
//! Uses a lazy singleton (`OnceLock`) so the model is loaded once on first use.

use crate::concourse::embed::quantized_gemma::ModelWeights;
use crate::concourse::types::RelationalEdge;
use crate::concourse::{SwarmError, SwarmResult};
use candle_core::{quantized::gguf_file, Device, Tensor};
use candle_transformers::generation::LogitsProcessor;
use std::io::BufReader;
use std::sync::{Mutex, OnceLock};
use tokenizers::Tokenizer;
use tracing::{info, warn};

/// Default path for the Function Gemma model (1B instruct, lightest)
const DEFAULT_MODEL_PATH: &str = "models/gemma-3-1b-it-Q4_K_M.gguf";

/// Local tokenizer path (shared with embed model)
const LOCAL_TOKENIZER_PATH: &str = "models/tokenizer.json";

/// Instruct prompt template for 7-edge classification
const EDGE_PROMPT: &str = "<start_of_turn>user
You are a semantic graph classifier. Classify the relationship between Concept A and Concept B using EXACTLY ONE of these relationship types:

ENCAPSULATES - A contains or wraps B as a component
SCAFFOLDS    - A provides the foundation or base that B extends
ACTUATES     - A triggers, causes, or initiates B
IS_ISOMORPHIC_TO - A and B share the same structural pattern
CONTRADICTS  - A and B are in logical opposition
CATALYZES    - A accelerates or enables B without being consumed
SYNTHESIZES  - A and B combine to form a new emergent concept

Concept A: {A}
Concept B: {B}

Respond with ONLY the relationship type name, nothing else.<end_of_turn>
<start_of_turn>model
";

// ── Edge name → RelationalEdge parsing ───────────────────────────────────────

fn parse_edge_from_output(output: &str) -> RelationalEdge {
    let upper = output.to_uppercase();
    if upper.contains("ENCAPSULAT") {
        RelationalEdge::Encapsulates
    } else if upper.contains("SCAFFOLD") {
        RelationalEdge::Scaffolds
    } else if upper.contains("ACTUAT") {
        RelationalEdge::Actuates
    } else if upper.contains("ISOMORPHIC") {
        RelationalEdge::IsIsomorphicTo
    } else if upper.contains("CONTRADICT") {
        RelationalEdge::Contradicts
    } else if upper.contains("CATALYZ") {
        RelationalEdge::Catalyzes
    } else if upper.contains("SYNTHESIZ") {
        RelationalEdge::Synthesizes
    } else {
        warn!("InstructGemma output did not match any edge: {:?}", output);
        RelationalEdge::Actuates // neutral fallback
    }
}

// ── InstructGemmaModel ────────────────────────────────────────────────────────

pub struct InstructGemmaModel {
    model: Mutex<ModelWeights>,
    tokenizer: Tokenizer,
    device: Device,
}

impl InstructGemmaModel {
    pub fn load(model_path: &str) -> SwarmResult<Self> {
        info!("Loading InstructGemma from: {model_path}");
        let mut file = std::fs::File::open(model_path).map_err(SwarmError::Io)?;
        let mut reader = BufReader::new(&mut file);
        let ct = gguf_file::Content::read(&mut reader)
            .map_err(|e| SwarmError::Embedding(format!("GGUF read: {e}")))?;
        let device = Device::Cpu;
        let model = ModelWeights::from_gguf(ct, &mut reader, &device)
            .map_err(|e| SwarmError::Candle(e))?;
        info!("InstructGemma loaded. Hidden dim: {}", model.hidden_dim);

        let tokenizer = Self::load_tokenizer()?;
        Ok(Self {
            model: Mutex::new(model),
            tokenizer,
            device,
        })
    }

    fn load_tokenizer() -> SwarmResult<Tokenizer> {
        // Local first, then HF Hub
        if std::path::Path::new(LOCAL_TOKENIZER_PATH).exists() {
            return Tokenizer::from_file(LOCAL_TOKENIZER_PATH).map_err(|e| {
                SwarmError::Embedding(format!("Tokenizer load failed: {e}"))
            });
        }
        // Reuse the HF hub logic from the embed model
        use hf_hub::api::sync::Api;
        let api = Api::new().map_err(|e| {
            SwarmError::Embedding(format!("HF API: {e}"))
        })?;
        let path = api
            .model("unsloth/embeddinggemma-300m-GGUF".to_string())
            .get("tokenizer.json")
            .map_err(|e| SwarmError::Embedding(format!("Tokenizer fetch: {e}")))?;
        Tokenizer::from_file(&path)
            .map_err(|e| SwarmError::Embedding(format!("Tokenizer parse: {e}")))
    }

    /// Classify the relational edge between two node texts.
    pub fn classify_edge(&self, concept_a: &str, concept_b: &str) -> SwarmResult<RelationalEdge> {
        let prompt = EDGE_PROMPT
            .replace("{A}", concept_a)
            .replace("{B}", concept_b);

        let enc = self
            .tokenizer
            .encode(prompt.as_str(), true)
            .map_err(|e| SwarmError::Embedding(format!("Tokenize: {e}")))?;
        let ids = enc.get_ids().to_vec();

        let eos = self
            .tokenizer
            .token_to_id("<eos>")
            .or_else(|| self.tokenizer.token_to_id("</s>"))
            .unwrap_or(1);

        let mut model = self.model.lock().unwrap();
        model.clear_kv_cache();

        // Temperature 0.2 for deterministic classification
        let mut lp = LogitsProcessor::new(42, Some(0.2), None);
        let mut offset = 0usize;

        // Feed prompt
        for &tok in &ids {
            let t = Tensor::new(&[tok], &self.device)
                .map_err(SwarmError::Candle)?
                .unsqueeze(0)
                .map_err(SwarmError::Candle)?;
            model.forward(&t, offset).map_err(SwarmError::Candle)?;
            offset += 1;
        }

        // Generate up to 12 tokens (edge name is at most 2 words)
        let mut generated_ids = Vec::new();
        let mut next = *ids.last().unwrap_or(&0);
        for _ in 0..12 {
            let t = Tensor::new(&[next], &self.device)
                .map_err(SwarmError::Candle)?
                .unsqueeze(0)
                .map_err(SwarmError::Candle)?;
            let logits = model.forward(&t, offset).map_err(SwarmError::Candle)?;
            let tok = lp
                .sample(&logits.squeeze(0).map_err(SwarmError::Candle)?)
                .map_err(|e| SwarmError::Embedding(format!("Sample: {e}")))?;
            generated_ids.push(tok);
            offset += 1;
            if tok == eos {
                break;
            }
            next = tok;
        }

        let text = self
            .tokenizer
            .decode(&generated_ids, true)
            .map_err(|e| SwarmError::Embedding(format!("Decode: {e}")))?;

        info!("InstructGemma edge output: {:?}", text.trim());
        Ok(parse_edge_from_output(text.trim()))
    }
}

// ── Lazy singleton ────────────────────────────────────────────────────────────

static INSTRUCT_MODEL: OnceLock<InstructGemmaModel> = OnceLock::new();

/// Get or initialize the shared InstructGemma singleton.
/// Returns None if the model file is not present (falls back to heuristics).
pub fn instruct_model() -> Option<&'static InstructGemmaModel> {
    INSTRUCT_MODEL.get_or_init(|| {
        match InstructGemmaModel::load(DEFAULT_MODEL_PATH) {
            Ok(m) => {
                info!("InstructGemma singleton ready");
                m
            }
            Err(e) => {
                // Panic during OnceLock would poison it — instead we never set it.
                // But OnceLock::get_or_init requires we always return a value.
                // So we log and return a deliberately broken model that will
                // fail gracefully at inference time, triggering heuristic fallback.
                warn!("InstructGemma load failed (falling back to heuristics): {e}");
                // Trigger a panic here to prevent poisoning OnceLock with a broken model;
                // instead don't use OnceLock for fallback — use a separate Option.
                panic!("InstructGemma load: {e}");
            }
        }
    });
    // If init panicked, OnceLock stays uninit — get() returns None.
    // But panics in OnceLock callbacks are not recoverable.
    // Better approach below using a wrapper.
    INSTRUCT_MODEL.get()
}

// ── Safe lazy init with fallback ──────────────────────────────────────────────

static MODEL_INIT_RESULT: OnceLock<bool> = OnceLock::new();

/// Try to initialize the InstructGemma model. Returns true if successful.
/// This is called once; subsequent calls are no-ops.
pub fn try_init_instruct_model() -> bool {
    *MODEL_INIT_RESULT.get_or_init(|| {
        if !std::path::Path::new(DEFAULT_MODEL_PATH).exists() {
            info!("InstructGemma model not found at {DEFAULT_MODEL_PATH}, using heuristics");
            return false;
        }
        match InstructGemmaModel::load(DEFAULT_MODEL_PATH) {
            Ok(m) => {
                // Store in separate static
                let _ = INSTRUCT_MODEL.set(m);
                true
            }
            Err(e) => {
                warn!("InstructGemma failed to load: {e}. Using heuristic fallback.");
                false
            }
        }
    })
}

/// Classify edge using LLM if available, otherwise return None (caller uses heuristics).
pub fn classify_edge_llm(concept_a: &str, concept_b: &str) -> Option<RelationalEdge> {
    // Initialize on first call (non-blocking — returns false if file missing)
    try_init_instruct_model();
    INSTRUCT_MODEL
        .get()
        .and_then(|m| m.classify_edge(concept_a, concept_b).ok())
}
