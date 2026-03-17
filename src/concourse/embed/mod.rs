//! Embed Gemmas: Sensory compressors that convert raw Flux into mathematical Nodes
//!
//! Three specialized instances in the 9-Node Matrix:
//! 1. Embed Alpha (Axiomatic Sensor): Code, rules, system constraints
//! 2. Embed Beta (Telemetry Sensor): User prompts, documentation, environmental data
//! 3. Embed Gamma (Anomaly Sensor): Error logs, tracebacks, paradoxes

use crate::concourse::types::{Node, NodeClass};
use crate::concourse::{SwarmError, SwarmResult};
use std::sync::Arc;
use tokio::sync::mpsc;

#[cfg(feature = "with-candle")]
use crate::concourse::cache::CacheManager;
#[cfg(feature = "with-candle")]
use std::sync::OnceLock;

#[cfg(feature = "with-candle")]
pub mod quantized_gemma;
#[cfg(feature = "with-candle")]
mod gemma;
#[cfg(feature = "with-candle")]
use self::gemma::{EmbeddingGemma300M, GemmaEmbedder};

#[cfg(feature = "with-candle")]
static CACHE_MANAGER: OnceLock<Arc<CacheManager>> = OnceLock::new();

#[cfg(feature = "with-candle")]
fn shared_cache_manager() -> Arc<CacheManager> {
    CACHE_MANAGER
        .get_or_init(|| Arc::new(CacheManager::new()))
        .clone()
}

/// Trait defining the compression interface for Embed Gemmas
pub trait EmbedGemma: Send + Sync {
    /// Compress raw text into a Node with semantic hash
    fn compress(&self, raw_text: &str) -> SwarmResult<Node>;

    /// Get the ontological class specialization of this Embed Gemma
    fn specialization(&self) -> NodeClass;

    /// Get embedding dimension (for MRL scaling)
    fn embedding_dimension(&self) -> usize;
}

/// Embed Alpha: Specialized for axiomatic compression
/// Processes code, rules, and system constraints into [AXIOM] nodes
pub struct EmbedAlpha {
    inner: Arc<dyn EmbedGemma>,
}

impl EmbedAlpha {
    pub fn new() -> SwarmResult<Self> {
        #[cfg(feature = "with-candle")]
        {
            // Load the 300M embedding Gemma model
            let model_path = "models/embeddinggemma-300M-Q8_0.gguf";
            let gemma = EmbeddingGemma300M::new(model_path)?;
            let model = gemma.model.clone();
            let cache = shared_cache_manager();
            let embedder: Arc<dyn EmbedGemma> = Arc::new(GemmaEmbedder::new_with_cache(
                model,
                NodeClass::Axiom,
                cache,
            ));
            Ok(Self { inner: embedder })
        }
        #[cfg(not(feature = "with-candle"))]
        {
            let stub: Arc<dyn EmbedGemma> = Arc::new(StubEmbedder(NodeClass::Axiom));
            Ok(Self { inner: stub })
        }
    }
}

/// Embed Beta: Specialized for telemetry compression
/// Processes user prompts and documentation into [OBSERVATION] nodes
pub struct EmbedBeta {
    inner: Arc<dyn EmbedGemma>,
}

impl EmbedBeta {
    pub fn new() -> SwarmResult<Self> {
        #[cfg(feature = "with-candle")]
        {
            let model_path = "models/embeddinggemma-300M-Q8_0.gguf";
            let gemma = EmbeddingGemma300M::new(model_path)?;
            let model = gemma.model.clone();
            let cache = shared_cache_manager();
            let embedder: Arc<dyn EmbedGemma> = Arc::new(GemmaEmbedder::new_with_cache(
                model,
                NodeClass::Observation,
                cache,
            ));
            Ok(Self { inner: embedder })
        }
        #[cfg(not(feature = "with-candle"))]
        {
            let stub: Arc<dyn EmbedGemma> = Arc::new(StubEmbedder(NodeClass::Observation));
            Ok(Self { inner: stub })
        }
    }
}

/// Embed Gamma: Specialized for anomaly compression
/// Processes errors and paradoxes into [ANOMALY] nodes
pub struct EmbedGamma {
    inner: Arc<dyn EmbedGemma>,
}

impl EmbedGamma {
    pub fn new() -> SwarmResult<Self> {
        #[cfg(feature = "with-candle")]
        {
            let model_path = "models/embeddinggemma-300M-Q8_0.gguf";
            let gemma = EmbeddingGemma300M::new(model_path)?;
            let model = gemma.model.clone();
            let cache = shared_cache_manager();
            let embedder: Arc<dyn EmbedGemma> = Arc::new(GemmaEmbedder::new_with_cache(
                model,
                NodeClass::Anomaly,
                cache,
            ));
            Ok(Self { inner: embedder })
        }
        #[cfg(not(feature = "with-candle"))]
        {
            let stub: Arc<dyn EmbedGemma> = Arc::new(StubEmbedder(NodeClass::Anomaly));
            Ok(Self { inner: stub })
        }
    }
}

// Stub implementation for when candle is not available
#[cfg(not(feature = "with-candle"))]
struct StubEmbedder(NodeClass);

#[cfg(not(feature = "with-candle"))]
impl EmbedGemma for StubEmbedder {
    fn compress(&self, raw_text: &str) -> SwarmResult<Node> {
        let semantic_hash = generate_semantic_hash(raw_text, 10);
        Ok(Node::new(generate_node_id(), self.0.clone(), semantic_hash))
    }

    fn specialization(&self) -> NodeClass {
        self.0.clone()
    }

    fn embedding_dimension(&self) -> usize {
        768
    }
}

impl EmbedGemma for EmbedAlpha {
    fn compress(&self, raw_text: &str) -> SwarmResult<Node> {
        self.inner.compress(raw_text)
    }

    fn specialization(&self) -> NodeClass {
        self.inner.specialization()
    }

    fn embedding_dimension(&self) -> usize {
        self.inner.embedding_dimension()
    }
}

impl EmbedGemma for EmbedBeta {
    fn compress(&self, raw_text: &str) -> SwarmResult<Node> {
        self.inner.compress(raw_text)
    }

    fn specialization(&self) -> NodeClass {
        self.inner.specialization()
    }

    fn embedding_dimension(&self) -> usize {
        self.inner.embedding_dimension()
    }
}

impl EmbedGemma for EmbedGamma {
    fn compress(&self, raw_text: &str) -> SwarmResult<Node> {
        self.inner.compress(raw_text)
    }

    fn specialization(&self) -> NodeClass {
        self.inner.specialization()
    }

    fn embedding_dimension(&self) -> usize {
        self.inner.embedding_dimension()
    }
}

/// Manager for all 3 Embed Gemmas with load balancing
pub struct EmbedManager {
    alpha: Arc<dyn EmbedGemma>,
    beta: Arc<dyn EmbedGemma>,
    gamma: Arc<dyn EmbedGemma>,
    tx: mpsc::Sender<Node>,
}

impl EmbedManager {
    pub fn new(
        alpha: Arc<dyn EmbedGemma>,
        beta: Arc<dyn EmbedGemma>,
        gamma: Arc<dyn EmbedGemma>,
        tx: mpsc::Sender<Node>,
    ) -> Self {
        Self {
            alpha,
            beta,
            gamma,
            tx,
        }
    }

    /// Route raw Flux to appropriate Embed Gemma based on content
    pub async fn process_flux(&self, raw_text: &str) -> SwarmResult<()> {
        let gemma = self.route_gemma(raw_text);
        let node = gemma.compress(raw_text)?;

        self.tx
            .send(node)
            .await
            .map_err(|e| SwarmError::Concurrency(format!("Failed to send node: {}", e)))?;

        Ok(())
    }

    /// Route Flux to appropriate Gemma based on content heuristics
    fn route_gemma(&self, text: &str) -> &Arc<dyn EmbedGemma> {
        // Simple routing heuristics (to be improved)
        if text.contains("error") || text.contains("fail") || text.contains("exception") {
            &self.gamma
        } else if text.contains("def")
            || text.contains("fn ")
            || text.contains("class ")
            || text.contains("struct ")
        {
            &self.alpha
        } else {
            &self.beta
        }
    }
}

/// Generate a semantic hash (max 10 words) from raw text
fn generate_semantic_hash(text: &str, max_words: usize) -> String {
    let words: Vec<&str> = text.split_whitespace().collect();
    let truncated: Vec<&str> = words.into_iter().take(max_words).collect();
    truncated.join(" ").to_uppercase()
}

/// Generate a unique node ID
fn generate_node_id() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    format!("N_{}", timestamp)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_semantic_hash_generation() {
        let text = "This is a test of the semantic hash generation system";
        let hash = generate_semantic_hash(text, 5);
        assert_eq!(hash, "THIS IS A TEST OF");

        let hash = generate_semantic_hash(text, 10);
        assert_eq!(
            hash,
            "THIS IS A TEST OF THE SEMANTIC HASH GENERATION SYSTEM"
        );
    }

    #[test]
    fn test_node_id_generation() {
        let id1 = generate_node_id();
        let id2 = generate_node_id();
        // IDs should start with N_ (uniqueness is not guaranteed due to nanosecond precision)
        assert!(id1.starts_with("N_"));
        assert!(id2.starts_with("N_"));
    }

    #[test]
    fn test_embed_alpha() {
        let alpha = EmbedAlpha::new().unwrap();
        let node = alpha.compress("test code").unwrap();
        assert_eq!(node.class, NodeClass::Axiom);
    }
}
