//! Model backend abstraction -- dispatches between Llama and Qwen3.5.

use candle_core::{Result, Tensor};

use crate::llama;
use crate::qwen35;

/// Unified model interface for physics steering.
#[allow(clippy::large_enum_variant)] // model weights are large by design
pub enum ModelBackend {
    Llama(llama::ModelWeights),
    Qwen35(qwen35::Qwen35Model),
}

impl ModelBackend {
    pub fn forward(&mut self, x: &Tensor, index_pos: usize) -> Result<Tensor> {
        match self {
            Self::Llama(m) => m.forward(x, index_pos),
            Self::Qwen35(m) => m.forward(x, index_pos),
        }
    }

    pub fn forward_with_hidden(
        &mut self,
        x: &Tensor,
        index_pos: usize,
    ) -> Result<(Tensor, Tensor)> {
        match self {
            Self::Llama(m) => m.forward_with_hidden(x, index_pos),
            Self::Qwen35(m) => m.forward_with_hidden(x, index_pos),
        }
    }

    pub fn project_to_logits(&self, hidden: &Tensor) -> Result<Tensor> {
        match self {
            Self::Llama(m) => m.project_to_logits(hidden),
            Self::Qwen35(m) => m.project_to_logits(hidden),
        }
    }

    pub fn token_embeddings(&self) -> &Tensor {
        match self {
            Self::Llama(m) => m.token_embeddings(),
            Self::Qwen35(m) => m.token_embeddings(),
        }
    }

    #[allow(dead_code)] // useful for debugging; not called in the hot path
    pub fn name(&self) -> &'static str {
        match self {
            Self::Llama(_) => "Llama",
            Self::Qwen35(_) => "Qwen3.5",
        }
    }

    #[allow(dead_code)] // used in tui.rs generate_text for chat template detection
    pub fn is_qwen35(&self) -> bool {
        matches!(self, Self::Qwen35(_))
    }
}
