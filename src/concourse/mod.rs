//! Concourse: 9-Node Matrix orchestration layer
//!
//! Imported from Lumina-Concourse (ouroboros_reactor).
//! Provides the Three Gemmas architecture for multi-LLM swarm coordination.

pub mod async_patterns;
pub mod cache;
pub mod embed;
pub mod function;
pub mod governor;
pub mod physics;
pub mod swarm;
pub mod types;
pub mod utils;

// Re-exports for convenient usage
pub use physics::{CognitiveState, VolumetricGovernor};
pub use types::{Edge, FluxTuple, Node, NodeClass, RelationalEdge};

/// Core error type for the Concourse swarm system
#[derive(thiserror::Error, Debug)]
pub enum SwarmError {
    #[error("Physics engine error: {0}")]
    Physics(String),

    #[error("Embedding error: {0}")]
    Embedding(String),

    #[error("Relational analysis error: {0}")]
    Relational(String),

    #[error("Governor error: {0}")]
    Governor(String),

    #[error("Persistence error: {0}")]
    Persistence(String),

    #[error("Concurrency error: {0}")]
    Concurrency(String),

    #[error("Configuration error: {0}")]
    Config(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    #[cfg(feature = "with-candle")]
    #[error("Candle error: {0}")]
    Candle(#[from] candle_core::Error),
}

/// Result type alias for Swarm operations
pub type SwarmResult<T> = std::result::Result<T, SwarmError>;
