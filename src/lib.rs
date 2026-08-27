//! Model-loading and forward-pass instrumentation shared between the swarm binary and the
//! `jlens-gguf` sidecar.
//!
//! This library is deliberately thin. Only the modules a second crate needs to load a GGUF
//! model, run it with hooks installed, and read out logits live here; the swarm's physics
//! (`field`, `memory`, `niodoo`, `ocean`, …) stays private to `main.rs`.
//!
//! The module graph is acyclic and shallow — `config` depends on nothing, `dim_assert` on
//! `config`, `hooks` on `dim_assert`, the three model forks and `jacobian` on `hooks`, and
//! `loader` on the model forks. Keep it that way: the point of the split is that a sidecar
//! can load hydro's models without dragging in the swarm.
//!
//! ## Licenses & attributions
//!
//! - Our code: MIT-0 (LICENSE)
//! - Candle loader code (`llama.rs`): Apache-2.0 OR MIT — NOT the same as model weights
//! - Model weights carry their own terms; see NOTICE in the repo root.

// The binary uses one subset of these APIs and the sidecar another, so items that look
// dead from inside the library are live across the workspace.
#![allow(dead_code)]

pub mod config;
pub mod dim_assert;
pub mod gemma;
pub mod gemma4;
pub mod hooks;
pub mod jacobian;
pub mod llama;
pub mod loader;
pub mod qwen35;
