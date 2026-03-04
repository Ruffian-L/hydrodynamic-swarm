//! Model + Tokenizer Loader (Phase 1)
//!
//! Reads `[models]` from `config.toml`, opens the GGUF file, auto-detects
//! architecture from its metadata, and returns a ready-to-use
//! `(ModelBackend, Tokenizer, arch)` triple.
//!
//! Keeping loading here means `main` never touches file-system paths directly —
//! change `config.toml` to switch models, no recompile needed.

use anyhow::Result;
use candle_core::quantized::gguf_file;
use candle_core::Device;
use std::io::BufReader;
use tokenizers::Tokenizer;

use crate::config::Config;
use crate::model::ModelBackend;
use crate::{llama, qwen35};

/// Load the GGUF model and its paired tokenizer from paths in `cfg.models`.
///
/// Returns `(model, tokenizer, arch)` where `arch` is the raw architecture
/// string extracted from GGUF metadata (e.g. `"llama"`, `"qwen35"`).
pub fn load(cfg: &Config, device: &Device) -> Result<(ModelBackend, Tokenizer, String)> {
    println!("\n--- Phase 1: Loading Model + Tokenizer ---");

    // ── Model ────────────────────────────────────────────────────────────────
    let model_path = &cfg.models.model_path;
    if !std::path::Path::new(model_path).exists() {
        return Err(anyhow::anyhow!(
            "Model not found: {}\n  → check [models] model_path in config.toml",
            model_path
        ));
    }
    println!("    Model:     {}", model_path);

    let mut file = std::fs::File::open(model_path)?;
    let mut reader = BufReader::new(&mut file);
    let ct = gguf_file::Content::read(&mut reader)?;

    // Auto-detect architecture from GGUF metadata
    let arch = ct
        .metadata
        .get("general.architecture")
        .and_then(|v| match v {
            gguf_file::Value::String(s) => Some(s.as_str()),
            _ => None,
        })
        .unwrap_or("llama")
        .to_string();
    println!("    Arch:      {}", arch);

    let model: ModelBackend = match arch.as_str() {
        "qwen35" => {
            let m = qwen35::Qwen35Model::from_gguf(ct, &mut reader, device)?;
            println!("    Backend:   Qwen3.5");
            ModelBackend::Qwen35(m)
        }
        _ => {
            let m = llama::ModelWeights::from_gguf(ct, &mut reader, device)?;
            println!("    Backend:   Llama");
            ModelBackend::Llama(m)
        }
    };

    // ── Tokenizer ────────────────────────────────────────────────────────────
    let tok_path = &cfg.models.tokenizer_path;
    if !std::path::Path::new(tok_path).exists() {
        return Err(anyhow::anyhow!(
            "Tokenizer not found: {}\n  → check [models] tokenizer_path in config.toml\n  \
             IMPORTANT: tokenizer must match the model family (Llama ≠ Qwen)",
            tok_path
        ));
    }
    let tokenizer =
        Tokenizer::from_file(tok_path).map_err(|e| anyhow::anyhow!("tokenizer: {}", e))?;
    println!("    Tokenizer: {}", tok_path);

    Ok((model, tokenizer, arch))
}
