//! Bidirectional UI ↔ Engine Channel Message Contracts.
//!
//! Enforces strict separation between the Ratatui / crossterm rendering thread
//! and the background physics/generation worker thread.

use crate::hud::HudFrame;

/// Commands sent from the UI thread to the background Engine worker thread.
#[derive(Debug, Clone)]
pub enum UiToEngineMsg {
    /// Load or hot-swap a GGUF model and tokenizer from disk.
    LoadModel {
        path: String,
        tokenizer: Option<String>,
    },
    /// Live update a parameter in NiodooEngine, LogitChain, or Sampling config.
    SetLiveParam {
        key: String,
        val: f32,
    },
    /// Adjust forward layer hook parameters (HookControls).
    SetHookControl {
        enabled: bool,
        site: f32,
        start_frac: f32,
        end_frac: f32,
        norm_fraction: f32,
    },
    /// Set or inject a new active system prompt.
    SetSystemPrompt(String),
    /// Initiate a generation turn with prompt, temperature, and max tokens.
    StartGeneration {
        prompt: String,
        temperature: f32,
        max_tokens: usize,
    },
    /// Immediately abort any active generation loop.
    AbortGeneration,
    /// Snapshot the active KV cache state (Choice-Driven KV cache).
    SnapshotKv,
    /// Restore the KV cache state from the last snapshot.
    RestoreKv,
    /// Clear the KV cache across all layers.
    ClearKv,
    /// Upsert or modify an entry in the persistent RememberStore.
    UpsertRememberLine {
        key: String,
        val: String,
    },
    /// Trigger side-by-side comparison with a vanilla endpoint.
    CompareVanilla {
        prompt: String,
        endpoint: String,
    },
}

/// Events and telemetry emitted from the Engine worker thread to the UI thread.
#[derive(Debug, Clone)]
pub enum EngineToUiMsg {
    /// Sent when the engine worker thread is booted and ready for commands.
    EngineReady,
    /// Progress notification during weight loading or initialization.
    ModelLoading {
        status: String,
        progress: f32,
    },
    /// Sent when a model has been successfully loaded into memory.
    ModelLoaded {
        name: String,
        n_layers: usize,
    },
    /// Per-token emission during generation with attached scalar telemetry.
    TokenGenerated {
        text: String,
        frame: HudFrame,
    },
    /// Notification that generation finished normally or reached EOS.
    GenerationComplete {
        total_tokens: usize,
        elapsed_sec: f32,
    },
    /// General error from backend operations (e.g. missing file, CUDA error).
    Error(String),
    /// Result payload from side-by-side vanilla vs hydro comparison.
    CompareResult {
        vanilla_text: String,
        hydro_text: String,
    },
    /// Updated list of all key-value entries in RememberStore.
    RememberStoreUpdated(Vec<(String, String)>),
    /// Status message from KV cache snapshot / restore / clear operations.
    KvSnapshotStatus {
        state: String,
    },
    /// Periodic or live telemetry update without token emission.
    TelemetryUpdate(HudFrame),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ui_to_engine_msg_variants() {
        let msg = UiToEngineMsg::StartGeneration {
            prompt: "Hello world".to_string(),
            temperature: 0.8,
            max_tokens: 128,
        };
        match &msg {
            UiToEngineMsg::StartGeneration { prompt, temperature, max_tokens } => {
                assert_eq!(prompt, "Hello world");
                assert_eq!(*temperature, 0.8);
                assert_eq!(*max_tokens, 128);
            }
            _ => panic!("Unexpected message variant"),
        }
    }

    #[test]
    fn test_engine_to_ui_msg_variants() {
        let frame = HudFrame {
            step: 10,
            temperature: 0.85,
            force_cap: 5.0,
            ..Default::default()
        };
        let msg = EngineToUiMsg::TokenGenerated {
            text: "token".to_string(),
            frame,
        };
        match &msg {
            EngineToUiMsg::TokenGenerated { text, frame } => {
                assert_eq!(text, "token");
                assert_eq!(frame.step, 10);
                assert_eq!(frame.force_cap, 5.0);
            }
            _ => panic!("Unexpected message variant"),
        }
    }
}
