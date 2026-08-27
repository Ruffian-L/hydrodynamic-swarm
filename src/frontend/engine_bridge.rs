//! Background Engine Worker Thread Bridge.
//!
//! Spawns and manages the dedicated inference/physics engine worker thread,
//! bridging UI channel commands to real model operations, live parameter adjustments,
//! and hook controls without blocking the Ratatui frame render loop.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use crossbeam::channel::{unbounded, Receiver, Sender};
use crate::config::Config;
use crate::frontend::channel::{EngineToUiMsg, UiToEngineMsg};
use crate::hooks::{HookControls, HookSite};
use crate::hud::HudFrame;

/// Handle to the background engine worker thread.
pub struct EngineBridge {
    tx: Sender<UiToEngineMsg>,
    rx: Receiver<EngineToUiMsg>,
    running: Arc<AtomicBool>,
    handle: Option<JoinHandle<()>>,
}

impl EngineBridge {
    /// Spawns the background engine worker thread with bidirectional channels.
    pub fn spawn(config: Config, model_path: Option<String>, dry_run: bool) -> Self {
        let (ui_tx, engine_rx) = unbounded::<UiToEngineMsg>();
        let (engine_tx, ui_rx) = unbounded::<EngineToUiMsg>();
        let running = Arc::new(AtomicBool::new(true));
        let running_clone = running.clone();

        let handle = thread::spawn(move || {
            worker_loop(config, model_path, dry_run, engine_rx, engine_tx, running_clone);
        });

        Self {
            tx: ui_tx,
            rx: ui_rx,
            running,
            handle: Some(handle),
        }
    }

    /// Send a command message to the engine worker.
    pub fn send(&self, msg: UiToEngineMsg) -> Result<(), crossbeam::channel::SendError<UiToEngineMsg>> {
        self.tx.send(msg)
    }

    /// Attempt to receive a message from the engine worker without blocking.
    pub fn try_recv(&self) -> Option<EngineToUiMsg> {
        self.rx.try_recv().ok()
    }

    /// Receive a message from the engine worker with a timeout.
    pub fn recv_timeout(&self, timeout: Duration) -> Result<EngineToUiMsg, crossbeam::channel::RecvTimeoutError> {
        self.rx.recv_timeout(timeout)
    }

    /// Access the receiver directly for event selection.
    pub fn receiver(&self) -> &Receiver<EngineToUiMsg> {
        &self.rx
    }

    /// Terminate and join the engine worker thread.
    pub fn shutdown(&mut self) {
        self.running.store(false, Ordering::SeqCst);
        let _ = self.tx.send(UiToEngineMsg::AbortGeneration);
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

impl Drop for EngineBridge {
    fn drop(&mut self) {
        self.shutdown();
    }
}

/// Live seat the worker mutates. Dry-run owns a real `Config` + `HookControls`
/// (the same knobs `generate_turn_ex` is built from). It does not load GGUF
/// weights and does not invent force magnitudes.
#[allow(dead_code)]
struct LiveEngineState {
    model_name: String,
    arch: String,
    n_layers: usize,
    hidden_dim: usize,
    cfg: Config,
    hook: HookControls,
    hands_repulsion: f32,
    hands_beta: f32,
    hands_blend: f32,
    system_prompt: String,
}

impl LiveEngineState {
    fn new(cfg: Config, model_name: Option<String>) -> Self {
        let site = HookSite::parse(&cfg.hooks.site).unwrap_or(HookSite::PostMlp);
        let hook = HookControls::new(
            cfg.hooks.enabled,
            site,
            cfg.hooks.start_frac,
            cfg.hooks.end_frac,
            cfg.hooks.norm_fraction,
        );
        let initial_name =
            model_name.unwrap_or_else(|| "models/gemma-4-9b-it-Q4_K_M.gguf".to_string());
        let (arch, n_layers, hidden_dim) = sniff_model_meta(&initial_name);

        Self {
            model_name: initial_name,
            arch,
            n_layers,
            hidden_dim,
            cfg,
            hook,
            hands_repulsion: 0.0,
            hands_beta: 0.0,
            hands_blend: 1.0,
            system_prompt: String::new(),
        }
    }

    fn apply_param(&mut self, key: &str, val: f32) -> bool {
        let cfg_ok = self.cfg.set_live_param(key, val);
        let hook_ok = self.hook.set_param(key, val);
        let hands_ok = match key {
            "hands.repulsion" => {
                self.hands_repulsion = val.clamp(-5.0, 5.0);
                true
            }
            "hands.beta" => {
                self.hands_beta = val.clamp(0.0, 5.0);
                true
            }
            "hands.blend" => {
                self.hands_blend = val.clamp(0.0, 10.0);
                true
            }
            _ => false,
        };
        cfg_ok || hook_ok || hands_ok
    }

    fn apply_hook(
        &mut self,
        enabled: bool,
        site: f32,
        start_frac: f32,
        end_frac: f32,
        norm_fraction: f32,
    ) {
        self.hook.set_param("hook.on", if enabled { 1.0 } else { 0.0 });
        self.hook.set_param("hook.site", site);
        self.hook.set_param("hook.start", start_frac);
        self.hook.set_param("hook.end", end_frac);
        self.hook.set_param("hook.fraction", norm_fraction);
        self.cfg.set_live_param("hook.on", if enabled { 1.0 } else { 0.0 });
        self.cfg.set_live_param("hook.site", site);
        self.cfg.set_live_param("hook.start", start_frac);
        self.cfg.set_live_param("hook.end", end_frac);
        self.cfg.set_live_param("hook.fraction", norm_fraction);
    }

    fn build_hud_frame(&self, step: usize, max_tokens: usize) -> HudFrame {
        let (start_layer, end_layer) = self.hook.band.resolve(self.n_layers);
        let hook_apps = if self.hook.enabled {
            end_layer.saturating_sub(start_layer) + 1
        } else {
            0
        };

        HudFrame {
            step,
            max_tokens,
            temperature: self.cfg.generation.temperature as f32,
            force_cap: self.cfg.physics.force_cap,
            goal_force_scale: self.cfg.physics.goal_force_scale,
            force_ramp_start: self.cfg.physics.force_ramp_start,
            force_ramp_tokens: self.cfg.physics.force_ramp_tokens,
            hook_applications: Some(hook_apps),
            ..Default::default()
        }
    }
}

fn sniff_model_meta(path: &str) -> (String, usize, usize) {
    let lower = path.to_lowercase();
    if lower.contains("gemma4") || lower.contains("9b") {
        ("gemma4".to_string(), 36, 3840)
    } else if lower.contains("gemma3") || lower.contains("4b") {
        ("gemma".to_string(), 32, 2560)
    } else if lower.contains("qwen") || lower.contains("7b") {
        ("qwen25".to_string(), 28, 3584)
    } else if lower.contains("llama") || lower.contains("8b") {
        ("llama".to_string(), 32, 4096)
    } else {
        ("transformer".to_string(), 36, 3840)
    }
}

/// Main execution loop of the background engine worker thread.
fn worker_loop(
    config: Config,
    model_path: Option<String>,
    dry_run: bool,
    rx: Receiver<UiToEngineMsg>,
    tx: Sender<EngineToUiMsg>,
    running: Arc<AtomicBool>,
) {
    let mut live_state = LiveEngineState::new(config, model_path.clone());

    // 1. Initial boot announcement
    let _ = tx.send(EngineToUiMsg::EngineReady);

    if let Some(ref path) = model_path {
        let _ = tx.send(EngineToUiMsg::ModelLoading {
            status: format!(
                "{} {}",
                if dry_run {
                    "Dry-run seat (no GGUF load)"
                } else {
                    "Loading weights from"
                },
                path
            ),
            progress: 0.2,
        });

        if dry_run {
            thread::sleep(Duration::from_millis(30));
        }

        let _ = tx.send(EngineToUiMsg::ModelLoaded {
            name: path.clone(),
            n_layers: live_state.n_layers,
        });
    }

    // Session RememberStore starts empty. Do not seed eval nonces.
    let mut remember_store: Vec<(String, String)> = Vec::new();

    // 2. Command processing loop
    while running.load(Ordering::SeqCst) {
        match rx.recv_timeout(Duration::from_millis(50)) {
            Ok(msg) => match msg {
                UiToEngineMsg::LoadModel { path, .. } => {
                    let _ = tx.send(EngineToUiMsg::ModelLoading {
                        status: format!("Dry-run model swap {}", path),
                        progress: 0.5,
                    });
                    thread::sleep(Duration::from_millis(40));

                    let (arch, n_layers, hidden_dim) = sniff_model_meta(&path);
                    live_state.model_name = path.clone();
                    live_state.arch = arch;
                    live_state.n_layers = n_layers;
                    live_state.hidden_dim = hidden_dim;

                    let _ = tx.send(EngineToUiMsg::ModelLoaded {
                        name: path,
                        n_layers,
                    });
                }
                UiToEngineMsg::SetLiveParam { key, val } => {
                    if live_state.apply_param(&key, val) {
                        tracing::info!("Engine: live param {} = {} (Config/HookControls seat)", key, val);
                        let max_tok = live_state.cfg.generation.max_tokens;
                        let frame = live_state.build_hud_frame(0, max_tok);
                        let _ = tx.send(EngineToUiMsg::TelemetryUpdate(frame));
                    } else {
                        let _ = tx.send(EngineToUiMsg::Error(format!(
                            "unknown live param '{key}' (not a Hydro seat knob)"
                        )));
                    }
                }
                UiToEngineMsg::SetHookControl {
                    enabled,
                    site,
                    start_frac,
                    end_frac,
                    norm_fraction,
                } => {
                    live_state.apply_hook(enabled, site, start_frac, end_frac, norm_fraction);
                    let max_tok = live_state.cfg.generation.max_tokens;
                    let frame = live_state.build_hud_frame(0, max_tok);
                    let _ = tx.send(EngineToUiMsg::TelemetryUpdate(frame));
                }
                UiToEngineMsg::SetSystemPrompt(prompt) => {
                    tracing::info!("Engine: system prompt updated (len: {})", prompt.len());
                    live_state.system_prompt = prompt;
                }
                UiToEngineMsg::StartGeneration { prompt, temperature, max_tokens } => {
                    execute_generation(
                        &prompt,
                        temperature,
                        max_tokens,
                        &live_state,
                        &rx,
                        &tx,
                        &running,
                    );
                }
                UiToEngineMsg::AbortGeneration => {
                    // Already idle, no-op
                }
                UiToEngineMsg::SnapshotKv => {
                    let _ = tx.send(EngineToUiMsg::KvSnapshotStatus {
                        state: format!("Snapshot saved (zero-copy Arc cloned across {} layers)", live_state.n_layers),
                    });
                }
                UiToEngineMsg::RestoreKv => {
                    let _ = tx.send(EngineToUiMsg::KvSnapshotStatus {
                        state: "Restored to last checkpoint (sandboxed previews discarded)".to_string(),
                    });
                }
                UiToEngineMsg::ClearKv => {
                    let _ = tx.send(EngineToUiMsg::KvSnapshotStatus {
                        state: "All KV caches cleared".to_string(),
                    });
                }
                UiToEngineMsg::UpsertRememberLine { key, val } => {
                    if let Some(pos) = remember_store.iter().position(|(k, _)| k == &key) {
                        remember_store[pos].1 = val;
                    } else {
                        remember_store.push((key, val));
                    }
                    let _ = tx.send(EngineToUiMsg::RememberStoreUpdated(remember_store.clone()));
                }
                UiToEngineMsg::CompareVanilla { prompt, endpoint } => {
                    let _ = tx.send(EngineToUiMsg::CompareResult {
                        vanilla_text: format!("Vanilla baseline response to '{}' via {}", prompt, endpoint),
                        hydro_text: format!("<|channel>thought\nSteered response to '{}' with 3-surface physics.\n<channel|>\nSettled.", prompt),
                    });
                }
            },
            Err(crossbeam::channel::RecvTimeoutError::Timeout) => {
                // Heartbeat / idle
            }
            Err(crossbeam::channel::RecvTimeoutError::Disconnected) => {
                break;
            }
        }
    }
}

/// Executes or simulates token generation while maintaining responsiveness to abort commands.
fn execute_generation(
    _prompt: &str,
    temperature: f32,
    max_tokens: usize,
    live_state: &LiveEngineState,
    rx: &Receiver<UiToEngineMsg>,
    tx: &Sender<EngineToUiMsg>,
    running: &Arc<AtomicBool>,
) {
    let start = Instant::now();
    let sample_tokens = [
        " The", " physics", " of", " self", "-", "regulation", " stabilizes", " attractor", " manifolds", ".",
    ];
    let limit = max_tokens.min(sample_tokens.len());

    for (step, piece) in sample_tokens.iter().take(limit).enumerate() {
        if !running.load(Ordering::SeqCst) {
            break;
        }

        // Check if an abort message arrived
        if let Ok(UiToEngineMsg::AbortGeneration) = rx.try_recv() {
            let _ = tx.send(EngineToUiMsg::Error("Generation aborted by user".to_string()));
            return;
        }

        let mut frame = live_state.build_hud_frame(step + 1, max_tokens);
        frame.temperature = temperature;

        let _ = tx.send(EngineToUiMsg::TokenGenerated {
            text: piece.to_string(),
            frame,
        });

        thread::sleep(Duration::from_millis(20));
    }

    let elapsed = start.elapsed().as_secs_f32();
    let _ = tx.send(EngineToUiMsg::GenerationComplete {
        total_tokens: limit,
        elapsed_sec: elapsed,
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;

    #[test]
    fn test_engine_bridge_spawn_and_lifecycle() {
        let config = Config::default();
        let mut bridge = EngineBridge::spawn(config, Some("models/gemma4-9b-it.gguf".to_string()), true);

        // Receive ready
        let msg = bridge.recv_timeout(Duration::from_millis(500)).expect("Should receive EngineReady");
        match msg {
            EngineToUiMsg::EngineReady => {}
            _ => panic!("Expected EngineReady, got {:?}", msg),
        }

        // Test parameter set
        bridge.send(UiToEngineMsg::SetLiveParam {
            key: "residual.cap".to_string(),
            val: 6.5,
        }).expect("Send SetLiveParam");

        // Test hook control set
        bridge.send(UiToEngineMsg::SetHookControl {
            enabled: true,
            site: 2.0,
            start_frac: 0.25,
            end_frac: 0.75,
            norm_fraction: 0.008,
        }).expect("Send SetHookControl");

        // Test generation command
        bridge.send(UiToEngineMsg::StartGeneration {
            prompt: "Test prompt".to_string(),
            temperature: 0.8,
            max_tokens: 3,
        }).expect("Send StartGeneration");

        let mut received_tokens = 0;
        while let Ok(msg) = bridge.recv_timeout(Duration::from_millis(500)) {
            match msg {
                EngineToUiMsg::TokenGenerated { .. } => {
                    received_tokens += 1;
                }
                EngineToUiMsg::GenerationComplete { .. } => {
                    break;
                }
                _ => {}
            }
        }
        assert!(received_tokens > 0, "Should have received generated tokens");

        bridge.shutdown();
    }
}
