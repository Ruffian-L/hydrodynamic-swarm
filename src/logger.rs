#![allow(dead_code)]
//! JSONL Telemetry Logger
//!
//! Logs every generation step and session summary to `logs/` as JSONL.
//! Each session gets its own file: `logs/YYYY-MM-DD_session_NNN.jsonl`
//!
//! v2 note: when we unify splat memory across prompts (Emergent Synthesis),
//! the logger will track which domain each splat originated from,
//! enabling cross-domain influence analysis.

use serde::Serialize;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

/// Per-step telemetry entry
#[derive(Serialize)]
pub struct StepEntry {
    pub step: usize,
    pub token_id: u32,
    pub token_text: String,
    pub steering_delta: f32,
    pub residual_norm: f32,
}

/// Session config snapshot
#[derive(Serialize)]
pub struct SessionConfig {
    pub dt: f32,
    pub viscosity: f32,
    pub kernel_sigma: f32,
    pub embedding_dim: usize,
    pub field_points: usize,
    pub model: String,
    pub backend: String,
}

/// Final session summary
#[derive(Serialize)]
pub struct SessionSummary {
    pub prompt: String,
    pub prompt_token_count: usize,
    pub generated_token_count: usize,
    pub goal_attractor_norm: f32,
    pub splat_count_before: usize,
    pub splat_count_after: usize,
    pub splat_type_added: String, // "pleasure", "pain", or "none"
    pub decoded_output: String,
    pub delta_min: f32,
    pub delta_max: f32,
    pub delta_mean: f32,
}

/// Top-level log entry — one per line in the JSONL file
#[derive(Serialize)]
pub struct LogEntry {
    pub timestamp: String,
    pub session_id: String,
    pub entry_type: String, // "config", "step", "summary"
    #[serde(skip_serializing_if = "Option::is_none")]
    pub config: Option<SessionConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub step: Option<StepEntry>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub summary: Option<SessionSummary>,
}

pub struct SessionLogger {
    file: fs::File,
    session_id: String,
    deltas: Vec<f32>,
}

impl SessionLogger {
    /// Create a new session logger. Creates `logs/` dir if needed.
    pub fn new() -> std::io::Result<Self> {
        let log_dir = Path::new("logs");
        fs::create_dir_all(log_dir)?;

        // Generate session ID: YYYY-MM-DD_HH-MM-SS
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        // Simple date formatting without chrono dependency
        let session_id = format!("session_{}", now);

        let path = log_dir.join(format!("{}.jsonl", &session_id));
        let file = fs::File::create(&path)?;

        println!("    📝 Logging to: {}", path.display());

        Ok(Self {
            file,
            session_id,
            deltas: Vec::new(),
        })
    }

    /// Log the session config
    pub fn log_config(&mut self, config: SessionConfig) -> std::io::Result<()> {
        let entry = LogEntry {
            timestamp: self.now_str(),
            session_id: self.session_id.clone(),
            entry_type: "config".to_string(),
            config: Some(config),
            step: None,
            summary: None,
        };
        self.write_entry(&entry)
    }

    /// Log a single generation step
    pub fn log_step(&mut self, step: StepEntry) -> std::io::Result<()> {
        self.deltas.push(step.steering_delta);
        let entry = LogEntry {
            timestamp: self.now_str(),
            session_id: self.session_id.clone(),
            entry_type: "step".to_string(),
            config: None,
            step: Some(step),
            summary: None,
        };
        self.write_entry(&entry)
    }

    /// Log final session summary
    pub fn log_summary(&mut self, mut summary: SessionSummary) -> std::io::Result<()> {
        // Compute delta stats from collected deltas
        if !self.deltas.is_empty() {
            summary.delta_min = self.deltas.iter().cloned().fold(f32::INFINITY, f32::min);
            summary.delta_max = self
                .deltas
                .iter()
                .cloned()
                .fold(f32::NEG_INFINITY, f32::max);
            summary.delta_mean = self.deltas.iter().sum::<f32>() / self.deltas.len() as f32;
        }
        let entry = LogEntry {
            timestamp: self.now_str(),
            session_id: self.session_id.clone(),
            entry_type: "summary".to_string(),
            config: None,
            step: None,
            summary: Some(summary),
        };
        self.write_entry(&entry)
    }

    /// Get the log file path
    pub fn path(&self) -> PathBuf {
        Path::new("logs").join(format!("{}.jsonl", &self.session_id))
    }

    fn write_entry(&mut self, entry: &LogEntry) -> std::io::Result<()> {
        let json = serde_json::to_string(entry).unwrap();
        writeln!(self.file, "{}", json)?;
        self.file.flush()
    }

    fn now_str(&self) -> String {
        let secs = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        format!("{}", secs)
    }
}
