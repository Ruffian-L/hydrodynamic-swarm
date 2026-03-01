#![allow(dead_code)]
//! JSONL Telemetry Logger
//!
//! Logs every generation step and session summary to `logs/` as JSONL.
//! Each session file: `logs/{date}_{time}_{label}.jsonl`
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
    pub prompt: String,
    pub dt: f32,
    pub viscosity: f32,
    pub kernel_sigma: f32,
    pub embedding_dim: usize,
    pub field_points: usize,
    pub model: String,
    pub model_variant: String,
    pub backend: String,
    pub splat_sigma: f32,
    pub splat_alpha: f32,
    pub force_cap: f32,
    pub temperature: f32,
    pub min_splat_dist: f32,
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

/// Top-level log entry -- one per line in the JSONL file
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
    log_path: PathBuf,
}

impl SessionLogger {
    /// Create a new session logger with a descriptive label.
    /// Filename: `logs/{YYYY-MM-DD}_{HH-MM-SS}_{label}.jsonl`
    pub fn new(label: &str) -> std::io::Result<Self> {
        let log_dir = Path::new("logs");
        fs::create_dir_all(log_dir)?;

        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        // Manual UTC date/time formatting (no chrono dependency)
        let secs_per_day: u64 = 86400;
        let days = now / secs_per_day;
        let day_secs = now % secs_per_day;
        let hours = day_secs / 3600;
        let minutes = (day_secs % 3600) / 60;
        let seconds = day_secs % 60;

        // Compute year/month/day from days since epoch
        let (year, month, day) = days_to_date(days);

        let date_str = format!("{:04}-{:02}-{:02}", year, month, day);
        let time_str = format!("{:02}-{:02}-{:02}", hours, minutes, seconds);

        // Sanitize label for filename
        let safe_label: String = label
            .chars()
            .map(|c| if c.is_alphanumeric() || c == '-' || c == '_' { c } else { '_' })
            .collect();

        let session_id = format!("{}_{}", date_str, safe_label);
        let filename = format!("{}_{}_{}.jsonl", date_str, time_str, safe_label);
        let log_path = log_dir.join(&filename);
        let file = fs::File::create(&log_path)?;

        println!("    Logging to: {}", log_path.display());

        Ok(Self {
            file,
            session_id,
            deltas: Vec::new(),
            log_path,
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
        self.log_path.clone()
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

/// Convert days since Unix epoch to (year, month, day).
fn days_to_date(mut days: u64) -> (u64, u64, u64) {
    // Simplified Gregorian calendar calculation
    let mut year = 1970;
    loop {
        let days_in_year = if is_leap(year) { 366 } else { 365 };
        if days < days_in_year {
            break;
        }
        days -= days_in_year;
        year += 1;
    }
    let months_days: [u64; 12] = if is_leap(year) {
        [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    } else {
        [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    };
    let mut month = 1;
    for &md in &months_days {
        if days < md {
            break;
        }
        days -= md;
        month += 1;
    }
    (year, month, days + 1)
}

fn is_leap(year: u64) -> bool {
    (year % 4 == 0 && year % 100 != 0) || year % 400 == 0
}
