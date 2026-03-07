#![allow(dead_code)]
//! JSONL Telemetry Logger
//!
//! Logs every generation step and session summary to `logs/` as JSONL.
//! Each session file: `logs/{date}_{time}_{label}.jsonl`
//!
//! v2 note: when we unify splat memory across prompts (Emergent Synthesis),
//! the logger will track which domain each splat originated from,
//! enabling cross-domain influence analysis.

use rusqlite::{params, Connection, Result as SqlResult};
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
    pub grad_force_mag: f32,
    pub splat_force_mag: f32,
    pub goal_force_mag: f32,
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
    pub model_variant: String,
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
    model_variant: String,
    deltas: Vec<f32>,
    log_path: PathBuf,
    taco: TacoDb,
}

impl SessionLogger {
    /// Create a new session logger with a descriptive label.
    /// Filename: `logs/{YYYY-MM-DD}_{HH-MM-SS}_{label}.jsonl`
    pub fn new(label: &str, model_variant: &str) -> std::io::Result<Self> {
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
            .map(|c| {
                if c.is_alphanumeric() || c == '-' || c == '_' {
                    c
                } else {
                    '_'
                }
            })
            .collect();

        let session_id = format!("{}_{}_{}", date_str, time_str, safe_label);
        let filename = format!("{}_{}_{}.jsonl", date_str, time_str, safe_label);
        let log_path = log_dir.join(&filename);
        let file = fs::File::create(&log_path)?;

        println!("    Logging to: {}", log_path.display());

        let taco = match TacoDb::new_persistent("logs/taco.db") {
            Ok(db) => {
                println!("    TACO DB: persistent logging enabled (logs/taco.db)");
                db
            }
            Err(e) => {
                eprintln!("    TACO DB init failed: {}, falling back to in-memory", e);
                TacoDb::new_in_memory().expect("in-memory TACO DB failed")
            }
        };

        Ok(Self {
            file,
            session_id,
            model_variant: model_variant.to_string(),
            deltas: Vec::new(),
            log_path,
            taco,
        })
    }

    /// Log the session config
    pub fn log_config(&mut self, config: SessionConfig) -> std::io::Result<()> {
        let entry = LogEntry {
            timestamp: self.now_str(),
            session_id: self.session_id.clone(),
            model_variant: self.model_variant.clone(),
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
        let _ = self.taco.log_step(&self.session_id, &step);
        let entry = LogEntry {
            timestamp: self.now_str(),
            session_id: self.session_id.clone(),
            model_variant: self.model_variant.clone(),
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
            model_variant: self.model_variant.clone(),
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

    /// Get the session ID string
    pub fn session_id(&self) -> &str {
        &self.session_id
    }

    /// Get TACO DB stats
    pub fn taco_stats(&self) -> String {
        self.taco
            .get_stats()
            .unwrap_or_else(|e| format!("TACO DB error: {}", e))
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
pub fn days_to_date(mut days: u64) -> (u64, u64, u64) {
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
    (year.is_multiple_of(4) && !year.is_multiple_of(100)) || year.is_multiple_of(400)
}

/// TACO DB - Telemetry, Action, Context, Observation SQLite backend
/// Supports in-memory (:memory:) or persistent file storage for structured querying
/// of session logs, splats, and cross-session memory insights.
pub struct TacoDb {
    conn: Connection,
}

impl TacoDb {
    /// Create in-memory TACO DB for testing/fast sessions
    pub fn new_in_memory() -> SqlResult<Self> {
        let conn = Connection::open_in_memory()?;
        Self::init_schema(&conn)?;
        Ok(Self { conn })
    }

    /// Create persistent TACO DB (e.g. "taco.db")
    pub fn new_persistent(path: &str) -> SqlResult<Self> {
        let conn = Connection::open(path)?;
        Self::init_schema(&conn)?;
        Ok(Self { conn })
    }

    fn init_schema(conn: &Connection) -> SqlResult<()> {
        conn.execute(
            "CREATE TABLE IF NOT EXISTS taco_entries (
                id INTEGER PRIMARY KEY,
                timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
                session_id TEXT,
                entry_type TEXT,
                token_id INTEGER,
                token_text TEXT,
                steering_delta REAL,
                residual_norm REAL,
                data TEXT
            )",
            [],
        )?;
        Ok(())
    }

    /// Log a step entry to TACO DB
    pub fn log_step(&self, session_id: &str, step: &StepEntry) -> SqlResult<usize> {
        self.conn.execute(
            "INSERT INTO taco_entries (session_id, entry_type, token_id, token_text, steering_delta, residual_norm, data) 
             VALUES (?1, 'step', ?2, ?3, ?4, ?5, ?6)",
            params![session_id, step.token_id, step.token_text, step.steering_delta, step.residual_norm, 
                   format!("grad_force:{} splat_force:{}", step.grad_force_mag, step.splat_force_mag)],
        )
    }

    /// Query recent steps for a session
    pub fn query_steps(
        &self,
        session_id: &str,
        limit: usize,
    ) -> SqlResult<Vec<(usize, String, f32)>> {
        let mut stmt = self.conn.prepare(
            "SELECT token_id, token_text, steering_delta FROM taco_entries 
             WHERE session_id = ?1 AND entry_type = 'step' ORDER BY id DESC LIMIT ?2",
        )?;
        let rows = stmt.query_map(params![session_id, limit], |row| {
            Ok((row.get(0)?, row.get(1)?, row.get(2)?))
        })?;
        rows.collect()
    }

    /// Get session stats
    pub fn get_stats(&self) -> SqlResult<String> {
        let count: usize = self
            .conn
            .query_row("SELECT COUNT(*) FROM taco_entries", [], |row| row.get(0))?;
        Ok(format!("TACO DB: {} entries stored", count))
    }
}

#[cfg(test)]
mod taco_tests {
    use super::*;

    #[test]
    fn test_taco_db_in_memory() {
        let db = TacoDb::new_in_memory().unwrap();
        let step = StepEntry {
            step: 1,
            token_id: 42,
            token_text: "test".to_string(),
            steering_delta: 0.5,
            residual_norm: 1.2,
            grad_force_mag: 0.3,
            splat_force_mag: 0.8,
            goal_force_mag: 0.4,
        };
        let session_id = "test_session_001";
        db.log_step(session_id, &step).unwrap();

        let steps = db.query_steps(session_id, 10).unwrap();
        assert_eq!(steps.len(), 1);
        assert_eq!(steps[0].0, 42);
        assert_eq!(steps[0].1, "test");
        assert_eq!(steps[0].2, 0.5);

        let stats = db.get_stats().unwrap();
        assert!(stats.contains("1 entries"));
        println!("TACO DB test passed: {}", stats);
    }
}
