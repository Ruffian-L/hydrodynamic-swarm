//! Seat remember-store JSONL: `{"payload":"key=value","key":"...","value":"..."}`.
//! Same shape as niodoo-adaptive-agency / niodoo `partner/remember.rs`.

use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RememberLine {
    pub payload: String,
    #[serde(default)]
    pub key: String,
    #[serde(default)]
    pub value: String,
}

pub fn remember_key(payload: &str) -> String {
    let compact = payload.trim();
    if let Some((key, _)) = compact.split_once('=') {
        return key.trim().to_ascii_lowercase();
    }
    compact
        .split_whitespace()
        .next()
        .unwrap_or(compact)
        .trim()
        .to_ascii_lowercase()
}

pub fn remember_value(payload: &str) -> String {
    payload
        .split_once('=')
        .map(|(_, value)| value.trim().to_string())
        .unwrap_or_default()
}

pub fn load_remember_lines(path: &Path) -> std::io::Result<Vec<RememberLine>> {
    if !path.exists() {
        return Ok(Vec::new());
    }
    let body = std::fs::read_to_string(path)?;
    let mut out = Vec::new();
    for line in body.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        if let Ok(parsed) = serde_json::from_str::<RememberLine>(trimmed) {
            out.push(parsed);
            continue;
        }
        if trimmed.contains('=') {
            out.push(RememberLine {
                payload: trimmed.to_string(),
                key: remember_key(trimmed),
                value: remember_value(trimmed),
            });
        }
    }
    Ok(out)
}

pub fn save_remember_lines(path: &Path, lines: &[RememberLine]) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }
    let mut out = String::new();
    for line in lines {
        out.push_str(
            &serde_json::to_string(line)
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?,
        );
        out.push('\n');
    }
    std::fs::write(path, out)
}

#[derive(Debug, Clone, Default)]
pub struct RememberStore {
    pub path: PathBuf,
    pub lines: Vec<RememberLine>,
}

impl RememberStore {
    pub fn open(path: &Path) -> std::io::Result<Self> {
        let lines = load_remember_lines(path)?;
        Ok(Self {
            path: path.to_path_buf(),
            lines,
        })
    }

    pub fn upsert(&mut self, payload: &str) -> std::io::Result<RememberLine> {
        let key = remember_key(payload);
        let value = remember_value(payload);
        let stored = if payload.contains('=') {
            payload.trim().to_string()
        } else {
            format!("{key}={value}")
        };
        if let Some(existing) = self.lines.iter_mut().find(|l| l.key == key) {
            existing.payload = stored;
            existing.value = value;
        } else {
            self.lines.push(RememberLine {
                payload: stored,
                key: key.clone(),
                value,
            });
        }
        if !self.path.as_os_str().is_empty() {
            save_remember_lines(&self.path, &self.lines)?;
        }
        Ok(self.get(&key).cloned().expect("upsert just wrote this key"))
    }

    pub fn reload(&mut self) -> std::io::Result<()> {
        self.lines = load_remember_lines(&self.path)?;
        Ok(())
    }

    pub fn get(&self, key: &str) -> Option<&RememberLine> {
        let k = key.trim().to_ascii_lowercase();
        self.lines.iter().find(|l| l.key == k)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn save_reload_survives_drop() {
        let dir = std::env::temp_dir().join(format!(
            "hydro_remember_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("seat_remember.jsonl");
        {
            let mut store = RememberStore::open(&path).unwrap();
            store.upsert("tuesday-boy=13/27").unwrap();
        }
        let mut revived = RememberStore::open(&path).unwrap();
        revived.reload().unwrap();
        let line = revived
            .get("tuesday-boy")
            .expect("key survived process drop");
        assert_eq!(line.value, "13/27");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn remember_survives_os_process_exit() {
        let dir = std::env::temp_dir().join(format!(
            "hydro_remember_proc_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("seat_remember.jsonl");
        if std::env::var("HYDRO_REMEMBER_CHILD").as_deref() == Ok("1") {
            let seat = std::env::var("HYDRO_REMEMBER_SEAT").expect("child seat path");
            let mut store = RememberStore::open(std::path::Path::new(&seat)).unwrap();
            store.upsert("loop-seat=alive").unwrap();
            return;
        }
        let exe = std::env::current_exe().expect("test exe");
        let status = std::process::Command::new(&exe)
            .arg("remember_store::tests::remember_survives_os_process_exit")
            .arg("--exact")
            .env("HYDRO_REMEMBER_CHILD", "1")
            .env("HYDRO_REMEMBER_SEAT", &path)
            .status()
            .expect("spawn remember child");
        assert!(
            status.success(),
            "child hydro remember process failed: {status}"
        );
        let store = RememberStore::open(&path).unwrap();
        let line = store
            .get("loop-seat")
            .expect("key survived OS process exit");
        assert_eq!(line.value, "alive");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
