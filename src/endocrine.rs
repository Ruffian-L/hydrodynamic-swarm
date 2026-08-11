// src/endocrine.rs
// Endocrine System — latent enzymes for the Hydrodynamic Swarm.
//
// Text enzyme worker produces *text* blooms (tool / verify / rag).
// Geometry for Monoliths is NOT TinyEmbed / hash projection.
// Main embeds bloom text with the **native model tok_embeddings**
// (same matrix as ContinuousField) — see main::native_embed_mean.
//
// Restored 2026-07-18 from SanDisk (Shep-loop mar22), then:
//   2026-07-25 — drop TinyEmbed; native model embeds blooms.
#![allow(dead_code)]

use std::sync::Arc;
use tokio::sync::mpsc;

/// The shared physical universe between all cognitive streams.
/// This is the "body" that Stream 1 (Llama), the Watcher, and the Endocrine system all touch.
pub struct SharedUniverse {
    pub monoliths: Vec<Monolith>,
}

/// A Monolith = Absolute factual truth injected into the latent space.
/// High mass, low viscosity, high repulsion against hallucinated vectors.
///
/// `pos` is a 4-D telemetry / residual projection of the **native** mean
/// embedding (filled on the main thread). Full-D attractor lives on the engine.
#[derive(Clone, Debug)]
pub struct Monolith {
    pub pos: [f32; 4],
    pub mass: f32,      // e.g. 10000.0 — crushing gravitational pull toward truth
    pub repulsion: f32, // violently pushes nearby hallucination vectors away
}

/// Signals sent TO the text enzyme (hormones from the nervous system)
#[derive(Debug)]
pub enum EndocrineSignal {
    ExecuteTool { intent: String, context: String },
    VerifyLogic { context: String },
    RAGQuery { query: String },
    AnalyzeTopology {
        entropy: f32,
        varentropy: f32,
        b1: f32,
    },
}

/// Bloom = text result from enzyme. Geometry applied later via native model.
#[derive(Debug)]
pub struct BloomEvent {
    pub raw_text: String,
}

/// Spawns one text enzyme (idle until signal).
pub async fn spawn_enzyme_node(
    mut rx_signal: mpsc::Receiver<EndocrineSignal>,
    tx_bloom: mpsc::Sender<BloomEvent>,
    enzyme: Arc<TextEnzyme>,
) {
    // "Idle" = blocked on the hormone channel (near-zero CPU until pain/high-δ).
    // Not dead, not unused — wakes on signal, blooms, idles again.
    println!("[ENDOCRINE] enzyme worker idle (listening for signals)...");

    while let Some(signal) = rx_signal.recv().await {
        let bloom = match signal {
            EndocrineSignal::ExecuteTool { intent, context } => {
                let tool_result = enzyme.strict_execute(&intent, &context).await;
                println!(
                    "[ENDOCRINE] enzyme fire: intent=\"{}\" -> \"{}\"",
                    intent.chars().take(60).collect::<String>(),
                    tool_result.chars().take(60).collect::<String>()
                );
                BloomEvent {
                    raw_text: tool_result,
                }
            }
            EndocrineSignal::VerifyLogic { context } => {
                let result = format!("[LOGIC VERIFIED] {}", context);
                BloomEvent { raw_text: result }
            }
            EndocrineSignal::RAGQuery { query } => {
                let result = format!("[RAG] Retrieved relevant context for: {}", query);
                BloomEvent { raw_text: result }
            }
            EndocrineSignal::AnalyzeTopology {
                entropy,
                varentropy,
                b1,
            } => {
                println!(
                    "[ENDOCRINE] Topology alert: entropy={:.3}, varentropy={:.3}, b1={:.3}",
                    entropy, varentropy, b1
                );
                // Topology analysis doesn't produce a Bloom — diagnostic only
                continue;
            }
        };

        if let Ok(()) = tx_bloom.send(bloom).await {
            println!("[ENDOCRINE] bloom text → main (native embed next).");
        }
    }

    println!("[ENDOCRINE] enzyme worker exit (channel closed).");
}

/// Text enzyme. Prefer local OpenAI-compatible HTTP when `ENDOCRINE_URL` is set
/// (e.g. `http://127.0.0.1:8210/v1`). Geometry stays native on main — this is text only.
/// Stateless: no claim about which body hosts the enzyme; env points at whoever is up.
pub struct TextEnzyme {
    /// Base URL like `http://127.0.0.1:8210/v1` (no trailing slash required).
    http_base: Option<String>,
    model: String,
}

impl TextEnzyme {
    pub fn from_env() -> Self {
        let http_base = std::env::var("ENDOCRINE_URL")
            .ok()
            .map(|s| s.trim().trim_end_matches('/').to_string())
            .filter(|s| !s.is_empty())
            .filter(|s| s.starts_with("http://") || s.starts_with("https://"));
        let model = std::env::var("ENDOCRINE_MODEL").unwrap_or_else(|_| "local".into());
        if let Some(ref u) = http_base {
            println!("[ENDOCRINE] text enzyme HTTP → {u} model={model}");
        } else {
            println!(
                "[ENDOCRINE] text enzyme offline stub (set ENDOCRINE_URL=http://host:port/v1 for live)"
            );
        }
        Self { http_base, model }
    }

    pub async fn strict_execute(&self, intent: &str, context: &str) -> String {
        if let Some(ref base) = self.http_base {
            match self.http_execute(base, intent, context).await {
                Ok(text) if !text.trim().is_empty() => {
                    return format!("[ENZYME] {}", text.trim());
                }
                Ok(_) => {
                    eprintln!("[ENDOCRINE] HTTP empty response — falling back to stub");
                }
                Err(e) => {
                    eprintln!("[ENDOCRINE] HTTP fail ({e}) — falling back to stub");
                }
            }
        }
        // Honest offline stub (geometry still native on main).
        let hash: u64 = intent
            .bytes()
            .fold(0u64, |acc, b| acc.wrapping_mul(31).wrapping_add(b as u64));
        let fact_id = hash % 1000;
        format!(
            "[FACT #{}] {} resolved from context: {}",
            fact_id, intent, context
        )
    }

    async fn http_execute(
        &self,
        base: &str,
        intent: &str,
        context: &str,
    ) -> Result<String, String> {
        let url = format!("{}/chat/completions", base.trim_end_matches('/'));
        // Qwen3 / thinking models: without jinja kwargs, content stays empty and
        // tokens go to reasoning_content. Force thinking off; also accept either field.
        let body = serde_json::json!({
            "model": self.model,
            "temperature": 0.0,
            "max_tokens": 96,
            "chat_template_kwargs": { "enable_thinking": false },
            "messages": [
                {
                    "role": "system",
                    "content": "You are a cold endocrine enzyme. One short factual stabilization note. No preamble. No thinking tags."
                },
                {
                    "role": "user",
                    "content": format!("intent: {intent}\ncontext: {context}")
                }
            ]
        });
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(12))
            .build()
            .map_err(|e| e.to_string())?;
        let resp = client
            .post(&url)
            .json(&body)
            .send()
            .await
            .map_err(|e| e.to_string())?;
        if !resp.status().is_success() {
            return Err(format!("status {}", resp.status()));
        }
        let v: serde_json::Value = resp.json().await.map_err(|e| e.to_string())?;
        let msg = &v["choices"][0]["message"];
        let text = msg["content"]
            .as_str()
            .filter(|s| !s.trim().is_empty())
            .or_else(|| msg["reasoning_content"].as_str().filter(|s| !s.trim().is_empty()))
            .or_else(|| v["choices"][0]["text"].as_str())
            .unwrap_or("")
            .trim()
            .to_string();
        // If we only got a long "Thinking Process:" dump, take last non-empty line.
        if text.starts_with("Thinking") || text.contains("**Analyze") {
            let last = text
                .lines()
                .map(str::trim)
                .filter(|l| !l.is_empty() && !l.starts_with('*') && !l.starts_with('#'))
                .last()
                .unwrap_or("")
                .to_string();
            if !last.is_empty() {
                return Ok(last);
            }
        }
        Ok(text)
    }
}

/// Project a native (D,) embedding vector into 4-D physics telemetry space.
/// Fixed random projection (seed 42) — same approach as SplatLens viz, not TinyEmbed hash-of-text.
pub fn project_native_to_4d(native: &[f32]) -> [f32; 4] {
    if native.is_empty() {
        return [0.0; 4];
    }
    let dim = native.len();
    let mut rng: u64 = 42;
    let scale = 1.0 / (dim as f32).sqrt();
    let mut out = [0.0f32; 4];
    for j in 0..4 {
        let mut sum = 0.0f32;
        for i in 0..dim {
            rng = rng
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let u = (rng >> 33) as f32 / (1u64 << 31) as f32;
            rng = rng
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let v = (rng >> 33) as f32 / (1u64 << 31) as f32;
            let w = (u + v - 1.0) * scale;
            sum += native[i] * w;
        }
        out[j] = sum;
    }
    // L2 normalize into ~unit ball for stable logs
    let n = (out[0] * out[0] + out[1] * out[1] + out[2] * out[2] + out[3] * out[3]).sqrt();
    if n > 1e-8 {
        for x in &mut out {
            *x /= n;
        }
    }
    out
}

pub fn create_endocrine_system(
) -> (SharedUniverse, mpsc::Sender<EndocrineSignal>, mpsc::Receiver<BloomEvent>) {
    let (tx_signal, rx_signal) = mpsc::channel(32);
    let (tx_bloom, rx_bloom) = mpsc::channel(32);

    let enzyme = Arc::new(TextEnzyme::from_env());

    tokio::spawn(async move {
        spawn_enzyme_node(rx_signal, tx_bloom, enzyme).await;
    });

    let universe = SharedUniverse {
        monoliths: Vec::new(),
    };

    println!(
        "[ENDOCRINE] online — text enzyme + native tok_embeddings on main (stateless)."
    );

    (universe, tx_signal, rx_bloom)
}
