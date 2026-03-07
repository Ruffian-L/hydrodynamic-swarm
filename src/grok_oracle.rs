//! Grok 4.20 Oracle Integration for Hydrodynamic Swarm
//! Native tool calling for structured decisions in the hydro storm / Niodoo physics engine

use reqwest::Client;
use serde_json::{json, Value};
use anyhow::{anyhow, Result};
use tracing::info;
use std::env;

pub struct GrokOracle {
    client: Client,
    api_key: String,
}

impl GrokOracle {
    pub fn new() -> Result<Self> {
        let api_key = env::var("XAI_API_KEY")
            .map_err(|_| anyhow!("XAI_API_KEY env var missing. export XAI_API_KEY=..."))?;
        
        Ok(Self {
            client: Client::new(),
            api_key,
        })
    }

    /// Classify or make structured decision using Grok tool calling
    pub async fn classify(&self, prompt: &str) -> Result<String> {
        let payload = json!({
            "model": "grok-beta",
            "messages": [{
                "role": "system",
                "content": "You are the oracle node in a hydrodynamic swarm. Use the tool for strict structured output."
            }, {
                "role": "user",
                "content": prompt
            }],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "swarm_decision",
                    "description": "Output structured decision for physics/memory steering.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "decision": { "type": "string", "description": "The decision/action" },
                            "justification": { "type": "string", "description": "Physical/thermodynamic justification" }
                        },
                        "required": ["decision", "justification"]
                    }
                }
            }],
            "tool_choice": { "type": "function", "function": { "name": "swarm_decision" } }
        });

        info!("Calling Grok 4.20 for structured decision...");

        let res = self.client.post("https://api.x.ai/v1/chat/completions")
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&payload)
            .send()
            .await?;

        let body: Value = res.json().await?;

        // Extract tool call arguments
        if let Some(choices) = body.get("choices").and_then(|c| c.as_array()) {
            if let Some(choice) = choices.first() {
                if let Some(msg) = choice.get("message") {
                    if let Some(tool_calls) = msg.get("tool_calls").and_then(|t| t.as_array()) {
                        if let Some(tc) = tool_calls.first() {
                            if let Some(args) = tc.get("function").and_then(|f| f.get("arguments")) {
                                if let Some(args_str) = args.as_str() {
                                    if let Ok(parsed) = serde_json::from_str::<Value>(args_str) {
                                        if let Some(dec) = parsed.get("decision").and_then(|d| d.as_str()) {
                                            if let Some(just) = parsed.get("justification").and_then(|j| j.as_str()) {
                                                info!("GROK JUSTIFICATION: {}", just);
                                            }
                                            return Ok(dec.to_string());
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        Ok("neutral".to_string()) // fallback
    }
}