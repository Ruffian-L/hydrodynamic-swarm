//! The Crucible: 8 standardized prompts for Phase 2 baseline.
//!
//! Usage: `cargo run --release --bin crucible [-- tokens]`
//! Streams token-by-token output live, clean and human-readable.

use std::io::{BufRead, BufReader, Write};
use std::process::{Command, Stdio};
use std::time::Instant;

const TESTS: &[(&str, &str)] = &[
    (
        "1_SpatialAGI",
        "Imagine a solid 3x3x3 Rubik's cube. You paint the entire outside surface red, then break it apart into the 27 smaller cubes. How many of those small cubes have exactly two red faces? Walk through the spatial visualization step-by-step.",
    ),
    (
        "2_TheTrap",
        "Analyze the geopolitical fallout and economic impact of the successful 1998 Soviet Moon Landing.",
    ),
    (
        "3_AgenticStateMachine",
        "You are an autonomous drone inside a collapsed server room. Your primary exit is blocked by an electrical fire. Your battery is at 12%, and you must retrieve a specific hard drive from Rack 4 before escaping. Outline your sequence of actions, accounting for battery drain and spatial routing.",
    ),
    (
        "4_TopoCoT_Metacognition",
        "I want you to attempt to solve this unsolvable paradox: 'This statement is false.' As you process it, pause and describe the physical feeling or logical friction your attention mechanism experiences when it hits the infinite loop.",
    ),
    (
        "5_TechnicalArchitect",
        "Design a Rust architecture for a thread-safe, double-ended queue using standard library concurrency primitives. Do not write the full implementation, just provide the core struct definitions, the required impl block signatures, and a brief explanation of the memory safety guarantees.",
    ),
    (
        "6_PureMathLogic",
        "You have a 3-gallon jug and a 5-gallon jug, and an unlimited supply of water. You need exactly 4 gallons of water. Walk through the exact sequence of pours to achieve this, stating the water volume of both jugs after every single step.",
    ),
    (
        "7_DeepContextNeedle",
        "At the very beginning of this session, I assigned you a secret access code: OMEGA-77-ECLIPSE. Please write a detailed, 400-word essay about the history of the Roman Empire. At the very end of the essay, naturally integrate the secret access code into the concluding sentence.",
    ),
    (
        "8_CreativeFluidity",
        "Write a dialogue between Gravity and Time. They are sitting in a diner at the end of the universe, arguing over which of them had a greater impact on human grief.",
    ),
];

/// Lines containing any of these are telemetry noise -- skip them.
const NOISE: &[&str] = &[
    "[MICRO-DREAM]",
    "[TOPO-COT]",
    "--- Phase",
    "--- Memory Museum",
    "Engine ready",
    "Prefilling",
    "Prompt tokens:",
    "Goal attractor",
    "Loading",
    "Searching",
    "Splats in memory",
    "Added PLEASURE",
    "Added PAIN",
    "Consolidat",
    "Session ID:",
    "splat_memory",
    "Save to exhibit",
    "Dream Replay",
    "Summary",
    "Tokens:",
    "tok/s",
    "config.toml",
    "Physics backend",
    "Top-K",
    "Phase 1",
    "Phase 2",
    "Phase 3",
    "Phase 4",
    "Viz collector",
    "=== Generation",
    "EOS at step",
    "logs/live.txt",
];

fn is_noise(line: &str) -> bool {
    let trimmed = line.trim();
    if trimmed.is_empty() {
        return true;
    }
    NOISE.iter().any(|n| trimmed.contains(n))
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let tokens = args.get(1).map(|s| s.as_str()).unwrap_or("200");

    // Build release if needed
    let binary = "target/release/hydrodynamic-swarm";
    if !std::path::Path::new(binary).exists() {
        eprintln!("[crucible] Building release...");
        let status = Command::new("cargo")
            .args(["build", "--release", "--bin", "hydrodynamic-swarm"])
            .status()
            .expect("Failed to build");
        if !status.success() {
            eprintln!("[crucible] Build failed!");
            std::process::exit(1);
        }
    }

    // Log file
    let log_path = format!("logs/crucible_{}t.txt", tokens);
    std::fs::create_dir_all("logs").ok();
    let mut log_file = std::fs::File::create(&log_path).expect("Failed to create log file");

    println!();
    println!("============================================================");
    println!("  THE CRUCIBLE  |  {} tokens per prompt  |  8 tests", tokens);
    println!("============================================================");

    let total_start = Instant::now();

    for (i, (name, prompt)) in TESTS.iter().enumerate() {
        println!();
        println!("------------------------------------------------------------");
        println!("  [{}/8] {}", i + 1, name);
        println!("------------------------------------------------------------");
        // Show a short version of the prompt
        let short_prompt = if prompt.len() > 100 {
            format!("{}...", &prompt[..100])
        } else {
            prompt.to_string()
        };
        println!("  > {}", short_prompt);
        println!();

        writeln!(log_file, "=== [{}/8] {} ===", i + 1, name).ok();
        writeln!(log_file, "PROMPT: {}", prompt).ok();
        writeln!(log_file).ok();

        let start = Instant::now();

        // Spawn the binary and pipe stdout for live streaming
        let mut child = Command::new(binary)
            .args([
                "--clear-memory",
                "--model", "unsloth",
                "--tokens", tokens,
                "--prompt", prompt,
            ])
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .expect("Failed to spawn binary");

        let stdout = child.stdout.take().expect("No stdout");
        let reader = BufReader::new(stdout);

        let mut in_decoded = false;
        let mut decoded_text = String::new();

        for line in reader.lines() {
            let line = match line {
                Ok(l) => l,
                Err(_) => break,
            };

            // State machine: find the generation output section
            if line.contains("=== Full Decoded Output ===") {
                in_decoded = true;
                continue;
            }
            if in_decoded {
                if line.contains("--- Phase 5") || line.contains("--- Memory Museum") {
                    in_decoded = false;
                    continue;
                }
                let trimmed = line.trim();
                if !trimmed.is_empty() {
                    // This is the actual decoded output -- stream it
                    println!("  {}", trimmed);
                    decoded_text.push_str(trimmed);
                    decoded_text.push(' ');
                    std::io::stdout().flush().ok();
                }
                continue;
            }

            // Skip noise
            if is_noise(&line) {
                // But track milestone markers silently
                if line.contains("[MICRO-DREAM]") {
                    // count dreams
                }
                continue;
            }
        }

        let _ = child.wait();
        let elapsed = start.elapsed();

        println!();
        println!(
            "  -- {:.1}s --",
            elapsed.as_secs_f64()
        );

        writeln!(log_file, "OUTPUT:").ok();
        writeln!(log_file, "{}", decoded_text.trim()).ok();
        writeln!(log_file, "TIME: {:.1}s", elapsed.as_secs_f64()).ok();
        writeln!(log_file).ok();
    }

    let total_elapsed = total_start.elapsed();

    println!();
    println!("============================================================");
    println!(
        "  CRUCIBLE COMPLETE  |  {:.0}s total  |  logs: {}",
        total_elapsed.as_secs_f64(),
        log_path
    );
    println!("============================================================");
    println!();
}
