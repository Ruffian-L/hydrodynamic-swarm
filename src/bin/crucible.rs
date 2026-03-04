//! The Crucible: 8 standardized prompts for Phase 2 baseline.
//!
//! Usage: `cargo run --release --bin crucible [-- tokens]`
//! Streams all output live -- you see tokens as they generate.

use std::io::Write;
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
    println!(
        "  THE CRUCIBLE  |  {} tokens per prompt  |  8 tests",
        tokens
    );
    println!("============================================================");

    let total_start = Instant::now();

    for (i, (name, prompt)) in TESTS.iter().enumerate() {
        println!();
        println!("------------------------------------------------------------");
        println!("  [{}/8] {}", i + 1, name);
        println!("------------------------------------------------------------");
        println!();

        writeln!(log_file, "=== [{}/8] {} ===", i + 1, name).ok();
        writeln!(log_file, "PROMPT: {}", prompt).ok();
        writeln!(log_file).ok();

        let start = Instant::now();

        // Inherit stdout/stderr so tokens stream live to terminal
        let status = Command::new(binary)
            .args([
                "--clear-memory",
                "--test",
                "--model",
                "unsloth",
                "--tokens",
                tokens,
                "--prompt",
                prompt,
            ])
            .stdout(Stdio::inherit())
            .stderr(Stdio::inherit())
            .status()
            .expect("Failed to run binary");

        let elapsed = start.elapsed();

        println!();
        println!(
            "  -- [{}/8] {} done in {:.1}s (exit: {}) --",
            i + 1,
            name,
            elapsed.as_secs_f64(),
            status
        );
        println!();

        writeln!(
            log_file,
            "TIME: {:.1}s | EXIT: {}",
            elapsed.as_secs_f64(),
            status
        )
        .ok();
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
