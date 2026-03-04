//! Session Finalization (Phases 5 & 6 + Summary)
//!
//! Everything that runs after the generation loop:
//!   - Phase 5: Splat scar tissue (pleasure/pain imprinting + evaporation)
//!   - Memory Museum: save to named exhibit or toss
//!   - Phase 6: Dream replay
//!   - Session summary banner + readable log

use anyhow::Result;
use candle_core::{Device, Tensor};
use std::io::Write;
use std::path::Path;

use crate::config::Config;
use crate::dream::DreamEngine;
use crate::logger::{SessionLogger, SessionSummary};
use crate::memory::SplatMemory;
use crate::niodoo::NiodooEngine;
use crate::splat::Splat;
use crate::viz::VizCollector;
use crate::viz_metal;

/// Everything produced by the generation loop that `finish` needs.
pub struct GenerationResult {
    pub tokens: Vec<u32>,
    pub full_text: String,
    pub goal_norm: f32,
    pub prompt_ids: Vec<u32>,
    /// The final steered latent position (for splat insertion).
    pub last_steered_pos: Option<Tensor>,
}

/// Run Phases 5 & 6 and emit the session summary.
///
/// Consumes `viz_collector` because Metal rendering does not return.
#[allow(clippy::too_many_arguments)]
pub fn finish(
    result: &GenerationResult,
    engine: &mut NiodooEngine,
    logger: &mut SessionLogger,
    device: &Device,
    model_variant: &str,
    prompt: &str,
    cfg: &Config,
    splat_file: &Path,
    test_mode: bool,
    viz_collector: Option<VizCollector>,
) -> Result<()> {
    // =========================================================
    // Phase 5: Splat Scar Tissue
    // =========================================================
    println!("\n--- Phase 5: Splat Scar Tissue ---");
    if let Some(ref final_pos) = result.last_steered_pos {
        let pos_1d = final_pos.squeeze(0)?; // (1, D) -> (D,)
        if result.tokens.len() > 15 {
            engine.memory_mut().add_splat(Splat::new(
                pos_1d,
                cfg.physics.splat_sigma,
                1.8, // positive scar (pleasure)
            ));
            println!(
                "    + Added PLEASURE splat (generation succeeded: {} tokens)",
                result.tokens.len()
            );
        } else {
            engine.memory_mut().add_splat(Splat::new(
                pos_1d,
                cfg.physics.splat_sigma,
                -0.9, // negative scar (pain)
            ));
            println!(
                "    x Added PAIN splat (generation too short: {} tokens)",
                result.tokens.len()
            );
        }
        println!("    Splats in memory: {}", engine.memory().len());
    }

    // Evaporation: time-based decay + cull dead splats
    engine.memory_mut().decay_step(cfg.memory.decay_rate);
    let culled = engine.memory_mut().cull(cfg.memory.prune_threshold);
    if culled > 0 {
        println!("    [EVAPORATE] Culled {} dead splats", culled);
    }

    // Consolidate and cap splat memory before saving
    let _ = engine
        .memory_mut()
        .consolidate(cfg.memory.consolidation_dist);
    engine.memory_mut().prune_to_limit(cfg.memory.max_splats);

    // Save persistent splat memory to disk
    engine.memory().save(splat_file)?;
    engine
        .memory()
        .save_metadata(splat_file, prompt, logger.session_id())?;

    // =========================================================
    // Memory Museum: Save to exhibit / Toss
    // =========================================================
    println!("\n--- Memory Museum ---");
    println!(
        "    Splats: {} | Source: \"{}\"",
        engine.memory().len(),
        prompt
    );

    // List existing exhibits
    let exhibits_dir = Path::new("exhibits");
    if exhibits_dir.exists() {
        if let Ok(entries) = std::fs::read_dir(exhibits_dir) {
            let names: Vec<String> = entries
                .filter_map(|e| e.ok())
                .filter(|e| {
                    e.path()
                        .extension()
                        .map(|ext| ext == "safetensors")
                        .unwrap_or(false)
                })
                .filter_map(|e| {
                    e.path()
                        .file_stem()
                        .map(|s| s.to_string_lossy().to_string())
                })
                .collect();
            if !names.is_empty() {
                println!("    Existing exhibits: {}", names.join(", "));
            }
        }
    }

    print!("    Save to exhibit? [name / n(ew) / t(oss)]: ");
    std::io::stdout().flush().ok();
    let museum_input = if test_mode {
        println!("(skipped -- test mode)");
        "t".to_string()
    } else {
        let mut input = String::new();
        std::io::stdin().read_line(&mut input)?;
        input
    };
    let museum_input = museum_input.trim();

    if !museum_input.is_empty() && museum_input != "t" && museum_input != "toss" {
        let exhibit_name = if museum_input == "n" || museum_input == "new" {
            print!("    Exhibit name: ");
            std::io::stdout().flush().ok();
            let mut name = String::new();
            std::io::stdin().read_line(&mut name)?;
            name.trim().to_string()
        } else {
            museum_input.to_string()
        };

        if !exhibit_name.is_empty() {
            let safe_name: String = exhibit_name
                .chars()
                .map(|c| {
                    if c.is_alphanumeric() || c == '-' || c == '_' {
                        c
                    } else {
                        '_'
                    }
                })
                .collect();

            std::fs::create_dir_all(exhibits_dir)?;
            let exhibit_path = exhibits_dir.join(format!("{}.safetensors", safe_name));
            let exhibit_meta = exhibits_dir.join(format!("{}.meta.json", safe_name));

            std::fs::copy(splat_file, &exhibit_path)?;
            let meta_src = splat_file.with_extension("meta.json");
            if meta_src.exists() {
                std::fs::copy(&meta_src, &exhibit_meta)?;
            }
            println!(
                "    Saved exhibit: {} ({} splats)",
                exhibit_path.display(),
                engine.memory().len()
            );
        } else {
            println!("    (empty name, skipping)");
        }
    } else {
        println!("    (tossed)");
    }

    // =========================================================
    // Phase 6: Dream Replay
    // =========================================================
    println!("\n--- Phase 6: Dream Replay ---");
    let dream_memory = SplatMemory::new(device.clone());
    let mut dream = DreamEngine::new(dream_memory);
    let dim = engine.field_dim();
    let traj = Tensor::randn(0.0f32, 1.0, (20, dim), device)?;
    dream.run(vec![traj], 0.05)?;

    // =========================================================
    // Session Summary
    // =========================================================
    let splat_type = if result.tokens.len() > 15 {
        "pleasure"
    } else {
        "pain"
    };
    logger.log_summary(SessionSummary {
        prompt: prompt.to_string(),
        prompt_token_count: result.prompt_ids.len(),
        generated_token_count: result.tokens.len(),
        goal_attractor_norm: result.goal_norm,
        splat_count_before: engine.memory().len(),
        splat_count_after: engine.memory().len(),
        splat_type_added: splat_type.to_string(),
        decoded_output: result.full_text.clone(),
        delta_min: 0.0,
        delta_max: 0.0,
        delta_mean: 0.0,
    })?;

    println!("\n========================================");
    println!("  SplatRAG v1.1 -- OPERATIONAL");
    println!("========================================");
    println!("  Model:    {}", cfg.models.model_path);
    println!("  Variant:  {}", model_variant);
    println!("  Prompt:   \"{}\"", prompt);
    println!("  Tokens:   {} generated", result.tokens.len());
    println!("  Log:      {}", logger.path().display());
    println!("  Backend:  {} + Niodoo physics", engine.backend_name());
    println!("========================================");

    // Append to human-readable session log
    write_readable_log(
        model_variant,
        prompt,
        result.tokens.len(),
        engine.memory().len(),
        &result.full_text,
    )?;

    // =========================================================
    // Visualization export + Metal window
    // =========================================================
    if let Some(collector) = viz_collector {
        let viz_path = logger.path().with_extension("viz.json");
        let _ = collector.export_json(&viz_path);
        let render_data = collector.into_render_data();
        viz_metal::launch(render_data); // does not return on macOS
    }

    Ok(())
}

/// Append one run entry to `logs/readable.txt`.
fn write_readable_log(
    model_variant: &str,
    prompt: &str,
    token_count: usize,
    splat_count: usize,
    full_text: &str,
) -> Result<()> {
    use std::io::Write;
    let readable_path = Path::new("logs/readable.txt");
    let mut f = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(readable_path)?;

    let timestamp = format_utc_now();
    writeln!(f, "=== Run: {} ===", timestamp)?;
    writeln!(
        f,
        "Model: {} | Tokens: {} | Splats: {}",
        model_variant, token_count, splat_count
    )?;
    writeln!(f, "Prompt: \"{}\"", prompt)?;
    writeln!(f)?;
    writeln!(f, "{}", full_text)?;
    writeln!(f)?;
    writeln!(f, "---")?;
    writeln!(f)?;
    Ok(())
}

/// Format the current UTC time as `YYYY-MM-DD HH:MM UTC`.
///
/// Uses only `std::time` — no external crate.
fn format_utc_now() -> String {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    let secs_per_day: u64 = 86400;
    let day_secs = now % secs_per_day;
    let hours = day_secs / 3600;
    let minutes = (day_secs % 3600) / 60;
    let mut days = now / secs_per_day;

    let mut y = 1970i64;
    loop {
        let days_in_year = if y % 4 == 0 && (y % 100 != 0 || y % 400 == 0) {
            366
        } else {
            365
        };
        if days < days_in_year {
            break;
        }
        days -= days_in_year;
        y += 1;
    }

    let month_days = [
        31u64,
        if y % 4 == 0 && (y % 100 != 0 || y % 400 == 0) {
            29
        } else {
            28
        },
        31,
        30,
        31,
        30,
        31,
        31,
        30,
        31,
        30,
        31,
    ];
    let mut m = 1u32;
    let mut remaining = days;
    for md in &month_days {
        if remaining < *md {
            break;
        }
        remaining -= md;
        m += 1;
    }
    let d = remaining + 1;

    format!("{}-{:02}-{:02} {:02}:{:02} UTC", y, m, d, hours, minutes)
}
