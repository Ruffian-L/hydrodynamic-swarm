//! Comprehensive E2E Test Suite for the Unified 6-Tab Ratatui Frontend.
//!
//! Covers:
//! - Tier 1: Feature Coverage (Scaffold, Concurrency, Channels, Tabs 1-6, Key Event Routing)
//! - Tier 2: Boundary & Corner Cases (Min/Max Sliders, Buffer Extremes, Unicode/Emoji, Tiny/Huge Terminals, Message Flooding)
//! - Tier 3: Cross-Feature Combinations (Concurrent Streaming + Physics Tuning, Model Swap + State Sync, KV Snapshotting)
//! - Tier 4: Real-World Workloads (Full Lifecycle Headless Walkthrough, Non-Panicking Startup, 500-Frame Stress Loop)
//!
//! Uses `ratatui::backend::TestBackend` for headless terminal verification without requiring a physical TTY.

use std::time::Duration;

use crossterm::event::{KeyCode, KeyEvent, KeyEventKind, KeyEventState, KeyModifiers};
use ratatui::backend::TestBackend;
use ratatui::buffer::Buffer;
use ratatui::Terminal;

// Re-export config and hooks at test crate root so `crate::config::Config` and `crate::hooks::HookSite` resolve in included modules
pub use hydrodynamic_swarm::config;
pub use hydrodynamic_swarm::hooks;

// Wire up the modules required for frontend integration testing
#[path = "../src/algo_scale.rs"]
#[allow(dead_code, unused_imports)]
pub mod algo_scale;

#[path = "../src/hud.rs"]
#[allow(dead_code, unused_imports)]
pub mod hud;

#[path = "../src/frontend/mod.rs"]
pub mod frontend;

use frontend::channel::{EngineToUiMsg, UiToEngineMsg};
use frontend::engine_bridge::EngineBridge;
use frontend::event::KeyRouter;
use frontend::tabs::Tab;
use frontend::App;
use hooks::HookSite;
use hydrodynamic_swarm::config::Config;

/// Helper to convert a Ratatui TestBackend buffer to a newline-separated String for easy pattern matching.
fn buffer_to_string(buffer: &Buffer) -> String {
    let mut out = String::new();
    for y in 0..buffer.area.height {
        for x in 0..buffer.area.width {
            let cell = buffer.cell((x, y)).unwrap();
            out.push_str(cell.symbol());
        }
        out.push('\n');
    }
    out
}

/// Helper to create a key press event.
fn press_key(code: KeyCode, modifiers: KeyModifiers) -> KeyEvent {
    KeyEvent {
        code,
        modifiers,
        kind: KeyEventKind::Press,
        state: KeyEventState::empty(),
    }
}

/// Helper to create a headless App instance with a dry-run EngineBridge.
fn create_test_app(width: u16, height: u16) -> (App, Terminal<TestBackend>) {
    let config = Config::default();
    let bridge = EngineBridge::spawn(config, Some("models/gemma4-9b-it.gguf".to_string()), true);
    let app = App::new(bridge, Some("Gemma 4 IT (Test)".to_string()));
    let backend = TestBackend::new(width, height);
    let terminal = Terminal::new(backend).expect("Failed to create TestBackend");
    (app, terminal)
}

// ============================================================================
// TIER 1: FEATURE COVERAGE
// ============================================================================

#[test]
fn test_tier1_app_initialization_defaults() {
    let (mut app, mut terminal) = create_test_app(120, 30);
    assert_eq!(app.active_tab, Tab::ModelLoader);
    assert_eq!(app.active_tab.index(), 0);
    assert!(!app.should_quit);
    assert_eq!(app.model_name, "Gemma 4 IT (Test)");
    assert!(!app.is_generating);
    assert!(app.generated_text.is_empty());
    assert!(!app.prompt_buffer.is_empty());

    // Initial render
    terminal.draw(|f| app.render(f)).expect("Render failed");
    let content = buffer_to_string(terminal.backend().buffer());
    assert!(content.contains("Hydrodynamic Swarm"));
    assert!(content.contains("Model & Config"));
    assert!(content.contains("STATUS:"));
}

#[test]
fn test_tier1_channel_bidirectional_messaging() {
    let config = Config::default();
    let bridge = EngineBridge::spawn(config, None, true);

    // Wait briefly for initial EngineReady
    let ready_msg = bridge.recv_timeout(Duration::from_millis(500));
    assert!(ready_msg.is_ok(), "Engine did not emit initial message");

    // UI -> Engine message send
    let send_res = bridge.send(UiToEngineMsg::SetLiveParam {
        key: "residual.cap".to_string(),
        val: 4.25,
    });
    assert!(send_res.is_ok());

    // UI -> Engine generation start
    let gen_send = bridge.send(UiToEngineMsg::StartGeneration {
        prompt: "Testing 1 2 3".to_string(),
        temperature: 0.8,
        max_tokens: 16,
    });
    assert!(gen_send.is_ok());

    // UI -> Engine KV snapshot
    let kv_send = bridge.send(UiToEngineMsg::SnapshotKv);
    assert!(kv_send.is_ok());
}

#[test]
fn test_tier1_key_router_global_quit_and_cancel() {
    let (mut app, _) = create_test_app(80, 24);

    // Press 'q' -> should quit
    app.handle_key_event(press_key(KeyCode::Char('q'), KeyModifiers::empty()));
    assert!(app.should_quit);

    // Reset and test Ctrl+C
    app.should_quit = false;
    app.handle_key_event(press_key(KeyCode::Char('c'), KeyModifiers::CONTROL));
    assert!(app.should_quit);

    // Reset and test Esc (Cancel / idle)
    app.should_quit = false;
    app.handle_key_event(press_key(KeyCode::Esc, KeyModifiers::empty()));
    assert!(!app.should_quit);
    assert!(app.status_message.contains("idle") || app.status_message.contains("Unfocused"));
}

#[test]
fn test_tier1_key_router_tab_cycling_forward_and_backward() {
    let (mut app, _) = create_test_app(80, 24);
    assert_eq!(app.active_tab, Tab::ModelLoader);

    // Tab -> Next (PhysicsBoard)
    app.handle_key_event(press_key(KeyCode::Tab, KeyModifiers::empty()));
    assert_eq!(app.active_tab, Tab::PhysicsBoard);

    // Tab -> Next (SystemDeck)
    app.handle_key_event(press_key(KeyCode::Tab, KeyModifiers::empty()));
    assert_eq!(app.active_tab, Tab::SystemDeck);

    // Tab -> Next (DebugMatrix)
    app.handle_key_event(press_key(KeyCode::Tab, KeyModifiers::empty()));
    assert_eq!(app.active_tab, Tab::DebugMatrix);

    // Tab -> Next (CompareArena)
    app.handle_key_event(press_key(KeyCode::Tab, KeyModifiers::empty()));
    assert_eq!(app.active_tab, Tab::CompareArena);

    // Tab -> Next (Misc)
    app.handle_key_event(press_key(KeyCode::Tab, KeyModifiers::empty()));
    assert_eq!(app.active_tab, Tab::Misc);

    // Tab -> Wrap around (ModelLoader)
    app.handle_key_event(press_key(KeyCode::Tab, KeyModifiers::empty()));
    assert_eq!(app.active_tab, Tab::ModelLoader);

    // Shift+Tab -> Prev (Misc)
    app.handle_key_event(press_key(KeyCode::BackTab, KeyModifiers::empty()));
    assert_eq!(app.active_tab, Tab::Misc);

    // Shift+Tab -> Prev (CompareArena)
    app.handle_key_event(press_key(KeyCode::Tab, KeyModifiers::SHIFT));
    assert_eq!(app.active_tab, Tab::CompareArena);
}

#[test]
fn test_tier1_key_router_direct_tab_navigation() {
    let (mut app, _) = create_test_app(80, 24);

    // Number keys '1' through '6'
    let tab_keys = [
        (KeyCode::Char('1'), Tab::ModelLoader),
        (KeyCode::Char('2'), Tab::PhysicsBoard),
        (KeyCode::Char('3'), Tab::SystemDeck),
        (KeyCode::Char('4'), Tab::DebugMatrix),
        (KeyCode::Char('5'), Tab::CompareArena),
        (KeyCode::Char('6'), Tab::Misc),
    ];

    for (key, expected_tab) in tab_keys {
        app.handle_key_event(press_key(key, KeyModifiers::empty()));
        assert_eq!(app.active_tab, expected_tab);
        assert_eq!(app.active_tab.index(), expected_tab.index());
    }

    // Function keys F1 through F6
    let f_keys = [
        (KeyCode::F(1), Tab::ModelLoader),
        (KeyCode::F(2), Tab::PhysicsBoard),
        (KeyCode::F(3), Tab::SystemDeck),
        (KeyCode::F(4), Tab::DebugMatrix),
        (KeyCode::F(5), Tab::CompareArena),
        (KeyCode::F(6), Tab::Misc),
    ];

    for (key, expected_tab) in f_keys {
        app.handle_key_event(press_key(key, KeyModifiers::empty()));
        assert_eq!(app.active_tab, expected_tab);
    }
}

#[test]
fn test_tier1_tab1_model_loader_render() {
    let (mut app, mut terminal) = create_test_app(120, 30);
    app.active_tab = Tab::ModelLoader;

    terminal.draw(|f| app.render(f)).expect("Render Tab 1 failed");
    let content = buffer_to_string(terminal.backend().buffer());

    assert!(content.contains("Model Browser"));
    assert!(content.contains("gemma-4"));
    assert!(content.contains("Active Model Architecture"));
    assert!(content.contains("Sampling & Size Scaling"));
    assert!(content.contains("Sampling & Context Configuration"));
    assert!(content.contains("Temperature"));
}

#[test]
fn test_tier1_tab2_physics_board_render() {
    let (mut app, mut terminal) = create_test_app(120, 30);
    app.active_tab = Tab::PhysicsBoard;

    terminal.draw(|f| app.render(f)).expect("Render Tab 2 failed");
    let content = buffer_to_string(terminal.backend().buffer());

    assert!(content.contains("Surface 1: Residual Forces"));
    assert!(content.contains("Surface 2: Logit Biases"));
    assert!(content.contains("Surface 3: Layer Hook"));
    assert!(content.contains("Stability Verdicts"));
}

#[test]
fn test_tier1_tab3_system_deck_render() {
    let (mut app, mut terminal) = create_test_app(120, 30);
    app.active_tab = Tab::SystemDeck;
    app.system_prompt_buffer = "You are a swarm coordinator.".to_string();

    terminal.draw(|f| app.render(f)).expect("Render Tab 3 failed");
    let content = buffer_to_string(terminal.backend().buffer());

    assert!(content.contains("Active System Prompt Deck"));
    assert!(content.contains("Packed Prompt"));
    assert!(content.contains("<spike>"));
    assert!(content.contains("<focus>"));
    assert!(content.contains("<remember>"));
    assert!(content.contains("<lock>"));
}

#[test]
fn test_tier1_tab4_debug_matrix_render() {
    let (mut app, mut terminal) = create_test_app(120, 30);
    app.active_tab = Tab::DebugMatrix;

    // Attach mock hud frame
    app.last_hud_frame = Some(hud::HudFrame {
        step: 42,
        entropy: Some(2.15),
        margin: Some(0.85),
        p_chosen: Some(0.72),
        hook_applications: Some(8),
        scars: 3,
        force_cap: 4.5,
        ..Default::default()
    });

    terminal.draw(|f| app.render(f)).expect("Render Tab 4 failed");
    let content = buffer_to_string(terminal.backend().buffer());

    assert!(content.contains("Entropy H(t)"));
    assert!(content.contains("Margin Δp(t)"));
    assert!(content.contains("Topological Data Analysis"));
    assert!(content.contains("Jacobian Sensitivity Matrix"));
    assert!(content.contains("Self-Regulation Phase State"));
}

#[test]
fn test_tier1_tab5_compare_arena_render() {
    let (mut app, mut terminal) = create_test_app(120, 30);
    app.active_tab = Tab::CompareArena;
    app.vanilla_compare_text = "Vanilla baseline response".to_string();
    app.hydro_compare_text = "Hydro physics-steered response".to_string();

    terminal.draw(|f| app.render(f)).expect("Render Tab 5 failed");
    let content = buffer_to_string(terminal.backend().buffer());

    assert!(content.contains("A/B Arena Controller"));
    assert!(content.contains("Vanilla Baseline"));
    assert!(content.contains("Hydro Swarm"));
    assert!(content.contains("Vanilla baseline response"));
    assert!(content.contains("Hydro physics-steered response"));
}

#[test]
fn test_tier1_tab6_misc_kv_cache_render() {
    let (mut app, mut terminal) = create_test_app(120, 30);
    app.active_tab = Tab::Misc;
    app.remember_items = vec![
        ("key_concept".to_string(), "Hydrodynamic stability".to_string()),
        ("eval_notice".to_string(), "Model knows it is tested".to_string()),
    ];

    terminal.draw(|f| app.render(f)).expect("Render Tab 6 failed");
    let content = buffer_to_string(terminal.backend().buffer());

    assert!(content.contains("Choice-Driven KV Cache Management"));
    assert!(content.contains("Persistent Remember Store"));
    assert!(content.contains("key_concept"));
    assert!(content.contains("eval_notice"));
}

#[test]
fn test_tier1_status_footer_and_header_rendering() {
    let (mut app, mut terminal) = create_test_app(140, 25);
    app.status_message = "Test status message active".to_string();

    terminal.draw(|f| app.render(f)).expect("Render header and footer failed");
    let content = buffer_to_string(terminal.backend().buffer());

    assert!(content.contains("Hydrodynamic Swarm"));
    assert!(content.contains("STATUS:"));
    assert!(content.contains("Test status message active"));
    assert!(content.contains("[Tab/1-6]"));
    assert!(content.contains("[Esc/q]"));
}

// ============================================================================
// TIER 2: BOUNDARY & CORNER CASES
// ============================================================================

#[test]
fn test_tier2_slider_boundary_limits_min_max_clamping() {
    let (mut app, _) = create_test_app(80, 24);

    // Rapid vertical navigation up beyond index 0
    for _ in 0..50 {
        app.handle_key_event(press_key(KeyCode::Up, KeyModifiers::empty()));
    }
    assert_eq!(app.selected_index, 0);

    // Rapid vertical navigation down
    for _ in 0..100 {
        app.handle_key_event(press_key(KeyCode::Down, KeyModifiers::empty()));
    }
    assert_eq!(app.selected_index, 100);

    // Horizontal adjust tests with Shift modifier
    let left_adj = KeyRouter::horizontal_adjust(&press_key(KeyCode::Left, KeyModifiers::empty()));
    assert_eq!(left_adj, Some(-1.0));

    let left_shift_adj = KeyRouter::horizontal_adjust(&press_key(KeyCode::Left, KeyModifiers::SHIFT));
    assert_eq!(left_shift_adj, Some(-10.0));

    let right_adj = KeyRouter::horizontal_adjust(&press_key(KeyCode::Right, KeyModifiers::empty()));
    assert_eq!(right_adj, Some(1.0));

    let right_shift_adj = KeyRouter::horizontal_adjust(&press_key(KeyCode::Right, KeyModifiers::SHIFT));
    assert_eq!(right_shift_adj, Some(10.0));
}

#[test]
fn test_tier2_empty_prompts_and_extreme_length_buffers() {
    let (mut app, mut terminal) = create_test_app(100, 30);

    // Empty prompt buffer
    app.prompt_buffer.clear();
    assert!(app.prompt_buffer.is_empty());
    terminal.draw(|f| app.render(f)).expect("Render with empty prompt failed");

    // Extreme length prompt buffer (100,000 characters)
    app.prompt_buffer = "A".repeat(100_000);
    assert_eq!(app.prompt_buffer.len(), 100_000);
    terminal.draw(|f| app.render(f)).expect("Render with huge prompt failed");

    // Huge generated text stream (50,000 characters)
    app.generated_text = "Token sequence generated by physics engine. ".repeat(1_000);
    terminal.draw(|f| app.render(f)).expect("Render with huge generated text failed");
}

#[test]
fn test_tier2_unicode_emoji_and_control_character_handling() {
    let (mut app, mut terminal) = create_test_app(100, 30);

    // Complex Unicode, CJK, Arabic, and emojis
    app.prompt_buffer = "🌊 Swarm 🦀 渦動 流体力学 <spike> こんにちは \t \n \r \0".to_string();
    app.system_prompt_buffer = "🛡️ God-tier: \u{1F980} \u{2202}\u{2207}\u{03A9}".to_string();
    app.vanilla_compare_text = "مرحبا بالعالم — Hydro test 🌟".to_string();

    terminal.draw(|f| app.render(f)).expect("Render with Unicode/Emoji failed");
    let content = buffer_to_string(terminal.backend().buffer());
    assert!(content.contains("Hydrodynamic Swarm"));
}

#[test]
fn test_tier2_terminal_dimension_extremes_tiny_and_gigantic() {
    // 1. Extremely small terminal (20 columns x 8 rows)
    let (mut app_tiny, mut term_tiny) = create_test_app(20, 8);
    term_tiny.draw(|f| app_tiny.render(f)).expect("Render on tiny terminal failed");

    // 2. Minimum non-zero terminal (10 x 5)
    let backend_min = TestBackend::new(10, 5);
    let mut term_min = Terminal::new(backend_min).unwrap();
    term_min.draw(|f| app_tiny.render(f)).expect("Render on 10x5 terminal failed");

    // 3. Gigantic terminal (400 columns x 200 rows)
    let (mut app_huge, mut term_huge) = create_test_app(400, 200);
    term_huge.draw(|f| app_huge.render(f)).expect("Render on huge terminal failed");
}

#[test]
fn test_tier2_channel_burst_message_flooding_no_deadlock() {
    let (mut app, _) = create_test_app(80, 24);

    // Flood 2,000 messages into handle_engine_msg
    for i in 0..2000 {
        let msg = if i % 4 == 0 {
            EngineToUiMsg::TokenGenerated {
                text: format!(" tok_{}", i),
                frame: hud::HudFrame {
                    step: i,
                    temperature: 0.8,
                    force_cap: 3.0 + (i as f32 * 0.001),
                    ..Default::default()
                },
            }
        } else if i % 4 == 1 {
            EngineToUiMsg::TelemetryUpdate(hud::HudFrame {
                step: i,
                entropy: Some(1.5),
                ..Default::default()
            })
        } else if i % 4 == 2 {
            EngineToUiMsg::KvSnapshotStatus {
                state: format!("Snapshot checkpoint #{}", i),
            }
        } else {
            EngineToUiMsg::RememberStoreUpdated(vec![
                ("key".to_string(), format!("val_{}", i)),
            ])
        };
        app.handle_engine_msg(msg);
    }

    assert!(app.is_generating);
    assert_eq!(app.remember_items.len(), 1);
    assert!(app.last_hud_frame.is_some());
}

#[test]
fn test_tier2_rapid_key_navigation_and_tab_overflow() {
    let (mut app, _) = create_test_app(80, 24);

    // Rapidly switch tabs 500 times
    for i in 0..500 {
        if i % 2 == 0 {
            app.handle_key_event(press_key(KeyCode::Tab, KeyModifiers::empty()));
        } else {
            app.handle_key_event(press_key(KeyCode::BackTab, KeyModifiers::empty()));
        }
    }
    assert!(Tab::ALL.contains(&app.active_tab));
}

// ============================================================================
// TIER 3: CROSS-FEATURE COMBINATIONS
// ============================================================================

#[test]
fn test_tier3_concurrent_token_streaming_with_tab_navigation() {
    let (mut app, mut terminal) = create_test_app(120, 30);

    // 1. Simulate starting on Tab 3 (System Deck)
    app.active_tab = Tab::SystemDeck;
    app.handle_key_event(press_key(KeyCode::Enter, KeyModifiers::empty()));

    // 2. Stream tokens in while navigating through all 6 tabs
    for (step, tab) in Tab::ALL.iter().enumerate() {
        app.active_tab = *tab;

        // Receive generated token from engine
        app.handle_engine_msg(EngineToUiMsg::TokenGenerated {
            text: format!(" token_{}", step),
            frame: hud::HudFrame {
                step,
                entropy: Some(1.8 - (step as f32 * 0.1)),
                margin: Some(0.6 + (step as f32 * 0.05)),
                force_cap: 5.0,
                scars: step,
                ..Default::default()
            },
        });

        // Render current tab under streaming load
        terminal.draw(|f| app.render(f)).expect("Render during stream failed");
    }

    // Complete generation
    app.handle_engine_msg(EngineToUiMsg::GenerationComplete {
        total_tokens: 6,
        elapsed_sec: 0.15,
    });

    assert!(!app.is_generating);
    assert_eq!(app.generated_text, " token_0 token_1 token_2 token_3 token_4 token_5");
    assert!(app.status_message.contains("Complete: 6 tokens"));
}

#[test]
fn test_tier3_live_parameter_tuning_while_generating() {
    let (mut app, _) = create_test_app(100, 30);

    // Generation active
    app.handle_engine_msg(EngineToUiMsg::TokenGenerated {
        text: "Starting steer...".to_string(),
        frame: hud::HudFrame {
            step: 1,
            force_cap: 3.5,
            ..Default::default()
        },
    });
    assert!(app.is_generating);

    // Switch to Tab 2 (Physics) and tweak sliders
    app.active_tab = Tab::PhysicsBoard;
    app.handle_key_event(press_key(KeyCode::Down, KeyModifiers::empty()));
    app.handle_key_event(press_key(KeyCode::Right, KeyModifiers::empty()));
    app.handle_key_event(press_key(KeyCode::Enter, KeyModifiers::empty()));

    assert_eq!(app.status_message, "Applied physics adjustment");

    // Hit Esc to abort generation while on physics tab
    app.handle_key_event(press_key(KeyCode::Esc, KeyModifiers::empty()));
    assert_eq!(app.status_message, "Generation aborted");
}

#[test]
fn test_tier3_model_load_event_synchronizes_ui_state() {
    let (mut app, mut terminal) = create_test_app(100, 30);

    // Receive ModelLoading progress
    app.handle_engine_msg(EngineToUiMsg::ModelLoading {
        status: "Loading tensor shard 2/4...".to_string(),
        progress: 0.5,
    });
    assert_eq!(app.status_message, "Loading tensor shard 2/4...");

    // Receive ModelLoaded
    app.handle_engine_msg(EngineToUiMsg::ModelLoaded {
        name: "Qwen 2.5 7B Instruct".to_string(),
        n_layers: 28,
    });
    assert_eq!(app.model_name, "Qwen 2.5 7B Instruct");
    assert_eq!(app.status_message, "Model loaded: Qwen 2.5 7B Instruct");

    // Render footer and check updated model name
    terminal.draw(|f| app.render(f)).expect("Render failed");
    let content = buffer_to_string(terminal.backend().buffer());
    assert!(content.contains("Qwen 2.5 7B Instruct"));
}

#[test]
fn test_tier3_kv_snapshot_and_restore_cycle_during_stream() {
    let (mut app, _) = create_test_app(100, 30);

    // Switch to Misc tab (Tab 6)
    app.active_tab = Tab::Misc;
    app.handle_key_event(press_key(KeyCode::Enter, KeyModifiers::empty()));
    assert_eq!(app.status_message, "KV Snapshot triggered");

    // Engine responds with snapshot status
    app.handle_engine_msg(EngineToUiMsg::KvSnapshotStatus {
        state: "Snapshot saved: layer_0..35 saved (142 MB)".to_string(),
    });
    assert_eq!(app.status_message, "Snapshot saved: layer_0..35 saved (142 MB)");
}

#[test]
fn test_tier3_compare_arena_roundtrip_with_remember_update() {
    let (mut app, mut terminal) = create_test_app(120, 30);
    app.active_tab = Tab::CompareArena;

    // Trigger compare
    app.handle_key_event(press_key(KeyCode::Enter, KeyModifiers::empty()));
    assert_eq!(app.status_message, "Comparing Vanilla vs Hydro Swarm...");

    // Receive CompareResult
    app.handle_engine_msg(EngineToUiMsg::CompareResult {
        vanilla_text: "Standard unsteered baseline output without physics forces.".to_string(),
        hydro_text: "Hydrodynamic steered output with continuous Diderot gradient.".to_string(),
    });
    assert_eq!(app.status_message, "Comparison complete");

    // Receive RememberStore update
    app.handle_engine_msg(EngineToUiMsg::RememberStoreUpdated(vec![
        ("experiment_a".to_string(), "Score 9.4".to_string()),
        ("experiment_b".to_string(), "Score 8.1".to_string()),
    ]));

    // Render Tab 5
    terminal.draw(|f| app.render(f)).expect("Render failed");
    let content = buffer_to_string(terminal.backend().buffer());
    assert!(content.contains("Standard unsteered baseline output"));
    assert!(content.contains("Hydrodynamic steered output"));
}

// ============================================================================
// TIER 4: REAL-WORLD WORKLOADS & LIFECYCLE
// ============================================================================

#[test]
fn test_tier4_full_application_lifecycle_headless_walkthrough() {
    let (mut app, mut terminal) = create_test_app(120, 32);

    // Step 1: Verify startup on Tab 1
    assert_eq!(app.active_tab, Tab::ModelLoader);
    terminal.draw(|f| app.render(f)).expect("Render 1 failed");

    // Step 2: Navigate to Tab 2 (Physics Board)
    app.handle_key_event(press_key(KeyCode::Char('2'), KeyModifiers::empty()));
    assert_eq!(app.active_tab, Tab::PhysicsBoard);
    terminal.draw(|f| app.render(f)).expect("Render 2 failed");

    // Step 3: Adjust physics controls
    app.handle_key_event(press_key(KeyCode::Down, KeyModifiers::empty()));
    app.handle_key_event(press_key(KeyCode::Right, KeyModifiers::SHIFT));
    app.handle_key_event(press_key(KeyCode::Enter, KeyModifiers::empty()));

    // Step 4: Navigate to Tab 3 (System Deck)
    app.handle_key_event(press_key(KeyCode::Char('3'), KeyModifiers::empty()));
    assert_eq!(app.active_tab, Tab::SystemDeck);
    app.system_prompt_buffer = "Coordinate swarm node #4".to_string();
    app.handle_key_event(press_key(KeyCode::Enter, KeyModifiers::empty()));
    terminal.draw(|f| app.render(f)).expect("Render 3 failed");

    // Step 5: Simulate incoming token generation
    for step in 0..10 {
        app.handle_engine_msg(EngineToUiMsg::TokenGenerated {
            text: format!(" word_{}", step),
            frame: hud::HudFrame {
                step,
                temperature: 0.85,
                force_cap: 4.0,
                scars: 2,
                ..Default::default()
            },
        });
    }

    // Step 6: Navigate to Tab 4 (Debug Matrix) while tokens arrived
    app.handle_key_event(press_key(KeyCode::Char('4'), KeyModifiers::empty()));
    assert_eq!(app.active_tab, Tab::DebugMatrix);
    terminal.draw(|f| app.render(f)).expect("Render 4 failed");

    // Step 7: Navigate to Tab 5 (Compare Arena)
    app.handle_key_event(press_key(KeyCode::Char('5'), KeyModifiers::empty()));
    assert_eq!(app.active_tab, Tab::CompareArena);
    terminal.draw(|f| app.render(f)).expect("Render 5 failed");

    // Step 8: Navigate to Tab 6 (Misc & KV)
    app.handle_key_event(press_key(KeyCode::Char('6'), KeyModifiers::empty()));
    assert_eq!(app.active_tab, Tab::Misc);
    app.handle_key_event(press_key(KeyCode::Enter, KeyModifiers::empty()));
    terminal.draw(|f| app.render(f)).expect("Render 6 failed");

    // Step 9: Send Quit command
    app.handle_key_event(press_key(KeyCode::Char('q'), KeyModifiers::empty()));
    assert!(app.should_quit);
}

#[test]
fn test_tier4_headless_startup_and_rendering_without_panic() {
    let (mut app, mut terminal) = create_test_app(100, 30);

    // Perform 100 consecutive update and draw iterations
    for i in 0..100 {
        app.update();

        if i % 10 == 0 {
            app.active_tab = app.active_tab.next();
        }

        terminal.draw(|f| app.render(f)).expect("Headless render failed");
    }
}

#[test]
fn test_tier4_high_throughput_500_frame_stress_workload() {
    let (mut app, mut terminal) = create_test_app(120, 30);

    for frame_idx in 0..500 {
        // Inject telemetry
        app.handle_engine_msg(EngineToUiMsg::TokenGenerated {
            text: format!(" t{}", frame_idx),
            frame: hud::HudFrame {
                step: frame_idx,
                force_cap: 3.0 + (frame_idx as f32 % 10.0) * 0.1,
                goal_force_scale: 0.15 + (frame_idx as f32 % 5.0) * 0.02,
                temperature: 0.7 + (frame_idx as f32 % 3.0) * 0.1,
                entropy: Some(1.2 + (frame_idx as f32 % 8.0) * 0.1),
                margin: Some(0.5 + (frame_idx as f32 % 4.0) * 0.1),
                p_chosen: Some(0.8),
                hook_applications: Some(frame_idx % 12),
                scars: frame_idx % 7,
                ..Default::default()
            },
        });

        // Periodically cycle tab and navigate list
        if frame_idx % 25 == 0 {
            app.active_tab = app.active_tab.next();
        }
        if frame_idx % 5 == 0 {
            app.selected_index = (app.selected_index + 1) % 10;
        }

        // Draw frame
        terminal.draw(|f| app.render(f)).expect("Stress draw failed");
    }

    assert_eq!(app.last_hud_frame.as_ref().map(|f| f.step), Some(499));
}

// ============================================================================
// TIER 5: MILESTONE 2 - TABS 1 & 2 INTERACTIVE CONTROLS & ENGINE BRIDGE SYNC
// ============================================================================

#[test]
fn test_m2_tab1_model_loader_navigation_and_hotkeys() {
    let (mut app, mut terminal) = create_test_app(120, 32);
    app.active_tab = Tab::ModelLoader;

    // Default state check
    assert_eq!(app.tab1_state.selected_field_idx, 0);
    assert_eq!(app.tab1_state.selected_model_idx, 0);
    assert_eq!(app.tab1_state.current_model().unwrap().name, "gemma-4-9b-it-Q4_K_M.gguf");

    // Cycle model with Right key
    app.handle_key_event(press_key(KeyCode::Right, KeyModifiers::empty()));
    assert_eq!(app.tab1_state.selected_model_idx, 1);
    assert_eq!(app.tab1_state.current_model().unwrap().name, "gemma-3-4b-it-Q4_K_M.gguf");

    // Cycle model with Left key
    app.handle_key_event(press_key(KeyCode::Left, KeyModifiers::empty()));
    assert_eq!(app.tab1_state.selected_model_idx, 0);

    // Navigate down to Temperature slider (field 1)
    app.handle_key_event(press_key(KeyCode::Down, KeyModifiers::empty()));
    assert_eq!(app.tab1_state.selected_field_idx, 1);
    let initial_temp = app.tab1_state.temperature;

    // Adjust temperature up
    app.handle_key_event(press_key(KeyCode::Right, KeyModifiers::empty()));
    assert!((app.tab1_state.temperature - (initial_temp + 0.05)).abs() < 1e-4);

    // Adjust temperature down with Shift (10x step = 0.20)
    app.handle_key_event(press_key(KeyCode::Left, KeyModifiers::SHIFT));
    assert!((app.tab1_state.temperature - (initial_temp + 0.05 - 0.20)).abs() < 1e-4);

    // Navigate to Max Tokens slider (field 3)
    app.handle_key_event(press_key(KeyCode::Down, KeyModifiers::empty())); // field 2: Rep Pen
    app.handle_key_event(press_key(KeyCode::Down, KeyModifiers::empty())); // field 3: Max Tokens
    assert_eq!(app.tab1_state.selected_field_idx, 3);
    let initial_tokens = app.tab1_state.max_tokens;
    app.handle_key_event(press_key(KeyCode::Right, KeyModifiers::empty()));
    assert_eq!(app.tab1_state.max_tokens, initial_tokens + 64);

    // Test Hotkey 'L' (Load selected model)
    app.handle_key_event(press_key(KeyCode::Char('l'), KeyModifiers::empty()));
    assert!(app.status_message.contains("Loading model"));

    // Test Hotkey 'U' (Unload model)
    app.handle_key_event(press_key(KeyCode::Char('u'), KeyModifiers::empty()));
    assert_eq!(app.model_name, "");
    assert_eq!(app.status_message, "Unloaded model");

    // Test Hotkey 'C' (Clear KV)
    app.handle_key_event(press_key(KeyCode::Char('c'), KeyModifiers::empty()));
    assert_eq!(app.status_message, "KV Cache cleared");

    // Render frame
    terminal.draw(|f| app.render(f)).expect("Render Tab 1 failed");
    let content = buffer_to_string(terminal.backend().buffer());
    assert!(content.contains("Model Browser"));
    assert!(content.contains("Temperature"));
    assert!(content.contains("Max Tokens"));
}

#[test]
fn test_m2_tab1_algo_scale_gain_ladder_and_decoupling_preview() {
    let (mut app, mut terminal) = create_test_app(120, 32);
    app.active_tab = Tab::ModelLoader;

    // 1. Test Piecewise rule (Current worktree: sqrt to 8B, log-soft above 8B, decoupled T)
    app.tab1_state.scaling_params_b = 9.0;
    app.tab1_state.scaling_model_type = algo_scale::ModelType::Instruct;
    app.tab1_state.scaling_size_rule = algo_scale::SizeRule::Piecewise;
    app.tab1_state.temperature = 0.85;

    let pred_piecewise = app.tab1_state.compute_scaling_preview();
    assert_eq!(pred_piecewise.transform_id, algo_scale::SizeRule::Piecewise);
    assert!(!pred_piecewise.temperature_coupled, "Piecewise must decouple temperature");
    assert_eq!(pred_piecewise.predicted_temperature, 0.85);
    assert!(pred_piecewise.sigma > 0.04 && pred_piecewise.sigma <= 0.50);

    // 2. Test Legacy 3B rule (coupled T)
    app.tab1_state.scaling_size_rule = algo_scale::SizeRule::Legacy;
    let pred_legacy = app.tab1_state.compute_scaling_preview();
    assert!(pred_legacy.temperature_coupled, "Legacy rule must couple temperature");
    assert!(pred_legacy.sigma <= 0.20, "Legacy rule has tight 0.20 sigma clamp");

    // 3. Test July 8B-Sqrt rule (coupled T)
    app.tab1_state.scaling_size_rule = algo_scale::SizeRule::EightBSqrt;
    let pred_8b = app.tab1_state.compute_scaling_preview();
    assert!(pred_8b.temperature_coupled, "July 8B rule must couple temperature");
    assert!(pred_8b.beta >= 70.0 && pred_8b.beta <= 220.0);

    // 4. Test Archetype multipliers (Thinking / Coding fragility)
    app.tab1_state.scaling_model_type = algo_scale::ModelType::Thinking;
    let pred_thinking = app.tab1_state.compute_scaling_preview();
    assert_eq!(pred_thinking.archetype_multiplier, 0.88); // July 8B multiplier for thinking

    // Render scaling panel
    terminal.draw(|f| app.render(f)).expect("Render scaling preview failed");
    let content = buffer_to_string(terminal.backend().buffer());
    assert!(content.contains("Sampling & Size Scaling"));
    assert!(content.contains("Predicted Knobs"));
    assert!(content.contains("T Coupling"));
}

#[test]
fn test_m2_tab2_physics_board_residual_and_logit_sliders() {
    let (mut app, mut terminal) = create_test_app(120, 32);
    app.active_tab = Tab::PhysicsBoard;

    // Field 0: residual.cap
    app.selected_index = 0;
    app.tab2_state.selected_field_idx = 0;
    let init_cap = app.tab2_state.residual_cap;
    app.handle_key_event(press_key(KeyCode::Right, KeyModifiers::empty()));
    assert!((app.tab2_state.residual_cap - (init_cap + 0.1)).abs() < 1e-4);

    // Field 1: residual.goal
    app.handle_key_event(press_key(KeyCode::Down, KeyModifiers::empty()));
    assert_eq!(app.tab2_state.selected_field_idx, 1);
    let init_goal = app.tab2_state.residual_goal;
    app.handle_key_event(press_key(KeyCode::Right, KeyModifiers::empty()));
    assert!((app.tab2_state.residual_goal - (init_goal + 0.01)).abs() < 1e-4);

    // Field 14: gov.on (toggle)
    app.selected_index = 14;
    app.tab2_state.selected_field_idx = 14;
    assert!(app.tab2_state.gov_on);
    app.handle_key_event(press_key(KeyCode::Enter, KeyModifiers::empty()));
    assert!(!app.tab2_state.gov_on);
    app.handle_key_event(press_key(KeyCode::Char('g'), KeyModifiers::empty()));
    assert!(app.tab2_state.gov_on);

    // Field 20: hands.repulsion
    app.selected_index = 20;
    app.tab2_state.selected_field_idx = 20;
    let init_rep = app.tab2_state.hands_repulsion;
    app.handle_key_event(press_key(KeyCode::Left, KeyModifiers::empty()));
    assert!((app.tab2_state.hands_repulsion - (init_rep - 0.1)).abs() < 1e-4);

    // Field 21: hands.beta
    app.selected_index = 21;
    app.tab2_state.selected_field_idx = 21;
    let init_beta = app.tab2_state.hands_beta;
    app.handle_key_event(press_key(KeyCode::Right, KeyModifiers::empty()));
    assert!((app.tab2_state.hands_beta - (init_beta + 0.05)).abs() < 1e-4);

    // Test 'r' Hotkey — adapt the selected transform onto the Hydro residual
    // seat. Must NOT dump formula-native σ onto residual.cap or Algo β onto hands.beta.
    let hands_beta_before_r = app.tab2_state.hands_beta;
    app.handle_key_event(press_key(KeyCode::Char('r'), KeyModifiers::empty()));
    let pred = app.tab1_state.compute_scaling_preview();
    let seat = app.tab1_state.predicted_hydro_seat();
    assert!(
        (app.tab2_state.residual_cap - seat.physics.force_cap).abs() < 1e-4,
        "R writes Hydro cap {}, not formula σ {}",
        app.tab2_state.residual_cap,
        pred.sigma
    );
    assert!((app.tab2_state.residual_cap - pred.sigma).abs() > 0.5);
    assert!((app.tab2_state.residual_goal - seat.physics.goal_force_scale).abs() < 1e-4);
    assert!((app.tab2_state.hands_beta - hands_beta_before_r).abs() < 1e-4);
    assert!(app.status_message.contains("apply_to_hydro_profile"));

    // Render Tab 2
    terminal.draw(|f| app.render(f)).expect("Render Tab 2 failed");
    let content = buffer_to_string(terminal.backend().buffer());
    assert!(content.contains("Surface 1: Residual Forces"));
    assert!(content.contains("Surface 2: Logit Biases"));
    assert!(content.contains("Surface 3: Layer Hook"));
}

#[test]
fn test_m2_tab2_hook_controls_interactive_panel() {
    let (mut app, mut terminal) = create_test_app(120, 32);
    app.active_tab = Tab::PhysicsBoard;

    // Field 23: hook.on
    app.selected_index = 23;
    app.tab2_state.selected_field_idx = 23;
    assert!(app.tab2_state.hook_on);
    app.handle_key_event(press_key(KeyCode::Char('h'), KeyModifiers::empty()));
    assert!(!app.tab2_state.hook_on);
    assert_eq!(app.status_message, "Hook: DISABLED");

    app.handle_key_event(press_key(KeyCode::Char('h'), KeyModifiers::empty()));
    assert!(app.tab2_state.hook_on);
    assert_eq!(app.status_message, "Hook: ENABLED");

    // Field 24: hook.site cycling
    app.selected_index = 24;
    app.tab2_state.selected_field_idx = 24;
    assert_eq!(app.tab2_state.hook_site, HookSite::PostMlp);
    app.handle_key_event(press_key(KeyCode::Right, KeyModifiers::empty()));
    assert_eq!(app.tab2_state.hook_site, HookSite::FinalNorm);
    app.handle_key_event(press_key(KeyCode::Right, KeyModifiers::empty()));
    assert_eq!(app.tab2_state.hook_site, HookSite::PreLayer);
    app.handle_key_event(press_key(KeyCode::Right, KeyModifiers::empty()));
    assert_eq!(app.tab2_state.hook_site, HookSite::PostAttn);

    // Field 25: hook.norm_fraction
    app.selected_index = 25;
    app.tab2_state.selected_field_idx = 25;
    let init_frac = app.tab2_state.hook_norm_fraction;
    app.handle_key_event(press_key(KeyCode::Right, KeyModifiers::empty()));
    assert!((app.tab2_state.hook_norm_fraction - (init_frac + 0.0005)).abs() < 1e-6);

    // Field 26: hook.start_frac & Field 27: hook.end_frac
    app.selected_index = 26;
    app.tab2_state.selected_field_idx = 26;
    app.tab2_state.hook_start_frac = 0.25;
    app.tab2_state.hook_end_frac = 0.75;

    // Render Tab 2 with Hook Controls
    terminal.draw(|f| app.render(f)).expect("Render Hook Controls failed");
    let content = buffer_to_string(terminal.backend().buffer());
    assert!(content.contains("Surface 3: Layer Hook"));
    assert!(content.contains("Hook Telemetry"));
    assert!(content.contains("Resolved Band"));
}

#[test]
fn test_m2_realtime_engine_bridge_bidirectional_state_sync() {
    let config = Config::default();
    let bridge = EngineBridge::spawn(config, Some("models/gemma-4-9b-it-Q4_K_M.gguf".to_string()), true);
    let mut app = App::new(bridge, Some("gemma-4-9b-it-Q4_K_M.gguf".to_string()));

    // 1. Drain initial ready and model loaded
    std::thread::sleep(Duration::from_millis(60));
    app.update();
    assert_eq!(app.model_name, "models/gemma-4-9b-it-Q4_K_M.gguf");

    // 2. Adjust parameter on Tab 2 -> verify message send
    app.active_tab = Tab::PhysicsBoard;
    app.selected_index = 0;
    app.tab2_state.selected_field_idx = 0;
    app.handle_key_event(press_key(KeyCode::Right, KeyModifiers::empty())); // residual.cap += 0.1
    assert_eq!(app.status_message, "Applied physics adjustment");

    // 3. Adjust hook control -> verify message send
    app.selected_index = 25;
    app.tab2_state.selected_field_idx = 25;
    app.handle_key_event(press_key(KeyCode::Right, KeyModifiers::empty())); // norm fraction += 0.0005

    // 4. Send LoadModel for Qwen
    let _ = app.engine_bridge.send(UiToEngineMsg::LoadModel {
        path: "models/qwen2.5-7b-instruct.gguf".to_string(),
        tokenizer: Some("data/qwen/tokenizer.json".to_string()),
    });

    std::thread::sleep(Duration::from_millis(80));
    app.update();

    assert_eq!(app.model_name, "models/qwen2.5-7b-instruct.gguf");
    assert_eq!(app.tab1_state.loaded_model_idx, Some(2)); // Qwen is index 2
}

#[test]
fn test_m2_slider_writes_config_seat_not_hashmap() {
    let config = Config::default();
    let start_cap = config.physics.force_cap;
    let bridge = EngineBridge::spawn(config, None, true);
    let mut app = App::new(bridge, None);
    std::thread::sleep(Duration::from_millis(40));
    app.update();

    app.active_tab = Tab::PhysicsBoard;
    app.selected_index = 0;
    app.tab2_state.selected_field_idx = 0;
    app.handle_key_event(press_key(KeyCode::Right, KeyModifiers::empty()));
    std::thread::sleep(Duration::from_millis(80));
    app.update();

    let frame_cap = app
        .last_hud_frame
        .as_ref()
        .expect("TelemetryUpdate after residual.cap nudge")
        .force_cap;
    assert!(
        (frame_cap - (start_cap + 0.1)).abs() < 1e-3,
        "live residual.cap must land on Config.physics.force_cap (got {}, start {})",
        frame_cap,
        start_cap
    );
    assert!(
        (frame_cap - 0.15).abs() > 0.5,
        "must not confuse Hydro cap with formula-native σ"
    );

    let _ = app.engine_bridge.send(UiToEngineMsg::SetLiveParam {
        key: "sigma".to_string(),
        val: 0.15,
    });
    std::thread::sleep(Duration::from_millis(80));
    app.update();
    assert!(
        app.status_message.contains("unknown live param"),
        "formula-native σ is rejected, got {}",
        app.status_message
    );
}

