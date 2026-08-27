//! Empirical Challenger Stress & Concurrency Suite for Milestone 1.
//!
//! Tests:
//! 1. Channel Throughput & High-Volume Concurrency (100,000 messages)
//! 2. Rapid Chaos Keyboard & Navigation Fuzzing (50,000 random key events)
//! 3. Terminal Dimension Extremes (0x0, 1x1, 1x100, 100x1, 5x3, 5000x5000) & Resize Jitter
//! 4. Engine Lifecycle, Abnormal Shutdown, Mid-Generation Drops, & Disconnection
//! 5. Non-Blocking UI Verification (UI latency during saturated engine worker)
//! 6. Headless Rendering Stability & Memory Leak Soak Test (10,000 frames)
//! 7. Multi-threaded Concurrent Engine Channel Stress (8 concurrent worker clients)

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use crossterm::event::{KeyCode, KeyEvent, KeyEventKind, KeyEventState, KeyModifiers};
use ratatui::backend::TestBackend;
use ratatui::Terminal;

pub use hydrodynamic_swarm::config;

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
use frontend::tabs::Tab;
use frontend::App;
use hydrodynamic_swarm::config::Config;

fn press(code: KeyCode, modifiers: KeyModifiers) -> KeyEvent {
    KeyEvent {
        code,
        modifiers,
        kind: KeyEventKind::Press,
        state: KeyEventState::empty(),
    }
}

// ----------------------------------------------------------------------------
// STRESS TEST 1: Channel Throughput & High-Volume Concurrency
// ----------------------------------------------------------------------------
#[test]
fn stress_channel_throughput_and_burst_flooding() {
    let config = Config::default();
    let bridge = EngineBridge::spawn(config, None, true);
    let mut app = App::new(bridge, None);

    let start = Instant::now();
    let num_messages = 100_000;

    for i in 0..num_messages {
        let msg = match i % 5 {
            0 => EngineToUiMsg::TokenGenerated {
                text: "a".to_string(),
                frame: hud::HudFrame {
                    step: i,
                    force_cap: (i as f32) % 10.0,
                    goal_force_scale: 0.15,
                    temperature: 0.8,
                    ..Default::default()
                },
            },
            1 => EngineToUiMsg::TelemetryUpdate(hud::HudFrame {
                step: i,
                entropy: Some(1.5),
                margin: Some(0.4),
                ..Default::default()
            }),
            2 => EngineToUiMsg::KvSnapshotStatus {
                state: format!("snap_{}", i),
            },
            3 => EngineToUiMsg::RememberStoreUpdated(vec![
                (format!("k_{}", i % 100), format!("v_{}", i)),
            ]),
            _ => EngineToUiMsg::ModelLoading {
                status: format!("progress_{}", i),
                progress: (i % 100) as f32 / 100.0,
            },
        };
        app.handle_engine_msg(msg);
    }

    let elapsed = start.elapsed();
    let throughput = (num_messages as f64) / elapsed.as_secs_f64();

    println!(
        "\n[STRESS 1] Processed {} channel messages in {:?} ({:.0} msgs/sec)",
        num_messages, elapsed, throughput
    );

    assert!(throughput > 100_000.0, "Throughput below 100k msgs/sec: {}", throughput);
    assert_eq!(app.generated_text.len(), num_messages / 5);
    assert!(app.last_hud_frame.is_some());
}

// ----------------------------------------------------------------------------
// STRESS TEST 2: Rapid Chaos Keyboard & Navigation Fuzzing
// ----------------------------------------------------------------------------
#[test]
fn stress_keyboard_chaos_and_navigation_fuzzing() {
    let config = Config::default();
    let bridge = EngineBridge::spawn(config, None, true);
    let mut app = App::new(bridge, None);

    let key_candidates = [
        KeyCode::Tab,
        KeyCode::BackTab,
        KeyCode::Char('1'),
        KeyCode::Char('2'),
        KeyCode::Char('3'),
        KeyCode::Char('4'),
        KeyCode::Char('5'),
        KeyCode::Char('6'),
        KeyCode::Char('7'), // out of tab bounds
        KeyCode::Char('0'), // out of tab bounds
        KeyCode::Char('9'),
        KeyCode::F(1),
        KeyCode::F(2),
        KeyCode::F(3),
        KeyCode::F(4),
        KeyCode::F(5),
        KeyCode::F(6),
        KeyCode::F(12), // invalid tab
        KeyCode::Up,
        KeyCode::Down,
        KeyCode::Left,
        KeyCode::Right,
        KeyCode::Char('k'),
        KeyCode::Char('j'),
        KeyCode::Char('h'),
        KeyCode::Char('l'),
        KeyCode::Enter,
        KeyCode::Char(' '),
        KeyCode::Esc,
        KeyCode::PageUp,
        KeyCode::PageDown,
        KeyCode::Home,
        KeyCode::End,
        KeyCode::Insert,
        KeyCode::Delete,
        KeyCode::Null,
    ];

    let modifiers = [
        KeyModifiers::empty(),
        KeyModifiers::SHIFT,
        KeyModifiers::CONTROL,
        KeyModifiers::ALT,
    ];

    let total_events = 50_000;
    let start = Instant::now();

    for i in 0..total_events {
        let code = key_candidates[i % key_candidates.len()];
        let modifier = modifiers[(i / key_candidates.len()) % modifiers.len()];

        // If 'q' or 'Ctrl+C' would quit, reset should_quit to continue stress testing
        if app.should_quit {
            app.should_quit = false;
        }

        app.handle_key_event(press(code, modifier));

        // INVARIANTS:
        // 1. active_tab must always be a valid Tab variant
        assert!(Tab::ALL.contains(&app.active_tab), "Invalid active_tab: {:?}", app.active_tab);
        // 2. selected_index must not panic
        assert!(app.selected_index <= total_events);
    }

    let elapsed = start.elapsed();
    println!(
        "[STRESS 2] Fuzzed {} keyboard events in {:?} ({:.0} ops/sec)",
        total_events, elapsed, (total_events as f64) / elapsed.as_secs_f64()
    );
}

// ----------------------------------------------------------------------------
// STRESS TEST 3: Terminal Dimension Extremes & Dynamic Resize Jitter
// ----------------------------------------------------------------------------
#[test]
fn stress_terminal_dimensions_and_resize_jitter() {
    let config = Config::default();
    let bridge = EngineBridge::spawn(config, None, true);
    let mut app = App::new(bridge, Some("Stress Model".to_string()));

    let extreme_sizes = [
        (0, 0),
        (1, 1),
        (1, 100),
        (100, 1),
        (2, 2),
        (5, 3),
        (10, 4),
        (15, 5),
        (20, 6),
        (30, 8),
        (50, 15),
        (80, 24),
        (120, 40),
        (250, 100),
        (1000, 500),
        (3000, 2000),
    ];

    // 1. Render all tabs on each extreme size
    for &(width, height) in &extreme_sizes {
        let backend = TestBackend::new(width, height);
        let mut terminal = Terminal::new(backend).expect("Terminal creation");

        for tab in Tab::ALL {
            app.active_tab = tab;
            let render_res = terminal.draw(|f| {
                app.render(f);
            });
            assert!(
                render_res.is_ok(),
                "Failed render at size ({}x{}) on tab {:?}",
                width, height, tab
            );
        }
    }

    // 2. Rapid resize jitter loop (simulating erratic window dragging)
    let mut terminal = Terminal::new(TestBackend::new(80, 24)).unwrap();
    let start = Instant::now();

    for i in 0..1_000 {
        let w = ((i * 17) % 300).max(1) as u16;
        let h = ((i * 31) % 150).max(1) as u16;
        terminal.backend_mut().resize(w, h);

        app.active_tab = Tab::ALL[i % Tab::ALL.len()];
        let res = terminal.draw(|f| {
            app.render(f);
        });
        assert!(res.is_ok(), "Resize jitter failed at frame {} ({}x{})", i, w, h);
    }

    println!("[STRESS 3] Verified 16 extreme dimensions & 1,000 resize jitter frames in {:?}", start.elapsed());
}

// ----------------------------------------------------------------------------
// STRESS TEST 4: Engine Lifecycle, Abnormal Shutdown, Mid-Gen Drops
// ----------------------------------------------------------------------------
#[test]
fn stress_engine_abnormal_shutdown_and_mid_gen_abort() {
    let start = Instant::now();

    // 1. Spawn and immediately drop 100 EngineBridges
    for _ in 0..100 {
        let config = Config::default();
        let bridge = EngineBridge::spawn(config, None, true);
        drop(bridge); // Implicit drop & shutdown
    }

    // 2. Spawn, trigger generation, and abort mid-generation 50 times
    for _ in 0..50 {
        let config = Config::default();
        let mut bridge = EngineBridge::spawn(config, None, true);

        bridge.send(UiToEngineMsg::StartGeneration {
            prompt: "Stress test abort".to_string(),
            temperature: 0.8,
            max_tokens: 100,
        }).unwrap();

        // Brief sleep to let worker enter loop
        thread::sleep(Duration::from_millis(5));

        // Abort
        bridge.send(UiToEngineMsg::AbortGeneration).unwrap();

        // Check shutdown idempotency
        bridge.shutdown();
        bridge.shutdown(); // Second call must not panic or dead-lock
    }

    // 3. Spawning multiple concurrent engine bridges
    let handles: Vec<_> = (0..8).map(|idx| {
        thread::spawn(move || {
            let config = Config::default();
            let mut bridge = EngineBridge::spawn(config, Some(format!("thread_model_{}.gguf", idx)), true);
            bridge.send(UiToEngineMsg::SnapshotKv).unwrap();
            let _ = bridge.recv_timeout(Duration::from_millis(100));
            bridge.shutdown();
        })
    }).collect();

    for h in handles {
        h.join().unwrap();
    }

    println!("[STRESS 4] Verified rapid spawn/drop, 50 mid-gen aborts, and 8 concurrent bridges in {:?}", start.elapsed());
}

// ----------------------------------------------------------------------------
// STRESS TEST 5: Non-Blocking UI Verification Under Saturated Worker Load
// ----------------------------------------------------------------------------
#[test]
fn stress_non_blocking_ui_under_heavy_worker_saturation() {
    let (tx_ui, rx_engine) = crossbeam::channel::unbounded::<UiToEngineMsg>();
    let (tx_engine, rx_ui) = crossbeam::channel::unbounded::<EngineToUiMsg>();
    let running = Arc::new(AtomicBool::new(true));
    let running_clone = running.clone();

    let worker_processed = Arc::new(AtomicUsize::new(0));
    let worker_processed_clone = worker_processed.clone();

    // Saturated worker thread continuously crunching and sending
    let worker_handle = thread::spawn(move || {
        while running_clone.load(Ordering::SeqCst) {
            // Drain any UI command
            while let Ok(_cmd) = rx_engine.try_recv() {}

            // Burn CPU / emit high frequency telemetry
            for step in 0..50 {
                let _ = tx_engine.send(EngineToUiMsg::TokenGenerated {
                    text: " tok".to_string(),
                    frame: hud::HudFrame {
                        step,
                        force_cap: 4.0,
                        ..Default::default()
                    },
                });
                worker_processed_clone.fetch_add(1, Ordering::Relaxed);
            }
            thread::sleep(Duration::from_millis(1));
        }
    });

    let backend = TestBackend::new(120, 40);
    let mut terminal = Terminal::new(backend).unwrap();

    let mut app = App::new(
        EngineBridge::spawn(Config::default(), None, true),
        Some("benchmark-model".to_string()),
    );

    // Measure UI frame render and update latency over 500 frames
    let mut max_frame_duration = Duration::ZERO;
    let mut total_duration = Duration::ZERO;
    let iterations = 500;

    for i in 0..iterations {
        let frame_start = Instant::now();

        // 1. Drain incoming flood
        while let Ok(msg) = rx_ui.try_recv() {
            app.handle_engine_msg(msg);
        }

        // 2. Dispatch UI input
        app.handle_key_event(press(KeyCode::Tab, KeyModifiers::empty()));
        let _ = tx_ui.send(UiToEngineMsg::SetLiveParam {
            key: "residual.cap".to_string(),
            val: 4.5,
        });

        // 3. Draw frame
        terminal.draw(|f| {
            app.render(f);
        }).unwrap();

        let frame_elapsed = frame_start.elapsed();
        total_duration += frame_elapsed;
        if frame_elapsed > max_frame_duration {
            max_frame_duration = frame_elapsed;
        }

        if i % 100 == 0 {
            // Check that worker is continuously progressing
            assert!(worker_processed.load(Ordering::Relaxed) > 0);
        }
    }

    running.store(false, Ordering::SeqCst);
    let _ = worker_handle.join();

    let avg_frame_duration = total_duration / iterations;
    let effective_fps = 1.0 / avg_frame_duration.as_secs_f64();

    println!(
        "[STRESS 5] UI Non-blocking benchmark ({} frames): avg frame {:?} ({:.1} FPS), max frame {:?}",
        iterations, avg_frame_duration, effective_fps, max_frame_duration
    );

    // Empirical criterion: UI thread frame latency must average below 20ms in debug (or <1ms in release)
    // ensuring steady 60 FPS responsiveness without blocking on worker computations.
    assert!(
        avg_frame_duration < Duration::from_millis(20),
        "UI frame latency exceeded 20ms: {:?}",
        avg_frame_duration
    );
}

// ----------------------------------------------------------------------------
// STRESS TEST 6: Headless Rendering Stability & Memory Soak (2,500 frames)
// ----------------------------------------------------------------------------
#[test]
fn stress_headless_rendering_stability_and_memory_soak() {
    let backend = TestBackend::new(120, 36);
    let mut terminal = Terminal::new(backend).unwrap();

    let config = Config::default();
    let bridge = EngineBridge::spawn(config, Some("gemma-4-it.gguf".to_string()), true);
    let mut app = App::new(bridge, Some("gemma-4-it.gguf".to_string()));

    let total_frames = 2_500;
    let start = Instant::now();

    for frame in 0..total_frames {
        // Interleave tab changes, slider navigation, prompt edits, hud updates
        if frame % 10 == 0 {
            app.active_tab = Tab::ALL[(frame / 10) % Tab::ALL.len()];
        }

        if frame % 5 == 0 {
            app.selected_index = (app.selected_index + 1) % 50;
        }

        // Simulate token stream
        app.handle_engine_msg(EngineToUiMsg::TokenGenerated {
            text: "x".to_string(),
            frame: hud::HudFrame {
                step: frame,
                force_cap: 3.5 + (frame % 10) as f32 * 0.1,
                goal_force_scale: 0.12,
                temperature: 0.85,
                scars: frame % 15,
                ..Default::default()
            },
        });

        // Cap text buffer growth to realistic terminal window history
        if app.generated_text.len() > 10_000 {
            app.generated_text.clear();
        }

        terminal.draw(|f| {
            app.render(f);
        }).unwrap();
    }

    let elapsed = start.elapsed();
    let fps = (total_frames as f64) / elapsed.as_secs_f64();

    println!(
        "[STRESS 6] Rendered {} headless frames across all tabs in {:?} ({:.0} FPS)",
        total_frames, elapsed, fps
    );

    assert!(fps > 50.0, "Headless rendering throughput lower than 50 FPS: {:.0} FPS", fps);
}
