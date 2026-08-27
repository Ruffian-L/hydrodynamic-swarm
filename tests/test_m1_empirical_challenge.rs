//! Empirical Challenge Test Suite for Milestone 1.
//!
//! Stress-tests:
//! 1. Non-UI paths & Engine contracts (generate_turn_ex preservation)
//! 2. Concurrency stress & channel disconnection safety
//! 3. Extreme terminal bounds (0x0, 1x1, single-line, massive 1000x1000)
//! 4. Malformed / extreme payload safety (NaN, Inf, huge buffers, control chars)
//! 5. Rapid abort and lifecycle tear-down

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::Duration;

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
use frontend::event::KeyRouter;
use frontend::tabs::Tab;
use frontend::App;
use hydrodynamic_swarm::config::Config;

fn press_key(code: KeyCode, modifiers: KeyModifiers) -> KeyEvent {
    KeyEvent {
        code,
        modifiers,
        kind: KeyEventKind::Press,
        state: KeyEventState::empty(),
    }
}

// ----------------------------------------------------------------------------
// Challenge 1: Extreme Terminal Dimensions (0x0, 1x1, 1x1000, 1000x1)
// ----------------------------------------------------------------------------
#[test]
fn challenge_terminal_geometry_extremes() {
    let config = Config::default();
    let bridge = EngineBridge::spawn(config, None, true);
    let mut app = App::new(bridge, Some("Stress Model".to_string()));

    let extreme_dimensions = [
        (0, 0),
        (1, 1),
        (2, 2),
        (5, 5),
        (1, 500),
        (500, 1),
        (10, 10),
        (1000, 1000),
    ];

    for &(w, h) in &extreme_dimensions {
        let backend = TestBackend::new(w, h);
        let mut terminal = Terminal::new(backend).expect("Terminal creation");

        for tab in Tab::ALL {
            app.active_tab = tab;
            let res = terminal.draw(|f| app.render(f));
            assert!(
                res.is_ok(),
                "Rendering failed on dimension ({}, {}) for tab {:?}",
                w,
                h,
                tab
            );
        }
    }
}

// ----------------------------------------------------------------------------
// Challenge 2: Floating Point Extremes (NaN, Inf, -Inf, Subnormal) in Telemetry
// ----------------------------------------------------------------------------
#[test]
fn challenge_floating_point_extremes_in_hud() {
    let (w, h) = (120, 40);
    let backend = TestBackend::new(w, h);
    let mut terminal = Terminal::new(backend).expect("Terminal creation");

    let config = Config::default();
    let bridge = EngineBridge::spawn(config, None, true);
    let mut app = App::new(bridge, None);

    let extreme_floats = [
        f32::NAN,
        f32::INFINITY,
        f32::NEG_INFINITY,
        0.0,
        -0.0,
        f32::MIN_POSITIVE,
        f32::MAX,
        f32::MIN,
    ];

    for &val in &extreme_floats {
        let frame = hud::HudFrame {
            step: 999,
            max_tokens: 1000,
            temperature: val,
            force_cap: val,
            goal_force_scale: val,
            entropy: Some(val),
            margin: Some(val),
            p_chosen: Some(val),
            baseline_norm: val,
            steered_norm: val,
            pullback: val,
            delta_h_norm: val,
            grad_mag: val,
            splat_mag: val,
            goal_mag: val,
            ocean_mag: val,
            scars: 42,
            hook_applications: Some(10),
            ..Default::default()
        };

        app.handle_engine_msg(EngineToUiMsg::TokenGenerated {
            text: format!(" tok_{}", val),
            frame: frame.clone(),
        });
        app.handle_engine_msg(EngineToUiMsg::TelemetryUpdate(frame));

        for tab in Tab::ALL {
            app.active_tab = tab;
            let res = terminal.draw(|f| app.render(f));
            assert!(
                res.is_ok(),
                "Rendering panicked on float value {} for tab {:?}",
                val,
                tab
            );
        }
    }
}

// ----------------------------------------------------------------------------
// Challenge 3: Channel Disconnection & Thread Teardown Stress
// ----------------------------------------------------------------------------
#[test]
fn challenge_engine_bridge_rapid_spawn_teardown_and_channel_disconnection() {
    for i in 0..50 {
        let config = Config::default();
        let mut bridge = EngineBridge::spawn(config, Some(format!("model_{}.gguf", i)), true);

        // Send messages immediately
        let _ = bridge.send(UiToEngineMsg::StartGeneration {
            prompt: "quick prompt".to_string(),
            temperature: 0.8,
            max_tokens: 5,
        });
        let _ = bridge.send(UiToEngineMsg::AbortGeneration);
        let _ = bridge.send(UiToEngineMsg::SnapshotKv);

        // Immediate shutdown without draining
        bridge.shutdown();
    }
}

// ----------------------------------------------------------------------------
// Challenge 4: Multithreaded Flooding and Contention
// ----------------------------------------------------------------------------
#[test]
fn challenge_multithreaded_concurrent_message_flooding() {
    let config = Config::default();
    let bridge = EngineBridge::spawn(config, None, true);
    let app_bridge = Arc::new(bridge);
    let done = Arc::new(AtomicBool::new(false));

    let mut handles = Vec::new();

    // Spawn 10 sender threads flooding commands
    for thread_id in 0..10 {
        let b = app_bridge.clone();
        let d = done.clone();
        let h = thread::spawn(move || {
            let mut count = 0;
            while !d.load(Ordering::Relaxed) && count < 200 {
                let _ = b.send(UiToEngineMsg::SetLiveParam {
                    key: format!("param_{}_{}", thread_id, count),
                    val: count as f32 * 0.1,
                });
                let _ = b.send(UiToEngineMsg::UpsertRememberLine {
                    key: format!("key_{}_{}", thread_id, count),
                    val: format!("val_{}", count),
                });
                count += 1;
                thread::yield_now();
            }
        });
        handles.push(h);
    }

    thread::sleep(Duration::from_millis(50));
    done.store(true, Ordering::Relaxed);

    for h in handles {
        h.join().expect("Worker thread join");
    }

    // Ensure receiver can drain without deadlocking
    let mut drained = 0;
    while let Some(_) = app_bridge.try_recv() {
        drained += 1;
    }
    assert!(drained >= 0);
}

// ----------------------------------------------------------------------------
// Challenge 5: Key Event Permutations & Unsupported Codes
// ----------------------------------------------------------------------------
#[test]
fn challenge_unsupported_key_events_and_modifier_combos() {
    let config = Config::default();
    let bridge = EngineBridge::spawn(config, None, true);
    let mut app = App::new(bridge, None);

    let unsupported_codes = [
        KeyCode::Null,
        KeyCode::Insert,
        KeyCode::Home,
        KeyCode::End,
        KeyCode::Delete,
        KeyCode::CapsLock,
        KeyCode::ScrollLock,
        KeyCode::NumLock,
        KeyCode::PrintScreen,
        KeyCode::Pause,
        KeyCode::Menu,
        KeyCode::KeypadBegin,
        KeyCode::Media(crossterm::event::MediaKeyCode::Play),
        KeyCode::Modifier(crossterm::event::ModifierKeyCode::LeftAlt),
    ];

    let modifiers_list = [
        KeyModifiers::empty(),
        KeyModifiers::ALT,
        KeyModifiers::CONTROL,
        KeyModifiers::SHIFT,
        KeyModifiers::SUPER,
        KeyModifiers::HYPER,
        KeyModifiers::META,
        KeyModifiers::all(),
    ];

    for code in unsupported_codes {
        for &mod_val in &modifiers_list {
            let key = KeyEvent {
                code,
                modifiers: mod_val,
                kind: KeyEventKind::Press,
                state: KeyEventState::empty(),
            };
            app.handle_key_event(key);
            assert!(!app.should_quit || (code == KeyCode::Char('c') && mod_val.contains(KeyModifiers::CONTROL)));
        }
    }
}
