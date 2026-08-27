//! Input Event Loop and Crossterm Event Polling.
//!
//! Provides non-blocking event polling and routing for keyboard navigation,
//! mouse events, resize events, engine messages, and render ticks.

use std::time::Duration;
use crossterm::event::{self, Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers, MouseEvent};
use crate::frontend::channel::EngineToUiMsg;

/// Unified event type processed by the application loop.
#[derive(Debug, Clone)]
pub enum AppEvent {
    /// Terminal keyboard event
    Key(KeyEvent),
    /// Terminal mouse event
    Mouse(MouseEvent),
    /// Terminal window resize event (width, height)
    Resize(u16, u16),
    /// Message emitted from background engine worker thread
    Engine(EngineToUiMsg),
    /// Periodic tick event (~60 FPS) for UI animations and updates
    Tick,
}

/// Helper functions for keyboard routing.
pub struct KeyRouter;

impl KeyRouter {
    /// Check if the key event represents a request to quit the application.
    pub fn is_quit(key: &KeyEvent) -> bool {
        if key.kind != KeyEventKind::Press {
            return false;
        }
        match key.code {
            KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => true,
            KeyCode::Char('q') if key.modifiers.is_empty() => true,
            _ => false,
        }
    }

    /// Check if the key event is an escape or cancel action.
    pub fn is_cancel(key: &KeyEvent) -> bool {
        if key.kind != KeyEventKind::Press {
            return false;
        }
        key.code == KeyCode::Esc
    }

    /// Check if the key event requests switching to next tab.
    pub fn is_next_tab(key: &KeyEvent) -> bool {
        if key.kind != KeyEventKind::Press {
            return false;
        }
        key.code == KeyCode::Tab && !key.modifiers.contains(KeyModifiers::SHIFT)
    }

    /// Check if the key event requests switching to previous tab.
    pub fn is_prev_tab(key: &KeyEvent) -> bool {
        if key.kind != KeyEventKind::Press {
            return false;
        }
        key.code == KeyCode::BackTab
            || (key.code == KeyCode::Tab && key.modifiers.contains(KeyModifiers::SHIFT))
    }

    /// Check if the key event is a direct tab switch (1-6 or F1-F6).
    pub fn direct_tab(key: &KeyEvent) -> Option<usize> {
        if key.kind != KeyEventKind::Press {
            return None;
        }
        match key.code {
            KeyCode::Char('1') | KeyCode::F(1) => Some(0),
            KeyCode::Char('2') | KeyCode::F(2) => Some(1),
            KeyCode::Char('3') | KeyCode::F(3) => Some(2),
            KeyCode::Char('4') | KeyCode::F(4) => Some(3),
            KeyCode::Char('5') | KeyCode::F(5) => Some(4),
            KeyCode::Char('6') | KeyCode::F(6) => Some(5),
            _ => None,
        }
    }

    /// Check for vertical navigation (Up/Down or k/j).
    pub fn vertical_nav(key: &KeyEvent) -> Option<i32> {
        if key.kind != KeyEventKind::Press {
            return None;
        }
        match key.code {
            KeyCode::Up | KeyCode::Char('k') => Some(-1),
            KeyCode::Down | KeyCode::Char('j') => Some(1),
            _ => None,
        }
    }

    /// Check for horizontal adjustment (Left/Right arrow keys or PageUp/PageDown).
    /// Returns multiplier (-1 for left, +1 for right), multiplied by 10 if Shift is held.
    pub fn horizontal_adjust(key: &KeyEvent) -> Option<f32> {
        if key.kind != KeyEventKind::Press {
            return None;
        }
        let scale = if key.modifiers.contains(KeyModifiers::SHIFT) { 10.0 } else { 1.0 };
        match key.code {
            KeyCode::Left => Some(-1.0 * scale),
            KeyCode::Right => Some(1.0 * scale),
            KeyCode::PageUp => Some(10.0),
            KeyCode::PageDown => Some(-10.0),
            _ => None,
        }
    }

    /// Check for action confirmation (Enter or Space).
    pub fn is_confirm(key: &KeyEvent) -> bool {
        if key.kind != KeyEventKind::Press {
            return false;
        }
        matches!(key.code, KeyCode::Enter | KeyCode::Char(' '))
    }
}

/// Polls crossterm for input events with a bounded timeout.
pub fn poll_crossterm_event(timeout: Duration) -> std::io::Result<Option<AppEvent>> {
    if event::poll(timeout)? {
        match event::read()? {
            Event::Key(key) => Ok(Some(AppEvent::Key(key))),
            Event::Mouse(mouse) => Ok(Some(AppEvent::Mouse(mouse))),
            Event::Resize(w, h) => Ok(Some(AppEvent::Resize(w, h))),
            _ => Ok(None),
        }
    } else {
        Ok(None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_key_router_direct_tab() {
        let key_1 = KeyEvent::new(KeyCode::Char('1'), KeyModifiers::empty());
        assert_eq!(KeyRouter::direct_tab(&key_1), Some(0));

        let key_6 = KeyEvent::new(KeyCode::Char('6'), KeyModifiers::empty());
        assert_eq!(KeyRouter::direct_tab(&key_6), Some(5));

        let key_f3 = KeyEvent::new(KeyCode::F(3), KeyModifiers::empty());
        assert_eq!(KeyRouter::direct_tab(&key_f3), Some(2));

        let key_a = KeyEvent::new(KeyCode::Char('a'), KeyModifiers::empty());
        assert_eq!(KeyRouter::direct_tab(&key_a), None);
    }

    #[test]
    fn test_key_router_tab_cycling() {
        let tab_key = KeyEvent::new(KeyCode::Tab, KeyModifiers::empty());
        assert!(KeyRouter::is_next_tab(&tab_key));
        assert!(!KeyRouter::is_prev_tab(&tab_key));

        let shift_tab_key = KeyEvent::new(KeyCode::Tab, KeyModifiers::SHIFT);
        assert!(!KeyRouter::is_next_tab(&shift_tab_key));
        assert!(KeyRouter::is_prev_tab(&shift_tab_key));

        let backtab_key = KeyEvent::new(KeyCode::BackTab, KeyModifiers::empty());
        assert!(KeyRouter::is_prev_tab(&backtab_key));
    }

    #[test]
    fn test_key_router_quit_and_cancel() {
        let ctrl_c = KeyEvent::new(KeyCode::Char('c'), KeyModifiers::CONTROL);
        assert!(KeyRouter::is_quit(&ctrl_c));

        let q_key = KeyEvent::new(KeyCode::Char('q'), KeyModifiers::empty());
        assert!(KeyRouter::is_quit(&q_key));

        let esc_key = KeyEvent::new(KeyCode::Esc, KeyModifiers::empty());
        assert!(KeyRouter::is_cancel(&esc_key));
        assert!(!KeyRouter::is_quit(&esc_key));
    }

    #[test]
    fn test_key_router_navigation_and_adjustment() {
        let up = KeyEvent::new(KeyCode::Up, KeyModifiers::empty());
        assert_eq!(KeyRouter::vertical_nav(&up), Some(-1));

        let down = KeyEvent::new(KeyCode::Down, KeyModifiers::empty());
        assert_eq!(KeyRouter::vertical_nav(&down), Some(1));

        let left = KeyEvent::new(KeyCode::Left, KeyModifiers::empty());
        assert_eq!(KeyRouter::horizontal_adjust(&left), Some(-1.0));

        let right_shift = KeyEvent::new(KeyCode::Right, KeyModifiers::SHIFT);
        assert_eq!(KeyRouter::horizontal_adjust(&right_shift), Some(10.0));
    }
}
