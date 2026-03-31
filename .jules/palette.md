## 2024-03-17 - Improve Accessible Label Connections
**Learning:** When modifying frontend HTML like `tools/splatlens_viewer.html`, disconnected `<label>` elements create warnings for screen readers. Using `for` attributes explicitly links them to inputs. For non-input elements, standard text elements like `<span class="setting-label">` should be used instead of `<label>`.
**Action:** Ensure all `<label>` tags use the `for` attribute or are styled spans for text groupings.
## 2024-03-21 - Smooth Accessible UI Panels
**Learning:** When implementing interactive UI panels or disclosure widgets (e.g., settings panels), abrupt `display: none`/`block` toggles reduce the perceived quality of the interface. Replacing them with smooth CSS transitions (e.g., using `opacity`, `visibility`, `transform`) greatly enhances the experience. Furthermore, standard accessibility must always accompany these interactions: set `aria-expanded` and `aria-controls` on the toggle button, update the ARIA state dynamically in JavaScript, and add an `Escape` key listener that closes the panel and returns keyboard focus to the toggle button.
**Action:** Always replace abrupt `display` toggles on interactive panels with CSS transitions, and implement ARIA state management alongside keyboard (Escape) closure.
## 2024-03-25 - ARIA States for Stateful Buttons
**Learning:** Stateful toggle buttons require `aria-pressed` to dynamically communicate their active state to screen readers. Grouping these controls with `role="group"` and `aria-labelledby` ensures context is provided when navigating via keyboard.
**Action:** Use `aria-pressed` to reflect state changes for all toggle buttons, and group related settings or playback controls structurally with `role="group"`.
