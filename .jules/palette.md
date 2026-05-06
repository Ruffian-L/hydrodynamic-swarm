## 2024-03-17 - Improve Accessible Label Connections
**Learning:** When modifying frontend HTML like `tools/splatlens_viewer.html`, disconnected `<label>` elements create warnings for screen readers. Using `for` attributes explicitly links them to inputs. For non-input elements, standard text elements like `<span class="setting-label">` should be used instead of `<label>`.
**Action:** Ensure all `<label>` tags use the `for` attribute or are styled spans for text groupings.
## 2024-03-21 - Smooth Accessible UI Panels
**Learning:** When implementing interactive UI panels or disclosure widgets (e.g., settings panels), abrupt `display: none`/`block` toggles reduce the perceived quality of the interface. Replacing them with smooth CSS transitions (e.g., using `opacity`, `visibility`, `transform`) greatly enhances the experience. Furthermore, standard accessibility must always accompany these interactions: set `aria-expanded` and `aria-controls` on the toggle button, update the ARIA state dynamically in JavaScript, and add an `Escape` key listener that closes the panel and returns keyboard focus to the toggle button.
**Action:** Always replace abrupt `display` toggles on interactive panels with CSS transitions, and implement ARIA state management alongside keyboard (Escape) closure.

## 2024-06-25 - Spacebar Keyboard Shortcut for Playback Control
**Learning:** Implementing a keyboard shortcut (Spacebar) to toggle play/pause states greatly improves keyboard accessibility and general UX for media visualization components.
**Action:** Use global `keydown` event listeners for ' ' (Spacebar), but explicitly verify `e.target.tagName` to ignore `INPUT`, `TEXTAREA`, and `BUTTON` to avoid interfering with native browser behaviors (like spacebar triggering focused buttons or adding spaces in inputs).
