## 2024-03-17 - Improve Accessible Label Connections
**Learning:** When modifying frontend HTML like `tools/splatlens_viewer.html`, disconnected `<label>` elements create warnings for screen readers. Using `for` attributes explicitly links them to inputs. For non-input elements, standard text elements like `<span class="setting-label">` should be used instead of `<label>`.
**Action:** Ensure all `<label>` tags use the `for` attribute or are styled spans for text groupings.
## 2024-03-19 - Smooth Accessibility for Disclosure Widgets
**Learning:** Abrupt display: none/block toggles are jarring and less accessible. Smooth transitions with opacity and visibility, combined with aria-expanded and Escape key listeners, significantly improve the micro-UX and accessibility of interactive settings panels.
**Action:** Use smooth CSS transitions and ensure standard ARIA and keyboard handling for all disclosure widgets instead of harsh display toggling.
