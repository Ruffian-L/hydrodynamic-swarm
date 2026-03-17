## 2024-03-17 - Improve Accessible Label Connections
**Learning:** When modifying frontend HTML like `tools/splatlens_viewer.html`, disconnected `<label>` elements create warnings for screen readers. Using `for` attributes explicitly links them to inputs. For non-input elements, standard text elements like `<span class="setting-label">` should be used instead of `<label>`.
**Action:** Ensure all `<label>` tags use the `for` attribute or are styled spans for text groupings.
