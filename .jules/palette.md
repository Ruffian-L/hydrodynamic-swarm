## 2024-03-17 - Improve Settings Panel Toggles and Triggers
**Learning:** When building custom widget dialogs or setting panels like the one in `tools/splatlens_viewer.html`, `display: none` creates abrupt UX and negatively impacts accessibility transitions.
**Action:** Replace `display: none` / `display: block` with smooth opacity/transform CSS transitions, and ensure standard accessibility patterns (updating `aria-expanded` and an Escape key hook for dismissing) are added.
