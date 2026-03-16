## 2025-05-18 - Improve SplatLens Viewer Accessibility
**Learning:** HTML labels and dynamic text can break screen readers without proper aria tags and explicit attributes.
**Action:** For all future frontend tools, always ensure labels correctly use the "for" tag connected to input elements and span tags to avoid screen reader warnings. Also add "aria-live=polite" to all dynamic text.
