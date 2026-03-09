## 2024-05-24 - Thematic Focus Indicators
**Learning:** The SplatLens dashboard relies on custom controls (like `.ctrl-slider`) that use `outline: none` for aesthetic reasons, which breaks keyboard accessibility. Using the primary data visualization color (Turquoise `#40E0D0` from the "Trail") for focus rings creates a cohesive thematic UX pattern without breaking the custom styling.
**Action:** Applied a global `:focus-visible` outline using the thematic Turquoise color, ensuring all interactive elements remain accessible via keyboard while retaining their custom mouse-driven designs.
