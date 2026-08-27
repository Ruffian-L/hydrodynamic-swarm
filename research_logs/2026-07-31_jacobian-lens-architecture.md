# Jacobian Lens — Architecture Spec

**Date:** 2026-07-31
**Author:** Shep
**Status:** Draft — awaiting team review
**Goal:** Measure how hidden-state dimensions map to output logits. The Jacobian is the *key* — it turns clusters into perm-addresses.

---

## 1. The Core Hypothesis

> The Jacobian matrix J = ∂logits/∂hidden is emergent during forward pass. It is not a module — it is a *measurement lens* applied to the hidden state. Each row of J tells us: "if I nudge hidden dimension d, how does output token t change?"

This is Jason's insight: the Jacobian *is* your memory key. It's what comes up emergently as you think. We don't build it; we measure it.

---

## 2. What We Measure

For a given input sequence, we compute:

```
J[i][d] = (logits(hidden + ε·e_d) - logits(hidden - ε·e_d)) / (2ε)
```

Where:
- `hidden` is the D-dimensional hidden state (pre-lm_head)
- `e_d` is the unit vector along dimension d
- `ε` is a small perturbation (start with 1e-4, sweep 1e-5 to 1e-3)
- `logits(h)` is the output from `model.project_hidden_to_logits(h)`

**Result:** A matrix of shape `(vocab_size, hidden_dim)` — or more practically, `(top_k_tokens, hidden_dim)` for the top-k most-sensitive tokens.

---

## 3. The Jacobian Lens Module

### 3.1 File: `src/jacobian.rs`

```rust
/// A Jacobian measurement session.
/// 
/// Perturbs hidden state along selected dimensions, measures output changes,
/// and returns a sensitivity map: which hidden dimensions drive which outputs.
pub struct JacobianLens {
    epsilon: f32,
    sites: Vec<HookSite>,        // where to measure: FinalNorm, PostMlp, etc.
    top_k: usize,                 // only measure top-k output tokens
    max_dims: Option<usize>,      // subsample hidden dims for efficiency
    trace: Option<HookTrace>,     // optional persistent log
}

/// A single sensitivity measurement at one hook site.
pub struct JacobianMeasurement {
    pub site: HookSite,
    pub layer_idx: usize,
    pub sensitivity: Tensor,       // shape: (top_k, hidden_dim)
    pub norm: f32,                 // ||sensitivity||_F for summary
    pub top_dimensions: Vec<(usize, f32)>, // (dim, |J|) sorted descending
    pub top_tokens: Vec<(usize, f32)>,     // (token_id, |J|) sorted descending
}

/// Summary statistics across all sites.
pub struct JacobianReport {
    pub measurements: Vec<JacobianMeasurement>,
    pub global_sensitivity: f32,   // mean ||J|| across all sites
    pub dominant_dimensions: Vec<usize>, // dims with highest avg |J|
    pub dominant_tokens: Vec<usize>,     // tokens most sensitive to perturbations
}
```

### 3.2 Public API

```rust
impl JacobianLens {
    /// Create a new lens. `sites` = which hook sites to measure at.
    pub fn new(epsilon: f32, sites: Vec<HookSite>, top_k: usize) -> Self;
    
    /// Measure Jacobian at a single forward pass.
    /// Takes hidden state (b, D) at a specific layer/site.
    /// Returns measurements for each requested site.
    pub fn measure(
        &mut self,
        model: &mut Model,
        tokens: &Tensor,
        index_pos: usize,
    ) -> Result<JacobianReport>;
    
    /// Measure Jacobian with physics hook applied (steered vs unsteered comparison).
    pub fn measure_with_hook(
        &mut self,
        model: &mut Model,
        tokens: &Tensor,
        index_pos: usize,
        hook: &mut dyn LayerHook,
    ) -> Result<(JacobianReport, JacobianReport)>; // (unsteered, steered)
    
    /// Compute the "Jacobian fingerprint" — a compact summary vector.
    /// This is the perm-address for a given hidden state.
    pub fn fingerprint(&self, report: &JacobianReport) -> Tensor;
}
```

---

## 4. Integration Points

### 4.1 In `main.rs` — Add to forward_decode_with_hook

```rust
// After forward_decode_with_hook returns (logits, hidden, report):
let jacobian = if let Some(lens) = &mut jacobian_lens {
    lens.measure_at_site(&model, &tokens, index_pos, HookSite::FinalNorm)?
} else {
    JacobianReport::default()
};
```

### 4.2 In `hooks.rs` — Add JacobianHook (optional)

A lightweight hook that measures sensitivity *during* the forward pass without full finite-difference:

```rust
/// Measures ∂h/∂h at each hook site using identity perturbation.
/// More efficient than full Jacobian but captures local sensitivity.
pub struct JacobianHook {
    pub epsilon: f32,
    pub baseline: Option<Tensor>,
}

impl LayerHook for JacobianHook {
    fn apply(&mut self, site: HookSite, layer_idx: usize, h: &Tensor) 
        -> Result<Option<Tensor>> 
    {
        // Perturb h by ε along each dimension, measure change
        // Return both perturbed and delta
    }
}
```

### 4.3 In `config.toml`

```toml
[jacobian]
enabled = false
epsilon = 1.0e-4
sites = ["final_norm", "post_mlp"]  # which hook sites to measure
top_k = 50                           # top-k output tokens to track
max_dims = 0                         # 0 = all dims; >0 = subsample
trace_path = ""                      # optional: log to file
```

---

## 5. The "Perm-Address" Concept

Jason said: *"Jacobian is what comes up emergently as you think — it's your literal memory key."*

The Jacobian fingerprint becomes a **perm-address**:
1. Given a hidden state H, compute J(H)
2. Reduce J(H) to a compact vector (e.g., top-k dimensions × top-k tokens)
3. This vector is the "address" of this thought-state in the memory manifold
4. Similar states have similar Jacobians → clustering happens naturally
5. The Jacobian *is* the cluster key

### 5.1 Fingerprint Reduction

```rust
impl JacobianLens {
    /// Reduce Jacobian report to a compact fingerprint.
    /// Strategy: take top-N dimensions by mean |J|, then project.
    pub fn fingerprint(&self, report: &JacobianReport) -> Tensor {
        // 1. Compute mean |J| across all tokens for each dimension
        // 2. Select top-N dimensions
        // 3. Return as D'-dimensional vector (D' << D)
    }
}
```

---

## 6. Acceptance Criteria

### Phase 1: Smoke Test (MVP)
- [ ] `JacobianLens::new()` creates without error
- [ ] `measure()` runs on a 4-dim dummy model, returns non-zero sensitivity
- [ ] Output is finite (no NaN/Inf)
- [ ] Runs in <100ms on CPU for hidden_dim=64

### Phase 2: Real Model Integration
- [ ] Integrates with 3surface `forward_decode_with_hook`
- [ ] Measures at `FinalNorm` site on Gemma4 model
- [ ] Produces JacobianReport with shape `(top_k, hidden_dim)`
- [ ] Fingerprint is deterministic for same input
- [ ] Fingerprint differs for different inputs (sensitivity to input)

### Phase 3: Physics Comparison
- [ ] `measure_with_hook()` compares steered vs unsteered Jacobians
- [ ] Quantifies how physics hooks change the sensitivity landscape
- [ ] Reports: "Hook changed sensitivity of dimension d by X%"

### Phase 4: Memory Addressing
- [ ] Fingerprint clusters similar states (test with known prompts)
- [ ] Perm-address is stable across minor input perturbations
- [ ] Perm-address changes meaningfully for semantically different inputs

---

## 7. Design Decisions

### 7.1 Finite Difference vs Analytic

**Chosen: Finite Difference** (2ε perturbation per dimension)
- Proven, simple, works with any model
- Cost: 2× hidden_dim forward passes per measurement
- Optimization: subsample dimensions (max_dims), use top-k tokens only

**Alternative: Analytic via backprop**
- Cost: 1 backward pass for all dimensions
- Requires modifying model to expose ∂logits/∂hidden
- More complex, but O(1) instead of O(D)

**Decision:** Start with finite difference for correctness. Add analytic as optimization later.

### 7.2 Where to Measure

**Primary: `FinalNorm`** (pre-lm_head residual stream)
- This is where Jason's "emergent thinking" lives
- Same space across all models (Gemma, Llama, Gemma4)
- Direct input to lm_head → direct output sensitivity

**Secondary: `PostMlp`** (per-layer block output)
- Shows how sensitivity evolves through the stack
- Can track "sensitivity propagation" from early to late layers

### 7.3 Efficiency

For a 27B model with hidden_dim=5376:
- Full Jacobian: 2 × 5376 = 10,752 forward passes
- With top_k=50 tokens: still 10,752 passes (we measure full hidden, reduce output)
- With max_dims=256: 512 passes → acceptable for offline analysis
- With max_dims=64: 128 passes → fast enough for interactive use

---

## 8. Relationship to Other Systems

### 8.1 vs Splat Lens
- Splat lens: steers hidden state to attract toward concepts
- Jacobian lens: measures *which* dimensions drive *which* outputs
- They're complementary: splat pushes, Jacobian tells you where the push went

### 8.2 vs Niodoo Physics
- Niodoo: applies directional forces in hidden space
- Jacobian: measures sensitivity to those forces
- Together: Niodoo pushes, Jacobian maps the terrain

### 8.3 vs Memory Clustering
- Memory clustering: groups similar states
- Jacobian fingerprint: *is* the clustering key
- The Jacobian turns "similar states" from a heuristic into a measurement

---

## 9. File Structure

```
src/
  jacobian.rs          # JacobianLens, JacobianMeasurement, JacobianReport
  hooks.rs             # + JacobianHook (optional, lightweight)
  main.rs              # + jacobian integration in forward_decode_with_hook
config.toml            # + [jacobian] section
```

---

## 10. Next Steps

1. **Draft `jacobian.rs`** — MVP with dummy model
2. **Integrate with 3surface** — hook into forward pass
3. **Smoke test on real model** — verify non-trivial output
4. **Compare steered vs unsteered** — physics hook impact
5. **Fingerprint clustering test** — does it work as a memory key?

---

*This is the lens. We don't build the cathedral — we lay one brick and see if it holds.*
