# IMMUTABLE RUN CONTRACT

**Owner:** Jason Van Pham  
**Date locked:** 2026-07-30  

For anyone who touches this tree (Claude · Grok · Gemini · others).

---

## The rule

**Daily driver = full stack ON.**  
Tune gains, rates, thresholds, and how often systems fire.  
**Do not turn subsystems off to “fix” generation.**

If you disable systems to get a clean story, you are not finishing Niodoo-physics.  
You are repeating the **655-claim** failure mode: too many systems off, unreproducible soup, cascading caps.

**Ablate only after the full stack has a tuned baseline.**  
Never ablate first.

---

## Full stack (all live together)

| System | Role |
|--------|------|
| Residual / force | Live residual shove |
| **Learned wills** | Memory in residual space (not scar / not poison) |
| Logit surface | Vocab bias / governor |
| Hooks | Mid-stack inject when config enables |
| Endocrine → bloom → monolith | Mid-gen recovery path — **tune**, don’t kill |
| Online will deposits | +will / −will while generating |
| TermSplat | Run weather |
| jsonl + taco | Receipts |

Daily config: `configs/gates/config.three_surface.toml`  
Ablation only under `configs/ablation/` with an explicit receipt of **what was off**.

---

## Allowed without Jason

- Cooldowns, thresholds, force_cap, ramp, scales, max biases  
- **One dial at a time**, with a short receipt  

## Requires Jason (and guilt)

- `--no-endocrine` / force zero / hooks zero **as the default chat**  
- A “clean” config whose cleanliness is dead systems  
- Ablation before a full-stack baseline receipt  

---

## Guilt line

> Full stack stays on. Tune, don’t amputate. Ablate only after the orchestra is in tune.  
> `IMMUTABLE_RUN_CONTRACT.md`

---

## Vocab

**Learned wills** — `docs/VOCAB.md`. No scar / poison product language.
