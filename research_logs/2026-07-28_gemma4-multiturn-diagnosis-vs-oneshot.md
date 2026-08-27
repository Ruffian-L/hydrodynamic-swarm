# Gemma 4 — multi-turn REPL vs one-shot smokes (baseline for tuning)

**Date:** 2026-07-28  
**Tree:** `/media/ruffianl/ghost_team/projects/hydrodynamic-swarm`  
**Authors:** Jason · Grok (xAI) co-engineer  

**Purpose of this note:** So someone else wiring a custom Gemma 4 path does not
spend days on **one-line smokes only**, conclude the stack “never makes sense,”
and keep digging without changing the test. We nearly did that. Multi-turn
showed we were closer than one-shot alone suggested — enough of a **baseline**
to ask “does more help? does less help?” on physics and generation settings.

Raw multi-turn transcripts stay local under gitignored `private/chats/` when
used. This log records **method and technical fixes**, not private dialogue.

---

## Finding

**One-shot smokes are necessary but not sufficient for a generation baseline.**

| What you run | What you can conclude | What you still do not know |
|--------------|----------------------|----------------------------|
| Short one-shot, non-English / symbol soup | Loader / RoPE / attention / wrap may be wrong | Multi-turn behavior |
| Short one-shot, readable English | Weights load; decode is not random | Stability over history; long-context SWA; wrap quality |
| Multi-turn REPL (several turns) | Whether coherence holds, frays, or crashes under history | Full residual/physics interaction (turn physics on only after this) |

**Recommendation:** After load works, run a **back-and-forth REPL** (`--chat` /
`scripts/chat_gemma4.sh`), not only `--prompt` one-liners. You may be closer
than single-shot results suggest. That multi-turn baseline is what lets you
tune more vs less (tokens, temperature, force scales, history length) with a
grounded read.

Chat here is a **generation diagnostic**, not a substitute for engineering.

---

## Mistakes we paid for (so you do not)

### 1. Full-attention RoPE without partial rotary

- **Symptom:** Model loads; short greedy completions are unreadable (random
  tokens / mixed scripts).  
- **Cause:** Full-attn layers need **partial** rotary (HF assets:
  `partial_rotary_factor ≈ 0.25`). Treating all `rope_freqs` pairs as active
  over full `head_dim` distorts positions.  
- **Fix:** Keep first `n_rot/2` inv-freq pairs; zero the rest (identity on
  non-rotated dims) in `src/gemma4.rs`.  
- **Check:** Load banner ~`n_rot_full=128 … keep_pairs=64 nonzero=64` on 12B
  with full head 512.

### 2. Config rejected `temperature = 0` → silent defaults

- **Symptom:** You intended near-vanilla (`force_cap=0`) but logs still show
  default force / field settings.  
- **Cause:** Validation required `T > 0` and fell back to full defaults.  
- **Fix:** Allow `T >= 0`; main uses argmax when `T ≈ 0`.  
- **Check:** No `[CONFIG] … using defaults` line; `force_cap` matches TOML.

### 3. IT wrap framing → exam / meta answers

- **Symptom:** Readable English, but list/exam style or “how to phrase…”
  instead of doing the request.  
- **Cause:** Extra instruction framing in the wrap (e.g. “answer in one short
  paragraph”) primes those modes. Observed with physics already near-off.  
- **Fix:** Minimal Gemma 4 wrap (user turn + empty thought channel per local
  assets). No extra system / “helpful” injection unless the operator adds it.  
- **Lesson:** Check the wrap before blaming residual physics.

### 4. SWA mask shape after window trim (multi-turn crash)

- **Symptom:** After long history re-prefill (~1k tokens):  
  `cannot broadcast [1039, 1039] to [1, 16, 1039, 1024]`.  
- **Cause:** Causal mask built as `[q, q]` while SWA trimmed K/V to window
  1024.  
- **Fix:** Mask shape `[q_len, kv_len]` with absolute key positions after trim
  (`make_mask` in `gemma4.rs`).  
- **Check:** Multi-turn past SWA window length should not crash (quality may
  still degrade — separate issue).

### 5. Multi-turn quality still open

- **Symptom:** Early turns coherent; later turns more noise / special-token
  debris while still loosely on topic.  
- **Status:** Not fully fixed. Use as the **tuning baseline** (more vs less
  history, T, max tokens, then residual scales).  
- **Do not** treat one-shot English as multi-turn ready.

---

## Suggested test ladder (baseline)

1. **Load banner** — `gemma4`, dual head dims, partial RoPE keep_pairs.  
2. **Near-vanilla one-shot** — `configs/gemma4/config.gemma4_greedy.toml`, clear memory,
   no endocrine: readable English.  
3. **Multi-turn REPL** — same config, 5–10+ turns: note fray vs crash.  
4. **Then** change one knob at a time (history length, T, force scales, √-scale
   residual, clamp) against that multi-turn baseline.

```bash
# one-shot
./target/release/hydrodynamic-swarm \
  --config configs/gemma4/config.gemma4_greedy.toml \
  --model /path/to/gemma-4-*-it-Q4_K_M.gguf \
  --tokenizer data/google/gemma4_assets/tokenizer.json \
  --prompt "…" --tokens 32 --clear-memory --no-endocrine

# multi-turn
./scripts/chat_gemma4.sh
```

---

## Related paths

| Path | Role |
|------|------|
| `src/gemma4.rs` | Loader, partial RoPE, SWA mask |
| `src/main.rs` | Minimal G4 wrap, `--chat`, optional local transcript |
| `src/config.rs` | `temperature >= 0` |
| `configs/gemma4/config.gemma4_greedy.toml` | Near-vanilla greedy probe |
| `scripts/chat_gemma4.sh` | REPL entry |
| `private/chats/` | Local transcripts only (gitignored) |

---

## Progression

- **Before:** Custom G4 load; one-shot only; easy to assume the whole path was
  wrong and never change the test.  
- **After:** Readable one-shot; multi-turn used as diagnostic baseline; SWA
  crash fixed; this note so others try REPL sooner.  
- **Next:** Tune from multi-turn baseline (more/less); residual/physics only
  after that baseline is understood; 31B A/B when ready.

---

## Open questions

- [ ] Sampling for multi-turn (slight T > 0 vs pure greedy)  
- [ ] History length / truncate under SWA  
- [ ] Empty thought channel for non-thinking IT  
- [ ] Residual scale / clamp after multi-turn is stable enough to measure  

---

**Authorship**  
- **Author:** Jason  
- **Co-engineer:** Grok (xAI)  
- **Intent:** Leave a clear trail so others do not rewalk the one-shot-only
  mistake. Professional wording; the failure is incomplete testing and our
  stack settings, not a judgment of the model or the operator.  

## 2026-07-28 target wiring addendum

The supported target launcher is now `scripts/chat_gemma4.sh`. Its defaults
resolve the worktree-local Gemma 4 31B GGUF, dedicated Gemma 4 tokenizer, and
the live three-surface config. `/tui` exposes all residual, logit, forward-hook,
and sampling controls in the loaded session. The earlier physics-off stable
baseline remains available with:

```bash
GEMMA4_CONFIG=configs/gemma4/config.gemma4_stable.toml \
  ./scripts/chat_gemma4.sh
```

A real CUDA/PTY smoke loaded 60 layers at hidden width 5376, entered `/tui`,
applied a live residual-cap edit, returned to chat, and generated `Ready`.
Gemma 4 also uses the complete EOS set from the local generation asset:
`[1, 106, 50]`.

An additional automatic-discovery run omitted both model and tokenizer flags.
It selected the Gemma 4 31B GGUF plus `gemma4_assets/tokenizer.json`; telemetry
at `logs/2026-07-29_03-29-27_gemma4_v3-forcecap3_T0_88_s12_a1_d25.jsonl`
records 30 `post_mlp` hook applications on the decode step.

**Addendum author:** Codex (OpenAI) + operator
