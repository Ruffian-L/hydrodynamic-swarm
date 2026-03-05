## Merge Request Checklist

**Before merging, verify ALL items below. This checklist prevents the kind of progressive physics-disabling incident documented in `.agent/rules.md` (March 3-4, 2026).**

### 🔬 Physics Safety (CRITICAL)

- [ ] **No physics parameters zeroed or disabled** (`force_cap`, `dt`, `viscosity_scale`, `splat_alpha` must be > 0)
- [ ] **No "fixes" that hide problems** (raising thresholds to 999, setting `should_dream = false`, etc.)
- [ ] **Parameter changes are intentional and documented** (if changing physics values, explain why in commit message)
- [ ] **Config.toml changes match code defaults** (never create config.toml with all zeros "for testing")

### 🧪 Testing & Verification

- [ ] **All tests pass** (`cargo test --release`)
- [ ] **Clippy warnings resolved** (`cargo clippy --release -- -D warnings`)
- [ ] **Telemetry verified** (if touching physics code, verify non-zero force magnitudes in test run)
- [ ] **No background commands or `/tmp` usage** (all work stays in project directory)

### 📝 Code Quality

- [ ] **Commit messages are accurate** (if touching 7 files, don't label it "docs:")
- [ ] **One logical change per commit** (no bulk refactors without discussion)
- [ ] **No more than 3 files changed** (or explicitly approved for larger changes)
- [ ] **Error handling preserved** (no `2>&1` redirection on cargo run, preserve stderr telemetry)

### 📚 Documentation

- [ ] **Breaking changes documented** (update relevant docs/ files if behavior changes)
- [ ] **New features have examples** (especially for public API changes)
- [ ] **Research logs updated** (for significant physics or architecture changes)

---

## Description

**What does this MR do?**
<!-- Describe the change in 1-2 sentences -->

**Why is this change needed?**
<!-- Link to issue, roadmap item, or explain the problem being solved -->

**How was this tested?**
<!-- Describe your testing approach -->

**Physics Impact Assessment:**
<!-- If touching physics code, describe expected impact on steering forces, splat memory, or generation quality -->

---

## Related Links

- 📖 **Read first:** [Foundation](../docs/foundation.md) - explains the physics model
- ⚠️ **Safety rules:** [Agent Rules](../.agent/rules.md) - hard rules for this project
- 🗺️ **Roadmap:** [Phase 2 Roadmap](../docs/roadmap.md) - current development priorities
- 📊 **Code map:** [CODE_MAP.md](../CODE_MAP.md) - architecture overview

---

## For Reviewers

**Key areas to focus on:**
<!-- Highlight specific files, functions, or concepts that need careful review -->

**Potential risks:**
<!-- Any areas where this change could break existing functionality -->

/label ~"needs review"
