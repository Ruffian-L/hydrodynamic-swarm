# Setup — does this run on *my* machine?

Three tiers. Most people only need **Tier 0**.

| Tier | What you get | Needs |
|------|----------------|-------|
| **0 · Museum** | Watch checked-in demos in a browser | `git clone`, `python3`, a browser |
| **1 · Unit tests** | `cargo test` physics/memory/config | Rust 1.75+, no GPU |
| **2 · Generate** | Live steering + splat memory | NVIDIA GPU + CUDA 12+, GGUF model (~2–6 GB for 4B) |

---

## Tier 0 — view only (any OS)

```bash
git clone <this-repo>
cd hydrodynamic-swarm
./splat-lens museum
# or: ./splat-lens   → menu → 1
```

- Opens `http://127.0.0.1:8765/museum/` with demos from `tools/museum/demos/`.
- **No CUDA. No model weights. No `cargo build`.**
- Requires `python3` only (serves static files; browsers block raw `file://` JSON).

Check what your machine has:

```bash
./splat-lens check
```

---

## Tier 1 — tests / library compile (no GPU)

Default features enable CUDA (for this project’s generation path). For CPU-only:

```bash
# Install Rust: https://rustup.rs
cargo test --no-default-features --features with-candle
cargo clippy --no-default-features --features with-candle --all-targets

# Continuity tooling (no GPU, no model):
python3 scripts/continuity_selftest.py
```

Unit tests use `Device::Cpu`. You do not need `nvcc` or a model for this tier.

---

## Tier 2 — live generation (NVIDIA)

### Hardware / toolchain

- NVIDIA GPU, **≥ ~8 GB** VRAM for Gemma 3 4B Q4 (more for 27B).
- CUDA **toolkit 12+** on `PATH` (`nvcc` optional if `kernels/decay.ptx` is present; driver + libs required at runtime).
- Rust 1.75+ stable.
- Linux is the well-tested path (developed on aarch64 Ubuntu + Blackwell). Windows/WSL2 with NVIDIA should work if Candle/CUDA do; Metal is **not** wired.

Scripts auto-discover CUDA via `scripts/cuda_env.sh` (`CUDA_HOME`, `/usr/local/cuda*`, etc.). Override if needed:

```bash
export CUDA_HOME=/path/to/cuda
export PATH="$CUDA_HOME/bin:$PATH"
```

### Model weights (not in git)

GGUF files are **gitignored** (~GB each). Place under `data/google/`:

| File | Role |
|------|------|
| `data/google/gemma-3-4b-it-Q4_K_M.gguf` | **Recommended starter** (~2–3 GB) |
| `data/google/tokenizer.json` | Matching tokenizer |
| `data/google/gemma-3-27b-it-Q4_K_M.gguf` | Larger / slower (use `configs/profiles/config.27b.toml`) |

Checksums for committed hashes: `data/google/SHA256SUMS`.

Example download (accept Gemma license on Hugging Face first):

```bash
pip install -U "huggingface_hub[cli]"
huggingface-cli login
# Pick a gemma-3-4b-it GGUF repo you can access, then e.g.:
huggingface-cli download <org/repo> gemma-3-4b-it-Q4_K_M.gguf \
  --local-dir data/google
# Also place tokenizer.json next to it (or at data/google/tokenizer.json)
```

Or run `./splat-lens check` and follow the model-help prompts.

### Config

```bash
cp config.example.toml config.toml   # local only; config.toml is gitignored
# optional profiles: configs/profiles/config.27b.toml, configs/profiles/config.force_off.toml, configs/profiles/config.ramp_off.toml
```

### Run

```bash
cargo build --release --bin hydrodynamic-swarm
./run_swarm.sh
# or:
./splat-lens generate
./scripts/chat.sh
```

CLI:

```bash
cargo run --release --bin hydrodynamic-swarm -- \
  --model data/google/gemma-3-4b-it-Q4_K_M.gguf \
  --tokenizer data/google/tokenizer.json \
  --prompt "Explain the Physics of Friendship in one short paragraph." \
  --tokens 65
```

---

## What is intentionally *not* portable yet

- **CPU generation** — main path forces `Device::new_cuda(0)`.
- **Metal / wgpu** — sketches in `kernels/`, not integrated.
- **Gemma 4 / Gemma 3n** — files may exist on disk; loaders not wired (clear error if you pass them).
- **CI default image** — uses CPU features only; full GPU build needs a CUDA runner.

---

## Quick troubleshooting

| Symptom | Likely fix |
|---------|------------|
| Museum blank / CORS | Use `./splat-lens museum`, not `file://` |
| `CUDA GPU required` | Install driver + toolkit; or stay on Tier 0 |
| Model not found | Put GGUF under `data/google/` or pass `--model` |
| `nvcc not found` at build | Checked-in `kernels/decay.ptx` should suffice; install CUDA toolkit if build panics |
| `gemm-f16` build fails on aarch64 | Upstream issue; CUDA runtime path still works with GPU features |
| Wrong machine paths in old docs | Prefer `data/google/…` and `SETUP.md`; no `/home/...` fallbacks in code |

---

*Continuity lane (public): [`docs/CONTINUITY.md`](docs/CONTINUITY.md). Local operator ledger: `LOAD.md` (gitignored if present).*
