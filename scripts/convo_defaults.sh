# Shared defaults for HUMAN talk and AI multi-turn smoke.
# Source this file — do not fork settings per person.
#
# Research: one-shot smokes are not multi-turn ready.
#   research_logs/2026-07-28_gemma4-multiturn-diagnosis-vs-oneshot.md
#
# Override anything with env before launching:
#   HYDRO_MODEL=... HYDRO_CONFIG=... HYDRO_TOKENS=80 ./scripts/talk.sh

# Config: same knobs for Jason, Grok, Shep, Echo.
# Isolation = pure forward baseline (force_cap 0). Full stack later:
#   HYDRO_CONFIG=configs/gates/config.three_surface.toml
: "${HYDRO_CONFIG:=configs/ablation/config_isolation_baseline.toml}"

# Model: Gemma 4 12B default (fits RAM; same card both surfaces).
# 31B: HYDRO_MODEL=data/google/bart_google_gemma-4-31B-it-Q4_K_M.gguf
: "${HYDRO_MODEL:=data/google/gemma-4-12b-it-Q4_K_M.gguf}"
if [[ ! -r "${HYDRO_MODEL}" && -r "${HOME}/models/gemma-4-12b-it-Q4_K_M.gguf" ]]; then
  HYDRO_MODEL="${HOME}/models/gemma-4-12b-it-Q4_K_M.gguf"
fi

: "${HYDRO_TOKENIZER:=data/google/gemma4_assets/tokenizer.json}"
# 64 cut mid-sentence constantly in long-form talk (Gemma then hyphen-glues on continue).
# 128 = room for short answers + one solid paragraph without wall-thrash.
: "${HYDRO_TOKENS:=128}"

# Shared chat flags (both surfaces).
# Isolation ablation wipes at process start. Full-stack three_surface is the
# ordinary seat: persist residual trails across death without KEEP.
#   HYDRO_KEEP_MEMORY=1  → persist even on isolation (compat)
#   HYDRO_KEEP_MEMORY=0  → wipe even on three_surface (measured clear arm)
# Do not use HYDRO_INJECT_TAG as continuity.
isolation_config=0
case "${HYDRO_CONFIG}" in
  *isolation*) isolation_config=1 ;;
esac
persist=0
if [[ "${HYDRO_KEEP_MEMORY:-}" == "1" ]]; then
  persist=1
elif [[ "${HYDRO_KEEP_MEMORY:-}" == "0" ]]; then
  persist=0
elif [[ "${isolation_config}" -eq 0 ]]; then
  persist=1
fi
if [[ "${persist}" -eq 1 ]]; then
  HYDRO_CHAT_FLAGS=(
    --chat
    --no-endocrine
    --no-termsplat
    --no-hud
  )
else
  HYDRO_CHAT_FLAGS=(
    --chat
    --clear-memory
    --no-save-memory
    --no-endocrine
    --no-termsplat
    --no-hud
  )
fi
