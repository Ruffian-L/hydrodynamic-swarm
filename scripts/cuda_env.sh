#!/usr/bin/env bash
# Portable CUDA toolkit discovery for launch scripts.
# Source from repo scripts:  source "$(dirname "$0")/cuda_env.sh"   (or via ROOT)
# Does nothing harmful if CUDA is absent (museum / unit-test path still works).

# Avoid re-sourcing
if [[ -n "${_HYDRO_CUDA_ENV_LOADED:-}" ]]; then
  return 0 2>/dev/null || true
fi
_HYDRO_CUDA_ENV_LOADED=1

hydro_prepend_path() {
  local d="$1"
  [[ -d "$d" ]] || return 0
  case ":${PATH:-}:" in
    *":$d:"*) ;;
    *) export PATH="$d${PATH:+:$PATH}" ;;
  esac
}

# Prefer explicit override, then common install layouts, then whatever is already on PATH.
if [[ -n "${CUDA_HOME:-}" ]]; then
  hydro_prepend_path "$CUDA_HOME/bin"
elif [[ -n "${CUDA_PATH:-}" ]]; then
  hydro_prepend_path "$CUDA_PATH/bin"
else
  for d in \
    /usr/local/cuda/bin \
    /usr/local/cuda-13.1/bin \
    /usr/local/cuda-13/bin \
    /usr/local/cuda-12/bin \
    /opt/cuda/bin
  do
    if [[ -x "$d/nvcc" ]] || [[ -x "$d/nvidia-smi" ]]; then
      hydro_prepend_path "$d"
      # Set CUDA_HOME from first hit if unset
      if [[ -z "${CUDA_HOME:-}" ]]; then
        export CUDA_HOME="$(cd "$d/.." && pwd)"
      fi
      break
    fi
  done
fi

# Lib path for dynamic loader (optional; some distros need this)
if [[ -n "${CUDA_HOME:-}" ]]; then
  for lib in "$CUDA_HOME/lib64" "$CUDA_HOME/lib"; do
    if [[ -d "$lib" ]]; then
      case ":${LD_LIBRARY_PATH:-}:" in
        *":$lib:"*) ;;
        *) export LD_LIBRARY_PATH="$lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" ;;
      esac
    fi
  done
fi
