#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/setup_conda_env.sh [auto|h100|legacy] [env_name]
# Examples:
#   bash scripts/setup_conda_env.sh auto
#   bash scripts/setup_conda_env.sh h100 pro-h100

MODE="${1:-auto}"
ENV_NAME="${2:-}"

if ! command -v conda >/dev/null 2>&1; then
    echo "[ERROR] conda not found in PATH."
    exit 1
fi

detect_mode() {
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        echo "legacy"
        return
    fi

    if nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | grep -qi "H100"; then
        echo "h100"
    else
        echo "legacy"
    fi
}

if [[ "$MODE" == "auto" ]]; then
    MODE="$(detect_mode)"
fi

if [[ "$MODE" != "h100" && "$MODE" != "legacy" ]]; then
    echo "[ERROR] MODE must be one of: auto, h100, legacy"
    exit 1
fi

if [[ -z "$ENV_NAME" ]]; then
    ENV_NAME="pro-${MODE}"
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${ROOT_DIR}/envs/pro-${MODE}.yml"

if [[ ! -f "$ENV_FILE" ]]; then
    echo "[ERROR] missing env file: $ENV_FILE"
    exit 1
fi

echo "[INFO] mode=$MODE env_name=$ENV_NAME"
echo "[INFO] env_file=$ENV_FILE"

# Keep channel priority strict for reproducibility.
conda config --set channel_priority strict >/dev/null 2>&1 || true

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "[INFO] updating existing env: $ENV_NAME"
    conda env update -n "$ENV_NAME" -f "$ENV_FILE" --prune
else
    echo "[INFO] creating env: $ENV_NAME"
    conda env create -n "$ENV_NAME" -f "$ENV_FILE"
fi

echo "[INFO] installing git-lfs hooks in env"
conda run -n "$ENV_NAME" git lfs install --skip-repo || true

echo "[INFO] verifying runtime"
conda run -n "$ENV_NAME" python "${ROOT_DIR}/scripts/verify_runtime.py"

echo "[OK] conda env is ready: $ENV_NAME"
echo "[NEXT] activate with: conda activate $ENV_NAME"
