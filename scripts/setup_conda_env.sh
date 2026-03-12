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

# Optional storage overrides for HPC/quota-limited environments.
# - CONDA_STORAGE_ROOT=/path/to/storage   -> sets both pkgs/envs roots
# - CONDA_PKGS_DIRS=/path/to/pkgs         -> conda package cache path
# - CONDA_ENVS_PATH=/path/to/envs         -> conda envs path
STORAGE_ROOT="${CONDA_STORAGE_ROOT:-}"
PKGS_DIRS="${CONDA_PKGS_DIRS:-}"
ENVS_PATH="${CONDA_ENVS_PATH:-}"

if [[ -n "$STORAGE_ROOT" ]]; then
    if [[ -z "$PKGS_DIRS" ]]; then
        PKGS_DIRS="${STORAGE_ROOT}/pkgs"
    fi
    if [[ -z "$ENVS_PATH" ]]; then
        ENVS_PATH="${STORAGE_ROOT}/envs"
    fi
fi

if [[ ! -f "$ENV_FILE" ]]; then
    echo "[ERROR] missing env file: $ENV_FILE"
    exit 1
fi

echo "[INFO] mode=$MODE env_name=$ENV_NAME"
echo "[INFO] env_file=$ENV_FILE"
if [[ -n "$PKGS_DIRS" ]]; then
    mkdir -p "$PKGS_DIRS"
    echo "[INFO] CONDA_PKGS_DIRS=$PKGS_DIRS"
fi
if [[ -n "$ENVS_PATH" ]]; then
    mkdir -p "$ENVS_PATH"
    echo "[INFO] CONDA_ENVS_PATH=$ENVS_PATH"
fi

CONDA_ENV_VARS=()
if [[ -n "$PKGS_DIRS" ]]; then
    CONDA_ENV_VARS+=("CONDA_PKGS_DIRS=$PKGS_DIRS")
fi
if [[ -n "$ENVS_PATH" ]]; then
    CONDA_ENV_VARS+=("CONDA_ENVS_PATH=$ENVS_PATH")
fi

conda_cmd() {
    env "${CONDA_ENV_VARS[@]}" conda "$@"
}

conda_run_in_env() {
    env "${CONDA_ENV_VARS[@]}" conda run -n "$ENV_NAME" "$@"
}

# Keep channel priority strict for reproducibility.
conda_cmd config --set channel_priority strict >/dev/null 2>&1 || true

if [[ "${CONDA_CLEAN_CACHE:-0}" == "1" ]]; then
    echo "[INFO] cleaning conda cache (CONDA_CLEAN_CACHE=1)"
    conda_cmd clean -a -y || true
fi

if conda_cmd env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "[INFO] updating existing env: $ENV_NAME"
    conda_cmd env update -n "$ENV_NAME" -f "$ENV_FILE" --prune
else
    echo "[INFO] creating env: $ENV_NAME"
    conda_cmd env create -n "$ENV_NAME" -f "$ENV_FILE"
fi

echo "[INFO] installing git-lfs hooks in env"
conda_run_in_env git lfs install --skip-repo || true

echo "[INFO] verifying runtime"
conda_run_in_env python "${ROOT_DIR}/scripts/verify_runtime.py"

echo "[OK] conda env is ready: $ENV_NAME"
echo "[NEXT] activate with: conda activate $ENV_NAME"
