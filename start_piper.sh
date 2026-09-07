#!/usr/bin/env bash
# ==============================================================================
# Piper Assistant Startup Script
# ==============================================================================

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}"
VENV_PATH="${PROJECT_ROOT}/.venv"
CONFIG_FILE="${PROJECT_ROOT}/config.yaml"

echo "=== Initializing Piper Voice Assistant ==="

# 1. Check & Activate Python Virtual Environment
if [[ ! -d "${VENV_PATH}" ]]; then
    echo "[Error] Virtual environment not found at ${VENV_PATH}"
    echo "Please create one and install dependencies before launching."
    exit 1
fi

source "${VENV_PATH}/bin/activate"
echo "[Setup] Virtual environment activated: $(python3 --version)"

# 2. Hardware Audio Configuration & Unmute
echo "[Audio] Checking PulseAudio sinks..."
if command -v pactl &> /dev/null; then
    # Unmute and set default sink to 100% volume
    pactl set-sink-mute 1 0 2>/dev/null || true
    pactl set-sink-volume 1 100% 2>/dev/null || true
    pactl set-default-sink 1 2>/dev/null || true
    echo "[Audio] Analog audio sink initialized and unmuted."
fi

# 3. Environment Variables for ONNX & CUDA
export ORT_LOGGING_LEVEL="3"
export TOKENIZERS_PARALLELISM="false"
export PYTHONUNBUFFERED="1"

# 4. Directory Structure Pre-flight Checks
mkdir -p "${PROJECT_ROOT}/data/checkpoints"
mkdir -p "${PROJECT_ROOT}/obsidian/Experiments"
mkdir -p "${PROJECT_ROOT}/obsidian/Syntheses"

# 5. Trap Termination Signals for Clean Exit
cleanup() {
    echo -e "\n[Shutdown] Caught termination signal. Exiting cleanly..."
    exit 0
}
trap cleanup SIGINT SIGTERM

# 6. Launch Main Runtime
echo "[Runtime] Launching Piper Brain (main.py)..."
python3 "${PROJECT_ROOT}/src/main.py" "$@"
