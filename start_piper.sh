#!/usr/bin/env bash
# ==============================================================================
# Piper Assistant - start the voice runtime (src/main.py)
#   ./start_piper.sh            normal start
# ==============================================================================
set -eo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PATH="${PROJECT_ROOT}/.venv"

echo "=== Starting Piper Voice Assistant ==="

# 1. Python virtual environment
if [[ ! -d "${VENV_PATH}" ]]; then
    echo "[Error] Virtual environment not found at ${VENV_PATH}"
    echo "        Create it and install requirements.txt first."
    exit 1
fi
source "${VENV_PATH}/bin/activate"
echo "[Setup] Virtual environment: $(python3 --version)"

# 2. Audio: unmute the speaker sink named by audio.speaker_device_hint (the SP-200), if
#    PulseAudio/PipeWire is running. The assistant itself picks its devices by name.
if command -v pactl &> /dev/null; then
    HINT="$(python3 -c 'import sys; sys.path.insert(0, "src"); from piper_brain.config import CONFIG; print(CONFIG["audio"]["speaker_device_hint"])' 2>/dev/null || true)"
    SINK="$(pactl list short sinks 2>/dev/null | grep -i -- "${HINT}" | head -1 | cut -f2 || true)"
    if [[ -n "${HINT}" && -n "${SINK}" ]]; then
        pactl set-sink-mute "${SINK}" 0 || true
        echo "[Audio] Unmuted ${SINK}"
    else
        echo "[Audio] No PulseAudio sink matches '${HINT}' - leaving audio settings alone"
    fi
fi

# 3. Environment
export ORT_LOGGING_LEVEL="3"
export TOKENIZERS_PARALLELISM="false"
export PYTHONUNBUFFERED="1"

# 4. The head link (piper-watch's ESP32) in the background; stopped when Piper exits
cd "${PROJECT_ROOT}"
HEAD_ENABLED="$(python3 -c 'import sys; sys.path.insert(0, "src"); from piper_brain.config import CONFIG; print(CONFIG["head"]["enabled"])' 2>/dev/null || echo False)"
if [[ "${HEAD_ENABLED}" == "True" ]]; then
    python3 src/piper_head/link.py &
    HEAD_PID=$!
    trap 'kill ${HEAD_PID} 2>/dev/null || true' EXIT
    echo "[Head] Link service started (pid ${HEAD_PID})"
fi

# 5. Run (Ctrl+C, or type q + Enter, to stop)
python3 src/main.py "$@"
