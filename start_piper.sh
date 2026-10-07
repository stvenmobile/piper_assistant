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

# 2. Audio: Piper opens the SP-200 directly (raw ALSA, all 6 mic channels). If PulseAudio/PipeWire
#    holds the card, the device is busy and silently missing from Piper's device list - so switch
#    the PulseAudio card named by audio.speaker_device_hint to profile "off" (releases it).
if command -v pactl &> /dev/null; then
    HINT="$(python3 -c 'import sys; sys.path.insert(0, "src"); from piper_brain.config import CONFIG; print(CONFIG["audio"]["speaker_device_hint"])' 2>/dev/null || true)"
    CARD="$(pactl list short cards 2>/dev/null | grep -i -- "${HINT}" | head -1 | cut -f2 || true)"
    if [[ -n "${HINT}" && -n "${CARD}" ]]; then
        pactl set-card-profile "${CARD}" off || true
        echo "[Audio] Released ${CARD} from PulseAudio for Piper"
    else
        echo "[Audio] PulseAudio has no card matching '${HINT}' - nothing to release"
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

# 5. Vision (camera, face tracking -> the ring's attention arc), also in the background
VISION_ENABLED="$(python3 -c 'import sys; sys.path.insert(0, "src"); from piper_brain.config import CONFIG; print(CONFIG["vision"]["enabled"])' 2>/dev/null || echo False)"
if [[ "${VISION_ENABLED}" == "True" ]]; then
    python3 src/piper_vision/service.py &
    VISION_PID=$!
    trap 'kill ${HEAD_PID:-} ${VISION_PID} 2>/dev/null || true' EXIT
    echo "[Vision] Service started (pid ${VISION_PID})"
fi

# 6. Research (in its scheduled hours - research.windows, default 20:00-08:00), also in the background
RESEARCH_ENABLED="$(python3 -c 'import sys; sys.path.insert(0, "src"); from piper_brain.config import CONFIG; print(CONFIG["research"]["enabled"])' 2>/dev/null || echo False)"
if [[ "${RESEARCH_ENABLED}" == "True" ]]; then
    python3 src/piper_research/service.py &
    RESEARCH_PID=$!
    trap 'kill ${HEAD_PID:-} ${VISION_PID:-} ${RESEARCH_PID} 2>/dev/null || true' EXIT
    echo "[Research] Service started (pid ${RESEARCH_PID})"
fi

# 7. Run (Ctrl+C, or type q + Enter, to stop)
python3 src/main.py "$@"
