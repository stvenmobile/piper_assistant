"""Shared by the hand-run audio tools in this folder: makes src/ importable and picks the
microphone / speaker from config.yaml's device hints, exactly as the assistant does."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from piper_brain.config import CONFIG          # noqa: E402
from piper_audio.devices import resolve        # noqa: E402


def mic() -> int | None:
    return resolve(CONFIG["audio"]["mic_device_hint"], "input", "Tool")


def speaker() -> int | None:
    return resolve(CONFIG["audio"]["speaker_device_hint"], "output", "Tool")
