"""
Piper Brain: Central configuration loader - the one place config.yaml is read.

Every setting has a default here; config.yaml overrides any of them (nested sections are
merged, so a partial section is fine), and a few environment variables override those.
"""

from pathlib import Path
import copy
import os
import yaml

try:
    from dotenv import load_dotenv
except ImportError:          # optional: only needed when a .env file is used
    load_dotenv = None

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = ROOT_DIR / "config.yaml"
ENV_PATH = ROOT_DIR / ".env"

DEFAULTS: dict = {
    "assistant": {
        "name": "Piper",
        "engaged_timeout_seconds": 20.0,
        "max_conversation_turns": 8,
    },
    "llm": {
        "provider": "ollama",
        "base_url": "http://192.168.1.150:11434",
        "model": "llama3.2:3b",
        "temperature": 0.4,
    },
    "audio": {
        # Substrings of the sounddevice names (run `python3 src/piper_audio/devices.py` to list
        # them). The SP-200 speakerphone is both the microphone and the speaker.
        "mic_device_hint": "usb",
        "speaker_device_hint": "usb",
        "hardware_rate": 48000,
        "whisper_rate": 16000,
        "whisper_model": "base.en",
        "whisper_compute": "int8",
        "tts_engine": "piper",            # "kokoro" (GPU) or "piper" (CPU)
        "kokoro_voice": "af_heart",
        "voice_model": "en_US-amy-medium.onnx",
        "volume": 0.45,
    },
    "weather": {
        "location": "Matthews,NC",
        "cache_minutes": 10,
    },
}

# environment variable -> (section, key)
ENV_OVERRIDES = {
    "PIPER_OLLAMA_URL": ("llm", "base_url"),
    "PIPER_LLM_MODEL": ("llm", "model"),
    "PIPER_VOICE_MODEL": ("audio", "voice_model"),
}


def deep_merge(base: dict, override: dict) -> dict:
    """Returns base updated with override; nested dicts are merged rather than replaced."""
    merged = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: Path = CONFIG_PATH, env: dict | None = None) -> dict:
    """Defaults <- config.yaml <- environment variables."""
    if env is None:
        if load_dotenv is not None:
            load_dotenv(ENV_PATH)
        env = os.environ

    user_cfg = {}
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            user_cfg = yaml.safe_load(f) or {}
    else:
        print(f"[Config] {path} not found - using defaults.")

    cfg = deep_merge(DEFAULTS, user_cfg)
    for var, (section, key) in ENV_OVERRIDES.items():
        if env.get(var):
            cfg[section][key] = env[var]
    return cfg


CONFIG = load_config()
