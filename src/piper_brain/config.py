"""
Piper Brain: Central configuration loader.
"""

from pathlib import Path
import os
import yaml
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = ROOT_DIR / "config.yaml"
ENV_PATH = ROOT_DIR / ".env"

load_dotenv(ENV_PATH)

def load_config() -> dict:
    """Loads config.yaml with environment variable overrides."""
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Configuration file not found at {CONFIG_PATH}")

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    # Defaults and environment overrides
    cfg.setdefault("llm", {})
    cfg["llm"]["base_url"] = os.getenv("PIPER_OLLAMA_URL", cfg["llm"].get("base_url", "http://192.168.1.150:11434"))
    cfg["llm"]["model"] = os.getenv("PIPER_LLM_MODEL", cfg["llm"].get("model", "llama3.2:3b"))

    cfg.setdefault("audio", {})
    cfg["audio"]["voice_model"] = os.getenv("PIPER_VOICE_MODEL", cfg["audio"].get("voice_model", "en_US-amy-medium.onnx"))

    return cfg

CONFIG = load_config()