"""
Piper Audio: Configurable Text-To-Speech with instant procedural wake chime.
"""

import os
os.environ["ORT_LOGGING_LEVEL"] = "3"

import sys
from pathlib import Path
import io
import wave
import numpy as np
import sounddevice as sd
from scipy.signal import resample
from piper import PiperVoice

SRC_DIR = Path(__file__).resolve().parent.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    from piper_brain.config import CONFIG
except ImportError:
    CONFIG = {}

MODELS_DIR = Path(__file__).resolve().parent / "models"
TARGET_HARDWARE_RATE = CONFIG.get("audio", {}).get("hardware_rate", 48000)

class PiperSpeaker:
    def __init__(self):
        audio_cfg = CONFIG.get("audio", {})
        raw_name = audio_cfg.get("voice_model", "en_US-ryan-high.onnx")
        
        # Normalize file stem and extension
        model_stem = raw_name[:-5] if raw_name.endswith(".onnx") else raw_name
        model_filename = f"{model_stem}.onnx"
        config_filename = f"{model_stem}.onnx.json"
        
        self.model_path = MODELS_DIR / model_filename
        self.config_path = MODELS_DIR / config_filename
        self.voice_name = model_stem
        
        if not self.model_path.exists():
            raise FileNotFoundError(f"Piper voice model not found: {self.model_path}")

        print(f"[Speaker] Loading Piper voice: {model_filename}...")
        self.voice = PiperVoice.load(str(self.model_path), config_path=str(self.config_path))
        
        self.volume = max(0.0, min(audio_cfg.get("volume", 0.45), 1.0))
        self.device_index = self._resolve_usb_device()
        self.chime_data = self._generate_chime()

    def _resolve_usb_device(self) -> int | None:
        devices = sd.query_devices()
        hint = CONFIG.get("audio", {}).get("speaker_device_hint", "usb2.0").lower()
        for idx, dev in enumerate(devices):
            name = dev["name"].lower()
            if (hint in name or "hw:0,0" in name) and dev["max_output_channels"] > 0:
                print(f"[Speaker] Output bound to [{idx}]: {dev['name']}")
                return idx
        for idx, dev in enumerate(devices):
            if "usb" in dev["name"].lower() and dev["max_output_channels"] > 0:
                return idx
        return None

    def _generate_chime(self) -> np.ndarray:
        duration = 0.08
        t = np.linspace(0, duration, int(TARGET_HARDWARE_RATE * duration), endpoint=False)
        envelope = np.sin(np.pi * t / duration) ** 2
        
        tone1 = np.sin(2 * np.pi * 880.0 * t) * envelope
        tone2 = np.sin(2 * np.pi * 1320.0 * t) * envelope
        
        chime = np.concatenate([tone1, tone2]) * 0.25 * 32767.0
        return chime.astype(np.int16)

    def play_chime(self):
        sd.play(self.chime_data, samplerate=TARGET_HARDWARE_RATE, device=self.device_index)
        sd.wait()

    def speak(self, text: str):
        if not text:
            return

        raw_wav_buffer = io.BytesIO()
        with wave.open(raw_wav_buffer, "wb") as wav_file:
            self.voice.synthesize_wav(text, wav_file)

        raw_wav_buffer.seek(0)
        with wave.open(raw_wav_buffer, "rb") as wf:
            orig_rate = wf.getframerate()
            frames = wf.readframes(wf.getnframes())

        if not frames:
            return

        audio_np = np.frombuffer(frames, dtype=np.int16).astype(np.float32)
        if orig_rate != TARGET_HARDWARE_RATE:
            num_samples = int(len(audio_np) * TARGET_HARDWARE_RATE / orig_rate)
            audio_np = resample(audio_np, num_samples)

        audio_final = (audio_np * self.volume).astype(np.int16)
        sd.play(audio_final, samplerate=TARGET_HARDWARE_RATE, device=self.device_index)
        sd.wait()


if __name__ == "__main__":
    speaker = PiperSpeaker()
    speaker.play_chime()
    
    # Extract clean persona name from model identifier (e.g., 'en_US-ryan-high' -> 'ryan')
    name_parts = speaker.voice_name.split("-")
    display_name = name_parts[1].capitalize() if len(name_parts) > 1 else speaker.voice_name
    
    speaker.speak(f"Testing speech synthesis with {display_name} voice.")