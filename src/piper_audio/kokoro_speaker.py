"""
Kokoro GPU Speaker for Piper Assistant.
"""

import os
os.environ["ORT_LOGGING_LEVEL"] = "3"

import time
import torch

# Bypass cuDNN mismatch by using standard CUDA kernels
torch.backends.cudnn.enabled = False

from pathlib import Path
import numpy as np
import sounddevice as sd
from scipy.signal import resample
from kokoro import KPipeline

from piper_brain.config import CONFIG

HARDWARE_SAMPLE_RATE = 48000
KOKORO_SAMPLE_RATE = 24000

class KokoroSpeaker:
    def __init__(self):
        audio_cfg = CONFIG.get("audio", {})
        self.voice = audio_cfg.get("kokoro_voice", "af_heart")
        self.volume = max(0.0, min(audio_cfg.get("volume", 0.45), 1.0))
        
        print(f"[KokoroSpeaker] Initializing KPipeline on GPU (voice: {self.voice})...")
        self.pipeline = KPipeline(lang_code="a", device="cuda")
        
        self.device_index = self._resolve_usb_device()
        self.chime_data = self._generate_chime()

    def _resolve_usb_device(self) -> int | None:
        devices = sd.query_devices()
        hint = CONFIG.get("audio", {}).get("speaker_device_hint", "usb2.0").lower()
        for idx, dev in enumerate(devices):
            name = dev["name"].lower()
            if (hint in name or "hw:0,0" in name) and dev["max_output_channels"] > 0:
                print(f"[KokoroSpeaker] Output bound to [{idx}]: {dev['name']}")
                return idx
        for idx, dev in enumerate(devices):
            if "usb" in dev["name"].lower() and dev["max_output_channels"] > 0:
                return idx
        return None

    def _generate_chime(self) -> np.ndarray:
        duration = 0.08
        t = np.linspace(0, duration, int(HARDWARE_SAMPLE_RATE * duration), endpoint=False)
        envelope = np.sin(np.pi * t / duration) ** 2
        tone1 = np.sin(2 * np.pi * 880.0 * t) * envelope
        tone2 = np.sin(2 * np.pi * 1320.0 * t) * envelope
        chime = np.concatenate([tone1, tone2]) * 0.25 * 32767.0
        return chime.astype(np.int16)

    def play_chime(self):
        sd.play(self.chime_data, samplerate=HARDWARE_SAMPLE_RATE, device=self.device_index)
        sd.wait()

    def speak(self, text: str):
        if not text:
            return

        generator = self.pipeline(text, voice=self.voice, speed=1.0)
        audio_chunks = []
        for gs, ps, audio in generator:
            if isinstance(audio, torch.Tensor):
                audio_np = audio.detach().cpu().numpy()
            else:
                audio_np = np.array(audio, dtype=np.float32)
            audio_chunks.append(audio_np)

        if not audio_chunks:
            return

        combined_audio = np.concatenate(audio_chunks)
        num_samples_48k = int(len(combined_audio) * HARDWARE_SAMPLE_RATE / KOKORO_SAMPLE_RATE)
        audio_48k = resample(combined_audio, num_samples_48k)
        audio_pcm = (audio_48k * self.volume * 32767.0).astype(np.int16)

        sd.play(audio_pcm, samplerate=HARDWARE_SAMPLE_RATE, device=self.device_index)
        sd.wait()