"""
Batch Voice Benchmark: Audition multiple Piper voice models sequentially.
"""

from pathlib import Path
import io
import wave
import numpy as np
import sounddevice as sd
from scipy.signal import resample
from piper import PiperVoice

MODELS_DIR = Path(__file__).resolve().parent / "src" / "piper_audio" / "models"
SAMPLE_TEXT = "Piper assistant is online and evaluating local neural voice quality."
TARGET_RATE = 48000

# Resolve output device (matches hw:0,0 USB DAC)
devices = sd.query_devices()
output_device = next((i for i, d in enumerate(devices) if "usb2.0" in d["name"].lower() and d["max_output_channels"] > 0), None)

for model_file in sorted(MODELS_DIR.glob("*.onnx")):
    config_file = model_file.with_suffix(".onnx.json")
    if not config_file.exists():
        continue

    print(f"\n--- Testing: {model_file.name} ---")
    voice = PiperVoice.load(str(model_file), config_path=str(config_file))
    
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        voice.synthesize_wav(SAMPLE_TEXT, wf)
    
    buf.seek(0)
    with wave.open(buf, "rb") as wf:
        rate = wf.getframerate()
        frames = wf.readframes(wf.getnframes())

    audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32)
    if rate != TARGET_RATE:
        audio = resample(audio, int(len(audio) * TARGET_RATE / rate))

    sd.play((audio * 0.45).astype(np.int16), samplerate=TARGET_RATE, device=output_device)
    sd.wait()
