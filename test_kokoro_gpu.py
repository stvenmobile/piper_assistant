"""
Kokoro-82M GPU Benchmark & Audio Output Test on Jetson Orin NX.
"""

import time
import os
import torch

# Bypass cuDNN version mismatch by falling back to standard CUDA kernels
torch.backends.cudnn.enabled = False

import numpy as np
import sounddevice as sd
from scipy.signal import resample
from kokoro import KPipeline

HARDWARE_SAMPLE_RATE = 48000
KOKORO_SAMPLE_RATE = 24000

# Resolve USB Speaker (hw:0,0)
devices = sd.query_devices()
output_device = next(
    (i for i, d in enumerate(devices) if ("usb2.0" in d["name"].lower() or "hw:0,0" in d["name"].lower()) and d["max_output_channels"] > 0),
    None
)
print(f"[Audio] Target output device index: {output_device}")

print("[Kokoro] Loading KPipeline model into Jetson Orin GPU memory...")
pipeline = KPipeline(lang_code="a", device="cuda")

test_phrases = [
    "Kokoro speech synthesis is fully operational on the Jetson Orin GPU.",
    "Evaluating prosody, inflection, and real-time generation speed."
]

voice = "af_heart"

for idx, text in enumerate(test_phrases):
    print(f"\n--- Synthesizing Phrase {idx + 1} ---")
    print(f"Text: \"{text}\"")
    
    t_start = time.perf_counter()
    generator = pipeline(text, voice=voice, speed=1.0)
    
    audio_chunks = []
    for gs, ps, audio in generator:
        if isinstance(audio, torch.Tensor):
            audio_np = audio.detach().cpu().numpy()
        else:
            audio_np = np.array(audio, dtype=np.float32)
        audio_chunks.append(audio_np)
        
    t_elapsed = (time.perf_counter() - t_start) * 1000
    
    if not audio_chunks:
        print("[Error] No audio generated.")
        continue

    combined_audio = np.concatenate(audio_chunks)
    duration_s = len(combined_audio) / KOKORO_SAMPLE_RATE
    rtf = (t_elapsed / 1000.0) / duration_s

    print(f"Inference Time: {t_elapsed:.1f} ms | Audio Duration: {duration_s:.2f} s | Real-Time Factor (RTF): {rtf:.3f}x")

    # Resample 24kHz -> 48kHz for USB DAC hardware
    num_samples_48k = int(len(combined_audio) * HARDWARE_SAMPLE_RATE / KOKORO_SAMPLE_RATE)
    audio_48k = resample(combined_audio, num_samples_48k)
    audio_pcm = (audio_48k * 0.45 * 32767.0).astype(np.int16)

    sd.play(audio_pcm, samplerate=HARDWARE_SAMPLE_RATE, device=output_device)
    sd.wait()

vram_used_mb = torch.cuda.memory_allocated() / (1024 ** 2)
print(f"\n[CUDA] Resident GPU VRAM allocated: {vram_used_mb:.1f} MB")