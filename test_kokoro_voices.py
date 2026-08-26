"""
Kokoro Voice Audition Tool for Jetson Orin NX.
Iterates through all primary Kokoro voices with real-time audio playback.
"""

import os
import time
import torch

# Bypass cuDNN mismatch
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
print(f"[Audio] Target device index: {output_device}")

print("[Kokoro] Initializing pipelines on CUDA...")
pipeline_us = KPipeline(lang_code="a", device="cuda")  # American English
pipeline_gb = KPipeline(lang_code="b", device="cuda")  # British English

VOICE_CATALOG = [
    # American Female
    ("af_heart", "American Female - Heart (Default expressive)", pipeline_us),
    ("af_bella", "American Female - Bella", pipeline_us),
    ("af_sarah", "American Female - Sarah", pipeline_us),
    ("af_nicole", "American Female - Nicole", pipeline_us),
    ("af_sky", "American Female - Sky", pipeline_us),
    
    # American Male
    ("am_adam", "American Male - Adam", pipeline_us),
    ("am_michael", "American Male - Michael", pipeline_us),
    ("am_george", "American Male - George", pipeline_us),
    ("am_eric", "American Male - Eric", pipeline_us),

    # British Female
    ("bf_emma", "British Female - Emma", pipeline_gb),
    ("bf_isabella", "British Female - Isabella", pipeline_gb),

    # British Male
    ("bm_george", "British Male - George", pipeline_gb),
    ("bm_lewis", "British Male - Lewis", pipeline_gb),
]

TEST_PHRASE = "Hello Steve! I am testing the natural inflection and cadence of this neural voice."

print(f"\nAuditioning {len(VOICE_CATALOG)} Kokoro voice presets...\n")

for voice_id, label, pipeline in VOICE_CATALOG:
    print(f"--- Playing: {label} [{voice_id}] ---")
    
    t0 = time.perf_counter()
    try:
        generator = pipeline(TEST_PHRASE, voice=voice_id, speed=1.0)
        chunks = []
        for gs, ps, audio in generator:
            if isinstance(audio, torch.Tensor):
                audio_np = audio.detach().cpu().numpy()
            else:
                audio_np = np.array(audio, dtype=np.float32)
            chunks.append(audio_np)
            
        elapsed_ms = (time.perf_counter() - t0) * 1000
        
        if not chunks:
            print(f"  [Error] No audio returned for {voice_id}.")
            continue

        combined = np.concatenate(chunks)
        duration_s = len(combined) / KOKORO_SAMPLE_RATE
        rtf = (elapsed_ms / 1000.0) / duration_s
        print(f"  Latency: {elapsed_ms:.1f}ms | Duration: {duration_s:.2f}s | RTF: {rtf:.3f}x")

        # Resample to 48kHz for USB DAC
        num_samples_48k = int(len(combined) * HARDWARE_SAMPLE_RATE / KOKORO_SAMPLE_RATE)
        audio_48k = resample(combined, num_samples_48k)
        audio_pcm = (audio_48k * 0.45 * 32767.0).astype(np.int16)

        sd.play(audio_pcm, samplerate=HARDWARE_SAMPLE_RATE, device=output_device)
        sd.wait()
        time.sleep(0.3)  # Brief pause between samples

    except Exception as e:
        print(f"  [Failed] Could not load or synthesize {voice_id}: {e}")

print("\nAudition complete.")