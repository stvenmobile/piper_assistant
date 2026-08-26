"""
Standalone Audio Diagnostic: Calibrate Noise, Test VAD, Transcribe, and Playback.
"""

import os
os.environ["ORT_LOGGING_LEVEL"] = "3"

import time
import tempfile
import wave
from collections import deque
import numpy as np
import sounddevice as sd
from scipy.signal import resample
from faster_whisper import WhisperModel

HARDWARE_RATE = 48000
WHISPER_RATE = 16000

MIC_INDEX = 1      # USB PnP Sound Device (hw:1,0)
SPEAKER_INDEX = 0  # USB2.0 Device (hw:0,0)

def calculate_rms(chunk: np.ndarray) -> float:
    return float(np.sqrt(np.mean(chunk.astype(np.float32) ** 2)))

def calibrate_ambient(duration: float = 2.0) -> float:
    print("\n[Step 1/3] Calibrating ambient noise on Mic [1] (remain silent)...")
    chunk_samples = int(HARDWARE_RATE * 0.1)
    samples_to_read = int(duration / 0.1)
    rms_values = []

    with sd.InputStream(samplerate=HARDWARE_RATE, channels=1, dtype="int16", device=MIC_INDEX) as stream:
        for _ in range(samples_to_read):
            chunk, _ = stream.read(chunk_samples)
            rms_values.append(calculate_rms(chunk))

    ambient_peak = float(np.max(rms_values))
    threshold = max(200.0, ambient_peak * 1.5)
    print(f" -> Ambient Peak RMS: {ambient_peak:.1f} | Trigger Threshold: {threshold:.1f}\n")
    return threshold

def record_voice_phrase(threshold: float, max_duration: float = 6.0, silence_timeout: float = 1.0) -> np.ndarray | None:
    print("[Step 2/3] Speak clearly into the microphone (e.g. 'Hello Piper')...\n")
    chunk_samples = int(HARDWARE_RATE * 0.1)
    pre_roll = deque(maxlen=3)
    speech_frames = []
    is_speaking = False
    silence_start = None
    start_time = time.time()

    with sd.InputStream(samplerate=HARDWARE_RATE, channels=1, dtype="int16", device=MIC_INDEX) as stream:
        while True:
            chunk, _ = stream.read(chunk_samples)
            rms = calculate_rms(chunk)
            
            bars = "#" * min(40, int(rms / 50))
            status = "RECORDING" if is_speaking else "LISTENING"
            print(f"\r[{status}] RMS: {rms:6.1f} | Threshold: {threshold:5.1f} | [{bars:<40}]", end="", flush=True)

            if rms > threshold:
                if not is_speaking:
                    is_speaking = True
                    speech_frames.extend(pre_roll)
                silence_start = None
                speech_frames.append(chunk)
            elif is_speaking:
                speech_frames.append(chunk)
                if silence_start is None:
                    silence_start = time.time()
                elif time.time() - silence_start > silence_timeout:
                    print("\n\n[Capture] Silence detected, processing...")
                    break
            else:
                pre_roll.append(chunk)

            if is_speaking and (time.time() - start_time > max_duration):
                print("\n\n[Capture] Max duration reached.")
                break
            if not is_speaking and (time.time() - start_time > 8.0):
                print("\n\n[Capture] Timeout: No speech detected.")
                return None

    if not speech_frames:
        return None

    return np.concatenate(speech_frames, axis=0).flatten()

def main():
    print("==========================================")
    print("      PIPER AUDIO PIPELINE DIAGNOSTIC     ")
    print("==========================================")

    threshold = calibrate_ambient(duration=2.0)
    raw_audio_48k = record_voice_phrase(threshold=threshold)
    
    if raw_audio_48k is None or len(raw_audio_48k) == 0:
        print("[Error] No audio captured.")
        return

    # Resample 48kHz -> 16kHz
    num_16k_samples = int(len(raw_audio_48k) * WHISPER_RATE / HARDWARE_RATE)
    audio_16k = resample(raw_audio_48k.astype(np.float32), num_16k_samples).astype(np.int16)

    print("[Step 3/3] Transcribing with faster-whisper...")
    model = WhisperModel("base.en", device="cpu", compute_type="int8")

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
        tmp_path = tmp_file.name
        with wave.open(tmp_path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(WHISPER_RATE)
            wf.writeframes(audio_16k.tobytes())

    try:
        segments, info = model.transcribe(
            tmp_path,
            beam_size=1,
            vad_filter=True,
            condition_on_previous_text=False,
            temperature=0.0
        )
        transcript = " ".join([s.text for s in segments]).strip()
        print("\n==========================================")
        print(f" TRANSCRIPTION: \"{transcript}\"")
        print(f" CONFIDENCE:    {info.language_probability * 100:.1f}%")
        print("==========================================\n")
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    print("[Playback] Playing back raw audio over Speaker [0]...")
    norm_audio = raw_audio_48k.astype(np.float32)
    peak = np.max(np.abs(norm_audio))
    if peak > 0:
        norm_audio = (norm_audio / peak) * 28000.0

    sd.play(norm_audio.astype(np.int16), samplerate=HARDWARE_RATE, device=SPEAKER_INDEX)
    sd.wait()
    print("[Done] Test complete.")

if __name__ == "__main__":
    main()