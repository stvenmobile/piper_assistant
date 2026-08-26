"""
Piper Audio: USB Microphone listener with Wake-Word activation and VAD command capture.
"""

import os
os.environ["ORT_LOGGING_LEVEL"] = "3"

from pathlib import Path
from collections import deque
import tempfile
import wave
import time
import re
import numpy as np
import sounddevice as sd
from scipy.signal import resample
from faster_whisper import WhisperModel

HARDWARE_SAMPLE_RATE = 48000
WHISPER_SAMPLE_RATE = 16000

# Common wake word variations and phonetic misspellings from Whisper
WAKE_WORDS = ["piper", "hey piper", "hi piper", "paper", "hey paper", "hi paper"]

class PiperListener:
    def __init__(
        self,
        model_size: str = "base.en",
        device: str = "cpu",
        compute_type: str = "int8",
        sample_rate: int = HARDWARE_SAMPLE_RATE
    ):
        print(f"[Listener] Loading faster-whisper ({model_size}) on {device} ({compute_type})...")
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)
        self.sample_rate = sample_rate
        self.input_device = self._resolve_usb_mic()
        self.energy_threshold = self._calibrate_noise_floor()

    def _resolve_usb_mic(self) -> int | None:
        """Finds sounddevice index for USB microphone hardware."""
        devices = sd.query_devices()
        for idx, dev in enumerate(devices):
            name = dev["name"].lower()
            if ("pnp" in name or "hw:1,0" in name) and dev["max_input_channels"] > 0:
                print(f"[Listener] Microphone bound to [{idx}]: {dev['name']}")
                return idx
        for idx, dev in enumerate(devices):
            if "usb" in dev["name"].lower() and dev["max_input_channels"] > 0:
                print(f"[Listener] Microphone fallback bound to [{idx}]: {dev['name']}")
                return idx
        return None

    def _calculate_rms(self, audio_chunk: np.ndarray) -> float:
        """Calculates RMS signal energy."""
        return float(np.sqrt(np.mean(audio_chunk.astype(np.float32) ** 2)))

    def _calibrate_noise_floor(self, duration: float = 1.0) -> float:
        """Calibrates baseline ambient noise."""
        print("[Listener] Calibrating ambient noise floor (remain silent)...")
        chunk_samples = int(self.sample_rate * 0.1)
        samples_to_read = int(duration / 0.1)
        rms_values = []

        with sd.InputStream(samplerate=self.sample_rate, channels=1, dtype="int16", device=self.input_device) as stream:
            for _ in range(samples_to_read):
                chunk, _ = stream.read(chunk_samples)
                rms_values.append(self._calculate_rms(chunk))

        ambient_peak = float(np.max(rms_values))
        ambient_mean = float(np.mean(rms_values))
        threshold = max(200.0, ambient_peak * 1.5)
        print(f"[Listener] Ambient Mean: {ambient_mean:.1f} | Peak: {ambient_peak:.1f} | Active Threshold: {threshold:.1f}")
        return threshold

    def _transcribe_buffer(self, raw_audio_48k: np.ndarray) -> str:
        """Helper to resample 48kHz PCM and transcribe using Whisper."""
        num_16k_samples = int(len(raw_audio_48k) * WHISPER_SAMPLE_RATE / self.sample_rate)
        audio_16k = resample(raw_audio_48k.astype(np.float32), num_16k_samples).astype(np.int16)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
            tmp_path = tmp_wav.name
            with wave.open(tmp_path, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(WHISPER_SAMPLE_RATE)
                wf.writeframes(audio_16k.tobytes())

        transcript = ""
        try:
            segments, info = self.model.transcribe(
                tmp_path,
                beam_size=1,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=300),
                condition_on_previous_text=False,
                temperature=0.0
            )
            raw_text = " ".join([seg.text for seg in segments]).strip()
            if raw_text and info.language_probability > 0.4:
                transcript = raw_text
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

        return transcript

    def _strip_wake_word(self, text: str) -> str:
        """Removes the wake word prefix from the prompt."""
        pattern = r"^(hey|hi|hello)?\s*(piper|paper)[,\.\?!]*\s*"
        return re.sub(pattern, "", text, flags=re.IGNORECASE).strip()

    def listen_for_wake_word_and_command(self, max_command_duration: float = 6.0, silence_timeout: float = 1.0, idle_timeout: float = 3.0) -> str:
        """
        Listens in non-blocking windows for wake-word activation.
        Yields control back every `idle_timeout` seconds if no speech is present.
        """
        chunk_duration = 0.1
        chunk_samples = int(self.sample_rate * chunk_duration)
        pre_roll = deque(maxlen=3)
        speech_frames = []
        is_speaking = False
        silence_start = None
        loop_start = time.time()

        with sd.InputStream(samplerate=self.sample_rate, channels=1, dtype="int16", device=self.input_device) as stream:
            while True:
                chunk, _ = stream.read(chunk_samples)
                rms = self._calculate_rms(chunk)

                if rms > self.energy_threshold:
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
                        break
                else:
                    pre_roll.append(chunk)

                if not is_speaking and (time.time() - loop_start > idle_timeout):
                    return ""
                if is_speaking and (time.time() - loop_start > max_command_duration):
                    break

        if not speech_frames or len(speech_frames) < 4:
            return ""

        raw_audio = np.concatenate(speech_frames, axis=0).flatten()
        heard_text = self._transcribe_buffer(raw_audio)
        if not heard_text:
            return ""

        print(f"[Listener Heard]: \"{heard_text}\"")

        clean_lower = heard_text.lower()
        wake_detected = any(w in clean_lower for w in WAKE_WORDS)

        if not wake_detected:
            return ""

        # Return full text so main.py handles chime and state transition
        return heard_text

    def listen_command_window(self, max_duration: float = 6.0, silence_timeout: float = 1.0) -> str:
        """Captures voice command follow-up during active engagement."""
        chunk_samples = int(self.sample_rate * 0.1)
        pre_roll = deque(maxlen=2)
        speech_frames = []
        is_speaking = False
        silence_start = None
        start_time = time.time()

        with sd.InputStream(samplerate=self.sample_rate, channels=1, dtype="int16", device=self.input_device) as stream:
            while True:
                chunk, _ = stream.read(chunk_samples)
                rms = self._calculate_rms(chunk)

                if rms > self.energy_threshold:
                    if not is_speaking:
                        is_speaking = True
                        print("[Listener] Follow-up voice detected, recording...")
                        speech_frames.extend(pre_roll)
                    silence_start = None
                    speech_frames.append(chunk)
                elif is_speaking:
                    speech_frames.append(chunk)
                    if silence_start is None:
                        silence_start = time.time()
                    elif time.time() - silence_start > silence_timeout:
                        break
                else:
                    pre_roll.append(chunk)

                if is_speaking and (time.time() - start_time > max_duration):
                    break
                if not is_speaking and (time.time() - start_time > 4.0):
                    return ""

        if not speech_frames:
            return ""

        raw_audio = np.concatenate(speech_frames, axis=0).flatten()
        return self._transcribe_buffer(raw_audio)