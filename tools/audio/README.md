# Audio tools

Hand-run checks for the Jetson's audio hardware (they need the speakerphone, and the Kokoro ones
need the GPU). Each uses the microphone / speaker named in `config.yaml`
(`audio.mic_device_hint`, `audio.speaker_device_hint`) - the same devices the assistant uses.

| Script | What it does |
|---|---|
| `audio_diagnostic.py` | calibrates the room noise, records until you stop talking, transcribes it with Whisper, and plays the recording back |
| `audition_piper_voices.py` | speaks a test sentence with every Piper voice model in `src/piper_audio/models/` |
| `audition_kokoro_voices.py` | speaks a test sentence with each Kokoro voice (American and British) |
| `kokoro_benchmark.py` | times Kokoro speech generation on the GPU and plays the result |
| `mic_channels.py` | for a multi-channel speakerphone (SP-200: 6 inputs): finds which input channel carries the echo-cancelled voice -> `audio.mic_channel` |

Run them from the repository root, e.g. `python3 tools/audio/audio_diagnostic.py`.
To list the audio devices first: `python3 src/piper_audio/devices.py`.
