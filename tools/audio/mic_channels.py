"""
Which input channel of a multi-channel speakerphone carries the echo-cancelled voice?

Plays a test tone through the speaker while recording every input channel, then records again
while you talk. The echo-cancelled channel is the one where the TONE is much quieter than on the
raw-microphone channels, but your VOICE is still clear. Put its number in config.yaml as
audio.mic_channel.

    python3 tools/audio/mic_channels.py
"""
import time
import numpy as np
import sounddevice as sd

from _devices import mic, speaker, channels_of, for_output, CONFIG

RATE = CONFIG["audio"]["hardware_rate"]
MIC, SPK = mic(), speaker()
IN_CH, OUT_CH = channels_of(MIC, "input"), channels_of(SPK, "output")


def record(seconds, play=None):
    """Record all input channels (optionally while playing `play`); RMS per channel."""
    frames = int(RATE * seconds)
    if play is not None:
        data = sd.playrec(for_output(play, OUT_CH), samplerate=RATE, channels=IN_CH, dtype="int16",
                          device=(MIC, SPK))
    else:
        data = sd.rec(frames, samplerate=RATE, channels=IN_CH, dtype="int16", device=MIC)
    sd.wait()
    data = data.astype(np.float32)
    skip = int(RATE * 0.3)                         # ignore the first 0.3 s (start-up)
    return np.sqrt(np.mean(data[skip:] ** 2, axis=0))


def show(title, rms):
    print(f"\n{title}")
    peak = max(rms.max(), 1.0)
    for ch, v in enumerate(rms):
        print(f"  channel {ch}: {v:8.1f}  {'#' * int(40 * v / peak)}")


print(f"Mic device [{MIC}] with {IN_CH} input channel(s), speaker [{SPK}] with {OUT_CH} output channel(s).")

print("\n1/3  Room noise - stay quiet for 3 s...")
quiet = record(3)
show("Room noise", quiet)

print("\n2/3  Test tone through the speaker - stay quiet...")
t = np.arange(int(RATE * 3)) / RATE
tone = (0.08 * 32767 * np.sin(2 * np.pi * 440 * t)).astype(np.int16)   # moderate level
toned = record(3, tone)
show("While the speaker plays a tone (the echo-cancelled channel should be LOW here)", toned)

print("\n3/3  Now talk normally for 4 s, starting... now")
time.sleep(0.3)
voiced = record(4)
show("While you talk (the channel you want should be clearly ABOVE the room noise)", voiced)

ratio = toned / np.maximum(voiced, 1.0)
best = int(np.argmin(ratio))
print(f"\nSuggestion: audio.mic_channel: {best}   (lowest tone-to-voice ratio: {ratio[best]:.2f}; "
      f"others {', '.join(f'{r:.2f}' for i, r in enumerate(ratio) if i != best)})")
