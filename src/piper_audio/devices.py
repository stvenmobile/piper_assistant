"""
Piper Audio: pick the microphone / speaker by name.

The SP-200 speakerphone shows up as one USB audio device that is both the microphone and the
speaker, so both hints in config.yaml usually name the same device. Run this file to list
what sounddevice sees and choose a hint (any unique part of the name):

    python3 src/piper_audio/devices.py
"""


def find_device(devices, hint: str, kind: str) -> int | None:
    """Index of the first device whose name contains `hint` (case-insensitive) and that has
    channels of the wanted kind ("input" or "output"); falls back to any USB device of that
    kind, then None (the system default)."""
    channels = "max_input_channels" if kind == "input" else "max_output_channels"
    usable = [(i, d) for i, d in enumerate(devices) if d.get(channels, 0) > 0]
    hint = (hint or "").lower()
    if hint:
        for i, d in usable:
            if hint in d["name"].lower():
                return i
    for i, d in usable:
        if "usb" in d["name"].lower():
            return i
    return None


def resolve(hint: str, kind: str, label: str = "Audio") -> int | None:
    """find_device() over the devices sounddevice can see, with a log line."""
    import sounddevice as sd
    devices = sd.query_devices()
    idx = find_device(devices, hint, kind)
    if idx is None:
        print(f"[{label}] No {kind} device matches '{hint}' or 'usb' - using the system default.")
    else:
        print(f"[{label}] {kind.capitalize()} bound to [{idx}]: {devices[idx]['name']}")
    return idx


if __name__ == "__main__":
    import sounddevice as sd
    for i, d in enumerate(sd.query_devices()):
        print(f"[{i:2}] in {d['max_input_channels']:2}  out {d['max_output_channels']:2}  "
              f"{int(d['default_samplerate'])} Hz  {d['name']}")
