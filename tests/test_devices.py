from piper_audio.devices import find_device

DEVICES = [
    {"name": "NVIDIA Jetson APE: - (hw:1,0)", "max_input_channels": 2, "max_output_channels": 2},
    {"name": "USB PnP Sound Device: Audio (hw:2,0)", "max_input_channels": 1, "max_output_channels": 0},
    {"name": "SP-200 Speakerphone: USB Audio (hw:3,0)", "max_input_channels": 1, "max_output_channels": 2},
    {"name": "default", "max_input_channels": 32, "max_output_channels": 32},
]


def test_hint_matches_case_insensitively():
    assert find_device(DEVICES, "sp-200", "input") == 2
    assert find_device(DEVICES, "SP-200", "output") == 2


def test_hint_must_have_the_right_kind_of_channels():
    # the PnP device has no output channels, so an output hint for it falls back to USB
    assert find_device(DEVICES, "pnp", "input") == 1
    assert find_device(DEVICES, "pnp", "output") == 2


def test_falls_back_to_any_usb_device():
    assert find_device(DEVICES, "nothing-like-this", "input") == 1


def test_none_when_nothing_matches():
    assert find_device([DEVICES[0]], "sp-200", "input") is None
