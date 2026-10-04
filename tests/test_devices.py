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


import numpy as np

from piper_audio.devices import for_output, open_channels, pick_channel


def test_raw_devices_open_all_channels_mixers_mono():
    assert open_channels(6) == 6          # SP-200 input (hw:2,0)
    assert open_channels(2) == 2          # SP-200 output
    assert open_channels(32) == 1         # 'pulse' / 'default'
    assert open_channels(0) == 1


def test_pick_channel_keeps_frames_by_one():
    block = np.arange(12).reshape(4, 3)
    assert pick_channel(block, 1).tolist() == [[1], [4], [7], [10]]
    assert pick_channel(block, 9).shape == (4, 1)        # out of range -> last channel


def test_for_output_duplicates_mono():
    mono = np.array([1, 2, 3], dtype=np.int16)
    assert for_output(mono, 2).tolist() == [[1, 1], [2, 2], [3, 3]]
    assert for_output(mono, 1).tolist() == [1, 2, 3]
