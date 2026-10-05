import math

import pytest

from piper_head import protocol
from piper_vision.tracker import Tracker, attention_deg

W, H = 1280, 720


def box(cx, cy, size=100, score=0.9):
    return (cx - size / 2, cy - size / 2, size, size, score)


def test_centred_face_is_straight_ahead():
    t = Tracker(W, H, hfov_deg=70.4)
    target = t.update([box(W / 2, H / 2)], 0.0)
    h, v = t.angles(target)
    assert abs(h) < 1e-6 and abs(v) < 1e-6


def test_frame_edge_is_half_the_field_of_view():
    t = Tracker(W, H, hfov_deg=70.4)
    target = t.update([box(W, H / 2)], 0.0)
    assert t.angles(target)[0] == pytest.approx(35.2, abs=0.01)


def test_picks_the_largest_face_then_sticks_with_it():
    t = Tracker(W, H, hfov_deg=70.4, smoothing=0.0)
    t.update([box(300, 360, 80), box(900, 360, 200)], 0.0)
    assert t.target.cx == pytest.approx(900)
    # a bigger face appears elsewhere: keep following the same person
    t.update([box(905, 360, 200), box(300, 360, 300)], 0.1)
    assert t.target.cx == pytest.approx(905)


def test_target_is_dropped_after_lost_s_then_reacquired():
    t = Tracker(W, H, hfov_deg=70.4, lost_s=0.8)
    t.update([box(640, 360)], 0.0)
    assert t.update([], 0.5) is not None          # briefly unseen: kept
    assert t.update([], 0.9) is None              # gone
    assert t.update([box(200, 360)], 1.0).cx == pytest.approx(200)


def test_smoothing_moves_part_way():
    t = Tracker(W, H, hfov_deg=70.4, smoothing=0.5)
    t.update([box(600, 360)], 0.0)
    t.update([box(640, 360)], 0.1)
    assert t.target.cx == pytest.approx(620)


def test_attention_glances_toward_the_person():
    # person to the camera's right = on the left as you face Piper = negative ring angle
    assert attention_deg(10, gain=2.5) == -25
    assert attention_deg(-10, gain=2.5) == 25
    assert attention_deg(80, gain=2.5) == -100     # clamped


def test_attention_message_leaves_state_alone():
    assert protocol.attention(-30) == {"t": "FACE", "attention": -30.0}
    assert protocol.attention(None) == {"t": "FACE", "attention": None}
