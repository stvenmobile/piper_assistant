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


# ---- face following ------------------------------------------------------------------------------
from piper_vision.follow import FollowConfig, Follower


def test_follow_turns_toward_a_face_on_the_right():
    f = Follower(FollowConfig(gain=0.5, deadband_deg=2))
    pan, tilt = f.update(0.0, 0.0, -33.0, 10.0, 0.0)       # face 10 deg to the camera's right
    assert pan == pytest.approx(-5.0) and tilt == -33.0      # pan + = left, so right is -


def test_follow_tilts_up_through_the_linkage_ratio():
    f = Follower(FollowConfig(gain=0.5, deadband_deg=2, tilt_ratio=1.6))
    pan, tilt = f.update(0.0, 0.0, -33.0, 0.0, 5.0)        # face 5 deg above the middle
    assert pan == 0.0 and tilt == pytest.approx(-33.0 + 5 * 1.6 * 0.5)


def test_follow_ignores_small_errors():
    f = Follower(FollowConfig(deadband_deg=2.5))
    assert f.update(0.0, 0.0, -33.0, 1.0, 1.0) is None


def test_follow_caps_the_step_and_respects_limits():
    f = Follower(FollowConfig(gain=1.0, max_step_deg=12, pan_limits=(-90, 90), tilt_limits=(-70, 6.3)))
    assert f.update(0.0, 0.0, -33.0, -30.0, 0.0)[0] == pytest.approx(12.0)
    assert f.update(0.1, 85.0, 5.0, -30.0, 20.0) == (90.0, 6.3)


def test_follow_goes_home_after_nobody_for_a_while():
    f = Follower(FollowConfig(home=(0.0, -33.0), home_after_s=4.0))
    f.update(0.0, 30.0, -20.0, 5.0, 0.0)
    assert f.update(2.0, 30.0, -20.0, None, None) is None
    assert f.update(4.5, 30.0, -20.0, None, None) == (0.0, -33.0)
    assert f.update(9.0, 0.0, -33.0, None, None) is None     # once
