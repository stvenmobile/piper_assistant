"""
Face following: turn the tracked face's angle in the camera into pan/tilt servo targets. Pure
logic - no camera, no serial port - so it can be unit-tested.

The camera rides on the head, so the tracker's angles are ERRORS: how far the face is from the
middle of the picture. Each update moves the servos part of the way (gain < 1, so a few frames of
camera lag don't make it overshoot), ignores small errors (deadband, so it doesn't hunt), and
caps each step. With nobody in view for `home_after_s` it goes back to looking straight ahead.

Angles are SERVO degrees in the robot convention: pan + = left, tilt + = the head up.
"""
from dataclasses import dataclass


@dataclass
class FollowConfig:
    gain: float = 0.6               # fraction of the error corrected per update
    deadband_deg: float = 2.5       # errors smaller than this are left alone
    max_step_deg: float = 12.0      # cap on one update's move (servo degrees)
    tilt_ratio: float = 1.6         # tilt servo degrees per degree of head tilt (the linkage, near level)
    pan_limits: tuple = (-90.0, 90.0)
    tilt_limits: tuple = (-70.0, 6.3)
    home: tuple = (0.0, -33.0)      # looking straight ahead, head level
    home_after_s: float = 4.0       # nobody in view this long -> go home


def _clamp(x, lim):
    return max(lim[0], min(lim[1], x))


def _step(err, gain, deadband, max_step):
    if abs(err) < deadband:
        return 0.0
    return max(-max_step, min(max_step, err * gain))


class Follower:
    def __init__(self, cfg: FollowConfig | None = None):
        self.cfg = cfg or FollowConfig()
        self.last_seen = None           # time of the last update with a target

    def update(self, now: float, pan: float, tilt: float, h_err: float | None, v_err: float | None):
        """pan/tilt: where the servos are now. h_err/v_err: the target's angle from the middle of
        the picture (tracker.angles: h + = to the camera's RIGHT, v + = UP), or None if nobody.
        Returns (pan, tilt) to move to, or None to leave the servos where they are."""
        c = self.cfg
        if h_err is None:
            if self.last_seen is not None and now - self.last_seen >= c.home_after_s:
                self.last_seen = None
                return c.home
            return None
        self.last_seen = now
        dp = -_step(h_err, c.gain, c.deadband_deg, c.max_step_deg)                 # face to the right -> pan right (-)
        dt = _step(v_err * c.tilt_ratio, c.gain, c.deadband_deg * c.tilt_ratio, c.max_step_deg)
        if dp == 0 and dt == 0:
            return None
        return _clamp(pan + dp, c.pan_limits), _clamp(tilt + dt, c.tilt_limits)
