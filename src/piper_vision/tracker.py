"""
Which face is Piper attending to, and where is it? Pure logic - no camera, no OpenCV - so it
can be unit-tested.

Faces come in as (x, y, w, h, score) boxes in frame pixels. The tracker keeps one TARGET:
  * it sticks with the same person while they're visible (nearest box to where they were),
  * it picks the largest (= nearest) face when it has no target,
  * it lets go after `lost_s` without a match,
  * it smooths the position so the attention arc (and later the head) doesn't jitter.
"""
import math
from dataclasses import dataclass


@dataclass
class Target:
    cx: float                  # smoothed centre, frame pixels
    cy: float
    size: float                # smoothed box width, pixels
    last_seen: float           # time.monotonic() of the last match
    first_seen: float


def box_centre(b):
    x, y, w, h = b[:4]
    return x + w / 2, y + h / 2


class Tracker:
    def __init__(self, frame_w: int, frame_h: int, hfov_deg: float, lost_s: float = 0.8,
                 smoothing: float = 0.5, match_frac: float = 0.6):
        self.w, self.h = frame_w, frame_h
        # pinhole model: focal length in pixels from the horizontal field of view
        self.f = (frame_w / 2) / math.tan(math.radians(hfov_deg) / 2)
        self.lost_s = lost_s
        self.alpha = smoothing          # 0 = no smoothing, closer to 1 = smoother/slower
        self.match_frac = match_frac    # a box within this * target size counts as the same person
        self.target: Target | None = None

    def update(self, boxes, now: float) -> Target | None:
        """Feed one frame's face boxes; returns the current target (or None)."""
        if self.target is not None:
            best, best_d = None, None
            for b in boxes:
                cx, cy = box_centre(b)
                d = math.hypot(cx - self.target.cx, cy - self.target.cy)
                if d <= self.match_frac * max(self.target.size, b[2]) and (best_d is None or d < best_d):
                    best, best_d = b, d
            if best is not None:
                cx, cy = box_centre(best)
                a = self.alpha
                t = self.target
                t.cx, t.cy = a * t.cx + (1 - a) * cx, a * t.cy + (1 - a) * cy
                t.size = a * t.size + (1 - a) * best[2]
                t.last_seen = now
            elif now - self.target.last_seen > self.lost_s:
                self.target = None
        if self.target is None and boxes:
            b = max(boxes, key=lambda b: b[2] * b[3])          # largest = nearest
            cx, cy = box_centre(b)
            self.target = Target(cx, cy, b[2], now, now)
        return self.target

    def angles(self, t: Target) -> tuple[float, float]:
        """(horizontal, vertical) angle of the target from the camera axis, degrees.
        Horizontal: + = to the camera's RIGHT. Vertical: + = UP."""
        h = math.degrees(math.atan((t.cx - self.w / 2) / self.f))
        v = math.degrees(math.atan((self.h / 2 - t.cy) / self.f))
        return h, v


def attention_deg(h_angle: float, gain: float, limit: float = 100.0) -> float:
    """Where on the light ring the attention arc goes (0 = top, clockwise as seen from the
    FRONT). A person to the camera's right is on the left as you face Piper, so the arc moves
    the other way: it 'glances' toward them."""
    return max(-limit, min(limit, -h_angle * gain))
