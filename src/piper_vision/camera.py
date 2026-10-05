"""
The C920 (or any UVC webcam), opened by NAME rather than /dev/videoN, as MJPEG.
Focus is locked (autofocus hunting looks bad while tracking); exposure stays automatic but
isn't allowed to drop the frame rate in dim light.
"""
import glob
import subprocess
from pathlib import Path

import cv2


def find_device(name_hint: str) -> str | None:
    """/dev/videoN of the first capture node whose name contains name_hint (case-insensitive).
    Webcams also expose a metadata node (e.g. /dev/video1) - the lowest index is the camera."""
    hits = []
    for node in sorted(glob.glob("/sys/class/video4linux/video*")):
        try:
            name = Path(node, "name").read_text().strip()
        except OSError:
            continue
        if name_hint.lower() in name.lower():
            hits.append(int(node.rsplit("video", 1)[1]))
    return f"/dev/video{min(hits)}" if hits else None


def set_controls(dev: str, focus: int | None):
    """Best effort: control names differ between kernels/firmware, so try both spellings and
    ignore the ones this camera doesn't have."""
    def ctl(*pairs):
        for p in pairs:
            subprocess.run(["v4l2-ctl", "-d", dev, "-c", p], capture_output=True)
    if focus is not None:
        ctl("focus_automatic_continuous=0", "focus_auto=0")
        ctl(f"focus_absolute={int(focus)}")
    ctl("exposure_dynamic_framerate=0", "exposure_auto_priority=0")


def open_camera(cfg: dict) -> tuple[cv2.VideoCapture, str]:
    dev = find_device(cfg["camera_name"]) or cfg.get("camera_device", "/dev/video0")
    cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg["width"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg["height"])
    cap.set(cv2.CAP_PROP_FPS, cfg["fps"])
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)          # always the newest frame, not a queued one
    if not cap.isOpened():
        raise RuntimeError(f"can't open camera {dev}")
    set_controls(dev, cfg.get("focus"))
    return cap, dev
