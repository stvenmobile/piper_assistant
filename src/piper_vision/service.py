"""
Vision service: camera -> face detection -> tracking -> the head.

    python3 src/piper_vision/service.py       (start_piper.sh starts it when vision.enabled)

For now (no pan motor yet) the tracked person's direction drives the light ring's ATTENTION
arc, so it glances toward whoever Piper is looking at. Later the same angle becomes the pan
target. A live preview with boxes is served on http://<jetson>:vision.preview_port.
"""
import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2

from piper_brain.config import CONFIG
from piper_head.client import HeadClient
from piper_vision.camera import open_camera
from piper_vision.detector import FaceDetector
from piper_vision.preview import Preview
from piper_vision.tracker import Tracker, attention_deg


def draw(frame, boxes, target, info: str):
    for (x, y, w, h, s) in boxes:
        cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), (90, 200, 90), 2)
        cv2.putText(frame, f"{s:.2f}", (int(x), int(y) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (90, 200, 90), 2)
    if target is not None:
        cv2.circle(frame, (int(target.cx), int(target.cy)), 8, (0, 200, 255), -1)
    h, w = frame.shape[:2]
    cv2.line(frame, (w // 2, 0), (w // 2, h), (80, 80, 80), 1)
    cv2.putText(frame, info, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)


def main():
    cfg = CONFIG["vision"]
    cap, dev = open_camera(cfg)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[Vision] Camera {dev} at {w}x{h}")
    detector = FaceDetector(w, h, cfg["detect_width"], cfg["score"])
    tracker = Tracker(w, h, cfg["hfov_deg"], cfg["lost_s"])
    head = HeadClient()
    preview = Preview(cfg["preview_port"]) if cfg["preview_port"] else None
    if preview:
        preview.start()
        print(f"[Vision] Preview on http://0.0.0.0:{cfg['preview_port']}/")

    sent = None                 # last attention sent (degrees or None)
    last_send = 0.0
    frames, fps, t_fps = 0, 0.0, time.monotonic()
    while True:
        ok, frame = cap.read()
        if not ok:
            print("[Vision] Camera read failed - retrying")
            time.sleep(0.5)
            continue
        now = time.monotonic()
        boxes = detector.detect(frame)
        target = tracker.update(boxes, now)

        # attention arc: send on change (>= 2 deg or appear/disappear), at most 10 times a second
        att = None
        if target is not None:
            h_ang, v_ang = tracker.angles(target)
            att = round(attention_deg(h_ang, cfg["attention_gain"]), 1)
        changed = (att is None) != (sent is None) or (att is not None and abs(att - sent) >= 2)
        if changed and now - last_send > 0.1:
            head.attention(att)
            sent, last_send = att, now

        frames += 1
        if now - t_fps >= 1:
            fps, frames, t_fps = frames / (now - t_fps), 0, now
        if preview:
            info = f"{fps:4.1f} fps  faces {len(boxes)}"
            if target is not None:
                info += f"  target {h_ang:+5.1f} deg  ring {att:+5.0f}"
            draw(frame, boxes, target, info)
            ok, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            if ok:
                preview.publish(jpg.tobytes())


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
