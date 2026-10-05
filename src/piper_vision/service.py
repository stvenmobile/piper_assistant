"""
Vision service: camera -> face detection -> tracking -> recognition -> the head + the assistant.

    python3 src/piper_vision/service.py       (start_piper.sh starts it when vision.enabled)

For now (no pan motor yet) the tracked person's direction drives the light ring's ATTENTION
arc, so it glances toward whoever Piper is looking at. Later the same angle becomes the pan
target. A live preview with boxes is served on http://<jetson>:vision.preview_port.

Recognition: a couple of times a second a clear, front-on look at the tracked person is turned
into a face print and compared with the people Piper has met (faces/, never in git). Who is
present goes to the assistant over the vision hub (hub.py); the assistant asks strangers their
name and sends back ENROLL, and the next few good looks are stored under that name.
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
from piper_vision.faces import FaceLibrary, Identity, usable_face
from piper_vision.hub import Hub
from piper_vision.preview import Preview
from piper_vision.tracker import Tracker, attention_deg


def draw(frame, boxes, target, info: str, label: str = ""):
    for (x, y, w, h, s) in boxes:
        cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), (90, 200, 90), 2)
        cv2.putText(frame, f"{s:.2f}", (int(x), int(y) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (90, 200, 90), 2)
    if target is not None:
        cv2.circle(frame, (int(target.cx), int(target.cy)), 8, (0, 200, 255), -1)
        if label:
            x = int(target.cx - target.size / 2)
            y = int(target.cy + target.size * 0.75)
            cv2.putText(frame, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 255), 2)
    h, w = frame.shape[:2]
    cv2.line(frame, (w // 2, 0), (w // 2, h), (80, 80, 80), 1)
    cv2.putText(frame, info, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)


class Enrolment:
    """Collecting looks at one tracked person to store under a name."""

    def __init__(self, name: str, track: int, until: float):
        self.name, self.track, self.until = name, track, until
        self.looks = []


def make_recognizer(cfg):
    if not cfg["recognize"]:
        return None
    try:
        from piper_vision.recognizer import FaceRecognizer
        return FaceRecognizer(cfg["min_sharpness"])
    except (FileNotFoundError, AttributeError, cv2.error) as e:
        print(f"[Vision] Face recognition off: {e}")
        return None


def label_for(ident, enrolling) -> str:
    if enrolling is not None:
        return f"learning {enrolling.name} ({len(enrolling.looks)})"
    if ident is None or ident.verdict is None:
        return "?"
    return ident.verdict or "stranger"


def main():
    cfg = CONFIG["vision"]
    cap, dev = open_camera(cfg)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[Vision] Camera {dev} at {w}x{h}")
    detector = FaceDetector(w, h, cfg["detect_width"], cfg["score"])
    tracker = Tracker(w, h, cfg["hfov_deg"], cfg["lost_s"])
    head = HeadClient()
    recognizer = make_recognizer(cfg)
    library = FaceLibrary(threshold=cfg["match_threshold"])
    if recognizer:
        known = ", ".join(library.names()) or "nobody yet"
        print(f"[Vision] Recognition on - knows {len(library.names())}: {known}")
    hub = Hub(cfg["events_port"], recognition=recognizer is not None)
    hub.start()
    preview = Preview(cfg["preview_port"]) if cfg["preview_port"] else None
    if preview:
        preview.start()
        print(f"[Vision] Preview on http://0.0.0.0:{cfg['preview_port']}/")

    track, ident = None, None   # the current target's id and who we think it is
    last_look = 0.0             # when we last took a face print
    enrolling = None
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

        # --- who is it? ------------------------------------------------------------------------
        if target is None:
            track, ident = None, None
            if enrolling is not None:
                hub.broadcast({"t": "ENROLL_FAILED", "name": enrolling.name, "track": enrolling.track,
                               "reason": "lost"})
                enrolling = None
        elif target.id != track:
            track, ident = target.id, Identity()

        while not hub.commands.empty():
            cmd = hub.commands.get_nowait()
            if cmd["t"] == "ENROLL":
                name = str(cmd.get("name", "")).strip()
                if recognizer is None or not name or target is None or cmd.get("track") != track:
                    reason = "no recognition" if recognizer is None else "not here"
                    hub.broadcast({"t": "ENROLL_FAILED", "name": name, "track": cmd.get("track"),
                                   "reason": reason})
                else:
                    enrolling = Enrolment(name, track, now + cfg["enroll_s"])
                    print(f"[Vision] Learning the face of {name}...")
            elif cmd["t"] == "FORGET":
                ok = library.forget(str(cmd.get("name", "")))
                print(f"[Vision] Forget {cmd.get('name')}: {'done' if ok else 'not known'}")
                if ok and ident is not None:
                    ident = Identity()
                hub.broadcast({"t": "FORGOTTEN", "name": cmd.get("name"), "ok": ok})

        gap = cfg["enroll_gap_s"] if enrolling else cfg["recognize_s"]
        if (recognizer and target is not None and target.box is not None and now - last_look >= gap
                and usable_face(target.box, cfg["min_face_px"])):
            emb = recognizer.embed(frame, target.box)
            if emb is not None:
                last_look = now
                if enrolling is not None:
                    enrolling.looks.append(emb)
                else:
                    name, sim = library.match(emb)
                    if ident.add(name):
                        print(f"[Vision] Track {track}: {ident.verdict or 'a stranger'} (best {sim:.2f})")

        if enrolling is not None and (len(enrolling.looks) >= cfg["enroll_looks"] or now > enrolling.until):
            if len(enrolling.looks) >= cfg["enroll_min_looks"]:
                n = library.add(enrolling.name, enrolling.looks)
                ident.settle(library.find(enrolling.name))
                print(f"[Vision] Learned {enrolling.name} ({n} looks stored)")
                hub.broadcast({"t": "ENROLLED", "name": enrolling.name, "track": track, "looks": n})
            else:
                hub.broadcast({"t": "ENROLL_FAILED", "name": enrolling.name, "track": track,
                               "reason": "no good look"})
            enrolling = None

        if target is None:
            hub.set_presence(False)
        else:
            hub.set_presence(True, track, ident.verdict)

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
        if preview and preview.viewers:
            info = f"{fps:4.1f} fps  faces {len(boxes)}"
            if target is not None:
                info += f"  target {h_ang:+5.1f} deg  ring {att:+5.0f}"
            draw(frame, boxes, target, info, label_for(ident, enrolling) if recognizer else "")
            ok, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            if ok:
                preview.publish(jpg.tobytes())


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
