"""
Face detection with OpenCV's YuNet (a small, fast, accurate face detector built into OpenCV
4.5.4+). Runs on a reduced copy of the frame; boxes come back in full-frame pixels.

The model file isn't in git: `python3 tools/download_models.py` fetches it into models/.
"""
from pathlib import Path

import cv2

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
YUNET = "face_detection_yunet_2023mar.onnx"


class FaceDetector:
    def __init__(self, frame_w: int, frame_h: int, detect_w: int = 640, score: float = 0.8):
        path = MODELS_DIR / YUNET
        if not path.exists():
            raise FileNotFoundError(f"{path} missing - run: python3 tools/download_models.py")
        self.scale = detect_w / frame_w
        self.size = (detect_w, round(frame_h * self.scale))
        self.net = cv2.FaceDetectorYN.create(str(path), "", self.size, score, 0.3, 50)

    def detect(self, frame) -> list[tuple]:
        """Faces as (x, y, w, h, score, landmarks) in the frame's own pixels, best first.
        landmarks is YuNet's full row (box, eyes, nose, mouth corners, score) scaled to the
        frame - what the recognizer needs to align a face."""
        small = cv2.resize(frame, self.size, interpolation=cv2.INTER_AREA)
        _, faces = self.net.detect(small)
        if faces is None:
            return []
        s = 1 / self.scale
        out = []
        for f in faces:
            row = f.copy()
            row[:14] *= s
            out.append((float(row[0]), float(row[1]), float(row[2]), float(row[3]), float(f[14]), row))
        return sorted(out, key=lambda b: -b[4])
