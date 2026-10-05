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
        """Faces as (x, y, w, h, score) in the frame's own pixels, best first."""
        small = cv2.resize(frame, self.size, interpolation=cv2.INTER_AREA)
        _, faces = self.net.detect(small)
        if faces is None:
            return []
        s = 1 / self.scale
        out = [(float(f[0] * s), float(f[1] * s), float(f[2] * s), float(f[3] * s), float(f[14])) for f in faces]
        return sorted(out, key=lambda b: -b[4])
