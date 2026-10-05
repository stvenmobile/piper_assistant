"""
Face recognition with OpenCV's SFace: a detected face (YuNet box + landmarks) -> a 128-number
embedding. Two looks at the same person give embeddings that point the same way (cosine
similarity well above ~0.4); different people's don't. faces.py does the comparing.

The model file isn't in git: `python3 tools/download_models.py` fetches it into models/.
"""
import cv2
import numpy as np

from piper_vision.detector import MODELS_DIR

SFACE = "face_recognition_sface_2021dec.onnx"


class FaceRecognizer:
    def __init__(self, min_sharpness: float = 15.0):
        path = MODELS_DIR / SFACE
        if not path.exists():
            raise FileNotFoundError(f"{path} missing - run: python3 tools/download_models.py")
        self.net = cv2.FaceRecognizerSF.create(str(path), "")
        self.min_sharpness = min_sharpness

    def embed(self, frame, face) -> np.ndarray | None:
        """Embedding of one face (x, y, w, h, score, landmarks) in `frame`, or None if the
        aligned crop is too blurry (motion blur ruins a face print)."""
        crop = self.net.alignCrop(frame, face[5])                 # 112 x 112, eyes levelled
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        if cv2.Laplacian(gray, cv2.CV_64F).var() < self.min_sharpness:
            return None
        return self.net.feature(crop).reshape(-1).copy()
