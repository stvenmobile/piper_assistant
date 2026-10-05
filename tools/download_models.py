"""
Fetch the model files that aren't kept in git into models/.

    python3 tools/download_models.py
"""
import sys
import urllib.request
from pathlib import Path

MODELS = Path(__file__).resolve().parents[1] / "models"

FILES = {
    # OpenCV Zoo's YuNet face detector (~230 KB, MIT licence)
    "face_detection_yunet_2023mar.onnx": [
        "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
        "https://huggingface.co/opencv/face_detection_yunet/resolve/main/face_detection_yunet_2023mar.onnx",
    ],
    # OpenCV Zoo's SFace face recogniser (~37 MB, Apache 2.0)
    "face_recognition_sface_2021dec.onnx": [
        "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx",
        "https://huggingface.co/opencv/face_recognition_sface/resolve/main/face_recognition_sface_2021dec.onnx",
    ],
}


def main():
    MODELS.mkdir(exist_ok=True)
    for name, urls in FILES.items():
        dest = MODELS / name
        if dest.exists() and dest.stat().st_size > 100_000:
            print(f"have {name}")
            continue
        for url in urls:
            try:
                print(f"downloading {name} ...")
                data = urllib.request.urlopen(url, timeout=60).read()
                if len(data) < 100_000:               # an LFS pointer or an error page, not the model
                    raise ValueError(f"only {len(data)} bytes")
                dest.write_bytes(data)
                print(f"  saved {dest} ({len(data) // 1024} KB)")
                break
            except Exception as e:
                print(f"  {url}: {e}")
        else:
            sys.exit(f"could not download {name}")


if __name__ == "__main__":
    main()
