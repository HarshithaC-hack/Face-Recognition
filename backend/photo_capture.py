"""
Photo capture and face preprocessing.
- Captures N frames from webcam
- Saves raw frames and cropped/resized face images
- Can be invoked from Flask/Tkinter or CLI
"""

from __future__ import annotations
import time
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
from tqdm import tqdm

from .config import DATASET_DIR, FACE_CASCADE_PATH, FACE_SIZE, NUM_IMAGES, CAPTURE_DELAY_SEC

def _ensure_user_dirs(user_name: str) -> Tuple[Path, Path]:
    root = DATASET_DIR / user_name
    raw_dir = root / "raw"
    cropped_dir = root / "cropped"
    raw_dir.mkdir(parents=True, exist_ok=True)
    cropped_dir.mkdir(parents=True, exist_ok=True)
    return raw_dir, cropped_dir

def _crop_largest_face(img_bgr: np.ndarray, face_size: Tuple[int, int]) -> np.ndarray:
    """Detect faces and return an RGB cropped+resized face (largest). Fallback to whole image if none."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)
    faces = cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    if len(faces) == 0:
        # fallback to whole image
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        return cv2.resize(rgb, face_size)

    # choose largest face
    x, y, w, h = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)[0]
    face = img_bgr[y:y+h, x:x+w]
    face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    return cv2.resize(face_rgb, face_size)

def capture_user_images(user_name: str, num_images: int = NUM_IMAGES, delay_sec: float = CAPTURE_DELAY_SEC) -> Tuple[Path, Path, int]:
    """
    Capture frames from default webcam for the given user.
    Returns (raw_dir, cropped_dir, count_captured).
    """
    raw_dir, cropped_dir = _ensure_user_dirs(user_name)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not access webcam. Check permissions or device.")

    print(f"[Camera] Capturing {num_images} images for '{user_name}' ...")
    time.sleep(2)  # small buffer

    count = 0
    for i in range(num_images):
        ok, frame = cap.read()
        if not ok:
            print("Warn: Failed to read frame; stopping capture.")
            break

        raw_path = raw_dir / f"img_{i+1}.jpg"
        cv2.imwrite(str(raw_path), frame)

        cropped_rgb = _crop_largest_face(frame, FACE_SIZE)
        cropped_bgr = cv2.cvtColor(cropped_rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str((cropped_dir / f"img_{i+1}.jpg")), cropped_bgr)

        count += 1
        time.sleep(delay_sec)

    cap.release()
    print(f"[Camera] Done. Captured {count} frames. Raw: {raw_dir}  Cropped: {cropped_dir}")
    return raw_dir, cropped_dir, count
