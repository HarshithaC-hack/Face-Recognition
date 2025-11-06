"""
Centralized configuration for the Eagle Access system.
You can override defaults by setting environment variables.
"""

import os
from pathlib import Path

# ---- Paths ----
BASE_DIR = Path(os.environ.get("EAGLE_BASE_DIR", Path.cwd()))
DATASET_DIR = Path(os.environ.get("EAGLE_DATASET_DIR", BASE_DIR / "dataset" / "Custom"))
CROPPED_DIR = Path(os.environ.get("EAGLE_CROPPED_DIR", BASE_DIR / "dataset" / "Custom"/"cropped"))

USERS_FILE = Path(os.environ.get("EAGLE_USERS_FILE", BASE_DIR / "users.json"))
EMBED_FILE = Path(os.environ.get("EAGLE_EMBED_FILE", BASE_DIR / "embeddings.json"))
LOG_FILE = Path(os.environ.get("EAGLE_LOG_FILE", BASE_DIR / "access_log.json"))

# Create directories/files if not present
DATASET_DIR.mkdir(parents=True, exist_ok=True)
for fpath, default in [(USERS_FILE, {}), (EMBED_FILE, {}), (LOG_FILE, [])]:
    if not fpath.exists():
        import json
        with open(fpath, "w", encoding="utf-8") as f:
            json.dump(default, f, indent=2)

# ---- Recognition / Capture ----
FACE_SIZE = (160, 160)  # standard Facenet input
NUM_IMAGES = int(os.environ.get("EAGLE_NUM_IMAGES", 30))
CAPTURE_DELAY_SEC = float(os.environ.get("EAGLE_CAPTURE_DELAY_SEC", 0.2))

# Cosine similarity threshold; higher is more lenient. Typical range ~0.45–0.65 for Facenet512.
SIMILARITY_THRESHOLD = float(os.environ.get("EAGLE_SIM_THRESHOLD", 0.50))

# OpenCV cascade path (fallback to bundled default)
import cv2
FACE_CASCADE_PATH = os.environ.get(
    "EAGLE_FACE_CASCADE_PATH",
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
