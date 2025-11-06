"""
Live recognition and similarity scoring.
- Loads stored embeddings
- Computes live frame embedding
- Compares using cosine, Euclidean, and Manhattan metrics
"""

from __future__ import annotations
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple
import os

import cv2
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import Normalizer
from scipy.spatial.distance import euclidean, cityblock
from deepface import DeepFace

from .config import EMBED_FILE, SIMILARITY_THRESHOLD, FACE_CASCADE_PATH, FACE_SIZE, LOG_FILE


# ---------------------------------------------------------------------
# Load Database
# ---------------------------------------------------------------------
def _load_db() -> Tuple[List[str], np.ndarray]:
    """Load embeddings.json and return (names, normalized_embeddings_2d)."""
    if not Path(EMBED_FILE).exists():
        raise FileNotFoundError("No embeddings.json found. Please register users first.")

    with open(EMBED_FILE, "r", encoding="utf-8") as f:
        db = json.load(f)

    if not db:
        raise RuntimeError("embeddings.json is empty. Register users first.")

    names = list(db.keys())
    mat = []
    for name in names:
        vecs = np.asarray(db[name], dtype="float32")
        mean_vec = vecs.mean(axis=0)
        mat.append(mean_vec)
    mat = np.asarray(mat, dtype="float32")

    in_encoder = Normalizer(norm="l2")
    mat = in_encoder.transform(mat)
    return names, mat


# ---------------------------------------------------------------------
# Face Cropping
# ---------------------------------------------------------------------
def _crop_largest_face(frame_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)
    faces = cascade.detectMultiScale(gray, 1.3, 5)
    if len(faces) == 0:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        return cv2.resize(rgb, FACE_SIZE)
    x, y, w, h = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)[0]
    face = frame_bgr[y:y+h, x:x+w]
    face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    return cv2.resize(face_rgb, FACE_SIZE)


# ---------------------------------------------------------------------
# Model Loader
# ---------------------------------------------------------------------
_model_cache = None
def _get_model():
    global _model_cache
    if _model_cache is None:
        print("[DeepFace] Building Facenet512 model (first call only)...")
        _model_cache = DeepFace.build_model("Facenet512")
    return _model_cache


# ---------------------------------------------------------------------
# Live Verification
# ---------------------------------------------------------------------
def verify_face_live(threshold: float = SIMILARITY_THRESHOLD, show_window: bool = True) -> Dict[str, str]:
    """
    Open webcam once, detect a face, compute embedding, compare to DB.
    Uses cosine similarity as the main decision metric, but logs
    Euclidean and Manhattan distances for analysis.
    """
    names, db_embeds = _load_db()
    in_encoder = Normalizer(norm="l2")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not access webcam.")

    print("[Access] Eagle is watching...")
    model = _get_model()

    decision = {
        "status": "denied",
        "name": None,
        "confidence": "0.00",
        "scores": {},
        "time": datetime.now().isoformat()
    }

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        face_rgb = _crop_largest_face(frame)

        try:
            temp_path = "temp_face.jpg"
            cv2.imwrite(temp_path, cv2.cvtColor(face_rgb, cv2.COLOR_RGB2BGR))

            rep = DeepFace.represent(
                img_path=temp_path,
                model_name="Facenet512",
                enforce_detection=False,
            )

            if rep and isinstance(rep, list) and "embedding" in rep[0]:
                emb = np.asarray(rep[0]["embedding"], dtype="float32").reshape(1, -1)
                emb = in_encoder.transform(emb)
                sims = cosine_similarity(emb, db_embeds)[0]
                max_idx = int(np.argmax(sims))

                # compute all metrics
                target_vec = db_embeds[max_idx]
                cos_score = float(sims[max_idx])
                euclid_score = float(1 / (1 + euclidean(emb.flatten(), target_vec)))
                manhattan_score = float(1 / (1 + cityblock(emb.flatten(), target_vec)))

                confidence = cos_score
                decision["scores"] = {
                    "cosine": round(cos_score, 4),
                    "euclidean": round(euclid_score, 4),
                    "manhattan": round(manhattan_score, 4),
                }

            else:
                confidence = 0.0

            if confidence > threshold:
                decision.update({
                    "status": "granted",
                    "name": names[max_idx],
                    "confidence": f"{float(confidence):.4f}"
                })
                label = f"{names[max_idx]} ({confidence*100:.1f}%)"
                color = (0, 255, 0)
            else:
                decision.update({
                    "status": "denied",
                    "name": "Unknown",
                    "confidence": f"{float(confidence):.4f}"
                })
                label = "Access Denied"
                color = (0, 0, 255)

        except Exception as e:
            print(f"[Error] DeepFace embedding failed: {e}")
            label, color = "No Face / Error", (0, 0, 255)
            confidence = 0.0

        if show_window:
            cv2.putText(frame, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            cv2.imshow("Eagle Access", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        if confidence > 0.0 or "Unknown" in label:
            time.sleep(1.5)
            break

    cap.release()
    if show_window:
        cv2.destroyAllWindows()

    try:
        os.remove("temp_face.jpg")
    except:
        pass

    # Ensure all values are Python-native for JSON
    decision = _convert_json_safe(decision)

    # Log the decision
    try:
        if Path(LOG_FILE).exists():
            with open(LOG_FILE, "r", encoding="utf-8") as f:
                log = json.load(f)
        else:
            log = []
        log.append(decision)
        with open(LOG_FILE, "w", encoding="utf-8") as f:
            json.dump(log, f, indent=2)
    except Exception as e:
        print(f"[Log] Failed to write access log: {e}")

    print(f"[Access] Decision: {decision}")
    return decision


# ---------------------------------------------------------------------
# JSON-safe conversion helper
# ---------------------------------------------------------------------
def _convert_json_safe(obj):
    """Recursively convert np.float32 and np.ndarray to JSON-safe types."""
    if isinstance(obj, dict):
        return {k: _convert_json_safe(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_convert_json_safe(v) for v in obj]
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj
