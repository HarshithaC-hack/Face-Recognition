"""
Embedding generation and storage utilities using DeepFace (Facenet512).
- Builds a single model once (call get_model())
- Generates embeddings for all images in a folder
- Averages them for stability
- Reads/writes to embeddings.json
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import List

import numpy as np
from deepface import DeepFace
from .config import EMBED_FILE

# --------------------------------------------------------------
# MODEL CACHE (so we don’t rebuild DeepFace model every time)
# --------------------------------------------------------------
_model_cache = None


def get_model():
    """Build or fetch a cached Facenet512 model."""
    global _model_cache
    if _model_cache is None:
        print("[DeepFace] Building Facenet512 model (first call only)...")
        _model_cache = DeepFace.build_model("Facenet512")
    return _model_cache


# --------------------------------------------------------------
# CORE: Compute embeddings for all images in a folder
# --------------------------------------------------------------
def compute_embeddings_for_folder(folder: Path) -> List[List[float]]:
    """
    Compute embeddings for all .jpg/.jpeg/.png files in the given folder.
    Returns a list of embedding vectors (Python lists).
    """
    model = get_model()
    out: List[List[float]] = []

    if not folder.exists():
        print(f"[Embed] Folder not found: {folder}")
        return out

    images = sorted([p for p in folder.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    if not images:
        print(f"[Embed] No images found in {folder}")
        return out

    print(f"[Embed] Generating embeddings for {len(images)} images in {folder} ...")

    for p in images:
        try:
            # ✅ No model argument (DeepFace handles it internally)
            rep = DeepFace.represent(
                img_path=str(p),
                model_name="Facenet512",
                enforce_detection=False
            )

            if rep and isinstance(rep, list) and "embedding" in rep[0]:
                out.append(rep[0]["embedding"])
            else:
                print(f"[Embed] No embedding returned for {p.name}")
        except Exception as e:
            print(f"[Embed] Skipping {p.name}: {e}")

    print(f"[Embed] Created {len(out)} embeddings from {len(images)} images.")
    return out


# --------------------------------------------------------------
# AVERAGING EMBEDDINGS (for more stable matching)
# --------------------------------------------------------------
def average_embeddings(embeddings: List[List[float]]) -> List[List[float]]:
    """
    Averages multiple embeddings into one mean vector.
    Returns a single-item list containing the averaged embedding.
    """
    if not embeddings:
        print("[Embed] No embeddings to average.")
        return []

    arr = np.array(embeddings, dtype=np.float32)
    mean_vec = arr.mean(axis=0)
    print(f"[Embed] Averaged {len(embeddings)} embeddings into one stable vector.")
    return [mean_vec.tolist()]


# --------------------------------------------------------------
# SAVE USER EMBEDDINGS TO JSON
# --------------------------------------------------------------
def save_user_embeddings(user_name: str, embeddings: List[List[float]]) -> int:
    """
    Append or replace embeddings for `user_name` in embeddings.json.
    Returns the number of vectors saved.
    """
    if not embeddings:
        print(f"[Embed] No embeddings to save for {user_name}.")
        return 0

    # Load existing data safely
    if EMBED_FILE.exists():
        try:
            with open(EMBED_FILE, "r", encoding="utf-8") as f:
                db = json.load(f)
        except json.JSONDecodeError:
            db = {}
    else:
        db = {}

    # Save (replace old data for that user)
    db[user_name] = embeddings
    with open(EMBED_FILE, "w", encoding="utf-8") as f:
        json.dump(db, f, indent=2)

    print(f"[Embed] ✅ Saved {len(embeddings)} embeddings for '{user_name}' to {EMBED_FILE}")
    return len(embeddings)


# --------------------------------------------------------------
# HIGH-LEVEL HELPER: Capture, average, and save together
# --------------------------------------------------------------
def generate_and_save_embeddings_for_user(user_name: str, cropped_folder: Path) -> None:
    """
    Convenience wrapper:
    - Computes embeddings from cropped images folder
    - Averages them
    - Saves result to embeddings.json
    """
    print(f"[Embed] Starting embedding generation for '{user_name}'...")
    embs = compute_embeddings_for_folder(cropped_folder)
    embs = average_embeddings(embs)
    saved = save_user_embeddings(user_name, embs)
    if saved:
        print(f"[Embed] 🎯 Embedding generation complete for '{user_name}'.")
    else:
        print(f"[Embed] ⚠️ No embeddings generated for '{user_name}'.")
