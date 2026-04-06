"""
extractor.py
Dynamically extracts instrument tip coordinates from ArthroPhase
color_mask PNG files. Fully adaptive — auto-detects dominant instrument
colour from the actual image rather than using hardcoded HSV ranges.
"""

import cv2
import numpy as np
import os
import json
from pathlib import Path
from typing import Optional


# ── Colour candidate ranges (BGR) — tried in order until one works ──
_COLOUR_CANDIDATES = [
    # Name,       Lower BGR,              Upper BGR
    ("pink",    np.array([100,  40, 140]), np.array([215, 170, 255])),
    ("salmon",  np.array([80,   60, 160]), np.array([200, 160, 255])),
    ("red",     np.array([0,    0,  140]), np.array([80,  80,  255])),
    ("green",   np.array([0,    80,   0]), np.array([80,  255,  80])),
    ("yellow",  np.array([0,   160, 160]), np.array([80,  255, 255])),
    ("cyan",    np.array([120, 120,   0]), np.array([255, 255,  80])),
]
_MIN_AREA = 30   # pixels — ignore tiny noise blobs


def _dominant_instrument_mask(img: np.ndarray) -> Optional[np.ndarray]:
    """
    Try each colour candidate and return the mask with the largest
    clean contour (most likely the instrument highlight region).
    """
    best_mask  = None
    best_area  = 0

    for _, lo, hi in _COLOUR_CANDIDATES:
        mask = cv2.inRange(img, lo, hi)
        # Morphological cleaning
        k    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            area = max(cv2.contourArea(c) for c in contours)
            if area > best_area and area >= _MIN_AREA:
                best_area = area
                best_mask = mask

    return best_mask


def _centroid(mask: np.ndarray) -> tuple[Optional[int], Optional[int]]:
    """Return centroid of the largest blob, or (None, None)."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < _MIN_AREA:
        return None, None
    M = cv2.moments(largest)
    if M["m00"] == 0:
        x, y, w, h = cv2.boundingRect(largest)
        return x + w // 2, y + h // 2
    return int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])


def extract_instrument_position(mask_path: str) -> tuple[Optional[int], Optional[int]]:
    img = cv2.imread(str(mask_path))
    if img is None:
        return None, None
    mask = _dominant_instrument_mask(img)
    if mask is None:
        return None, None
    return _centroid(mask)


def extract_coordinates_from_folder(folder_path: str,
                                     fps: float = 30.0) -> list[dict]:
    """
    Extract (x, y, t) from ALL color_mask PNGs in a segment folder.
    Sorted numerically by frame number. Returns every frame — no skipping.
    """
    folder = Path(folder_path)
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    mask_files = sorted(
        folder.glob("*_endo_color_mask.png"),
        key=lambda p: int("".join(filter(str.isdigit, p.name.split("_endo")[0].split("_")[-1])) or "0")
    )

    if not mask_files:
        # Fallback: try any png with "color" in name
        mask_files = sorted(folder.glob("*color*.png"))

    if not mask_files:
        raise ValueError(f"No color mask PNGs found in: {folder_path}")

    records = []
    for mf in mask_files:
        # Parse frame number robustly
        digits = "".join(filter(str.isdigit, mf.stem.split("endo")[0]))
        frame_num = int(digits) if digits else len(records)
        cx, cy = extract_instrument_position(str(mf))
        records.append({
            "frame":     frame_num,
            "timestamp": round(frame_num / fps, 6),
            "tip_x":     cx,
            "tip_y":     cy,
        })

    return records


def process_entire_archive(archive_root: str,
                            fps: float = 30.0) -> dict:
    """
    Walk archive → videoXX → segmentXX → extract all.
    Returns nested dict for baseline computation.
    """
    archive = Path(archive_root)
    if not archive.exists():
        return {}

    dataset = {}
    for video_dir in sorted(d for d in archive.iterdir() if d.is_dir()):
        vid_id = video_dir.name
        dataset[vid_id] = {}
        for seg_dir in sorted(d for d in video_dir.iterdir() if d.is_dir()):
            try:
                recs = extract_coordinates_from_folder(str(seg_dir), fps)
                if recs:
                    dataset[vid_id][seg_dir.name] = recs
            except Exception:
                pass
    return dataset


def save_coordinates(records: list[dict], path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(records, f, indent=2)


def load_coordinates(path: str) -> list[dict]:
    with open(path) as f:
        return json.load(f)
