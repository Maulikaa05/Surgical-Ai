"""
phase_reader.py
Reads ArthroPhase annotation txt files.
"""

from pathlib import Path
from typing import Optional

PHASE_NAMES = {
    "0": "Preparation",
    "1": "Diagnosis",
    "2": "Femoral Tunnel Creation",
    "3": "Tibial Tunnel Creation",
    "4": "ACL Reconstruction",
}


def load_phase_annotations(path: str) -> dict:
    anns = {}
    p = Path(path)
    if not p.exists():
        return anns
    with open(p) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                try:
                    anns[int(parts[0])] = PHASE_NAMES.get(parts[1], f"Phase {parts[1]}")
                except ValueError:
                    pass
    return anns


def get_dominant_phase(frame_numbers: list, annotations: dict) -> str:
    if not annotations:
        return "Unknown"
    counts: dict = {}
    ann_keys = sorted(annotations.keys())
    for fn in frame_numbers:
        # binary-search nearest annotated frame
        nearest = min(ann_keys, key=lambda k: abs(k - fn))
        phase = annotations[nearest]
        counts[phase] = counts.get(phase, 0) + 1
    return max(counts, key=counts.get) if counts else "Unknown"


def find_phase_file(archive_root: str, video_id: str) -> Optional[str]:
    root = Path(archive_root)
    candidates = [
        root.parent / f"{video_id}-phase.txt",
        root         / f"{video_id}-phase.txt",
        root.parent.parent / f"{video_id}-phase.txt",
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    return None
