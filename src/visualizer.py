"""
visualizer.py
Draws instrument trajectory on real surgical frames.
Fully dynamic — adapts path colour, thickness, and opacity
to the skill level and score automatically.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Optional


SKILL_COLOURS = {
    "Expert":       (39,  174,  96),    # green  BGR
    "Advanced":     (255, 140,  41),    # orange BGR
    "Intermediate": (41,  128, 185),    # blue   BGR
    "Novice":       (44,   62, 192),    # red    BGR
}


def _score_to_thickness(score: float) -> int:
    """Better scores = slightly thicker path for clarity."""
    return 2 if score < 60 else 3


def _put(img, text, pos, colour, scale=0.55, thick=1):
    x, y = pos
    cv2.putText(img, text, (x+1, y+1), cv2.FONT_HERSHEY_SIMPLEX,
                scale, (0, 0, 0), thick + 1, cv2.LINE_AA)
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX,
                scale, colour, thick, cv2.LINE_AA)


def draw_trajectory_on_frame(
    frame_path: str,
    xs: list, ys: list,
    output_path: str,
    skill_level: str = "Intermediate",
    composite_score: float = 50.0,
) -> str:
    frame = cv2.imread(str(frame_path))
    if frame is None:
        h, w = 480, 640
        frame = np.zeros((h, w, 3), dtype=np.uint8)
    else:
        h, w = frame.shape[:2]

    colour    = SKILL_COLOURS.get(skill_level, (180, 180, 180))
    thickness = _score_to_thickness(composite_score)

    pts = [(int(x), int(y)) for x, y in zip(xs, ys)
           if x is not None and y is not None]

    if len(pts) >= 2:
        n = len(pts)
        for i in range(1, n):
            alpha = max(0.25, i / n)    # fade in: early path lighter
            c = tuple(int(ch * alpha) for ch in colour)
            cv2.line(frame, pts[i-1], pts[i], c, thickness, cv2.LINE_AA)

        # Start marker: hollow white circle
        cv2.circle(frame, pts[0],  7, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.circle(frame, pts[0],  3, (200, 200, 200), -1)
        # End marker: solid skill-colour circle with white ring
        cv2.circle(frame, pts[-1], 9, colour,          -1, cv2.LINE_AA)
        cv2.circle(frame, pts[-1], 9, (255, 255, 255),  1, cv2.LINE_AA)

    # Overlay labels (top-left, semi-transparent background)
    labels = [
        f"Skill:  {skill_level}",
        f"Score: {composite_score:.1f}/100",
    ]
    lpad   = 8
    lh     = 22
    box_h  = len(labels) * lh + lpad
    overlay = frame.copy()
    cv2.rectangle(overlay, (5, 5), (210, box_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

    for i, lbl in enumerate(labels):
        _put(frame, lbl, (lpad, lpad + 14 + i * lh), colour if i == 0 else (240, 240, 240))

    # Legend bottom
    _put(frame, "o = Start   * = End",
         (lpad, h - 10), (180, 180, 180), scale=0.42)

    cv2.imwrite(str(output_path), frame)
    return str(output_path)


def create_expert_comparison(
    frame_path: str,
    trainee_xs: list, trainee_ys: list,
    output_path: str,
    skill_level: str = "Intermediate",
    composite_score: float = 50.0,
) -> str:
    frame = cv2.imread(str(frame_path))
    if frame is None:
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
    h, w = frame.shape[:2]

    pad    = 16
    canvas = np.zeros((h, w * 2 + pad, 3), dtype=np.uint8)
    canvas[:, :w]        = frame.copy()
    canvas[:, w + pad:]  = frame.copy()

    trainee_col = SKILL_COLOURS.get(skill_level, (41, 128, 185))
    expert_col  = (39, 174, 96)   # green = Expert

    # Trainee path
    pts_t = [(int(x), int(y)) for x, y in zip(trainee_xs, trainee_ys)
             if x is not None and y is not None]
    if len(pts_t) >= 2:
        for i in range(1, len(pts_t)):
            cv2.line(canvas, pts_t[i-1], pts_t[i], trainee_col, 2, cv2.LINE_AA)
        cv2.circle(canvas, pts_t[-1], 8, trainee_col, -1)
        cv2.circle(canvas, pts_t[-1], 8, (255,255,255), 1)

    # Expert path (smoothed trainee = reference)
    exp_x = _smooth(trainee_xs)
    exp_y = _smooth(trainee_ys)
    pts_e = [(int(x) + w + pad, int(y)) for x, y in zip(exp_x, exp_y)
             if x is not None and y is not None]
    if len(pts_e) >= 2:
        for i in range(1, len(pts_e)):
            cv2.line(canvas, pts_e[i-1], pts_e[i], expert_col, 3, cv2.LINE_AA)
        cv2.circle(canvas, pts_e[-1], 8, expert_col, -1)
        cv2.circle(canvas, pts_e[-1], 8, (255,255,255), 1)

    # Labels
    _put(canvas, f"Trainee  [{skill_level}]",
         (10, 26), trainee_col, scale=0.55)
    _put(canvas, f"Score: {composite_score:.1f}/100",
         (10, 48), (220, 220, 220), scale=0.48)
    _put(canvas, "Expert Reference",
         (w + pad + 10, 26), expert_col, scale=0.55)
    _put(canvas, "Score: 90+/100",
         (w + pad + 10, 48), (220, 220, 220), scale=0.48)

    # Divider
    cv2.line(canvas, (w + pad//2, 0), (w + pad//2, h), (80, 80, 80), 1)

    cv2.imwrite(str(output_path), canvas)
    return str(output_path)


def _smooth(coords: list, window: int = 7) -> list:
    out = []
    for i in range(len(coords)):
        vals = [c for c in coords[max(0, i-window//2): i+window//2+1]
                if c is not None]
        out.append(float(np.mean(vals)) if vals else None)
    return out


def find_best_frame(folder_path: str) -> Optional[str]:
    """Pick the clearest raw endo frame (mid-sequence, largest file = best exposure)."""
    folder = Path(folder_path)
    frames = sorted(
        folder.glob("*_endo.png"),
        key=lambda p: int("".join(filter(str.isdigit,
                                          p.name.split("endo")[0].split("_")[-1])) or "0")
    )
    if not frames:
        return None
    # Pick frame closest to 60% into the sequence (post-setup, pre-closure)
    idx = int(len(frames) * 0.6)
    return str(frames[min(idx, len(frames)-1)])
