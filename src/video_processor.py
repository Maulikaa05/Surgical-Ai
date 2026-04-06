"""
video_processor.py
Handles video download (YouTube / direct URL) and frame extraction.
Integrates output frames as color-analysable PNGs for the main pipeline.
"""

import os
import cv2
import numpy as np
from pathlib import Path
from typing import Optional


def download_video_frames(url: str, output_dir: str) -> str:
    """
    Download a video from YouTube or any direct URL.
    Returns path to the downloaded video file.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Direct file URL (not YouTube)
    if not any(yt in url for yt in ["youtube.com", "youtu.be"]):
        try:
            import urllib.request
            video_file = str(output_path / "video.mp4")
            urllib.request.urlretrieve(url, video_file)
            return video_file
        except Exception as e:
            raise RuntimeError(f"Direct URL download failed: {e}")

    # YouTube URL
    try:
        import yt_dlp
    except ImportError:
        raise RuntimeError(
            "yt-dlp is not installed. Run: pip install yt-dlp\n"
            "Then restart the app and try again."
        )

    video_file = str(output_path / "video.mp4")

    ydl_opts = {
        # Prefer mp4 at ≤720p; fall back to any available format
        "format": (
            "bestvideo[ext=mp4][height<=720]+bestaudio[ext=m4a]"
            "/bestvideo[height<=720]+bestaudio"
            "/best[height<=720]"
            "/best"
        ),
        "outtmpl": str(output_path / "video.%(ext)s"),
        "merge_output_format": "mp4",
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,          # Only download single video, not playlist
        "socket_timeout": 30,
        # Bypass age-gate and region issues
        "age_limit": None,
        # Retry on transient errors
        "retries": 3,
        "fragment_retries": 3,
        # Use a realistic user-agent to avoid bot detection
        "http_headers": {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            )
        },
        # Post-processing
        "postprocessors": [{
            "key": "FFmpegVideoConvertor",
            "preferedformat": "mp4",
        }],
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
    except Exception as e:
        # Re-try with more permissive format selector
        ydl_opts["format"] = "best"
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
        except Exception as e2:
            raise RuntimeError(
                f"YouTube download failed.\n"
                f"Primary error: {e}\n"
                f"Fallback error: {e2}\n\n"
                f"Tips:\n"
                f"  • Make sure yt-dlp is up to date: pip install -U yt-dlp\n"
                f"  • Some videos are region-locked or require login\n"
                f"  • Try downloading the video manually and uploading as a file"
            )

    # Find the downloaded file (extension may vary)
    candidates = sorted(output_path.glob("video.*"), key=lambda p: p.stat().st_size, reverse=True)
    if candidates:
        best = candidates[0]
        # Rename to .mp4 for cv2 compatibility if needed
        if best.suffix.lower() not in (".mp4", ".avi", ".mov", ".mkv"):
            new_path = output_path / "video.mp4"
            best.rename(new_path)
            return str(new_path)
        return str(best)

    return video_file


def extract_frames_from_video(
    video_path: str,
    output_dir: str,
    max_frames: int = 300,
    frame_step: Optional[int] = None,
) -> int:
    """
    Extract frames from a video file and save them as PNG files
    compatible with the ArthroPhase pipeline.
    
    Creates both raw frames (frame_NNN_endo.png) and synthetic
    color masks (frame_NNN_endo_color_mask.png) by applying
    instrument-detection preprocessing.
    
    Returns number of frames extracted.
    """
    # Clean quote marks if the user accidentally included them
    video_path_clean = str(video_path).strip("\"'")
    cap = cv2.VideoCapture(video_path_clean)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path_clean}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    # Calculate step to get at most max_frames
    if frame_step is None:
        frame_step = max(1, total_frames // max_frames)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    count = 0
    frame_idx = 0

    while True:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break

        frame_num = frame_idx
        # Save raw endoscopic frame
        raw_path = output_path / f"frame_{frame_num:04d}_endo.png"
        cv2.imwrite(str(raw_path), frame)

        # Generate synthetic color mask by detecting bright/instrument regions
        color_mask = _generate_color_mask(frame)
        mask_path = output_path / f"frame_{frame_num:04d}_endo_color_mask.png"
        cv2.imwrite(str(mask_path), color_mask)

        count += 1
        frame_idx += frame_step

        if count >= max_frames:
            break

    cap.release()
    return count


def _generate_color_mask(frame: np.ndarray) -> np.ndarray:
    """
    Generate a synthetic color mask from a video frame.
    Detects the brightest/most salient instrument-like regions
    and highlights them in pink/salmon (matching ArthroPhase convention).
    
    Uses multiple detection strategies:
    1. Bright specular highlights (instrument reflections)
    2. High-saturation coloured regions
    3. Edge-dense compact blobs (instrument shaft)
    """
    h, w = frame.shape[:2]
    mask = np.zeros((h, w, 3), dtype=np.uint8)

    # Strategy 1: Detect instrument highlight via HSV saturation + value
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # High-value, low-saturation = specular reflection (metal instrument)
    bright_mask = cv2.inRange(hsv,
                               np.array([0, 0, 200]),
                               np.array([180, 60, 255]))
    
    # Strategy 2: Detect pink/red/salmon instrument markers
    # Pink range
    pink_mask = cv2.inRange(hsv,
                             np.array([140, 40, 100]),
                             np.array([180, 255, 255]))
    # Red range (wraps around in HSV)
    red1 = cv2.inRange(hsv, np.array([0, 50, 50]), np.array([15, 255, 255]))
    red2 = cv2.inRange(hsv, np.array([165, 50, 50]), np.array([180, 255, 255]))

    # Strategy 3: Compact bright blob detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

    # Combine all masks
    combined = cv2.bitwise_or(bright_mask, pink_mask)
    combined = cv2.bitwise_or(combined, red1)
    combined = cv2.bitwise_or(combined, red2)
    combined = cv2.bitwise_or(combined, thresh)

    # Morphological cleaning
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)

    # Find the largest blob (most likely the instrument)
    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        # Take top 3 blobs by area (handles multi-instrument scenarios)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:3]
        result_mask = np.zeros_like(combined)
        for c in contours:
            if cv2.contourArea(c) > 50:
                cv2.drawContours(result_mask, [c], -1, 255, -1)
        # Paint instrument region as pink/salmon on black background
        # This matches ArthroPhase color_mask convention
        mask[result_mask > 0] = [170, 130, 220]  # BGR: salmon/pink

    return mask
