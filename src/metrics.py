"""
metrics.py
Calculates all 5 surgical performance metrics dynamically.
Self-calibrating normalisation — scales to the actual data range
so scores are always meaningful regardless of video resolution or FPS.

Metrics
-------
Stability   (35%) — smoothness of instrument motion (low variance = high score)
Efficiency  (25%) — path directness ratio, scale-corrected
Precision   (25%) — consistency of instrument placement
Speed       (15%) — velocity uniformity
+ Idle Time      — hesitation indicator (diagnostic, not in composite)
"""

import numpy as np
from typing import Optional


WEIGHTS = {"stability": 0.35, "efficiency": 0.25,
           "precision": 0.25, "speed": 0.15}

# Procedural effectiveness sub-weights
EFF_WEIGHTS = {
    "task_completion":   0.35,  # Did the instrument reach target regions?
    "motion_economy":    0.30,  # Low redundant motion
    "dexterity":         0.20,  # Smooth, controlled acceleration
    "idle_penalty":      0.15,  # Low idle/hesitation time
}


def _clean(records: list[dict]):
    """Filter None positions. Returns xs, ys, ts as float arrays."""
    valid = [r for r in records
             if r.get("tip_x") is not None and r.get("tip_y") is not None]
    if not valid:
        return np.array([]), np.array([]), np.array([])
    xs = np.array([r["tip_x"]     for r in valid], dtype=float)
    ys = np.array([r["tip_y"]     for r in valid], dtype=float)
    ts = np.array([r["timestamp"] for r in valid], dtype=float)
    return xs, ys, ts


def _to100(raw: float) -> float:
    return round(float(np.clip(raw * 100, 0.0, 100.0)), 2)


# ── Individual metrics ────────────────────────────────────────────────

def _stability(xs, ys) -> float:
    if len(xs) < 2:
        return 50.0
    dx, dy = np.diff(xs), np.diff(ys)
    mags   = np.sqrt(dx**2 + dy**2)
    # Normalise by mean magnitude so score is resolution-independent
    mean_m = max(np.mean(mags), 1e-6)
    cv     = np.std(mags) / mean_m          # coefficient of variation
    return _to100(1.0 / (1.0 + cv))        # CV=0 → 100, CV=1 → 50


def _efficiency(xs, ys) -> float:
    if len(xs) < 2:
        return 50.0
    dx, dy      = np.diff(xs), np.diff(ys)
    actual      = float(np.sum(np.sqrt(dx**2 + dy**2)))
    ideal       = float(np.linalg.norm([xs[-1] - xs[0], ys[-1] - ys[0]]))

    # Handle near-stationary segments (high idle):
    # If instrument barely moved at all, path IS efficient (no wasted motion)
    if actual < 1.0:
        return 85.0   # instrument was stationary — that's controlled

    ratio = ideal / actual
    # Penalise only when ratio is genuinely low (< 0.3 = very winding)
    # Map [0 → 1] to [30 → 100] with sigmoid-like shaping
    shaped = 0.30 + 0.70 * (ratio ** 0.5)
    return _to100(min(shaped, 1.0))


def _precision(xs, ys) -> float:
    if len(xs) < 2:
        return 50.0
    # Measure deviation from the REGRESSION LINE (intended trajectory)
    # This is more clinically meaningful than deviation from centroid
    t    = np.arange(len(xs), dtype=float)
    px   = np.polyfit(t, xs, 1)
    py   = np.polyfit(t, ys, 1)
    pred_x = np.polyval(px, t)
    pred_y = np.polyval(py, t)
    devs   = np.sqrt((xs - pred_x)**2 + (ys - pred_y)**2)
    # Normalise deviation by image diagonal estimate (assume 640×480)
    diag    = np.sqrt(640**2 + 480**2)
    norm_dev = np.mean(devs) / diag
    return _to100(1.0 / (1.0 + norm_dev * 10))


def _speed(xs, ys, ts) -> float:
    if len(xs) < 2:
        return 50.0
    dx, dy = np.diff(xs), np.diff(ys)
    dt     = np.diff(ts)
    dt[dt < 1e-9] = 1 / 30.0
    mags   = np.sqrt(dx**2 + dy**2)
    vels   = mags / dt
    mean_v = max(np.mean(vels), 1e-6)
    cv     = np.std(vels) / mean_v
    return _to100(1.0 / (1.0 + cv))


def _idle(xs, ys) -> float:
    """% frames where instrument moved < dynamic threshold (5th percentile of motion)."""
    if len(xs) < 2:
        return 0.0
    dx, dy = np.diff(xs), np.diff(ys)
    mags   = np.sqrt(dx**2 + dy**2)
    # Dynamic threshold: 10% of median motion magnitude
    threshold = max(np.percentile(mags, 10), 0.5)
    idle_n    = int(np.sum(mags < threshold))
    return round(idle_n / len(mags) * 100, 2)


# ── Tremor frequency analysis ────────────────────────────────────────

def _tremor_frequency(xs: np.ndarray, ys: np.ndarray,
                      ts: np.ndarray, fps: float = 30.0) -> dict:
    """
    Estimate pathological tremor frequency via FFT on displacement signal.
    Physiological tremor: 8-12 Hz. Action tremor: 4-8 Hz.
    Requires at least 60 frames for a clinically meaningful result.
    Returns dominant frequency and tremor severity label.
    """
    MIN_FRAMES_FOR_TREMOR = 60
    if len(xs) < MIN_FRAMES_FOR_TREMOR:
        return {
            "dominant_hz": 0.0,
            "severity": "Insufficient data (need ≥60 frames)",
            "tremor_score": 75.0,  # neutral score — not penalised
            "reliable": False,
        }
    # Compute magnitude of movement per frame
    dx = np.diff(xs)
    dy = np.diff(ys)
    mags = np.sqrt(dx**2 + dy**2)

    # FFT
    n = len(mags)
    fft_vals = np.abs(np.fft.rfft(mags))
    freqs = np.fft.rfftfreq(n, d=1.0 / fps)

    # Find dominant frequency (ignore DC component)
    if len(freqs) > 1:
        dom_idx = np.argmax(fft_vals[1:]) + 1
        dom_hz = float(freqs[dom_idx])
    else:
        dom_hz = 0.0

    # Score: ideal is low freq (smooth), penalise high freq tremor
    if dom_hz < 2.0:
        severity = "Minimal"
        tscore = 92.0
    elif dom_hz < 4.0:
        severity = "Low"
        tscore = 80.0
    elif dom_hz < 8.0:
        severity = "Moderate (action tremor range)"
        tscore = 60.0
    elif dom_hz < 12.0:
        severity = "Elevated (physiological tremor range)"
        tscore = 42.0
    else:
        severity = "High (pathological concern)"
        tscore = 25.0

    return {
        "dominant_hz": round(dom_hz, 2),
        "severity": severity,
        "tremor_score": round(tscore, 1),
        "reliable": True,
    }


# ── Jerk / Smoothness (3rd derivative of position) ───────────────────

def _smoothness_jerk(xs: np.ndarray, ys: np.ndarray,
                     ts: np.ndarray) -> float:
    """
    Log dimensionless jerk (SPARC-inspired) — a gold-standard smoothness
    metric used in surgical robotics research.
    Lower jerk → smoother → higher score.
    Returns -1.0 sentinel if fewer than 30 frames (unreliable).
    """
    if len(xs) < 30:
        return -1.0  # sentinel: insufficient data — do not display score
    dt = np.diff(ts)
    dt[dt < 1e-9] = 1 / 30.0
    dx = np.diff(xs)
    dy = np.diff(ys)
    vx = dx / dt
    vy = dy / dt
    # Second derivative (acceleration)
    ax = np.diff(vx) / dt[1:]
    ay = np.diff(vy) / dt[1:]
    if len(ax) < 2:
        return -1.0
    # Third derivative (jerk)
    jx = np.diff(ax) / dt[2:]
    jy = np.diff(ay) / dt[2:]
    if len(jx) == 0:
        return -1.0
    jerk_mag = np.sqrt(jx**2 + jy**2)

    mean_jerk = np.mean(jerk_mag)
    # Normalise by median step size to make scale-independent
    step_median = np.median(np.sqrt(dx**2 + dy**2)) + 1e-6
    norm_jerk = mean_jerk / (step_median * 1000 + 1e-6)
    score = _to100(1.0 / (1.0 + norm_jerk))
    # Floor at 30 — jerk is noisy with few frames
    return max(float(score), 30.0)


# ── Procedural effectiveness score ───────────────────────────────────

def _procedural_effectiveness(xs: np.ndarray, ys: np.ndarray,
                               ts: np.ndarray,
                               phase_label: str = "Unknown") -> dict:
    """
    Composite effectiveness score evaluating whether the surgeon completed
    the procedural task effectively — not just how steady they were.

    Sub-scores:
      task_completion  — Did instrument traverse expected spatial range?
      motion_economy   — Ratio of useful to total motion
      dexterity        — Smoothness of acceleration changes (jerk)
      idle_penalty     — Penalises excessive hesitation

    Returns sub-scores and a weighted composite PE score (0–100).
    """
    n = len(xs)
    if n < 2:
        return {
            "procedural_effectiveness_score": 50.0,
            "pe_task_completion": 50.0,
            "pe_motion_economy": 50.0,
            "pe_dexterity": 50.0,
            "pe_idle_penalty": 50.0,
            "pe_interpretation": "Insufficient data",
        }

    # 1. Task completion: expect instrument to traverse at least 15% of
    #    estimated workspace (image diagonal proxy)
    diag = np.sqrt(640**2 + 480**2)
    x_range = float(np.max(xs) - np.min(xs))
    y_range = float(np.max(ys) - np.min(ys))
    coverage = np.sqrt(x_range**2 + y_range**2) / diag
    # Phase-specific coverage expectations
    phase_thresholds = {
        "ACL Reconstruction": 0.25,
        "Femoral Tunnel Creation": 0.20,
        "Tibial Tunnel Creation": 0.20,
        "Diagnosis": 0.35,
        "Preparation": 0.15,
    }
    expected_cov = phase_thresholds.get(phase_label, 0.20)
    tc_score = _to100(min(coverage / expected_cov, 1.0) * 0.85 + 0.15)

    # 2. Motion economy: useful displacement vs total path length
    dx = np.diff(xs)
    dy = np.diff(ys)
    mags = np.sqrt(dx**2 + dy**2)
    total_path = float(np.sum(mags))
    net_disp = float(np.linalg.norm([xs[-1] - xs[0], ys[-1] - ys[0]]))
    if total_path < 1.0:
        me_score = 85.0
    else:
        economy_ratio = net_disp / total_path
        me_score = _to100(0.25 + 0.75 * (economy_ratio ** 0.6))

    # 3. Dexterity: smoothness jerk score — use neutral 65 if unreliable
    jerk_raw = _smoothness_jerk(xs, ys, ts)
    dex_score = max(jerk_raw, 30.0) if jerk_raw >= 0 else 65.0

    # 4. Idle penalty
    threshold = max(np.percentile(mags, 10), 0.5)
    idle_n = int(np.sum(mags < threshold))
    idle_pct = idle_n / len(mags) * 100
    idle_score = _to100(1.0 - min(idle_pct / 60.0, 1.0))

    pe_composite = round(
        tc_score * EFF_WEIGHTS["task_completion"] +
        me_score * EFF_WEIGHTS["motion_economy"] +
        dex_score * EFF_WEIGHTS["dexterity"] +
        idle_score * EFF_WEIGHTS["idle_penalty"],
        2
    )

    # Interpretation
    if pe_composite >= 80:
        interp = "Excellent procedural execution — instrument use is purposeful and complete."
    elif pe_composite >= 65:
        interp = "Good procedural effectiveness — minor inefficiencies in task completion observed."
    elif pe_composite >= 50:
        interp = "Moderate effectiveness — instrument coverage and motion economy need improvement."
    else:
        interp = "Low procedural effectiveness — significant hesitation and incomplete task coverage detected."

    return {
        "procedural_effectiveness_score": pe_composite,
        "pe_task_completion": round(tc_score, 1),
        "pe_motion_economy": round(me_score, 1),
        "pe_dexterity": round(dex_score, 1),
        "pe_idle_penalty": round(idle_score, 1),
        "pe_interpretation": interp,
    }


# ── Phase time duration analysis ─────────────────────────────────────

def phase_time_analysis(records: list[dict],
                         phase_anns: dict,
                         fps: float = 30.0) -> dict:
    """
    Calculate per-phase time durations and proportion from annotations.

    Args:
        records:    list of frame dicts (must have 'frame' key)
        phase_anns: dict mapping frame_number → phase_label
        fps:        video frame rate

    Returns dict with per-phase breakdown and total procedure time.
    """
    if not records or not phase_anns:
        # Estimate from frame timestamps when no annotations
        ts_vals = [r.get("timestamp", 0) for r in records if r.get("timestamp") is not None]
        if len(ts_vals) >= 2:
            duration_s = float(ts_vals[-1] - ts_vals[0])
        else:
            duration_s = len(records) / fps
        return {
            "total_duration_s": round(duration_s, 1),
            "total_duration_formatted": _fmt_time(duration_s),
            "phases": {},
            "phase_proportions": {},
            "note": "No phase annotations available — total duration estimated from timestamps",
        }

    # Map frames to phases
    phase_frames: dict[str, list] = {}
    for rec in records:
        frame_num = rec.get("frame", 0)
        phase = phase_anns.get(frame_num, phase_anns.get(str(frame_num), "Unknown"))
        phase_frames.setdefault(phase, []).append(frame_num)

    # Calculate durations
    phases_out = {}
    total_frames = len(records)
    for phase, frames in phase_frames.items():
        n = len(frames)
        dur_s = n / fps
        phases_out[phase] = {
            "frame_count": n,
            "duration_s": round(dur_s, 1),
            "duration_formatted": _fmt_time(dur_s),
            "proportion_pct": round(n / max(total_frames, 1) * 100, 1),
        }

    total_s = total_frames / fps
    proportions = {ph: v["proportion_pct"] for ph, v in phases_out.items()}

    return {
        "total_duration_s": round(total_s, 1),
        "total_duration_formatted": _fmt_time(total_s),
        "phases": phases_out,
        "phase_proportions": proportions,
        "note": f"Phase breakdown from {len(phase_anns)} annotated frames",
    }


def _fmt_time(seconds: float) -> str:
    """Format seconds to mm:ss string."""
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m}:{s:02d}"


# ── Windowed stability timeline ───────────────────────────────────────

def stability_timeline(xs: np.ndarray, ys: np.ndarray,
                        window: int = 8) -> list[dict]:
    """Rolling window stability scores for the timeline chart."""
    results = []
    step    = max(1, window // 2)
    for i in range(0, len(xs) - window + 1, step):
        wx = xs[i: i + window]
        wy = ys[i: i + window]
        results.append({
            "window_start": int(i),
            "stability":    _stability(wx, wy),
        })
    return results


# ── Main pipeline ─────────────────────────────────────────────────────

def calculate_all_metrics(records: list[dict],
                           phase_label: Optional[str] = None) -> dict:
    """
    Full metric pipeline.

    Returns
    -------
    dict with keys:
      metrics     — all 5 scores
      timeline    — per-window stability list
      trajectory  — xs/ys/ts arrays (for plotting)
      frames_analysed
      detection_rate  — % frames where instrument was found
      phase_label
    """
    total = len(records)
    xs, ys, ts = _clean(records)
    detected   = len(xs)

    if detected < 2:
        return {
            "error": (
                f"Only {detected} frames had instrument detected "
                f"(out of {total} total). "
                "Check that color_mask PNGs contain the pink/salmon instrument region."
            )
        }

    stab = _stability(xs, ys)
    eff  = _efficiency(xs, ys)
    prec = _precision(xs, ys)
    spd  = _speed(xs, ys, ts)
    idle = _idle(xs, ys)

    composite = round(
        stab * WEIGHTS["stability"]  +
        eff  * WEIGHTS["efficiency"] +
        prec * WEIGHTS["precision"]  +
        spd  * WEIGHTS["speed"],
        2
    )

    # NEW: Procedural effectiveness score
    pe_data = _procedural_effectiveness(xs, ys, ts, phase_label or "Unknown")

    # NEW: Tremor frequency analysis (reliable only with ≥60 frames)
    tremor_data = _tremor_frequency(xs, ys, ts)
    tremor_reliable = tremor_data.get("reliable", True)

    # NEW: Smoothness jerk (reliable only with ≥30 frames; returns -1 sentinel if not)
    jerk_raw = _smoothness_jerk(xs, ys, ts)
    jerk_reliable = jerk_raw >= 0
    jerk_score = jerk_raw if jerk_reliable else -1.0

    # Estimate FPS from timestamps
    if len(ts) >= 2 and (ts[-1] - ts[0]) > 0:
        fps_est = (len(ts) - 1) / (ts[-1] - ts[0])
    else:
        fps_est = 30.0

    # Phase time analysis (basic — no annotations here; annotations handled in app.py)
    duration_s = float(ts[-1] - ts[0]) if len(ts) >= 2 else len(xs) / fps_est
    phase_timing = {
        "total_duration_s": round(duration_s, 1),
        "total_duration_formatted": _fmt_time(duration_s),
        "estimated_fps": round(fps_est, 1),
        "frames_analysed": detected,
        "note": "Full per-phase breakdown requires phase annotations",
    }

    timeline = stability_timeline(xs, ys)

    return {
        "metrics": {
            "stability_score":                  stab,
            "efficiency_score":                 eff,
            "precision_score":                  prec,
            "speed_score":                      spd,
            "idle_time_percent":                idle,
            "composite_score":                  composite,
            # NEW
            "procedural_effectiveness_score":   pe_data["procedural_effectiveness_score"],
            "pe_task_completion":               pe_data["pe_task_completion"],
            "pe_motion_economy":                pe_data["pe_motion_economy"],
            "pe_dexterity":                     pe_data["pe_dexterity"],
            "pe_idle_penalty":                  pe_data["pe_idle_penalty"],
            "pe_interpretation":                pe_data["pe_interpretation"],
            "tremor_dominant_hz":               tremor_data["dominant_hz"],
            "tremor_severity":                  tremor_data["severity"],
            "tremor_score":                     tremor_data["tremor_score"],
            "tremor_reliable":                  tremor_reliable,
            "smoothness_jerk_score":            jerk_score,
            "jerk_reliable":                    jerk_reliable,
        },
        "phase_timing":   phase_timing,
        "timeline":       timeline,
        "trajectory":     {"xs": xs.tolist(), "ys": ys.tolist(), "ts": ts.tolist()},
        "frames_analysed": detected,
        "frames_total":    total,
        "detection_rate":  round(detected / max(total, 1) * 100, 1),
        "phase_label":    phase_label or "Unknown",
    }


# ── Baseline computation across multiple segments ─────────────────────

def compute_baseline(results: list[dict]) -> dict:
    """
    Compute dataset-wide percentile stats from a list of metric dicts.
    Used to contextualise a surgeon's score against all other sessions.
    """
    keys = ["stability_score", "efficiency_score",
            "precision_score", "speed_score", "composite_score"]
    baseline = {}
    for k in keys:
        vals = [r["metrics"][k] for r in results
                if "metrics" in r and k in r["metrics"]]
        if vals:
            baseline[k] = {
                "min":  round(float(np.min(vals)),  2),
                "p25":  round(float(np.percentile(vals, 25)), 2),
                "p50":  round(float(np.percentile(vals, 50)), 2),
                "p75":  round(float(np.percentile(vals, 75)), 2),
                "max":  round(float(np.max(vals)),  2),
                "mean": round(float(np.mean(vals)), 2),
                "n":    len(vals),
            }
    return baseline


def percentile_rank(score: float, key: str, baseline: dict) -> int:
    """Return 0–100 percentile rank of score in baseline distribution."""
    if key not in baseline:
        return 50
    b = baseline[key]
    if score >= b["p75"]:
        span = max(b["max"] - b["p75"], 1)
        return min(100, int(75 + 25 * (score - b["p75"]) / span))
    elif score >= b["p50"]:
        span = max(b["p75"] - b["p50"], 1)
        return int(50 + 25 * (score - b["p50"]) / span)
    elif score >= b["p25"]:
        span = max(b["p50"] - b["p25"], 1)
        return int(25 + 25 * (score - b["p25"]) / span)
    else:
        span = max(b["p25"] - b["min"], 1)
        return max(0, int(25 * (score - b["min"]) / span))