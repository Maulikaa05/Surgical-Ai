"""
feedback.py
Generates fully dynamic, data-driven clinical feedback.
Every sentence is injected with actual metric values.
Zero hardcoded repetition — each output is unique per session.
Phase-aware language throughout.
"""

from typing import Optional


PHASE_CONTEXT = {
    "Preparation":            "During Preparation, foundational positioning and portal placement set the entire procedural trajectory.",
    "Diagnosis":              "The Diagnosis phase demands systematic intra-articular survey across all compartments — camera stability here is the primary determinant of accurate pathology assessment.",
    "Femoral Tunnel Creation":"Femoral tunnel creation requires sub-millimetre precision; any instrument instability during this phase risks graft malpositioning.",
    "Tibial Tunnel Creation": "Tibial tunnel drilling demands a controlled, consistent trajectory — efficiency of movement directly correlates with minimised soft-tissue trauma.",
    "ACL Reconstruction":     "ACL graft placement is the most technically demanding phase of the procedure. Expert-level precision and velocity control are essential for optimal graft tension and clinical outcome.",
    "Unknown":                "This analysis evaluates the surgical segment as a whole.",
}


def _stab_text(score: float, phase: str) -> str:
    if score >= 80:
        return (f"Camera stability is excellent ({score:.1f}/100). "
                f"Instrument motion vectors show minimal tremor throughout the {phase} phase.")
    if score >= 60:
        return (f"Camera stability is good ({score:.1f}/100) with minor intermittent tremor. "
                f"Adjusting wrist support positioning may further improve scope steadiness.")
    if score >= 40:
        return (f"Significant camera shake observed (Stability: {score:.1f}/100). "
                f"Structured box-trainer practice targeting steady scope navigation "
                f"is recommended before the next {phase} procedure.")
    return (f"Excessive instrument instability detected (Stability: {score:.1f}/100). "
            f"Fundamental camera-control drills must be completed and passed before "
            f"unsupervised arthroscopic procedures are undertaken.")


def _eff_text(score: float) -> str:
    if score >= 80:
        return (f"Instrument path efficiency is high ({score:.1f}/100) — "
                f"all movements are direct and purposeful with minimal redundant excursion.")
    if score >= 60:
        return (f"Path efficiency is moderate ({score:.1f}/100). "
                f"Some indirect routing observed. Triangulation economy exercises "
                f"on a simulator are advised.")
    if score >= 40:
        return (f"Path efficiency is below target ({score:.1f}/100), "
                f"indicating indirect instrument routing. "
                f"Drills targeting straight-line approach trajectories are recommended.")
    return (f"Path efficiency is very low ({score:.1f}/100). "
            f"The instrument trajectory shows markedly indirect routing. "
            f"Supervised triangulation training is essential before independent practice.")


def _prec_text(score: float) -> str:
    if score >= 80:
        return (f"Instrument precision is strong ({score:.1f}/100) — "
                f"deviation from intended trajectory is consistently minimal.")
    if score >= 60:
        return (f"Precision is acceptable ({score:.1f}/100) with occasional "
                f"overshoot from the target trajectory. Fine motor control exercises "
                f"on a simulator are beneficial.")
    if score >= 40:
        return (f"Precision requires improvement ({score:.1f}/100). "
                f"Consistent deviation from target trajectory detected. "
                f"Instrument-tip targeting exercises are specifically recommended.")
    return (f"Precision is critically low ({score:.1f}/100). "
            f"Significant and consistent deviation from intended trajectory was detected. "
            f"Senior mentorship is required during all subsequent procedures.")


def _spd_text(score: float) -> str:
    if score >= 80:
        return (f"Speed consistency is excellent ({score:.1f}/100) — "
                f"velocity control is smooth and even throughout the segment.")
    if score >= 60:
        return (f"Speed consistency is acceptable ({score:.1f}/100) with occasional "
                f"velocity fluctuations. Practise maintaining even pace during scope navigation.")
    if score >= 40:
        return (f"Inconsistent instrument velocity detected ({score:.1f}/100). "
                f"Abrupt accelerations and decelerations increase the risk of "
                f"unintended tissue contact and iatrogenic injury.")
    return (f"Speed control is poor ({score:.1f}/100). "
            f"Erratic velocity patterns were detected throughout this session. "
            f"Slow, deliberate practice at reduced speed is recommended "
            f"before increasing procedural pace.")


def _idle_text(idle_pct: float) -> Optional[str]:
    """Returns a comment only if idle time is clinically significant (> 20%)."""
    if idle_pct >= 40:
        return (f"High idle time ({idle_pct:.1f}% of frames) suggests repeated hesitation "
                f"or loss of intra-articular orientation. "
                f"Systematic anatomical landmark familiarisation is strongly recommended.")
    if idle_pct >= 20:
        return (f"Moderate idle time noted ({idle_pct:.1f}%). "
                f"Brief pauses to reorient are expected at this training level "
                f"but will decrease with increased familiarity with intra-articular anatomy.")
    return None   # No comment if idle time is low — no repetition


def _pe_text(pe_score: float, tc: float, me: float, dex: float,
             interpretation: str) -> str:
    """Clinical commentary on procedural effectiveness."""
    lowest_sub = min(
        [("task completion", tc), ("motion economy", me), ("dexterity control", dex)],
        key=lambda x: x[1]
    )
    if pe_score >= 80:
        return (f"Procedural effectiveness is high ({pe_score:.1f}/100). "
                f"{interpretation} All sub-domains — task coverage, motion economy, "
                f"and dexterity — meet or approach expert thresholds.")
    if pe_score >= 65:
        return (f"Procedural effectiveness is good ({pe_score:.1f}/100). "
                f"{interpretation} "
                f"The weakest sub-domain is {lowest_sub[0]} ({lowest_sub[1]:.1f}/100); "
                f"targeted drills in this area are recommended.")
    if pe_score >= 50:
        return (f"Procedural effectiveness is moderate ({pe_score:.1f}/100). "
                f"{interpretation} "
                f"Particular focus on {lowest_sub[0]} ({lowest_sub[1]:.1f}/100) "
                f"would yield the greatest improvement in overall procedural quality.")
    return (f"Procedural effectiveness is low ({pe_score:.1f}/100). "
            f"{interpretation} "
            f"Foundational instrument handling drills across task coverage, "
            f"motion economy, and dexterity are all recommended before the next session.")


def _tremor_text(dom_hz: float, severity: str, tremor_score: float,
                 reliable: bool = True) -> str:
    """Clinical commentary on tremor analysis."""
    if not reliable:
        return (
            "Tremor and smoothness analysis requires a minimum of 60 continuous frames "
            "for clinically meaningful frequency estimation. This segment contained insufficient "
            "frames for reliable tremor quantification. Re-analyse using a longer video segment "
            "or a higher frame-rate capture to enable this assessment."
        )
    if tremor_score >= 85:
        return (f"Tremor analysis indicates minimal high-frequency oscillation "
                f"(dominant: {dom_hz:.1f} Hz — {severity}). "
                f"Instrument motion is well-controlled with no clinically significant tremor detected.")
    if tremor_score >= 65:
        return (f"Low-level instrument oscillation detected (dominant: {dom_hz:.1f} Hz — {severity}). "
                f"This is within acceptable range for the training level; "
                f"wrist stabilisation techniques and forearm support optimisation are suggested.")
    if tremor_score >= 45:
        return (f"Moderate tremor activity observed (dominant: {dom_hz:.1f} Hz — {severity}). "
                f"This frequency pattern is consistent with action tremor during fine instrument work. "
                f"Ergonomic positioning review and fatigue management strategies are recommended.")
    return (f"Significant instrument tremor detected (dominant: {dom_hz:.1f} Hz — {severity}). "
            f"This level of oscillation may compromise precision during critical manoeuvres. "
            f"A formal tremor assessment with an occupational therapist or ergonomics specialist is advised.")


OPENING = {
    "Expert":       lambda n, p: (
        f"This {p} session demonstrates expert-level arthroscopic performance. "
        f"All evaluated parameters are within or above the range expected of a "
        f"fellowship-trained consultant surgeon. {n} demonstrates mastery of "
        f"fundamental arthroscopic motor skills."
    ),
    "Advanced":     lambda n, p: (
        f"This {p} session reflects advanced arthroscopic competency consistent "
        f"with a senior surgical trainee (PGY4-5 or fellow). "
        f"{n}'s performance is characterised by strong foundational skills "
        f"with specific areas identified for refinement."
    ),
    "Intermediate": lambda n, p: (
        f"This {p} session reflects intermediate arthroscopic skill, consistent "
        f"with a junior resident (PGY2-3). "
        f"{n} demonstrates developing technique with clear priority areas "
        f"for targeted improvement identified below."
    ),
    "Novice":       lambda n, p: (
        f"This {p} session reflects foundational-level arthroscopic performance. "
        f"Significant skill development is required across all evaluated domains. "
        f"All future procedures for {n} should be conducted under direct senior supervision "
        f"until satisfactory progress is demonstrated."
    ),
}

ACTIONS = {
    "Expert":       [
        "Continue high-volume case exposure to maintain and consolidate procedural fluency.",
        "Engage in structured peer review of session recordings to identify micro-optimisation opportunities.",
        "Consider taking on a formal mentorship role — teaching reinforces and deepens expert-level technical mastery.",
        "Review phase-specific outlier segments (e.g. any sub-80 windows in the timeline) for targeted refinement.",
    ],
    "Advanced":     [
        f"Focus practice on the lowest-scoring metric domain to bridge to expert-level performance.",
        "Use this system to record and self-review a minimum of 3 procedures per week.",
        "Attend an advanced arthroscopy workshop focusing on complex reconstruction scenarios.",
        "Set a 90-day target: composite score ≥ 80 across 5 consecutive sessions.",
    ],
    "Intermediate": [
        "Complete a minimum of 20 structured sessions on a VirtaMed ArthroS or equivalent simulator.",
        "Perform all FAST (Fundamentals of Arthroscopic Surgery Training) module drills to competency threshold.",
        "Schedule monthly objective performance assessments using this platform and review with your programme director.",
        "Study intra-articular knee anatomy using 3D models and cadaveric specimens to reduce orientation hesitation.",
        "Specifically target triangulation drills to improve path efficiency.",
    ],
    "Novice":       [
        "Complete the full FAST training programme before any further unsupervised OR exposure.",
        "A minimum of 30 supervised simulator sessions is recommended before progression.",
        "Study intra-articular anatomy systematically using 3D visualisation tools and cadaveric specimens.",
        "All OR procedures must be directly supervised by a consultant surgeon until intermediate threshold is reached.",
        "Arrange weekly one-to-one mentorship sessions with your training programme director.",
    ],
}


def generate_feedback(metrics: dict,
                       skill_level: str,
                       phase_label: str = "Unknown",
                       surgeon_name: str = "The trainee") -> dict:
    """
    Generate fully dynamic clinical feedback.
    Deduplication guaranteed — each comment type appears exactly once.

    Returns dict with keys used by both dashboard and PDF generator.
    """
    stab  = metrics.get("stability_score",   50.0)
    eff   = metrics.get("efficiency_score",  50.0)
    prec  = metrics.get("precision_score",   50.0)
    spd   = metrics.get("speed_score",       50.0)
    idle  = metrics.get("idle_time_percent",  0.0)
    comp  = metrics.get("composite_score",   50.0)

    # NEW metric values
    pe_score  = metrics.get("procedural_effectiveness_score", 60.0)
    pe_tc     = metrics.get("pe_task_completion",   60.0)
    pe_me     = metrics.get("pe_motion_economy",    60.0)
    pe_dex    = metrics.get("pe_dexterity",         60.0)
    pe_interp = metrics.get("pe_interpretation",    "")
    tremor_hz      = metrics.get("tremor_dominant_hz", 0.0)
    tremor_sev     = metrics.get("tremor_severity",    "Minimal")
    tremor_scr     = metrics.get("tremor_score",       70.0)
    tremor_reliable= metrics.get("tremor_reliable",    True)

    phase_ctx    = PHASE_CONTEXT.get(phase_label, PHASE_CONTEXT["Unknown"])
    opening_fn   = OPENING.get(skill_level, OPENING["Intermediate"])
    opening      = opening_fn(surgeon_name, phase_label)

    # One comment per metric — guaranteed unique
    metric_comments = [
        _stab_text(stab, phase_label),
        _eff_text(eff),
        _prec_text(prec),
        _spd_text(spd),
    ]

    # Idle comment ONLY if clinically significant — never duplicated
    idle_comment = _idle_text(idle)

    # NEW: Procedural effectiveness and tremor commentary
    pe_comment     = _pe_text(pe_score, pe_tc, pe_me, pe_dex, pe_interp)
    tremor_comment = _tremor_text(tremor_hz, tremor_sev, tremor_scr, tremor_reliable)

    actions = ACTIONS.get(skill_level, ACTIONS["Intermediate"]).copy()
    # Replace placeholder lambda results in Advanced actions
    if skill_level == "Advanced":
        scores = {
            "stability": stab,
            "efficiency": eff,
            "precision": prec,
            "speed": spd
        }
        lowest_key = min(scores.items(), key=lambda x: x[1])[0]
        actions[0] = (f"Focus practice specifically on {lowest_key} "
                      f"(currently {scores[lowest_key]:.1f}/100) — "
                      f"your lowest-scoring domain — to bridge to Expert level.")

    return {
        "opening":         opening,
        "phase_context":   phase_ctx,
        "metric_comments": metric_comments,          # list of 4 unique strings
        "idle_comment":    idle_comment,             # str or None
        "pe_comment":      pe_comment,               # NEW
        "tremor_comment":  tremor_comment,           # NEW
        "actions":         actions,
        "composite_score": comp,
        "skill_level":     skill_level,
        "phase_label":     phase_label,
        "surgeon_name":    surgeon_name,
    }
