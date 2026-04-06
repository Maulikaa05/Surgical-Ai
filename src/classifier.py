"""
classifier.py
Assigns skill level dynamically.
• With baseline: uses percentile rank against actual dataset distribution.
• Without baseline: uses ASSET-aligned rule-based thresholds.
Optional ML upgrade with SVM on metric vectors.
"""

import numpy as np
from typing import Optional

try:
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    SKLEARN_OK = True
except ImportError:
    SKLEARN_OK = False

# ASSET-aligned thresholds
THRESHOLDS = [
    ("Expert",       80),
    ("Advanced",     60),
    ("Intermediate", 40),
    ("Novice",        0),
]

LEVEL_META = {
    "Expert":       {"colour": "#27AE60", "badge": "#1E8449", "emoji": "🏆",
                     "clinical": "Fellowship-trained consultant surgeon"},
    "Advanced":     {"colour": "#2980B9", "badge": "#1A5276", "emoji": "⭐",
                     "clinical": "Senior trainee (PGY4-5 / Fellow)"},
    "Intermediate": {"colour": "#D68910", "badge": "#9A6600", "emoji": "📈",
                     "clinical": "Junior resident (PGY2-3)"},
    "Novice":       {"colour": "#C0392B", "badge": "#922B21", "emoji": "🎓",
                     "clinical": "Medical student / PGY1"},
}

_pipeline = None   # cached ML model


def _rule(composite: float) -> str:
    for level, threshold in THRESHOLDS:
        if composite >= threshold:
            return level
    return "Novice"


def _percentile(composite: float, baseline: dict) -> str:
    if "composite_score" not in baseline:
        return _rule(composite)
    b = baseline["composite_score"]
    if composite >= b["p75"]:   return "Expert"
    if composite >= b["p50"]:   return "Advanced"
    if composite >= b["p25"]:   return "Intermediate"
    return "Novice"


def classify(metrics: dict, baseline: Optional[dict] = None) -> dict:
    composite = metrics.get("composite_score", 50.0)

    # ML path
    if SKLEARN_OK and _pipeline is not None:
        vec   = [_feature_vec(metrics)]
        label = _pipeline.predict(vec)[0]
        proba = float(max(_pipeline.predict_proba(vec)[0]))
        return {"skill_level": label, "confidence": round(proba, 3), "method": "ml"}

    # Percentile path
    if baseline:
        label = _percentile(composite, baseline)
        return {"skill_level": label, "confidence": 0.78, "method": "percentile"}

    # Rule-based
    label = _rule(composite)
    conf  = abs(composite - 50) / 50 * 0.5 + 0.5   # higher near extremes
    return {"skill_level": label, "confidence": round(conf, 3), "method": "rule_based"}


def train_classifier(results: list[dict]) -> bool:
    if not SKLEARN_OK or len(results) < 4:
        return False
    X, y = [], []
    for r in results:
        if "metrics" not in r:
            continue
        label = _rule(r["metrics"]["composite_score"])
        X.append(_feature_vec(r["metrics"]))
        y.append(label)
    if len(set(y)) < 2:
        return False
    global _pipeline
    _pipeline = Pipeline([
        ("sc",  StandardScaler()),
        ("svm", SVC(kernel="rbf", C=10, gamma="scale", probability=True)),
    ])
    _pipeline.fit(X, y)
    return True


def _feature_vec(metrics: dict) -> list:
    return [
        metrics.get("stability_score",   50),
        metrics.get("efficiency_score",  50),
        metrics.get("precision_score",   50),
        metrics.get("speed_score",       50),
        metrics.get("idle_time_percent",  0),
    ]


def skill_color(level: str) -> str:
    return LEVEL_META.get(level, LEVEL_META["Intermediate"])["colour"]

def skill_badge_color(level: str) -> str:
    return LEVEL_META.get(level, LEVEL_META["Intermediate"])["badge"]

def skill_emoji(level: str) -> str:
    return LEVEL_META.get(level, LEVEL_META["Intermediate"])["emoji"]

def skill_clinical_equiv(level: str) -> str:
    return LEVEL_META.get(level, LEVEL_META["Intermediate"])["clinical"]

def score_band(score: float) -> str:
    return _rule(score)

def score_colour(score: float) -> str:
    return skill_color(score_band(score))
