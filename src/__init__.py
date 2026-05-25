"""
src — Behavioral Fingerprint Anomaly Detection

Modules:
    features  — raw event feature extraction
    profiler  — per-user behavioral profile and EER evaluation
    scoring   — enrollment sensitivity and threshold optimization
"""

from src.features import (
    extract_keystroke_features,
    extract_mouse_features,
    extract_session_features
)

from src.profiler import (
    compute_eer,
    build_profile,
    score_session,
    evaluate_user
)

from src.scoring import (
    enrollment_sensitivity,
    optimal_threshold
)