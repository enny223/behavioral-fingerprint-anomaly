"""
profiler.py

Per-user behavioral profile builder and anomaly scorer.
Trains one-class models on legitimate sessions only and evaluates
using False Rejection Rate, False Acceptance Rate, and Equal Error Rate.
"""

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import roc_curve


def compute_eer(y_true, anomaly_scores):
    """
    Compute Equal Error Rate (EER) from binary labels and anomaly scores.

    Parameters
    ----------
    y_true : array-like
        Binary labels — 1 = legitimate, 0 = impostor.
    anomaly_scores : array-like
        Anomaly scores — higher = more anomalous = more likely impostor.

    Returns
    -------
    eer : float
        Equal Error Rate — threshold where FRR == FAR.
    frr : array
        False Rejection Rate at each threshold.
    far : array
        False Acceptance Rate at each threshold.
    thresholds : array
        Threshold values corresponding to frr/far.

    Notes
    -----
    Scores are negated before passing to roc_curve because roc_curve
    treats high values as evidence of the positive class (legitimate=1),
    but high anomaly scores indicate impostors. Negating corrects the
    direction. Failure to negate produces EER > 50% — worse than random.
    See notebooks/03_modeling.ipynb for full diagnosis of this pitfall.
    """
    # Negate: roc_curve expects high = positive (legitimate)
    # but high anomaly score = impostor
    fpr, tpr, thresholds = roc_curve(y_true, -anomaly_scores)

    frr = 1 - tpr  # false rejection rate
    far = fpr       # false acceptance rate

    eer_idx = np.argmin(np.abs(frr - far))
    eer = (frr[eer_idx] + far[eer_idx]) / 2

    return eer, frr, far, thresholds


def build_profile(legitimate_features, contamination=0.05):
    """
    Train all three anomaly detection models on legitimate sessions.

    Parameters
    ----------
    legitimate_features : np.ndarray
        Feature matrix of legitimate sessions only — shape (n_sessions, n_features).
    contamination : float
        Expected fraction of outliers in training data. Keep low (0.05)
        with small training sets to avoid assuming legitimate sessions are anomalous.

    Returns
    -------
    dict of fitted model objects
    """
    models = {}

    models['isolation_forest'] = IsolationForest(
        contamination=contamination,
        random_state=42,
        n_estimators=100
    ).fit(legitimate_features)

    models['one_class_svm'] = OneClassSVM(
        nu=contamination,
        kernel='rbf',
        gamma='scale'
    ).fit(legitimate_features)

    models['lof'] = LocalOutlierFactor(
        n_neighbors=5,
        novelty=True,
        contamination=contamination
    ).fit(legitimate_features)

    return models


def score_session(models, session_features):
    """
    Score a single session against all three models.

    Parameters
    ----------
    models : dict
        Fitted model objects from build_profile.
    session_features : np.ndarray
        Feature vector for one session — shape (1, n_features).

    Returns
    -------
    dict of float anomaly scores — higher = more suspicious
    """
    scores = {}
    for name, model in models.items():
        scores[name] = float(-model.decision_function(session_features))
    return scores


def evaluate_user(user_data, feature_cols, contamination=0.05):
    """
    Train and evaluate all 3 models for a single user.
    Models trained on legitimate sessions only.
    Evaluated on all sessions using EER.

    Parameters
    ----------
    user_data : pd.DataFrame
        All sessions for one user with 'label' column.
    feature_cols : list of str
        Feature columns to use.
    contamination : float
        Passed to build_profile.

    Returns
    -------
    dict — EER and curve data per model
    """
    legitimate = user_data[user_data['label'] == 'legitimate'][feature_cols].values
    impostor = user_data[user_data['label'] == 'impostor'][feature_cols].values

    all_features = np.vstack([legitimate, impostor])
    all_labels = np.array([1] * len(legitimate) + [0] * len(impostor))

    models = build_profile(legitimate, contamination)

    results = {}
    for name, model in models.items():
        scores = -model.decision_function(all_features)
        eer, frr, far, thresholds = compute_eer(all_labels, scores)
        results[name] = {
            'eer': eer,
            'frr': frr,
            'far': far,
            'thresholds': thresholds,
            'scores': scores,
            'labels': all_labels
        }

    return results