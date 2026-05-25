"""
scoring.py

Enrollment sensitivity analysis and threshold optimization utilities.
Answers the question: how does performance degrade as enrollment data decreases?
"""

import numpy as np
import pandas as pd
from sklearn.svm import OneClassSVM

from src.profiler import compute_eer


def enrollment_sensitivity(
    df,
    feature_cols,
    enrollment_sizes=None,
    contamination=0.05
):
    """
    Measure EER degradation as enrollment session count decreases.

    Trains One-Class SVM on the first n_enroll legitimate sessions,
    evaluates on remaining legitimate + all impostor sessions.

    Parameters
    ----------
    df : pd.DataFrame
        Full feature matrix with 'user_id' and 'label' columns.
    feature_cols : list of str
        Feature columns to use.
    enrollment_sizes : list of int, optional
        Number of enrollment sessions to test. Defaults to [10,8,6,4,2,1].
    contamination : float
        Passed to OneClassSVM as nu parameter.

    Returns
    -------
    pd.DataFrame with columns:
        n_enroll, mean_eer, median_eer, std_eer, degradation_pct
    """
    if enrollment_sizes is None:
        enrollment_sizes = [10, 8, 6, 4, 2, 1]

    results = []
    baseline_mean = None

    for n_enroll in enrollment_sizes:
        size_eers = []

        for user_id, user_data in df.groupby('user_id'):
            legitimate = user_data[user_data['label'] == 'legitimate']
            impostor = user_data[user_data['label'] == 'impostor']

            # Train on first n_enroll legitimate sessions
            train = legitimate.iloc[:n_enroll][feature_cols].values

            # Evaluate on remaining legitimate + all impostors
            eval_leg = legitimate.iloc[n_enroll:]
            if len(eval_leg) == 0:
                eval_leg = legitimate.iloc[-1:]

            eval_data = pd.concat([eval_leg, impostor])
            eval_features = eval_data[feature_cols].values
            eval_labels = np.array(
                [1] * len(eval_leg) + [0] * len(impostor)
            )

            try:
                model = OneClassSVM(
                    nu=contamination,
                    kernel='rbf',
                    gamma='scale'
                ).fit(train)
                scores = -model.decision_function(eval_features)
                eer, _, _, _ = compute_eer(eval_labels, scores)
                size_eers.append(eer)
            except Exception:
                size_eers.append(0.5)

        mean_eer = np.mean(size_eers)

        if baseline_mean is None:
            baseline_mean = mean_eer

        degradation = (mean_eer - baseline_mean) / baseline_mean * 100

        results.append({
            'n_enroll': n_enroll,
            'mean_eer': mean_eer,
            'median_eer': np.median(size_eers),
            'std_eer': np.std(size_eers),
            'degradation_pct': degradation
        })

    return pd.DataFrame(results)


def optimal_threshold(frr, far, thresholds, strategy='eer'):
    """
    Select an operational threshold based on a deployment strategy.

    Parameters
    ----------
    frr : array
        False Rejection Rate at each threshold.
    far : array
        False Acceptance Rate at each threshold.
    thresholds : array
        Threshold values.
    strategy : str
        'eer'        — minimize |FRR - FAR| (balanced)
        'low_far'    — FAR <= 0.05 with lowest FRR (high security)
        'low_frr'    — FRR <= 0.05 with lowest FAR (low friction)

    Returns
    -------
    float — selected threshold value
    """
    if strategy == 'eer':
        idx = np.argmin(np.abs(frr - far))

    elif strategy == 'low_far':
        candidates = np.where(far <= 0.05)[0]
        if len(candidates) == 0:
            idx = np.argmin(far)
        else:
            idx = candidates[np.argmin(frr[candidates])]

    elif strategy == 'low_frr':
        candidates = np.where(frr <= 0.05)[0]
        if len(candidates) == 0:
            idx = np.argmin(frr)
        else:
            idx = candidates[np.argmin(far[candidates])]

    else:
        raise ValueError(f"Unknown strategy: {strategy}. "
                         f"Choose from 'eer', 'low_far', 'low_frr'.")

    return float(thresholds[idx])