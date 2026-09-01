"""
isolation_forest.py — Isolation Forest Anomaly Detector
========================================================

Wraps scikit-learn's IsolationForest for the GroundWatch pipeline.

Training strategy:
    Trained ONLY on "normal"-labelled windows from the training split.
    This is unsupervised anomaly detection: the model learns the
    distribution of normal sensor readings and flags anything that
    deviates as anomalous.

    The contamination parameter (expected fraction of anomalies) is
    initially set from config, then TUNED using the validation set's
    known anomaly rate to find the threshold that maximises separation.

Output:
    Per-window anomaly score in [0, 1] where:
        0 = very normal
        1 = very anomalous
    (scikit-learn's raw scores are negative; we normalise them.)
"""

import numpy as np
from sklearn.ensemble import IsolationForest as SklearnIF


class IsolationForestModel:
    """Wraps sklearn IsolationForest with GroundWatch-specific interface."""

    def __init__(self, n_estimators=150, max_features=0.8,
                 contamination="auto", random_state=42):
        """
        Args:
            n_estimators: number of isolation trees. 150 balances
                detection power vs model size for edge deployment.
            max_features: fraction of features per tree. 0.8 provides
                good diversity while keeping each tree informative.
            contamination: expected anomaly fraction. Set to "auto" for
                initial training; tuned on validation set afterward.
            random_state: reproducibility seed.
        """
        self.model = SklearnIF(
            n_estimators=n_estimators,
            max_features=max_features,
            contamination=contamination,
            random_state=random_state,
            n_jobs=-1,  # use all cores during training
        )
        self._is_fitted = False

    def fit(self, X_normal):
        """Fit on NORMAL-only data.

        Args:
            X_normal: feature matrix containing ONLY normal-labelled windows.
                      Shape (n_samples, n_features).
        """
        self.model.fit(X_normal)
        self._is_fitted = True

    def anomaly_scores(self, X):
        """Compute normalised anomaly scores in [0, 1].

        sklearn's decision_function returns NEGATIVE scores for anomalies
        and positive for inliers. We normalise:
            score = 1 − (raw − raw_min) / (raw_max − raw_min)
        so that 1 = maximally anomalous, 0 = maximally normal.
        """
        assert self._is_fitted, "Model not fitted yet"
        raw = self.model.decision_function(X)
        # Normalise to [0, 1] — higher = more anomalous
        raw_min, raw_max = raw.min(), raw.max()
        if raw_max - raw_min < 1e-10:
            return np.zeros(len(X))
        normalised = 1.0 - (raw - raw_min) / (raw_max - raw_min)
        return normalised

    def tune_contamination(self, X_val, y_val):
        """Tune contamination using validation set's known anomaly rate.

        The optimal contamination parameter should match the fraction of
        actually anomalous windows in the data. Since we know the true
        labels on the validation set, we can set this exactly.

        Args:
            X_val: validation feature matrix.
            y_val: validation labels (string labels from window_labels.csv).

        Returns:
            Optimal contamination value.
        """
        actual_anomaly_rate = np.mean(y_val != "normal")
        # Clamp to valid range for sklearn (0, 0.5]
        optimal = float(np.clip(actual_anomaly_rate, 0.01, 0.5))
        return optimal
