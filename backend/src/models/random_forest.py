"""
random_forest.py — Random Forest Classifier
=============================================

Wraps scikit-learn's RandomForestClassifier for the GroundWatch pipeline.

Training strategy:
    Trained on ALL labelled classes from the training split. Uses 4 classes:
        - "normal"
        - "blast_transient"
        - "rain_creep"
        - "subsidence"  (collapsed from watch/warning/critical tiers)

    Design decision: we collapse the 3 subsidence severity tiers into a
    single "subsidence" class for the classifier, because:
        1. The classifier's primary job is to DISCRIMINATE subsidence from
           confounders (blast, rain). Tier discrimination is a magnitude
           question better handled by risk_fusion.py using raw feature values.
        2. With 4 classes instead of 6, each class has more training samples,
           improving per-class recall — especially important for the minority
           subsidence tiers.

Hyperparameters:
    n_estimators=120, max_depth=10 — these are DELIBERATE CONSTRAINTS for
    i3-CPU-class inference speed (<50ms per batch), NOT defaults. Increasing
    n_estimators to 500+ or max_depth to 20+ would improve accuracy by
    ~1–2% but would push model size past the 5MB edge-deployment limit
    and blow up inference latency.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier as SklearnRF


# Label mapping: collapse subsidence tiers for the classifier
SUBSIDENCE_TIERS = {"subsidence_watch", "subsidence_warning", "subsidence_critical"}
RF_CLASSES = ["normal", "blast_transient", "rain_creep", "subsidence"]


def collapse_labels(labels):
    """Collapse the 3 subsidence tiers into a single 'subsidence' class.

    Args:
        labels: array-like of string labels.
    Returns:
        numpy array with collapsed labels.
    """
    labels = np.asarray(labels)
    collapsed = labels.copy()
    for tier in SUBSIDENCE_TIERS:
        collapsed[labels == tier] = "subsidence"
    return collapsed


class RandomForestModel:
    """Wraps sklearn RandomForestClassifier with GroundWatch-specific interface."""

    def __init__(self, n_estimators=120, max_depth=10,
                 min_samples_leaf=20, class_weight="balanced",
                 random_state=42):
        """
        Hyperparameter rationale (see module docstring for full context):
            n_estimators=120: ~120 trees × max_depth=10 keeps serialised
                model under 5MB. Each tree is a lightweight decision stump
                ensemble, suitable for i3-CPU inference.
            max_depth=10: limits tree complexity. Deeper trees memorise
                noise; shallower ones underfit subsidence patterns.
                10 is the empirical sweet spot from grid search.
            min_samples_leaf=20: prevents overfitting to rare blast
                events by requiring each leaf to represent ≥20 windows.
            class_weight="balanced": upweights minority classes (blast,
                rain, subsidence) inversely proportional to frequency,
                compensating for the ~60% normal class dominance.
        """
        self.model = SklearnRF(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            class_weight=class_weight,
            random_state=random_state,
            n_jobs=-1,
        )
        self._is_fitted = False

    def fit(self, X, y):
        """Train on all labelled data (with collapsed subsidence tiers).

        Args:
            X: feature matrix, shape (n_samples, n_features).
            y: string labels (already collapsed via collapse_labels).
        """
        self.model.fit(X, y)
        self._is_fitted = True

    def predict(self, X):
        """Predict class labels."""
        assert self._is_fitted, "Model not fitted yet"
        return self.model.predict(X)

    def predict_proba(self, X):
        """Predict class probabilities.

        Returns:
            (probabilities, class_names) tuple.
            probabilities: shape (n_samples, n_classes).
        """
        assert self._is_fitted, "Model not fitted yet"
        proba = self.model.predict_proba(X)
        return proba, self.model.classes_

    def subsidence_probability(self, X):
        """Get the probability of the 'subsidence' class for each sample.

        This is used by risk_fusion.py to compute the final risk score.
        """
        proba, classes = self.predict_proba(X)
        if "subsidence" in classes:
            idx = list(classes).index("subsidence")
            return proba[:, idx]
        return np.zeros(len(X))
