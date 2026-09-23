"""
xgboost_classifier.py -- XGBoost Multi-Class Severity Classifier
=================================================================

Replaces the previous Random Forest classifier with XGBoost's
gradient-boosted trees. The switch was motivated by model-size
constraints: the Random Forest consistently serialized to 6-12 MB
(exceeding the 5MB edge-deployment limit), while XGBoost's
histogram-based trees achieve comparable or better accuracy at
2-4 MB serialized size.

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
           improving per-class recall -- especially important for the minority
           subsidence tiers.

Hyperparameters:
    n_estimators=80, max_depth=6, learning_rate=0.1 -- these are
    DELIBERATE CONSTRAINTS for i3-CPU-class inference speed (<50ms per
    batch) AND the 5MB edge-deployment model-size limit.
    tree_method="hist" is mandatory for fast CPU training (do NOT use
    the default exact splitter on large datasets -- it is O(n*features)
    per split vs hist's O(bins*features)).
"""

import numpy as np
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier


# Label mapping: collapse subsidence tiers for the classifier
SUBSIDENCE_TIERS = {"subsidence_watch", "subsidence_warning", "subsidence_critical"}
XGB_CLASSES = ["normal", "blast_transient", "rain_creep", "subsidence"]


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


class XGBoostModel:
    """Wraps xgboost.XGBClassifier with GroundWatch-specific interface.

    Exposes the same fit/predict/predict_proba/subsidence_probability
    API as the previous RandomForestModel, so train.py, risk_fusion.py,
    and evaluate.py require only import-level changes.
    """

    def __init__(self, n_estimators=80, max_depth=6, learning_rate=0.1,
                 tree_method="hist", early_stopping_rounds=20,
                 subsample=1.0, colsample_bytree=1.0, min_child_weight=1,
                 random_state=42):
        """
        Hyperparameter rationale:
            n_estimators=80: 80 boosted trees (not 500+) -- each boosted
                tree is more informative than a single bagged RF tree, so
                fewer are needed. 80 keeps serialized size well under 5MB.
            max_depth=6: shallower than RF's typical 8-10 because boosting
                builds depth iteratively -- each tree corrects the previous
                ensemble's errors, so individual trees can be shallower.
            learning_rate=0.1: standard starting point; lower values (0.01)
                need more trees (blowing up model size), higher (0.3+)
                risk overfitting to noise.
            tree_method="hist": histogram-based splitting, O(bins*features)
                per split instead of O(n*features). MANDATORY for fast
                training on i3-class CPUs with 500k+ samples.
            early_stopping_rounds=20: stop if validation loss doesn't
                improve for 20 rounds -- prevents wasted training time
                on the i3 CPU and acts as implicit regularization.
            subsample: fraction of training rows per tree (stochastic
                gradient boosting). Values < 1.0 add diversity.
            colsample_bytree: fraction of features per tree. Values < 1.0
                reduce feature correlation between trees.
            min_child_weight: minimum sum of instance weight in a child.
                Higher values prevent splits on very small sample groups.
        """
        self._label_encoder = LabelEncoder()
        self._early_stopping_rounds = early_stopping_rounds
        self.model = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            tree_method=tree_method,
            objective="multi:softprob",
            eval_metric="mlogloss",
            use_label_encoder=False,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            min_child_weight=min_child_weight,
            random_state=random_state,
            n_jobs=-1,
            verbosity=0,  # suppress XGBoost's own logging
        )
        self._is_fitted = False

    @property
    def label_encoder(self):
        """Expose the label encoder for saving alongside the model."""
        return self._label_encoder

    def fit(self, X, y, X_val=None, y_val=None):
        """Train on all labelled data (with collapsed subsidence tiers).

        Args:
            X: feature matrix, shape (n_samples, n_features).
            y: string labels (already collapsed via collapse_labels).
            X_val: optional validation feature matrix for early stopping.
            y_val: optional validation string labels (already collapsed).
        """
        # Encode string labels to integers for XGBoost
        y_encoded = self._label_encoder.fit_transform(y)

        # Compute sample weights to balance classes -- equivalent to
        # sklearn's class_weight="balanced". Without this, XGBoost
        # ignores minority classes (rain_creep at 6.6% → near-zero recall).
        # Formula: weight(class_i) = n_samples / (n_classes * n_samples_i)
        classes, counts = np.unique(y_encoded, return_counts=True)
        n_samples = len(y_encoded)
        n_classes = len(classes)
        class_weights = {c: n_samples / (n_classes * cnt)
                         for c, cnt in zip(classes, counts)}
        sample_weights = np.array([class_weights[yi] for yi in y_encoded])

        # Set up early stopping with validation set if provided
        fit_params = {"sample_weight": sample_weights}
        if X_val is not None and y_val is not None:
            y_val_encoded = self._label_encoder.transform(y_val)
            fit_params["eval_set"] = [(X_val, y_val_encoded)]
            # XGBoost 2.x uses callbacks for early stopping
            self.model.set_params(
                early_stopping_rounds=self._early_stopping_rounds
            )

        self.model.fit(X, y_encoded, **fit_params)
        self._is_fitted = True

    def predict(self, X):
        """Predict class labels (returns string labels, not integers)."""
        assert self._is_fitted, "Model not fitted yet"
        y_pred_encoded = self.model.predict(X)
        return self._label_encoder.inverse_transform(y_pred_encoded)

    def predict_proba(self, X):
        """Predict class probabilities.

        Returns:
            (probabilities, class_names) tuple.
            probabilities: shape (n_samples, n_classes).
            class_names: array of string class labels matching column order.

        NOTE: class order follows LabelEncoder's alphabetical sorting,
        which matches sklearn RF's default. The caller should use the
        returned class_names to index into the probability matrix,
        NOT hardcode column positions.
        """
        assert self._is_fitted, "Model not fitted yet"
        proba = self.model.predict_proba(X)
        return proba, self._label_encoder.classes_

    def subsidence_probability(self, X):
        """Get the probability of the 'subsidence' class for each sample.

        This is used by risk_fusion.py to compute the final risk score.
        The class order comes from LabelEncoder (alphabetical), so we
        look up 'subsidence' by name, not by hardcoded index.
        """
        proba, classes = self.predict_proba(X)
        classes_list = list(classes)
        if "subsidence" in classes_list:
            idx = classes_list.index("subsidence")
            return proba[:, idx]
        return np.zeros(len(X))
