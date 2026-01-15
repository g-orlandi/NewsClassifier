"""
Classification model-level training, prediction and complexity
measurement routines. 
This module does not manage experiment orchestration, which is
handled separately by the evaluator.

This module implements:

- model-specific training/evaluation wrappers
- structural complexity measures for interpretable models
- a unified metric computation interface

Functions here return dictionaries of evaluation metrics and (when applicable)
structural complexity values to allow uniform comparison across models in
the paper.
"""

from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    precision_recall_fscore_support,
    accuracy_score,
    confusion_matrix,
    balanced_accuracy_score,
)


# ============================================================================
#                            METRIC COMPUTATION
# ============================================================================

def classification_metrics_full(y_true, y_pred, labels=None):
    """
    Compute a comprehensive set of classification metrics.

    Returns both aggregated scores and diagnostic structures.
    """

    # --- Global metrics ---
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )

    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="micro", zero_division=0
    )

    accuracy = accuracy_score(y_true, y_pred)
    balanced_accuracy = balanced_accuracy_score(y_true, y_pred)

    # --- Per-class metrics ---
    precision_c, recall_c, f1_c, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )

    # --- Confusion matrix ---
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    # --- Structured output ---
    results = {
        # Global
        "Precision_macro": precision_macro,
        "Recall_macro": recall_macro,
        "f1-macro": f1_macro,
        "f1-micro": f1_micro,
        "Accuracy": accuracy,
        "Balanced_accuracy": balanced_accuracy,

        # Per-class
        "Per_class": {
            "precision": precision_c,
            "recall": recall_c,
            "f1": f1_c,
            "support": support,
        },

        # Structure
        "Confusion_matrix": cm,
    }

    return results

# ============================================================================
#                              MODEL WRAPPERS
# ============================================================================

from sklearn.exceptions import NotFittedError
import numpy as np

def train_model(model_name, hyperparams, X_train, X_test, y_train, y_test, submission=False):
    try:
        match model_name:
            case "logistic_regression":
                model = LogisticRegression(**hyperparams)
                model.fit(X_train, y_train)

            case "naive_bayes":
                # ComplementNB richiede X >= 0
                if X_train.min() < 0 or X_test.min() < 0:
                    raise ValueError("Negative values in X for ComplementNB.")
                model = ComplementNB(**hyperparams)
                model.fit(X_train, y_train)

            case "xgboost":
                model = xgb.XGBClassifier(**hyperparams)
                model.fit(X_train, y_train)

            case "linear_svm":
                model = LinearSVC(**hyperparams)
                model.fit(X_train, y_train)

            case _:
                raise RuntimeError(f"{model_name} is not a valid model name.")

        y_pred = model.predict(X_test)

    except (ValueError, NotFittedError, xgb.core.XGBoostError) as e:
        if submission:
            raise  # in submission vuoi sapere che è andato male
        return {"f1-macro": 0.0, "error": str(e)}

    if submission:
        return y_pred

    return classification_metrics_full(y_test, y_pred)
