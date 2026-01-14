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
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB
from sklearn.svm import LinearSVC

# ============================================================================
#                            METRIC COMPUTATION
# ============================================================================

def classification_metric(y_true, y_pred):
    """
    Compute evaluation metrics for classification models.

    Parameters
    ----------
    y_true : array-like
        Ground-truth labels.
    y_pred : array-like
        Predicted labels.
    model : sklearn-like estimator, optional
        Model instance to store on disk (if requested).
    model_file_name : str, optional
        Filename used when saving the model.

    Returns
    -------
    dict
        Dictionary containing macro-averaged precision, recall, F-beta,
        accuracy, and model complexity (if available).
    """
    # Main performance metrics (macro-averaged)
    precision, recall, fbeta, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=True
    )

    accuracy = accuracy_score(y_true, y_pred)

    results = {
        "Precision": precision,
        "Recall": recall,
        "f1-macro": fbeta,
        "Accuracy": accuracy,
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

    return classification_metric(y_test, y_pred)
