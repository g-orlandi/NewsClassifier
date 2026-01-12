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
import pandas as pd

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
        "Fbeta": fbeta,
        "Accuracy": accuracy,
        "f1-macro": f1_score(y_true, y_pred, average='macro')
    }

    return results


# ============================================================================
#                              MODEL WRAPPERS
# ============================================================================


def logistic_regression_performances(hyperparams, X_train, X_test, y_train, y_test, submission=False):
    """Train and evaluate Logistic Regression."""
    from sklearn.linear_model import LogisticRegression

    model = LogisticRegression(**hyperparams).fit(X_train, y_train)
    y_pred = model.predict(X_test)
    if submission:
        return y_pred
    return classification_metric(y_test, y_pred)

# ============================================================================
#                               NAIVE BAYES
# ============================================================================

def naive_bayes_performances(hyperparams, X_train, X_test, y_train, y_test, submission=False):
    """Train and evaluate Complement Naive Bayes"""
    from sklearn.naive_bayes import ComplementNB

    model = ComplementNB(**hyperparams).fit(X_train, y_train)
    y_pred = model.predict(X_test)
    if submission:
        return y_pred
    return classification_metric(y_test, y_pred)

# ============================================================================
#                                XGBOOST
# ============================================================================

def xgboost_performances(hyperparams, X_train, X_test, y_train, y_test, submission=False):
    model = xgb.XGBClassifier(**hyperparams)
    y_tr = y_train.to_numpy() if hasattr(y_train, "to_numpy") else y_train
    model.fit(X_train, y_tr)

    y_pred = model.predict(X_test)
    if submission:
        return y_pred
    return classification_metric(y_test, y_pred)


def linear_svm_performances(hyperparams, X_train, X_test, y_train, y_test, submission=False):
    from sklearn.svm import LinearSVC
    model = LinearSVC(**hyperparams).fit(X_train, y_train)
    y_pred = model.predict(X_test)

    if submission:
        return y_pred
    return classification_metric(y_test, y_pred)
