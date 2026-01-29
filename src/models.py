"""
Simple wrapper to train different classification models and get predictions/metrics.
"""

from sklearn.linear_model import LogisticRegression, SGDClassifier, RidgeClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import ComplementNB
import xgboost as xgb

from src.evaluation import classification_metrics_full
from .config import XGB_WEIGHTS


def train_model(model_name, hyperparams, X_train, X_test, y_train, y_test, submission=False):
    """
    Train a model selected by its name, using the given hyperparameters.
    Returns predictions or evaluation metrics based on submission param.
    """

    match model_name:
        case "logistic_regression":
            model = LogisticRegression(**hyperparams)
            model.fit(X_train, y_train)

        case "naive_bayes":
            # IMPORTANT: requires non-negative features
            if X_train.min() < 0 or X_test.min() < 0:
                raise ValueError("Negative values in X for ComplementNB.")
            model = ComplementNB(**hyperparams)
            model.fit(X_train, y_train)

        case "xgboost":
            model = xgb.XGBClassifier(**hyperparams)
            # Manual class weighting for xgboost
            w = {0: 1.0, 2: 1.8, 1: 2.4}
            sw = y_train.map(lambda c: w[int(c)]).to_numpy()
            model.fit(X_train, y_train, sample_weight=sw)

        case "linear_svm":
            # Linear Support Vector Machine
            model = LinearSVC(**hyperparams)
            model.fit(X_train, y_train)

        case "sgd":
            model = SGDClassifier(**hyperparams)
            model.fit(X_train, y_train)

        case "ridge":
            # Ridge classifier (linear model with L2 regularization)
            model = RidgeClassifier(**hyperparams)
            model.fit(X_train, y_train)

        case _:
            # Invalid model name
            raise RuntimeError(f"{model_name} is not a valid model name.")

    y_pred = model.predict(X_test)

    # Return only predictions for submission mode (y_test=None),
    # otherwise return metrics and predictions.
    if submission:
        return y_pred
    else:
        return classification_metrics_full(y_test, y_pred), y_pred
