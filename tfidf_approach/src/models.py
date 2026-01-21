"""

"""
from sklearn.metrics import (
    precision_recall_fscore_support,
    confusion_matrix,
    precision_recall_fscore_support
)

from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.svm import LinearSVC
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB


# ============================================================================
#                            METRIC COMPUTATION
# ============================================================================

def classification_metrics_full(y_true, y_pred, labels=None):
    """
    Compute a comprehensive set of classification metrics.
    """

    # Aggregated metrics
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )

    # Per class metrics
    precision_c, recall_c, f1_c, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )

    cm = confusion_matrix(y_true, y_pred, labels=labels)

    results = {
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1-macro": f1_macro,
        "per_class": {
            "precision": precision_c,
            "recall": recall_c,
            "f1": f1_c,
            "support": support,
        },
        "confusion_matrix": cm,
    }

    return results

# ============================================================================
#                              MODEL WRAPPER
# ============================================================================

def train_model(model_name, hyperparams, X_train, X_test, y_train, y_test, submission=False):
    match model_name:
        case "logistic_regression":
            model = LogisticRegression(**hyperparams)
            model.fit(X_train, y_train)

        case "naive_bayes":
            if X_train.min() < 0 or X_test.min() < 0:
                raise ValueError("Negative values in X for ComplementNB.")
            model = ComplementNB(**hyperparams)
            model.fit(X_train, y_train)

        case "xgboost":
            w = {0:1.0, 5:1.8, 2:2.1, 1:2.2, 3:2.4, 4:2.7, 6:7.6}
            sw = y_train.map(lambda c: w[int(c)]).to_numpy()
            model = xgb.XGBClassifier(**hyperparams)
            model.fit(X_train, y_train, sample_weight=sw)

        case "linear_svm":
            model = LinearSVC(**hyperparams)
            model.fit(X_train, y_train)

        case "sgd":
            model = SGDClassifier(**hyperparams)
            model.fit(X_train, y_train)

        case "ridge":
            model = RidgeClassifier(**hyperparams)
            model.fit(X_train, y_train)
        
        case _:
            raise RuntimeError(f"{model_name} is not a valid model name.")

    y_pred = model.predict(X_test)

    if submission:
        return y_pred
    else:
        return classification_metrics_full(y_test, y_pred)
