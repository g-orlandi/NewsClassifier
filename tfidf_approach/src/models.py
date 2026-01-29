from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.svm import LinearSVC
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB

from src.evaluation import classification_metrics_full

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
            # w = {0:1.0, 5:1.8, 2:2.1, 1:2.2, 3:2.4, 4:2.7, 6:7.6}
            w = {0:1.0, 2:1.8, 1:2.4}
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
        return classification_metrics_full(y_test, y_pred), y_pred
