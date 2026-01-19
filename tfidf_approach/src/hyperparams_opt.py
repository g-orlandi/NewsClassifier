"""
Hyperparameter search configuration and Optuna optimization utilities.

This module defines:

- the full hyperparameter grid specification for all models used in the study
- dynamically adjusted search ranges depending on dataset size
- a unified Optuna-based optimizer (`optuna_hyp_opt`) that returns
  the best hyperparameter set based on the selected scoring metric

No training or evaluation happens here: the optimizer interacts with external
model-performance functions passed as callables.
"""

import optuna
from sklearn.model_selection import StratifiedKFold
import numpy as np

from .config import *
from .preprocessing import *
from .models import train_model

# ============================================================================
#                 MODEL-SPECIFIC HYPERPARAMETER CONFIGURATION
# ============================================================================

def get_models_optuna_config(version):
    """
    Return the hierarchical Optuna search space for all models.

    Returns
    -------
    dict
        A nested dictionary specifying parameter types and ranges
        for each model under 'clf' and 'regr'.
    """
    if version == 0:
        return {
            "logistic_regression": {
                "solver": {"fixed": "saga"},
                "l1_ratio": {"fixed": 0},
                "C": {"range": (0.1, 10.0), "type": "logfloat"},
                "max_iter": {"fixed": 5000},
            },

            "naive_bayes": {
                "alpha": {"range": (0.1, 5.0), "type": "logfloat"}
            },

            "linear_svm": {
                "C": {"range": (0.1, 10.0), "type": "logfloat"},
                "max_iter": {"fixed": 5000},
                "class_weight": {"fixed": {0:1.0, 1:1.0, 2:1.0, 3:2.0, 4:1.0, 5:2.0, 6:1.5}},
            },

            "xgboost": {
                "objective": {"fixed": "multi:softprob"},
                "num_class": {"fixed": 7},
                "eval_metric": {"fixed": "mlogloss"},
                "tree_method": {"fixed": "hist"},
                "n_jobs": {"fixed": -1},

                "learning_rate": {"range": (0.03, 0.2), "type": "logfloat"},
                "n_estimators": {"range": [300, 600, 1000, 1500], "type": "categorical"},
                "max_depth": {"range": (2, 4), "type": "int"},
                "min_child_weight": {"range": (1, 20), "type": "int"},

                "subsample": {"fixed": 0.8},
                "colsample_bytree": {"fixed": 0.6},
                "gamma": {"range": (0.0, 2.0), "type": "float"},
                "reg_alpha": {"range": (1e-8, 1e-1), "type": "logfloat"},
                "reg_lambda": {"range": (0.1, 50.0), "type": "logfloat"},
            }
        }
    if version == 1:

        return {
                "logistic_regression": {
                    "solver": {"fixed": "saga"},
                    "C": {"range": (0.001, 10.0), "type": "float"},
                    "l1_ratio": {"range": [0, 1], "type": "categorical"},
                    "max_iter": {"fixed": 10000},
                },
                "naive_bayes": {
                    "alpha": {"range": (1e-3, 10.0), "type": "logfloat"}
                },
                "xgboost": {
                    "objective": {"fixed": "multi:softprob"},
                    "num_class": {"fixed": 7},
                    "tree_method": {"fixed": "hist"},
                    "n_jobs": {"fixed": -1},

                    "learning_rate": {"range": (0.03, 0.2), "type": "logfloat"},
                    "n_estimators": {
                        "range": [300, 400, 600, 800, 1000, 1200, 1500],
                        "type": "categorical"
                    },
                    "max_depth": {"range": (2, 5), "type": "int"},
                    "min_child_weight": {"range": (2, 6), "type": "int"},

                    "subsample": {"range": (0.75, 0.95), "type": "float"},
                    "colsample_bytree": {"range": (0.4, 0.8), "type": "float"},

                    "gamma": {"range": (0.0, 1), "type": "float"},
                    "reg_alpha": {"range": (0.0, 0.8), "type": "float"},
                    "reg_lambda": {"range": (0.2, 3.0), "type": "float"},
                },
                "linear_svm": {
                    "C": {"range": (0.05, 10.0), "type": "logfloat"},
                    "max_iter": {"fixed": 10000},
                    "class_weight": {"fixed": {0:1.0, 1:1.0, 2:1.0, 3:2.0, 4:1.0, 5:2.0, 6:1.5}},
                }

        }


# ============================================================================
#                           OPTUNA OPTIMIZATION LOGIC
# ============================================================================

def optuna_hyp_opt(model, X, y, version):
    """
    Run Optuna hyperparameter optimization using the model-specific search space.

    Parameters
    ----------
    model : str
        Model identifier (e.g., 'knn', 'ebm', 'symbolic_regression').
    function : callable
        A function with signature f(params, X_train, X_val, y_train, y_val)
        returning a score dictionary.
    X, y : array-like or DataFrame
        Dataset used for optimization.

    Returns
    -------
    dict
        Best hyperparameters found by Optuna.
    """

    def objective(trial):
        # Construct parameter dictionary from config
        params = {}
        param_config = get_models_optuna_config(version)[model]

        for key, cfg in param_config.items():
            if "fixed" in cfg:
                params[key] = cfg["fixed"]

            elif "range" in cfg and "type" in cfg:
                if cfg["type"] == "int":
                    params[key] = trial.suggest_int(key, *cfg["range"])

                elif cfg["type"] == "float":
                    params[key] = trial.suggest_float(key, *cfg["range"])

                elif cfg["type"] == "logfloat":
                    params[key] = trial.suggest_float(
                        key, *cfg["range"], log=True
                    )

                elif cfg["type"] == "categorical":
                    params[key] = trial.suggest_categorical(
                        key, cfg["range"]
                    )
                else:
                    raise ValueError(
                        f"Unsupported parameter type for '{key}': {cfg['type']}"
                    )

            else:
                raise ValueError(
                    f"Invalid parameter configuration: {key} -> {cfg}"
                )

        # Train/validation split

        # cv = StratifiedKFold(n_splits=OPTUNA_KSPLITS, shuffle=True, random_state=SEED)
        cv = StratifiedKFold(n_splits=OPTUNA_KSPLITS, shuffle=True)

        scores = []
        for tr_idx, val_idx in cv.split(X, y):
            Xtr = X.iloc[tr_idx] if hasattr(X, "iloc") else X[tr_idx]
            Xval = X.iloc[val_idx] if hasattr(X, "iloc") else X[val_idx]
            ytr = y.iloc[tr_idx] if hasattr(y, "iloc") else y[tr_idx]
            yval = y.iloc[val_idx] if hasattr(y, "iloc") else y[val_idx]

            preprocess = build_preprocess(model)
            Xtr = preprocess.fit_transform(Xtr)
            Xval = preprocess.transform(Xval)
            print(Xtr.shape[1])

            score_dict = train_model(model, params, Xtr, Xval, ytr, yval)

            scores.append(score_dict["f1-macro"])

        trial.set_user_attr("full_params", params)

        return float(np.mean(scores))

    optuna.logging.set_verbosity(optuna.logging.ERROR)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler()
    )

    if model == 'linear_svm':
        optuna_trials = 30
    elif model == 'xgboost':
        optuna_trials = 30

    study.optimize(objective, n_trials=optuna_trials, show_progress_bar=True)

    return study.best_trial.user_attrs["full_params"]
