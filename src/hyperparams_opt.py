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
from .preprocessing import Preprocessor
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
                "max_iter": {"fixed": 5000}
            },

            "xgboost": {
                "objective": {"fixed": "multi:softprob"},
                "num_class": {"fixed": 7},
                "eval_metric": {"fixed": "mlogloss"},
                "tree_method": {"fixed": "hist"},
                "n_jobs": {"fixed": -1},

                # pochi parametri davvero influenti
                "learning_rate": {"range": (0.05, 0.2), "type": "logfloat"},
                "n_estimators": {"range": [300, 500, 800], "type": "categorical"},
                "max_depth": {"range": (3, 5), "type": "int"},
                "min_child_weight": {"range": (1, 5), "type": "int"},

                # fissi per baseline
                "subsample": {"fixed": 0.8},
                "colsample_bytree": {"fixed": 0.6},
                "gamma": {"fixed": 0.0},
                "reg_alpha": {"fixed": 0.0},
                "reg_lambda": {"fixed": 1.0},
            }
        }
    if version == 1:

        return {
                "logistic_regression": {
                    "solver": {"fixed": "saga"},
                    "C": {"range": (0.01, 10.0), "type": "logfloat"},
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

                    "learning_rate": {"range": (0.03, 0.15), "type": "logfloat"},
                    "n_estimators": {"range": [300, 350, 400, 450, 500, 600, 700, 800, 900, 1000, 1100, 1200], "type": "categorical"},

                    "max_depth": {"range": (3, 6), "type": "int"},
                    "min_child_weight": {"range": (1, 8), "type": "int"},

                    "subsample": {"range": (0.7, 1.0), "type": "float"},
                    "colsample_bytree": {"range": (0.3, 0.9), "type": "float"},

                    "gamma": {"range": (0.0, 0.7), "type": "float"},
                    "reg_alpha": {"range": (0.0, 1.5), "type": "float"},
                    "reg_lambda": {"range": (0.8, 4.0), "type": "float"},
                },
                "linear_svm": {
                    "C": {"range": (0.05, 10.0), "type": "logfloat"},
                    "max_iter": {"fixed": 5000}
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

        cv = StratifiedKFold(n_splits=OPTUNA_KSPLITS, shuffle=True, random_state=True)

        scores = []
        for tr_idx, val_idx in cv.split(X, y):
            Xtr = X.iloc[tr_idx] if hasattr(X, "iloc") else X[tr_idx]
            Xval = X.iloc[val_idx] if hasattr(X, "iloc") else X[val_idx]
            ytr = y.iloc[tr_idx] if hasattr(y, "iloc") else y[tr_idx]
            yval = y.iloc[val_idx] if hasattr(y, "iloc") else y[val_idx]


            prep = Preprocessor()

            Xtr, idxs = prep.fit_transform(Xtr.copy())
            ytr = ytr.loc[idxs]

            Xval, idxs = prep.transform(Xval.copy())
            yval = yval.loc[idxs]

            score_dict = train_model(model, params, Xtr, Xval, ytr, yval)

            scores.append(score_dict["f1-macro"])

        trial.set_user_attr("full_params", params)

        return float(np.mean(scores))

    optuna.logging.set_verbosity(optuna.logging.ERROR)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler()
    )

    study.optimize(objective, n_trials=OPTUNA_TRIALS, show_progress_bar=True)

    return study.best_trial.user_attrs["full_params"]
