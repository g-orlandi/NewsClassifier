"""
Hyperparameters search and Optuna optimization utilities.
"""

import optuna
from sklearn.model_selection import StratifiedKFold
import numpy as np

from src.config import *
from src.preprocessing import build_preprocess
from src.models import train_model

# ============================================================================
#                 Utility functions for get the params
# ============================================================================

def get_search_space(version):
    """
    Return the hierarchical Optuna search space for all models, but of the specified version.
    See the grid config/search_spaces.py
    """
    if version == 0:
        return HYPERPARAMS_V0
    if version == 1:
        return HYPERPARAMS_V1

def get_params(model, version, trial):

    params = {}
    param_config = get_search_space(version)[model]

    for key, cfg in param_config.items():
        if "fixed" in cfg:
            params[key] = cfg["fixed"]
        elif "range" in cfg and "type" in cfg:
            if cfg["type"] == "int":
                params[key] = trial.suggest_int(key, *cfg["range"])

            elif cfg["type"] == "float":
                params[key] = trial.suggest_float(key, *cfg["range"])

            elif cfg["type"] == "logfloat":
                params[key] = trial.suggest_float(key, *cfg["range"], log=True)

            elif cfg["type"] == "categorical":
                params[key] = trial.suggest_categorical(key, cfg["range"])
            else:
                raise ValueError(f"Unsupported parameter type for '{key}': {cfg['type']}")
        else:
            raise ValueError(f"Invalid parameter configuration: {key} -> {cfg}")

    if model == 'linear_svm':
        cw_id = trial.suggest_int("class_weight_id", 0, len(CLASS_WEIGHT_CHOICES)-1)
        params["class_weight"] = CLASS_WEIGHT_CHOICES[cw_id]

    params["random_state"] = SEED

    return params

# ============================================================================
#                           OPTUNA OPTIMIZATION LOGIC
# ============================================================================

def optuna_hyp_opt(model, X, y, version, big):
    """
    Run Optuna hyperparameter optimization using the model-specific search space.
    Returns best hyperparameters found by Optuna.
    """
    optuna.logging.set_verbosity(optuna.logging.ERROR)
    cv = StratifiedKFold(n_splits=OPTUNA_KSPLITS, shuffle=True, random_state=SEED)

    def objective(trial):
        params = get_params(model, version, trial)

        scores = []
        for tr_idx, val_idx in cv.split(X, y):
            Xtr = X.iloc[tr_idx] if hasattr(X, "iloc") else X[tr_idx]
            Xval = X.iloc[val_idx] if hasattr(X, "iloc") else X[val_idx]
            ytr = y.iloc[tr_idx] if hasattr(y, "iloc") else y[tr_idx]
            yval = y.iloc[val_idx] if hasattr(y, "iloc") else y[val_idx]

            preprocess = build_preprocess(model, big=big)
            Xtr = preprocess.fit_transform(Xtr)
            Xval = preprocess.transform(Xval)

            score_dict = train_model(model, params, Xtr, Xval, ytr, yval)
            scores.append(score_dict["aggregated"]["f1macro"])

        trial.set_user_attr("full_params", params)
        print(params)
        return float(np.mean(scores))

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=OPTUNA_TRIALS, show_progress_bar=True, n_jobs=4)
    return study.best_trial.user_attrs["full_params"]
