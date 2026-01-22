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
    Return the hierarchical Optuna search space for all models. See the grid on config.py
    """
    if version == 0:
        return HYPERPARAMS_V0
    if version == 1:
        return HYPERPARAMS_V1

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
            scores.append(score_dict["f1-macro"])

        trial.set_user_attr("full_params", params)
        print(params)
        
        return float(np.mean(scores))

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=OPTUNA_TRIALS, show_progress_bar=True, n_jobs=4)

    return study.best_trial.user_attrs["full_params"]
