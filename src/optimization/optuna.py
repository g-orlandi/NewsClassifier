"""
Hyperparameters search and Optuna optimization utilities.
This implementation support both the optimization of models' hyperparameters and also the params of the
preprocessor. 
"""

import optuna
from sklearn.model_selection import StratifiedKFold
import numpy as np

from src.config import *
from src.preprocessing import build_preprocess
from src.models import train_model
from .params_utils import *


def optuna_hyp_opt(model, X, y, svd=False, all_cls=True):
    """
    Run Optuna hyperparameter optimization using the model-specific search space.
    Returns best hyperparameters found by Optuna.
    """
    optuna.logging.set_verbosity(optuna.logging.ERROR)
    cv = StratifiedKFold(n_splits=OPTUNA_KSPLITS, shuffle=True, random_state=SEED)

    def objective(trial):
        
        pipeline_params, include_title, big, svd_params = get_params_preprocessor(trial, svd)
        model_params = get_params_model(model, trial, all_cls)

        scores = []
        for tr_idx, val_idx in cv.split(X, y):
            Xtr = X.iloc[tr_idx] if hasattr(X, "iloc") else X[tr_idx]
            Xval = X.iloc[val_idx] if hasattr(X, "iloc") else X[val_idx]
            ytr = y.iloc[tr_idx] if hasattr(y, "iloc") else y[tr_idx]
            yval = y.iloc[val_idx] if hasattr(y, "iloc") else y[val_idx]

            preprocess = build_preprocess(model, big=big, svd=svd_params, config=pipeline_params, include_title=include_title)
            Xtr = preprocess.fit_transform(Xtr)
            Xval = preprocess.transform(Xval)

            score_dict, _ = train_model(model, model_params, Xtr, Xval, ytr, yval)
            scores.append(score_dict["aggregated"]["f1_macro"])

        full_params = {
            "model_params": model_params,
            "pipeline_params": pipeline_params,
            "svd_params": svd_params,
            "include_title": include_title,
            "big": big
        }

        trial.set_user_attr("full_params", full_params)
        return float(np.mean(scores))

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=OPTUNA_TRIALS, show_progress_bar=True, n_jobs=16)
    return study.best_trial.user_attrs["full_params"]
