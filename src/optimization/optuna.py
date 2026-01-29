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
def parse_ngram(v):
    if isinstance(v, str) and "," in v:
        a, b = v.split(",")
        return (int(a), int(b))
    return v


def get_search_space(version):
    """
    Return the hierarchical Optuna search space for all models, but of the specified version.
    See the grid config/search_spaces.py
    """
    if version == 0:
        return HYPERPARAMS_V0
    if version == 1:
        return HYPERPARAMS_V1
    if version == 2:
        return PREP_SEARCH
    if version == 3:
        return SVD_SEARCH

def get_generic_params(param_config, trial, prefix=""):
    params = {}
    for key, cfg in param_config.items():
        opt_name = f"{prefix}{key}"
        if "fixed" in cfg:
            params[key] = cfg["fixed"]
        elif "range" in cfg and "type" in cfg:
            if cfg["type"] == "int":
                params[key] = trial.suggest_int(opt_name, *cfg["range"])

            elif cfg["type"] == "float":
                params[key] = trial.suggest_float(opt_name, *cfg["range"])

            elif cfg["type"] == "logfloat":
                params[key] = trial.suggest_float(opt_name, *cfg["range"], log=True)

            elif cfg["type"] == "categorical":
                params[key] = trial.suggest_categorical(opt_name, cfg["range"])
            else:
                raise ValueError(f"Unsupported parameter type for '{key}': {cfg['type']}")
        else:
            raise ValueError(f"Invalid parameter configuration: {key} -> {cfg}")
        
    # post-process
    if "ngram_range" in params and isinstance(params["ngram_range"], str):
        a, b = params["ngram_range"].split(",")
        params["ngram_range"] = (int(a), int(b))

    return params

def get_params_model(model, version, trial):

    param_config = get_search_space(version)[model]

    params = get_generic_params(param_config, trial)
    if model == 'linear_svm' or model == 'sgd':
        cw_id = trial.suggest_int("class_weight_id", 0, len(CLASS_WEIGHT_CHOICES)-1)
        params["class_weight"] = CLASS_WEIGHT_CHOICES[cw_id]

    params["random_state"] = SEED

    return params

# ============================================================================
#                           OPTUNA OPTIMIZATION LOGIC
# ============================================================================

def optuna_hyp_opt(model, X, y, version):
    """
    Run Optuna hyperparameter optimization using the model-specific search space.
    Returns best hyperparameters found by Optuna.
    """
    optuna.logging.set_verbosity(optuna.logging.ERROR)
    cv = StratifiedKFold(n_splits=OPTUNA_KSPLITS, shuffle=True, random_state=SEED)

    def objective(trial):

        model_params = get_params_model(model, version, trial)
        title_vec = get_generic_params(get_search_space(2)['title_vec'], trial, "title__")
        include_title = title_vec['include']
        del title_vec["include"]

        pipeline_params = {
            "article_vec": get_generic_params(get_search_space(2)['article_vec'], trial, "article__"),
            "title_vec": title_vec,
            "article_char_vec": get_generic_params(get_search_space(2)['article_char_vec'], trial, "articlechar__")
        }
        # svd_params = get_generic_params(get_search_space(3), trial)
        big = get_generic_params(get_search_space(2)['big'], trial)['flag']

        scores = []
        for tr_idx, val_idx in cv.split(X, y):
            Xtr = X.iloc[tr_idx] if hasattr(X, "iloc") else X[tr_idx]
            Xval = X.iloc[val_idx] if hasattr(X, "iloc") else X[val_idx]
            ytr = y.iloc[tr_idx] if hasattr(y, "iloc") else y[tr_idx]
            yval = y.iloc[val_idx] if hasattr(y, "iloc") else y[val_idx]

            preprocess = build_preprocess(model, big=big, svd=None, config=pipeline_params, include_title=include_title)
            Xtr = preprocess.fit_transform(Xtr)
            Xval = preprocess.transform(Xval)

            score_dict, _ = train_model(model, model_params, Xtr, Xval, ytr, yval)
            scores.append(score_dict["aggregated"]["f1_macro"])

        full_params = {
            "model_params": model_params,
            "pipeline_params": pipeline_params,
            # "svd_params": svd_params,
            "include_title": include_title,
            "big": big
        }

        trial.set_user_attr("full_params", full_params)
        return float(np.mean(scores))

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=OPTUNA_TRIALS, show_progress_bar=True, n_jobs=16)
    return study.best_trial.user_attrs["full_params"]
