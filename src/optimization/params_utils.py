"""
Helper functions to build hyperparameters search space for Optuna.
"""

from src.config import *


def parse_ngram(v):
    """
    Convert "a,b" to a tuple (a, b) for n-gram ranges
    """
    if isinstance(v, str) and "," in v:
        a, b = v.split(",")
        return (int(a), int(b))
    return v

def get_generic_params(param_config, trial, prefix=""):
    """
    Build a parameter dict by sampling from a generic Optuna config
    Supports fixed values, numeric ranges, and categorical choices
    """

    params = {}

    for key, cfg in param_config.items():
        opt_name = f"{prefix}{key}"

        # Fixed parameter (no optimization)
        if "fixed" in cfg:
            params[key] = cfg["fixed"]
        # Searchable parameter
        elif "range" in cfg and "type" in cfg:
            match cfg["type"]:
                case "int":
                    params[key] = trial.suggest_int(opt_name, *cfg["range"])
                case "float":
                    params[key] = trial.suggest_float(opt_name, *cfg["range"])
                case "logfloat":
                    params[key] = trial.suggest_float(opt_name, *cfg["range"], log=True)
                case "categorical":
                    params[key] = trial.suggest_categorical(opt_name, cfg["range"])
                case _:
                    raise ValueError(f"Unsupported parameter type for '{key}': {cfg['type']}")
        else:
            raise ValueError(f"Invalid parameter configuration: {key} -> {cfg}")

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
        
    # Manage ngram_range case
    if "ngram_range" in params and isinstance(params["ngram_range"], str):
        a, b = params["ngram_range"].split(",")
        params["ngram_range"] = (int(a), int(b))

    return params

def get_params_model(model, trial, all_cls, also_weights):
    """
    Sample hyperparameters for a specific model.
    Also handles class weights for linear SVM and SGD
    """

    param_config = MODELS_SEARCH[model]

    params = get_generic_params(param_config, trial)
    # if model == 'linear_svm' or model == 'sgd':
    if also_weights:
        if all_cls:
            cw = ALL_CLASS_WEIGHT_CHOICES
        else:
            cw = CLASS_WEIGHT_CHOICES
        cw_id = trial.suggest_int("class_weight_id", 0, len(cw)-1)
        params["class_weight"] = cw[cw_id]
    
    if model != 'naive_bayes':
        params["random_state"] = SEED

    return params

def get_params_preprocessor(trial, svd):
    """
    Get parameters for the preprocessor (tfidf, svd, ...)
    """
    title_vec = get_generic_params(PREP_SEARCH['title_vec'], trial, "title__")
    include_title = title_vec['include']
    del title_vec["include"]

    pipeline_params = {
        "article_vec": get_generic_params(PREP_SEARCH['article_vec'], trial, "article__"),
        "title_vec": title_vec,
        "article_char_vec": get_generic_params(PREP_SEARCH['article_char_vec'], trial, "articlechar__")
    }
    if svd:
        svd_params = get_generic_params(SVD_SEARCH, trial)
    else:
        svd_params = None

    big = get_generic_params(PREP_SEARCH['big'], trial)['flag']

    return pipeline_params, include_title, big, svd_params