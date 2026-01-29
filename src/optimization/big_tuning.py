"""
Utilities to search for the best hyperparameters on the whole development dataset.
"""

from src.utils import load_data
from src.config import *
from .optuna import optuna_hyp_opt


def best_hyperparams(models_name, svd):
    """ 
    Search the best hyperparameters using the full development dataset.
    No final train/test split is performed here
    """
    news_df = load_data(DEVELOPMENT_PATH)

    X = news_df.drop(columns=["y"])
    y = news_df["y"]
    
    if isinstance(models_name, str):
        models_name = [models_name]

    all_hyperparams = {}
    for model_name in models_name:
        hyperparams = optuna_hyp_opt(model_name, X, y, svd)
        all_hyperparams[model_name] = hyperparams

    return all_hyperparams
