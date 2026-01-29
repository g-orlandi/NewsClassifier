from src.utils import load_data
from src.config import *
from .optuna import optuna_hyp_opt


def best_hyperparams(models_name, version, big):
    """
    Function that search the best hyperparams on the whole DEVELOPMENT dataset,
    without splitting for the final testing.
    """
    news_df = load_data(DEVELOPMENT_PATH)

    X = news_df.drop(columns=['y'])
    y = news_df['y']

    all_hyperparams = {}

    if isinstance(models_name, str):
        models_name = [models_name]

    for model_name in models_name:

        hyperparams = optuna_hyp_opt(model_name, X, y, version, big)
        
        all_hyperparams[model_name] = hyperparams

    return all_hyperparams