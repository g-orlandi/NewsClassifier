"""
Main functions to run experiments.

This module includes:
- a function to find the best hyperparameters on the development set and evaluate the final model
- a function to quickly evaluate a single model with fixed hyperparameters (development set only)
- a function to generate submission files given a model and fixed hyperparameters
"""

from time import time
import pandas as pd
from sklearn.model_selection import train_test_split

from src.preprocessing import build_preprocess, initial_prep
from src.utils import load_data
from src.config import *
from src.models import train_model
from src.optimization import optuna_hyp_opt


def optimize_and_evaluate(models_name, also_weights=False, svd=False, all_cls=True, also_pipe=False):
    """
    Complete function that:
    - perform an holdout on the DEVELOPMENT dataset (train-valid vs test)
    - hyperparameters tuning on train-valid
    - produce the final performance on a retrained model on the whole train-valid set
    """
    df = load_data(DEVELOPMENT_PATH)
    X = df.drop(columns=['y'])
    y = df['y']
    Xtr_val, X_test, ytr_val, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)

    all_models_results = {}
    all_hyperparams = {}

    if isinstance(models_name, str):
        models_name = [models_name]
    
    for model_name in models_name:
        start = time()
        best_params = optuna_hyp_opt(model_name, Xtr_val, ytr_val, also_weights=also_weights, svd=svd, all_cls=all_cls, also_pipe=also_pipe)

        preprocess = build_preprocess(model_name, svd=best_params["svd_params"], big=best_params["big"], include_title=best_params["include_title"], config=best_params["pipeline_params"])
    
        print(f'Beginning shapes: \nXtr_val: {Xtr_val.shape} | X_test: {X_test.shape} | ytr_val: {ytr_val.shape} | y_test: {y_test.shape}')
        Xtr_val_prep = preprocess.fit_transform(Xtr_val)
        X_test_prep = preprocess.transform(X_test)
        print(f'Prep shapes: \nXtr_val: {Xtr_val_prep.shape} | X_test: {X_test_prep.shape} | ytr_val: {ytr_val.shape} | y_test: {y_test.shape}')

        result, _ = train_model(model_name, best_params["model_params"], Xtr_val_prep, X_test_prep, ytr_val, y_test)
        end = time()
        result['time'] = end - start
        
        all_models_results[model_name] = result
        all_hyperparams[model_name] = best_params
        print(model_name)
        print(result)
        print(best_params["model_params"])

    models_results_df = pd.DataFrame(all_models_results).T

    return models_results_df, all_hyperparams

def performance(model_name, hyperparams, big, include_title):
    """
    Given a single model name and a dict with the params,
    it returns the performance computed on a single holdout split, only on the DEVELOPMENT dataset.
    """
    news_df = load_data(DEVELOPMENT_PATH)
    X = news_df.drop(columns=['y'])
    y = news_df['y']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)

    preprocess = build_preprocess(model_name, big=big, include_title=include_title)
    X_train = preprocess.fit_transform(X_train)
    X_test = preprocess.transform(X_test)
    print(f'X_train shape after prep: {X_train.shape}')

    result, y_pred = train_model(model_name, hyperparams, X_train, X_test, y_train, y_test)
    return result, y_pred, y_test.index

def produce_submissions(model_name, hyperparams, big, include_title, output_filename):
    """
    Given a single model and a parameter grid, it trains the model on the whole DEVELOPMENT dataset
    and produce prediciton for the EVALUATION dataset; it stores these predictions on 'output_filename'. 
    """
    development = load_data(DEVELOPMENT_PATH)
    X_train = development.drop(columns=['y'])
    y_train = development['y']
    X_test = pd.read_csv(EVALUATION_PATH, index_col=0, na_values=NAN_PATTERNS)
    X_test = initial_prep(X_test, dev=False)
    idxs = X_test.index

    preprocess = build_preprocess(model_name, big=big, include_title=include_title)
    X_train = preprocess.fit_transform(X_train)
    X_test = preprocess.transform(X_test)

    y_pred = train_model(model_name, hyperparams, X_train, X_test, y_train, y_test=None, submission=True)

    submission_df = pd.DataFrame({"Id": idxs, "Predicted": y_pred})

    submission_df.to_csv(f'{SUBMISSION_PATH}/{output_filename}', index=False)
    print(f'Prediction saved in {output_filename}')

    return submission_df