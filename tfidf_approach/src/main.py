from time import time
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import *
from src.utils import load_data
from src.preprocessing import build_preprocess
from src.models import train_model
from src.optimization import optuna_hyp_opt


def main(models_name, version, big):
    """
    Complete function that:
    - perform an holdout on the DEVELOPMENT dataset (train-valid vs test)
    - hyperparameters tuning on train-valid
    - produce the final performance on a retrained model on the whole train-valid set
    """
    news_df = load_data(DEVELOPMENT_PATH)

    X = news_df.drop(columns=['y'])
    y = news_df['y']

    Xtr_val, X_test, ytr_val, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    
    preprocess = build_preprocess(model_name, big=big)
    Xtr_val_prep = preprocess.fit_transform(Xtr_val)
    X_test_prep = preprocess.transform(X_test)

    print(f'Beginning shapes: \nXtr_val: {Xtr_val.shape} | X_test: {X_test.shape} | ytr_val: {ytr_val.shape} | y_test: {y_test.shape}')


    all_models_results = {}
    all_hyperparams = {}

    if isinstance(models_name, str):
        models_name = [models_name]

    for model_name in models_name:
        

        print(f'Prep shapes: \nXtr_val: {Xtr_val_prep.shape} | X_test: {X_test_prep.shape} | ytr_val: {ytr_val.shape} | y_test: {y_test.shape}')

        start = time()

        hyperparams = optuna_hyp_opt(model_name, Xtr_val, ytr_val, version, big=big)

        result = train_model(model_name, hyperparams, Xtr_val_prep, X_test_prep, ytr_val, y_test)

        end = time()

        result['time'] = end - start
        
        all_models_results[model_name] = result
        all_hyperparams[model_name] = hyperparams

    models_results_df = pd.DataFrame(all_models_results).T

    return models_results_df, all_hyperparams