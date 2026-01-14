from sklearn.model_selection import train_test_split
from scipy import sparse
import pandas as pd
import time

from .config import *
from .preprocessing import *
from .utils import load_data
from .models import train_model
from .hyperparams_opt import optuna_hyp_opt

def evaluate(models_name=['linear_svm', 'xgboost'], version=0):
    news_df = load_data(DEVELOPMENT_PATH)
    prep = Preprocessor()

    X = news_df.drop(columns=['y'])
    y = news_df['y']

    Xtr_val, X_test, ytr_val, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    

    print(f'Beginning shapes: \nXtr_val: {Xtr_val.shape} | X_test: {X_test.shape} | ytr_val: {ytr_val.shape} | y_test: {y_test.shape}')

    Xtr_val_prep, idxs = prep.fit_transform(Xtr_val.copy())
    ytr_val_prep = ytr_val.loc[idxs]

    X_test_prep, idxs = prep.transform(X_test.copy())
    y_test_prep = y_test.loc[idxs]

    print(f'Prep shapes: \nXtr_val: {Xtr_val_prep.shape} | X_test: {X_test_prep.shape} | ytr_val: {ytr_val_prep.shape} | y_test: {y_test_prep.shape}')

    all_models_results = {}
    all_hyperparams = {}

    if isinstance(models_name, str):
        models_name = [models_name]

    for model_name in models_name:
        start = time.time()

        # TO-DO: eventually scale

        hyperparams = optuna_hyp_opt(model_name, Xtr_val, ytr_val, version)

        result = train_model(model_name, hyperparams, Xtr_val_prep, X_test_prep, ytr_val_prep, y_test_prep)

        end = time.time()

        result['time'] = end - start
        
        all_models_results[model_name] = result
        all_hyperparams[model_name] = hyperparams

    models_results_df = pd.DataFrame(all_models_results).T

    return models_results_df, all_hyperparams

def produce_submissions(model_name, hyperparams, output_filename):
    development = load_data(DEVELOPMENT_PATH)
    prep = Preprocessor()

    X_train = development.drop(columns=['y'])
    y_train = development['y']
    X_train, idxs = prep.fit_transform(X_train.copy())
    y_train = y_train.loc[idxs]

    evaluation = pd.read_csv(EVALUATION_PATH, index_col=0, na_values='\\N')

    X_test, idxs = prep.transform(evaluation.copy())


    y_pred = train_model(model_name, hyperparams, X_train, X_test, y_train, y_test=None, submission=True)

    submission_df = pd.DataFrame(
        {
            "Id": idxs,
            "Predicted": y_pred
        },
        index=idxs
    )
    submission_df.to_csv(output_filename, index=False)
    print(f'Prediction saved in {output_filename}')


