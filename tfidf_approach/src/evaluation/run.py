import pandas as pd
from sklearn.model_selection import train_test_split

from src.preprocessing import build_preprocess
from src.utils import load_data
from src.config import *
from src.models import train_model
from src.preprocessing import initial_prep


def performance(model_name, hyperparams, big):
    """
    Given a single model name and a dict with the params,
    it returns the performance computed on a single holdout split, only on the DEVELOPMENT dataset.
    """
    news_df = load_data(DEVELOPMENT_PATH)
    X = news_df.drop(columns=['y'])
    y = news_df['y']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)

    preprocess = build_preprocess(model_name, big=big)
    X_train = preprocess.fit_transform(X_train)
    X_test = preprocess.transform(X_test)
    print(X_train.shape)

    result, y_pred = train_model(model_name, hyperparams, X_train, X_test, y_train, y_test)
    return result, y_pred, y_test.index

def produce_submissions(model_name, hyperparams, output_filename, big):
    """
    Given a single model and a parameter grid, it trains the model on the whole DEVELOPMENT dataset
    and produce prediciton for the EVALUATION dataset; it stores these predictions on 'output_filename'. 
    """
    development = load_data(DEVELOPMENT_PATH)
    X_train = development.drop(columns=['y'])
    y_train = development['y']
    X_test = pd.read_csv(EVALUATION_PATH, index_col=0, na_values='\\N')
    X_test = initial_prep(X_test, dev=False)
    idxs = X_test.index

    preprocess = build_preprocess(model_name, big=big)
    X_train = preprocess.fit_transform(X_train)
    X_test = preprocess.transform(X_test)

    y_pred = train_model(model_name, hyperparams, X_train, X_test, y_train, y_test=None, submission=True)

    submission_df = pd.DataFrame({"Id": idxs, "Predicted": y_pred})

    submission_df.to_csv(SUBMISSION_PATH + '/' + output_filename, index=False)
    print(f'Prediction saved in {output_filename}')

    return submission_df


