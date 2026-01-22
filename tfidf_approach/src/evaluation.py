from sklearn.model_selection import train_test_split
from scipy import sparse
import pandas as pd
import time

from .config import *
from .preprocessing import *
from .utils import load_data
from .models import train_model
from .hyperparams_opt import optuna_hyp_opt


def best_hyperparams(models_name, version, is_w2v, big):
    news_df = load_data(DEVELOPMENT_PATH, is_w2v=is_w2v)

    X = news_df.drop(columns=['y'])
    y = news_df['y']

    all_hyperparams = {}

    if isinstance(models_name, str):
        models_name = [models_name]

    for model_name in models_name:

        hyperparams = optuna_hyp_opt(model_name, X, y, version, big, is_w2v)
        
        all_hyperparams[model_name] = hyperparams

    return all_hyperparams


def evaluate(models_name, version, is_w2v, big):
    news_df = load_data(DEVELOPMENT_PATH, is_w2v=is_w2v)

    X = news_df.drop(columns=['y'])
    y = news_df['y']

    Xtr_val, X_test, ytr_val, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    

    print(f'Beginning shapes: \nXtr_val: {Xtr_val.shape} | X_test: {X_test.shape} | ytr_val: {ytr_val.shape} | y_test: {y_test.shape}')


    all_models_results = {}
    all_hyperparams = {}

    if isinstance(models_name, str):
        models_name = [models_name]

    for model_name in models_name:
        
        preprocess = build_preprocess(model_name, is_w2v=is_w2v, big=big)
        Xtr_val_prep = preprocess.fit_transform(Xtr_val)
        X_test_prep = preprocess.transform(X_test)

        print(f'Prep shapes: \nXtr_val: {Xtr_val_prep.shape} | X_test: {X_test_prep.shape} | ytr_val: {ytr_val.shape} | y_test: {y_test.shape}')

        start = time.time()

        hyperparams = optuna_hyp_opt(model_name, Xtr_val, ytr_val, version, big=big, is_w2v=is_w2v)

        result = train_model(model_name, hyperparams, Xtr_val_prep, X_test_prep, ytr_val, y_test)

        end = time.time()

        result['time'] = end - start
        
        all_models_results[model_name] = result
        all_hyperparams[model_name] = hyperparams

    models_results_df = pd.DataFrame(all_models_results).T

    return models_results_df, all_hyperparams


def produce_submissions(model_name, hyperparams, output_filename, big, is_w2v, calibrated=False, t0=None):
    development = load_data(DEVELOPMENT_PATH, is_w2v=is_w2v)

    X_train = development.drop(columns=['y'])
    y_train = development['y']
    X_test = pd.read_csv(EVALUATION_PATH, index_col=0, na_values='\\N')
    X_test = initial_prep(X_test, is_w2v=is_w2v, dev=False)
    idxs = X_test.index

    preprocess = build_preprocess(model_name, big=big, is_w2v=is_w2v)
    X_train = preprocess.fit_transform(X_train)
    X_test = preprocess.transform(X_test)

    if calibrated:
        if t0 is None:
            raise ValueError("If calibrate=True you must pass t0 (threshold for class 0).")

        base = LinearSVC(**hyperparams)
        calib = CalibratedClassifierCV(
            estimator=base,          
            method="sigmoid",
            cv=3
        )
        calib.fit(X_train, y_train)
        proba = calib.predict_proba(X_test)
        pred = np.argmax(proba, axis=1)

        # se P(0) < t0, vieta la predizione 0 e scegli la migliore non-0
        mask = proba[:, 0] < t0
        pred[mask] = np.argmax(proba[mask, 1:], axis=1) + 1
        y_pred = pred
    else:
        y_pred = train_model(model_name, hyperparams, X_train, X_test, y_train, y_test=None, submission=True)

    submission_df = pd.DataFrame(
        {
            "Id": idxs,
            "Predicted": y_pred
        }
    )
    submission_df.to_csv(SUBMISSION_PATH + '/' + output_filename, index=False)
    print(f'Prediction saved in {output_filename}')

    return submission_df


def performance(model_name, hyperparams, is_w2v, big=False):

    news_df = load_data(DEVELOPMENT_PATH, is_w2v=is_w2v)

    preprocess = build_preprocess(model_name, is_w2v=is_w2v, big=big)

    X = news_df.drop(columns=['y'])
    y = news_df['y']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)

    X_train = preprocess.fit_transform(X_train)
    X_test = preprocess.transform(X_test)

    print(X_train.shape)

    result = train_model(model_name, hyperparams, X_train, X_test, y_train, y_test)

    return result


from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score


def choose_threshold(model_name, hyperparams, is_w2v, big):
    news_df = load_data(DEVELOPMENT_PATH, is_w2v=is_w2v)

    X = news_df.drop(columns=["y"])
    y = news_df["y"]

    # holdout interno dal development (questa è la tua validation per t0)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED
    )

    # fit preprocess SOLO su train split (no leakage)
    preprocess = build_preprocess(model_name, is_w2v=is_w2v, big=big)
    X_tr = preprocess.fit_transform(X_tr)
    X_val = preprocess.transform(X_val)

    # base estimator NON fit
    base = LinearSVC(**hyperparams)

    # calibrazione: rifitta internamente il base su fold di X_tr
    calib = CalibratedClassifierCV(
        estimator=base,          # se hai sklearn vecchio: base_estimator=base
        method="sigmoid",
        cv=3
    )
    calib.fit(X_tr, y_tr)

    # probabilità sulla validation holdout
    proba = calib.predict_proba(X_val)  # shape (n, 7)

    # ricerca soglia anti-0
    t_grid = np.linspace(0.30, 0.85, 56)  # step ~0.01
    best_t, best_f1 = None, -1.0

    for t0 in t_grid:
        pred = np.argmax(proba, axis=1)

        # se P(0) < t0, vieta la predizione 0 e scegli la migliore non-0
        mask = proba[:, 0] < t0
        pred[mask] = np.argmax(proba[mask, 1:], axis=1) + 1

        f1 = f1_score(y_val, pred, average="macro")
        if f1 > best_f1:
            best_f1 = f1
            best_t = float(t0)

    return best_t, best_f1