import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from src.utils import load_data
from src.config import *
from src.preprocessing import build_preprocess


def choose_threshold(model_name, hyperparams, big):
    news_df = load_data(DEVELOPMENT_PATH)

    X = news_df.drop(columns=["y"])
    y = news_df["y"]

    # holdout interno dal development (questa è la tua validation per t0)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED
    )

    # fit preprocess SOLO su train split (no leakage)
    preprocess = build_preprocess(model_name, big=big)
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


def produce_calibrated_submission(model_name, hyperparams, output_filename, big, t0):
    development = load_data(DEVELOPMENT_PATH)

    X_train = development.drop(columns=['y'])
    y_train = development['y']
    X_test = pd.read_csv(EVALUATION_PATH, index_col=0, na_values='\\N')
    X_test = initial_prep(X_test, dev=False)
    idxs = X_test.index

    preprocess = build_preprocess(model_name, big=big)
    X_train = preprocess.fit_transform(X_train)
    X_test = preprocess.transform(X_test)

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

    submission_df = pd.DataFrame(
        {
            "Id": idxs,
            "Predicted": y_pred
        }
    )
    submission_df.to_csv(SUBMISSION_PATH + '/' + output_filename, index=False)
    print(f'Prediction saved in {output_filename}')

    return submission_df
