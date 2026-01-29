"""
Utilities for probability calibration and threshold selection.

This file contains:
- a function to choose an optimal threshold for class 0 using a validation split
- a function to generate calibrated predictions for submission

NOTE: This is one of the approaches tested but not used in the final solution; therefore, it may not be fully refined.
"""

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from src.utils import load_data
from src.config import *
from src.preprocessing import build_preprocess, initial_prep


def choose_threshold(model_name, hyperparams, big):
    """
    Function used to choose the threshold for class 0 (the most problematic); LinearSVC model fixed because
    it's the one that achieved better results.
    """
    news_df = load_data(DEVELOPMENT_PATH)
    X = news_df.drop(columns=["y"])
    y = news_df["y"]

    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)

    preprocess = build_preprocess(model_name, big=big)
    X_tr = preprocess.fit_transform(X_tr)
    X_val = preprocess.transform(X_val)

    base = LinearSVC(**hyperparams)
    # Calibrate the classifier to obtain probabilities
    calib = CalibratedClassifierCV(estimator=base, method="sigmoid", cv=3)
    calib.fit(X_tr, y_tr)
    proba = calib.predict_proba(X_val)

    # Search for the best threshold for class 0
    t_grid = np.arange(0.1, 0.9, 0.01)
    best_t, best_f1 = None, -1.0

    for t0 in t_grid:
        # Get all the probability of class 0 < t0 and put them to -inf;
        # in this way, when we select the argmax it will be surely not 0 in the case is below t0.
        proba[proba[:, 0] < t0, 0] = -np.inf
        pred = np.argmax(proba, axis=1)

        f1 = f1_score(y_val, pred, average="macro")
        if f1 > best_f1:
            best_f1 = f1
            best_t = t0

    return best_t, best_f1

def produce_calibrated_submissions(model_name, hyperparams, output_filename, big, t0):
    """
    Produce submission for calibrated models, with a given fixed threshold of class 0 prob.
    It is the calibrated version of produce_submissions in src/evaluation/run.py
    """
    development = load_data(DEVELOPMENT_PATH)
    X_train = development.drop(columns=["y"])
    y_train = development["y"]

    X_test = pd.read_csv(EVALUATION_PATH, index_col=0, na_values=NAN_PATTERNS)
    X_test = initial_prep(X_test, dev=False)
    idxs = X_test.index

    preprocess = build_preprocess(model_name, big=big)
    X_train = preprocess.fit_transform(X_train)
    X_test = preprocess.transform(X_test)

    # Train and calibrate the classifier
    base = LinearSVC(**hyperparams)
    calib = CalibratedClassifierCV(estimator=base, method="sigmoid", cv=3)
    calib.fit(X_train, y_train)

    proba = calib.predict_proba(X_test)

    proba[proba[:, 0] < t0, 0] = -np.inf
    y_pred = np.argmax(proba, axis=1)

    submission_df = pd.DataFrame({"Id": idxs, "Predicted": y_pred})
    
    submission_df.to_csv(f'{SUBMISSION_PATH}/{output_filename}', index=False)
    print(f"Prediction saved in {output_filename}")

    return submission_df
