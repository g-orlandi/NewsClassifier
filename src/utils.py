"""
Small utility functions used across the project.
"""

import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay

from .config import *
from .preprocessing import initial_prep


def load_data(path):
    """
    Load a CSV file and apply the initial preprocessing steps.
    The target column is renamed to 'y' for consistency and avoiding problem with tfidf.
    Used only for load development dataset.
    """

    df = pd.read_csv(path, index_col=0, na_values=NAN_PATTERNS)
    df["y"] = df["label"]
    df.drop(columns="label", inplace=True)

    df = initial_prep(df, dev=True)

    return df

def plot_cm(cm, normalize=True, labels=range(0, 7)):
    """ 
    Plot a confusion matrix.
    It can be normalized by ROW and supports custom class labels
    """
    if normalize:
        cm = cm / cm.sum(axis=1, keepdims=True)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot()
