import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay

from .config import *
from .preprocessing import initial_prep


def load_data(path):
    df = pd.read_csv(path, index_col=0, na_values='\\N')
    df["y"] = df["label"]
    df.drop(columns="label", inplace=True)

    df = initial_prep(df, dev=True)

    return df

def plot_cm(cm, normalize=True, labels=range(0,7)):
    if normalize:
        cm = cm / cm.sum(axis=1, keepdims=True)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=labels)
    disp.plot()