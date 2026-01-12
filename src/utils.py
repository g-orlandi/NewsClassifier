import pandas as pd

from .config import *


def load_data(path):
    df = pd.read_csv(path, index_col=0, na_values='\\N')
    df["y"] = df["label"]
    df.drop(columns="label", inplace=True)
    return df
