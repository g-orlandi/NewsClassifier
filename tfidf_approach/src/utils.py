import pandas as pd

from .config import *
from .preprocessing import initial_prep

def load_data(path, is_w2v):
    df = pd.read_csv(path, index_col=0, na_values='\\N')
    df["y"] = df["label"]
    df.drop(columns="label", inplace=True)

    df = initial_prep(df, is_w2v=is_w2v, dev=True)

    return df