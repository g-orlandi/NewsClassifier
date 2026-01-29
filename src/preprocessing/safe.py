"""
Basic preprocessing utilities used before vectorization and model training.

This module includes:
- initial preprocessing of the raw dataframe (text cleaning, timestamp parsing, simple feature creation)
- duplicate removal rules (development set only)
- timestamp-based feature engineering
- page-rank mapping to a smaller set of categories
"""

import pandas as pd
import numpy as np

from .text import *


def initial_prep(df, dev=True):
    """
    Apply the first preprocessing steps used in the whole project
    It cleans text, parses timestamps, creates time features, and maps page_rank
    """

    # Text cleaning
    df["source"] = df["source"].fillna("MISSING")
    df["title"] = df["title"].fillna("")
    df["article"] = df["article"].fillna("")
    
    ts = pd.to_datetime(df['timestamp'], errors="coerce") 
    df["timestamp"] = ts
    
    if dev:
        df = remove_duplicates(df)

    df = text_cleaner_wrapper(df)
    df = timestamp_features(df)
    df = map_pagerank(df)
    return df

def remove_duplicates(df):
    """ 
    Remove duplicated rows using a set of rules to handle conflicting labels.
    NOTE: This could be made more efficient, but I prefer to keep it clear and easy to understand.
    """

    # If all cols match and also the target => keep only one row
    df.drop_duplicates(inplace=True)

    # If all cols match but NOT the target => drop all rows
    df.drop_duplicates(subset=['source', 'title', 'article', 'page_rank', 'timestamp'], keep=False, inplace=True)

    # If everything matches except timestamp, keep one row with majority label
    # and keep the earliest timestamp among rows with that label
    for a, g in (df[df.duplicated(subset=['source', 'title', 'article', 'page_rank'], keep=False)]
                 .sort_values(['title', 'timestamp']).
                 groupby(['source', 'title', 'article', 'page_rank'], dropna=False)):
        y = g['y'].value_counts().index[0]
        timestamp = g[g['y']==y]['timestamp'].min()
        df.loc[g.index[0], 'y'] = y
        df.loc[g.index[0], 'timestamp'] = timestamp
        df.drop(index=g.index[1:], inplace=True)

    # If everything matches except page_rank, keep one row with majority label
    # and set page_rank to the most frequent value
    for a, g in (df[df.duplicated(subset=['source', 'title', 'article', 'timestamp'], keep=False)]
                 .groupby(['source', 'title', 'article', 'timestamp'], dropna=False)):
        y = g['y'].value_counts().index[0]
        page_rank = g['page_rank'].mode()[0]
        df.loc[g.index[0], 'y'] = y
        df.loc[g.index[0], 'page_rank'] = page_rank
        df.drop(index=g.index[1:], inplace=True)

    # If everything matches except source, keep one row with majority label
    # and set source to the most frequent value among rows with that label
    for a, g in df[df.duplicated(subset=['timestamp', 'title', 'article', 'page_rank'], keep=False)].groupby(['timestamp', 'title', 'article', 'page_rank'], dropna=False):
        y = g['y'].value_counts().index[0]
        source = g[g['y']==y]['source'].mode()[0]
        df.loc[g.index[0], 'y'] = y
        df.loc[g.index[0], 'source'] = source
        df.drop(index=g.index[1:], inplace=True)
    
    return df

def timestamp_features(df):
    """ 
    Create simple features from the timestamp column
    Missing timestamps are handled with a dedicated missingness flag and 0 in all the features.
    """
    df["timestamp_missing"] = df["timestamp"].isna().astype(int)

    valid = df["timestamp"].notna()
    
    # is_weekend: keep as 0/1
    df["is_weekend"] = 0
    df.loc[valid, "is_weekend"] = (df.loc[valid, "timestamp"].dt.day_of_week.isin([5, 6]).astype(int))

    # dayofweek: 1 to 7
    df["dayofweek"] = 0
    df.loc[valid, "dayofweek"] = df.loc[valid, "timestamp"].dt.day_of_week + 1

    # hour: both the raw value betwenn 1-24 and the cyclical encoding with sin and cos
    df["hour_sin"] = 0.0
    df["hour_cos"] = 0.0
    df["hour"] = 0
    h = df.loc[valid, "timestamp"].dt.hour.astype(float)
    df.loc[valid, "hour_sin"] = np.sin(2 * np.pi * h / 24)
    df.loc[valid, "hour_cos"] = np.cos(2 * np.pi * h / 24)
    df.loc[valid, "hour"] = h + 1

    # month: the raw value between 1-12 and the cyclical encoding with sin and cos
    df["month_sin"] = 0.0
    df["month_cos"] = 0.0
    df["month"] = 0
    m = df.loc[valid, "timestamp"].dt.month.astype(float)
    df.loc[valid, "month_sin"] = np.sin(2 * np.pi * m / 12)
    df.loc[valid, "month_cos"] = np.cos(2 * np.pi * m / 12)
    df.loc[valid, "month"] = m

    # Year
    df["year"] = 0
    df.loc[valid, "year"] = df.loc[valid, "timestamp"].dt.year.astype(int)

    return df

def map_pagerank(df):
    """ 
    Map page_rank values to a smaller set of discrete categories
    """

    df['page_rank'] = df['page_rank'].map({2:1, 3:1, 4:2, 5:3})
    df['page_rank'] = df['page_rank'].fillna(0)

    return df


