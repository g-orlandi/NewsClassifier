import pandas as pd
import numpy as np

from .text import *


def initial_prep(df, dev=True):
    df.drop(columns=['page_rank', 'timestamp', 'source'])
    # Text cleaning
    df["source"] = df["source"].fillna("MISSING")
    df["title"] = df["title"].fillna("")
    df["article"] = df["article"].fillna("")
    if dev:
        df = remove_duplicates(df)

    df = text_cleaner_wrapper(df)

    # Timestamp formatting
    ts = df["timestamp"].replace("0000-00-00 00:00:00", pd.NA)
    ts = pd.to_datetime(ts, errors="coerce")  # invalid -> NaT
    df["timestamp"] = ts

    df = timestamp_features(df)
    df = map_pagerank(df)
    return df

def remove_duplicates(df):
    # DUPLICATES
    # If all cols match and also the target => keep only one row
    df.drop_duplicates(inplace=True)

    # If all cols match but NOT the target => drop all rows
    df.drop_duplicates(subset=['source', 'title', 'article', 'page_rank', 'timestamp'], keep=False, inplace=True)

    # If title, article, source, page_rank match but NOT timestamp => keep only one row with y=majority_voting and timestamp the first/not null
    for a, g in df[df.duplicated(subset=['source', 'title', 'article', 'page_rank'], keep=False)].sort_values(['title', 'timestamp']).groupby(['source', 'title', 'article', 'page_rank'], dropna=False):
        
        y = g['y'].value_counts().index[0]
        timestamp = g[g['y']==y]['timestamp'].min()
        df.loc[g.index[0], 'y'] = y
        df.loc[g.index[0], 'timestamp'] = timestamp
        df.drop(index=g.index[1:], inplace=True)

    # If title, article, source, timestamp match but NOT page_rank => keep only one row with y=majority_voting and page_rank the mode value
    for a, g in df[df.duplicated(subset=['source', 'title', 'article', 'timestamp'], keep=False)].groupby(['source', 'title', 'article', 'timestamp'], dropna=False):
        y = g['y'].value_counts().index[0]
        page_rank = g['page_rank'].mode()[0]
        df.loc[g.index[0], 'y'] = y
        df.loc[g.index[0], 'page_rank'] = page_rank
        df.drop(index=g.index[1:], inplace=True)

    # If title, article, page_rank, timestamp match but NOT source => keep only one row with y=majority_voting and source the mode value
    for a, g in df[df.duplicated(subset=['timestamp', 'title', 'article', 'page_rank'], keep=False)].groupby(['timestamp', 'title', 'article', 'page_rank'], dropna=False):
        y = g['y'].value_counts().index[0]
        source = g[g['y']==y]['source'].mode()[0]
        df.loc[g.index[0], 'y'] = y
        df.loc[g.index[0], 'source'] = source
        df.drop(index=g.index[1:], inplace=True)
    
    return df

def timestamp_features(df):
    # 2) Missingness flag (important!)
    df["timestamp_missing"] = df["timestamp"].isna().astype(int)

    # 3) Create time features ONLY where timestamp is valid
    valid = df["timestamp"].notna()

    # is_weekend: keep as 0/1; set to 0 for missing (or leave NaN if you prefer)
    df["is_sunday"] = 0
    df.loc[valid, "is_sunday"] = (
        df.loc[valid, "timestamp"].dt.day_of_week.isin([6]).astype(int)
    )
    df["dayofweek"] = 0
    df.loc[valid, "dayofweek"] = df.loc[valid, "timestamp"].dt.day_of_week + 1

    # hour sin/cos
    df["hour_sin"] = 0.0
    df["hour_cos"] = 0.0
    df["hour"] = 0
    h = df.loc[valid, "timestamp"].dt.hour.astype(float)
    df.loc[valid, "hour_sin"] = np.sin(2 * np.pi * h / 24)
    df.loc[valid, "hour_cos"] = np.cos(2 * np.pi * h / 24)
    df.loc[valid, "hour"] = h + 1


    # month sin/cos
    df["month_sin"] = 0.0
    df["month_cos"] = 0.0
    df["month"] = 0
    m = df.loc[valid, "timestamp"].dt.month.astype(float)
    df.loc[valid, "month_sin"] = np.sin(2 * np.pi * m / 12)
    df.loc[valid, "month_cos"] = np.cos(2 * np.pi * m / 12)
    df.loc[valid, "month"] = m

    df["year"] = 0
    df.loc[valid, "year"] = df.loc[valid, "timestamp"].dt.year.astype(int)
    return df

def map_pagerank(df):
    df['page_rank'] = df['page_rank'].map({2:1, 3:1, 4:2, 5:3})
    df['page_rank'] = df['page_rank'].fillna(0)

    return df


