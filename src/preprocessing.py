import pandas as pd
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from scipy.sparse import csr_matrix, hstack

from .config import *


class Preprocessor:
    
    def __init__(self):
        self.df = None
        self.is_fit = False
        self.vectorizer = None
        self.ohe = None
        self.top_50 = None
        self.title_cols_to_keep_idxs = []


    def timestamp_management(self): 
        null_timestamp_idxs = self.df[self.df['timestamp'] == '0000-00-00 00:00:00'].index 
        self.df.loc[null_timestamp_idxs, 'timestamp'] = '1970-01-01 00:00:00' 
        self.df['timestamp'] = pd.to_datetime(self.df['timestamp']) 
        self.df['dayofweek'] = self.df['timestamp'].dt.day_of_week 
        self.df['month'] = self.df['timestamp'].dt.month 
        self.df['hour'] = self.df['timestamp'].dt.hour 
        self.df['year'] = self.df['timestamp'].dt.year 

    def na_management(self): 
        null_title_idx = self.df[self.df['title'].isna()].index 
        self.df.drop(null_title_idx, inplace=True)
        
    def title_management(self):
        if self.is_fit:
            self.vectorizer = TfidfVectorizer(stop_words="english", min_df=10)
            tfidf_matrix = self.vectorizer.fit_transform(self.df["title"].fillna(""))
            pattern = r"^\d+$"
            feat = self.vectorizer.get_feature_names_out()

            matches = [m[0] for w in feat if (m := re.findall(pattern, w))]
            title_cols_to_drop = [c for c in feat if len(c) <= 2] + matches    
            self.title_cols_to_keep_idxs = [i for i,c in enumerate(feat) if c not in title_cols_to_drop]

        else:
            tfidf_matrix = self.vectorizer.transform(self.df["title"].fillna(""))

        tfidf_matrix = tfidf_matrix[:, self.title_cols_to_keep_idxs]
        return tfidf_matrix

    def source_management(self):
        df = self.df["source"].fillna("MISSING")

        if self.is_fit:
            self.top_50 = df.value_counts()[:50].index.to_list()
            self.ohe = OneHotEncoder(categories=[self.top_50], handle_unknown="ignore", sparse_output=True)
            X = df.where(df.isin(self.top_50), np.nan).to_frame()
            X_ohe = self.ohe.fit_transform(X)
        else:
            X = df.where(df.isin(self.top_50), np.nan).to_frame()
            X_ohe = self.ohe.transform(X)
        
        return X_ohe

    def full_prep(self,df):
        self.df = df
        self.na_management()
        self.timestamp_management()
        X_ohe = self.source_management()
        tfidf_matrix = self.title_management()
        output = self.df.copy()
        idxs = output.index
        self.df = None


        output.drop(columns=COLUMNS_TO_DROP, inplace=True, errors='ignore')
        output = csr_matrix(output.to_numpy(dtype=np.float32))
        X = hstack([tfidf_matrix, X_ohe, output], format="csr")
        return X, idxs


    def fit(self, df):
        self.is_fit = True
        return self.full_prep(df)

    def fit_transform(self, df):
        if self.vectorizer is None or self.ohe is None or self.top_50 is None:
            raise RuntimeError("Preprocessor.fit_transform called before fit.")
        
        self.is_fit = False
        return self.full_prep(df)