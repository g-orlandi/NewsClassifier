import pandas as pd
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder

from .config import *

def load_data():
    return pd.read_csv(DEVELOPMENT_PATH, index_col=0, na_values='\\N')

class Preprocessor:
    
    def __init__(self, df):
        self.df = df.copy()
        self.df["y"] = self.df["label"]
        self.df.drop(columns="label", inplace=True)
        self.is_fit = False
        self.vectorizer = None
        self.ohe = None
        self.top_50 = None

    def title_management(self):
        if self.is_fit:
            self.vectorizer = TfidfVectorizer(stop_words="english", min_df=10)
            tfidf_matrix = self.vectorizer.fit_transform(self.df["title"].fillna(""))
        else:
            tfidf_matrix = self.vectorizer.transform(self.df["title"].fillna(""))

        tfidf_df = pd.DataFrame(
            tfidf_matrix.toarray(),  # <-- qui ancora densifica: ok solo se piccola; meglio evitare del tutto
            columns=self.vectorizer.get_feature_names_out(),
            index=self.df.index,
        )

        pattern = r"^\d+$"
        matches = [m[0] for w in self.vectorizer.get_feature_names_out() if (m := re.findall(pattern, w))]
        tfidf_df.drop(columns=matches, inplace=True, errors="ignore")
        tfidf_df.drop(columns=tfidf_df.columns[tfidf_df.columns.str.len() <= 2], inplace=True)

        self.df = pd.concat([self.df, tfidf_df], axis=1)

    def source_management(self):
        self.df["source"] = self.df["source"].fillna("MISSING")

        if self.is_fit:
            self.top_50 = self.df["source"].value_counts()[:50].index.to_list()
            self.ohe = OneHotEncoder(categories=[self.top_50], handle_unknown="ignore", sparse_output=False)
            X = self.df["source"].where(self.df["source"].isin(self.top_50), np.nan).to_frame()
            X_ohe = self.ohe.fit_transform(X)
        else:
            X = self.df["source"].where(self.df["source"].isin(self.top_50), np.nan).to_frame()
            X_ohe = self.ohe.transform(X)

        X_ohe_df = pd.DataFrame(X_ohe, columns=self.ohe.get_feature_names_out(), index=self.df.index)
        self.df = pd.concat([self.df, X_ohe_df], axis=1)

    def fit(self):
        self.is_fit = True
        self.full_prep()
        return self

    def transform(self, df=None):
        if df is not None:
            self.df = df.copy()
            self.df["y"] = self.df["label"]
            self.df.drop(columns="label", inplace=True)
        self.is_fit = False
        self.full_prep()
        return self.df
