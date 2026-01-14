import pandas as pd
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from scipy.sparse import csr_matrix, hstack
import unicodedata
import html

from src.config import *


class Preprocessor:
    
    def __init__(self):
        self.df = None
        self.is_fit = False
        self.vectorizer = None
        self.ohe = None
        self.top_50 = None

    def timestamp_management(self):
        # 1) Parse timestamp: treat invalid placeholders as missing (NaT)
        ts = self.df["timestamp"].replace("0000-00-00 00:00:00", pd.NA)
        ts = pd.to_datetime(ts, errors="coerce")  # invalid -> NaT
        self.df["timestamp"] = ts

        # 2) Missingness flag (important!)
        self.df["timestamp_missing"] = self.df["timestamp"].isna().astype(int)

        # 3) Create time features ONLY where timestamp is valid
        valid = self.df["timestamp"].notna()

        # is_weekend: keep as 0/1; set to 0 for missing (or leave NaN if you prefer)
        self.df["is_weekend"] = 0
        self.df.loc[valid, "is_weekend"] = (
            self.df.loc[valid, "timestamp"].dt.day_of_week.isin([5, 6]).astype(int)
        )

        # hour sin/cos
        self.df["hour_sin"] = 0.0
        self.df["hour_cos"] = 0.0
        h = self.df.loc[valid, "timestamp"].dt.hour.astype(float)
        self.df.loc[valid, "hour_sin"] = np.sin(2 * np.pi * h / 24)
        self.df.loc[valid, "hour_cos"] = np.cos(2 * np.pi * h / 24)

        # month sin/cos
        self.df["month_sin"] = 0.0
        self.df["month_cos"] = 0.0
        m = self.df.loc[valid, "timestamp"].dt.month.astype(float)
        self.df.loc[valid, "month_sin"] = np.sin(2 * np.pi * m / 12)
        self.df.loc[valid, "month_cos"] = np.cos(2 * np.pi * m / 12)

        # year (numeric): set to -1 for missing (or keep NaN)
        self.df["year"] = 0
        self.df.loc[valid, "year"] = self.df.loc[valid, "timestamp"].dt.year.astype(int)

    def pagerank_manegement(self):
        self.df['page_rank'] = self.df['page_rank'].map({2:0, 3:0, 4:1, 5:2})

    def na_management(self): 
        null_title_idx = self.df[self.df['title'].isna()].index 
        self.df.drop(null_title_idx, inplace=True)

    @staticmethod
    def clean_number(s):
        year_pattern = r"\b(19|20)\d{2}s?\b"
        pct_pattern = (r"(\b\d+(\.\d+)?\s?%)"
                    r"|(%\s?\d+(\.\d+)?\b)"
                    r"|\b\d+(?:\.\d+)?\s?percent(?:age)?s?\b"
                        r"\b\d+(?:\.\d+)?\s*[-–]\s*\d+(?:\.\d+)?\s*(?:%|percent(?:age)?s?)\b"
                        r"|\b\d+(?:\.\d+)?\s*%\b"
                        r"|%\s*\d+(?:\.\d+)?\b"
                        r"|\b\d+(?:\.\d+)?\s*percent(?:age)?s?\b"               
        )
        unit_pattern = r"\b\d+(?:\.\d+)?\s?(?:GB|MB|TB|MP|kg|g|km|m|cm|mm)\b"
        money_pattern = r"((\d+((\.|,)\d*)?[$€£]+)|([$€£]+\d+((\.|,)\d*)?))(M|B)*"
        score_pattern = r"\b\d+\s?[-–]\s?\d+\b"
        quarter_pattern = r"\b(?:[1-4]Q|Q[1-4])\b"
        ord_pattern = r"\b\d+(?:st|nd|rd|th)\b"
        num_pattern = r"(?<![A-Za-z])\b\d{1,3}(?:,\d{3})*(?:\.\d+)?\b(?![A-Za-z])"
        iso_pattern = r"\b\d{4,5}:\d{4}\b"

        replacements = [
            (iso_pattern,     "ISOTOKEN"),       # 13485:2003
            (quarter_pattern, "QUARTERTOKEN"),   # 3Q, Q4
            (score_pattern,   "SCORETOKEN"),     # 3-1
            (pct_pattern,     "PCTTOKEN"),       # 6%, %6, 6 percent, 6.0-6.5 percent
            (money_pattern,   "MONEYTOKEN"),     # €100, $100.5, 100€ (vedi nota sotto)
            (unit_pattern,    "UNITTOKEN"),      # 8GB, 80kg, 100m
            (year_pattern,    "YEARTOKEN"),      # 1990, 1990s
            (ord_pattern,     "ORDTOKEN"),       # 3rd, 10th
            (num_pattern,     "NUMTOKEN"),       # numeri standalone rimanenti
        ]
        for pattern, token in replacements:
            s = s.str.replace(
            pattern,
            token,
            regex=True,
            flags=re.IGNORECASE
            )
        return s
    
    @staticmethod
    def clean_text(s):
        s = s.apply(html.unescape)
        s = s.apply(lambda x: unicodedata.normalize("NFKC", x))
        return s

    def title_management(self):
        self.df["title"] = Preprocessor.clean_text(self.df['title'])
        self.df['title'] = Preprocessor.clean_number(self.df['title'])

        titles = self.df["title"].fillna("")

        if self.is_fit:
            KEEP_2CHAR = {
                "uk","us","eu","un","ny","nj","nh","la","tv","ip","xp","hp","hq",
                "f1","g7","g8","u2","vw","gm","bp","ft","cd"
            }
            keep2 = "|".join(sorted(KEEP_2CHAR))
            token_pattern = rf"(?u)\b(?!\d+\b)(?:[A-Za-z]{{3,}}|(?:{keep2})|[A-Za-z]\w+)\b"

            self.vectorizer = TfidfVectorizer(stop_words="english", min_df=10, token_pattern=token_pattern, ngram_range=(1,3), strip_accents="unicode")
            return  self.vectorizer.fit_transform(titles)
        else:
            return self.vectorizer.transform(titles)

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
        self.pagerank_manegement()
        X_ohe = self.source_management()
        tfidf_matrix = self.title_management()
        output = self.df.copy()
        idxs = output.index
        self.df = None

        output.drop(columns=COLUMNS_TO_DROP, inplace=True, errors='ignore')
        output = csr_matrix(output.to_numpy(dtype=np.float32))
        X = hstack([tfidf_matrix, X_ohe, output], format="csr")

        return X, idxs


    def fit_transform(self, df):
        self.is_fit = True
        return self.full_prep(df)

    def transform(self, df):
        if self.vectorizer is None or self.ohe is None or self.top_50 is None:
            raise RuntimeError("Preprocessor.fit_transform called before fit.")
        
        self.is_fit = False
        return self.full_prep(df)