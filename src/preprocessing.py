import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer

from .config import *



class Preprocessor():

    def __init__(self):
        self.df = self.load_data()  

    def load_data(self):
        return pd.read_csv(DEVELOPMENT_PATH, index_col=0)

    def timestamp_management(self):
        null_timestamp_idxs = self.df[self.df['timestamp'] == '0000-00-00 00:00:00'].index
        self.df.loc[null_timestamp_idxs, 'timestamp'] = pd.NA
        self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
        self.df['dayofweek'] = self.df['timestamp'].dt.day_of_week
        self.df['month'] = self.df['timestamp'].dt.month
        self.df['hour'] = self.df['timestamp'].dt.hour
        self.df['year'] = self.df['timestamp'].dt.year

    def na_management(self):
        null_article_idx = self.df[self.df['title'].isna()].index
        self.df.drop(null_article_idx, inplace=True)
    
    def title_management(self):
        vectorizer = TfidfVectorizer(stop_words='english', min_df=10)
        tfidf_matrix = vectorizer.fit_transform(self.df['title'])
        import pdb;pdb.set_trace()
        tfidf_df = pd.DataFrame(tfidf_matrix, columns=vectorizer.get_feature_names_out())

        pattern = r'^\d+$'
        matches = [
            m[0]
            for word in vectorizer.get_feature_names_out()
            if (m := re.findall(pattern, word))
        ]
        tfidf_df.drop(columns=matches, inplace=True, errors='ignore')

        tfidf_df.drop(columns=tfidf_df.columns[(tfidf_df.columns.str.len() <= 2)], inplace=True)
        
        return pd.concat([self.df, tfidf_df], axis=1)



    def full_prep(self):
        self.timestamp_management()
        self.na_management()
        return self.title_management()
