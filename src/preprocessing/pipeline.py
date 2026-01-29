"""
Preprocessing pipeline builder.

This module defines:
- default TF-IDF configurations for text fields
- a function to build a ColumnTransformer for text, categorical and numeric features
- optional SVD compression for high-dimensional TF-IDF representations
"""

from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, Normalizer, MinMaxScaler, StandardScaler

from src.config import SEED, DEFAULT_PIPELINE_CONFIG


def build_preprocess(scheme, big, svd=None, include_title=True, config=DEFAULT_PIPELINE_CONFIG):
    """
    Build the preprocessing pipeline used by the models
    It vectorizes text fields, encodes categorical source and scales numeric features   
    """

    source_ohe = OneHotEncoder(handle_unknown="ignore")

    title_conf = config["title_vec"]
    article_vec_conf = config["article_vec"]
    article_char_conf = config["article_char_vec"]

    title_vec = TfidfVectorizer(**title_conf, stop_words="english")
    article_vec = TfidfVectorizer(**article_vec_conf, stop_words="english")
    article_char_vec = TfidfVectorizer(**article_char_conf, analyzer="char_wb")
    
    # NOTE: cyclical encoding excluded because showed worser results
    numeric_cols = [
        "page_rank",
        "timestamp_missing",
        "year",
        "len_article",
        "len_title",
        "dayofweek",
        "month",
        "hour"
    ]

    if scheme == 'naive_bayes':
        num_scal = MinMaxScaler(clip=True)
    else:
        num_scal = StandardScaler()

    if svd is not None:

        source_ohe = OneHotEncoder(max_categories=svd['source_cat'], handle_unknown="ignore")
    
        title_vec = make_pipeline(
            title_vec,
            TruncatedSVD(n_components=svd['k_title'], random_state=SEED),
            Normalizer(copy=False),
        )

        article_vec = make_pipeline(
            article_vec,
            TruncatedSVD(n_components=svd['k_article'], random_state=SEED),
            Normalizer(copy=False),
        )

        article_char_vec = make_pipeline(
            article_char_vec,
            TruncatedSVD(n_components=svd['k_char'], random_state=SEED),
            Normalizer(copy=False),
        )

    steps = [("article", article_vec, "article"),
            ("source", source_ohe, ["source"]),
            ("num", num_scal, numeric_cols)]

    if include_title:
        steps.append(("title", title_vec, "title"))

    if big:
        steps.append(("article_char", article_char_vec, "article"))

    preprocess = ColumnTransformer(transformers=steps)
    return preprocess
    