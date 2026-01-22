from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, Normalizer, MinMaxScaler, StandardScaler

from src.config import SEED

def _build_preprocess(big, svd=False, nb=False):
    source_ohe = OneHotEncoder(handle_unknown="ignore")

    title_vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=2,
        max_df=0.8, norm="l2", sublinear_tf=True)

    article_vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 3), min_df=6, 
        max_df=0.9, norm="l2", sublinear_tf=True)
    
    article_char_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=5, 
        max_df=0.9, sublinear_tf=True, norm="l2")
    
    numeric_cols = [
        "page_rank",
        "timestamp_missing",
        "is_weekend",
        "hour_sin",
        "hour_cos",
        "month_sin",
        "month_cos",
        "year",
        "len_article",
        "len_title"
    ]

    if nb:
        num_scal = MinMaxScaler(clip=True)
    else:
        num_scal = StandardScaler()

    if svd:
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

    if big:

    
        preprocess = ColumnTransformer(
            transformers=[
                ("title_word",   title_vec,        "title"),
                ("article_word", article_vec,      "article"),
                ("article_char", article_char_vec, "article"),
                ("source",       source_ohe,       ["source"]),
                ("num",          num_scal,    numeric_cols),
            ])
    else:
        preprocess = ColumnTransformer(
            transformers=[
                ("title",   title_vec,   "title"),
                ("article", article_vec, "article"),
                ("source",  source_ohe,  ["source"]),
                ("num",     num_scal, numeric_cols),
            ])

    return preprocess

def build_preprocess(model_name, big=False):
    if model_name == 'naive_bayes':
        return _build_preprocess(big, nb=True)
    else:
        return _build_preprocess(big)