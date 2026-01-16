from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline

def build_preprocess_linear():
    source_ohe = OneHotEncoder(handle_unknown="ignore")

    title_vec = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=7,
        max_df=0.8,
        norm="l2",
        sublinear_tf=True,
    )

    article_vec = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=10,
        max_df=0.9,
        norm="l2",
        sublinear_tf=True,
    )

    numeric_cols = [
        "page_rank",
        "timestamp_missing",
        "is_weekend",
        "hour_sin",
        "hour_cos",
        "month_sin",
        "month_cos",
        "year",
    ]

    preprocess = ColumnTransformer(
        transformers=[
            ("title",   title_vec,   "title"),
            ("article", article_vec, "article"),
            ("source",  source_ohe,  ["source"]),
            ("num",     "passthrough", numeric_cols),
        ],
        remainder="drop",
    )

    return preprocess



def build_preprocess_xgb(
    n_components_title=100,
    n_components_article=250,
):
    
    source_ohe = OneHotEncoder(handle_unknown="ignore", max_categories=30)

    title_vec = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=7,
        max_df=0.8,
        sublinear_tf=True,
    )

    article_vec = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=10,
        max_df=0.9,
        sublinear_tf=True,
    )

    title_pipe = Pipeline(
        steps=[
            ("tfidf", title_vec),
            ("svd", TruncatedSVD(n_components=n_components_title)),
        ]
    )

    article_pipe = Pipeline(
        steps=[
            ("tfidf", article_vec),
            ("svd", TruncatedSVD(n_components=n_components_article)),
        ]
    )

    numeric_cols = [
        "page_rank",
        "timestamp_missing",
        "is_weekend",
        "hour_sin",
        "hour_cos",
        "month_sin",
        "month_cos",
        "year",
    ]

    preprocess = ColumnTransformer(
        transformers=[
            ("title",   title_pipe,   "title"),
            ("article", article_pipe, "article"),
            ("source",  source_ohe,  ["source"]),
            ("num",     "passthrough", numeric_cols),
        ],
        remainder="drop",
    )

    return preprocess


def build_preprocess(model_name):
    if model_name == "linear_svm":
        return build_preprocess_linear()
    elif model_name == "xgboost":
        return build_preprocess_xgb()
    else:
        raise ValueError(f"Unknown model: {model_name}")
