from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline

from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, Normalizer

def build_preprocess_linear(big):
    source_ohe = OneHotEncoder(handle_unknown="ignore")

    title_vec = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 4),
        min_df=2,
        max_df=0.8,
        norm="l2",
        sublinear_tf=True,
    )

    article_vec = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 3),
        min_df=6,
        max_df=0.9,
        norm="l2",
        sublinear_tf=True,
    )

    article_char_vec = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=5,
        max_df=0.9,
        sublinear_tf=True,
        norm="l2",
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
        "len_article",
        "len_title"
    ]
    if big:

        preprocess = ColumnTransformer(
            transformers=[
                ("title_word",   title_vec,        "title"),
                ("article_word", article_vec,      "article"),
                ("article_char", article_char_vec, "article"),
                ("source",       source_ohe,       ["source"]),
                ("num",          "passthrough",    numeric_cols),
            ],
            remainder="drop",
            sparse_threshold=0.3,
        )
    else:
        preprocess = ColumnTransformer(
            transformers=[
                ("title",   title_vec,   "title"),
                ("article", article_vec, "article"),
                ("source",  source_ohe,  ["source"]),
                ("num",     StandardScaler(), numeric_cols),
            ],
            remainder="drop",
    )

    return preprocess

def build_preprocess_linear_svd(big, k_title=400, k_article=600, k_char=600):
    source_ohe = OneHotEncoder(handle_unknown="ignore")

    title_vec = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 4),
        min_df=2,
        max_df=0.8,
        sublinear_tf=True,
    )

    article_vec = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 3),
        min_df=6,
        max_df=0.9,
        sublinear_tf=True,
    )

    article_char_vec = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=5,
        max_df=0.9,
        sublinear_tf=True,
    )

    # LSA blocks: TF-IDF -> SVD -> Normalizer
    title_lsa = make_pipeline(
        title_vec,
        TruncatedSVD(n_components=k_title, random_state=42),
        Normalizer(copy=False),
    )

    article_lsa = make_pipeline(
        article_vec,
        TruncatedSVD(n_components=k_article, random_state=42),
        Normalizer(copy=False),
    )

    char_lsa = make_pipeline(
        article_char_vec,
        TruncatedSVD(n_components=k_char, random_state=42),
        Normalizer(copy=False),
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
        "len_article",
        "len_title"
    ]

    if big:
        preprocess = ColumnTransformer(
            transformers=[
                ("title_lsa",   title_lsa,  "title"),
                ("article_lsa", article_lsa,"article"),
                ("char_lsa",    char_lsa,   "article"),
                ("source",      source_ohe, ["source"]),
                ("num",         "passthrough", numeric_cols),
            ],
            remainder="drop",
            # qui puoi anche rimuoverlo: dopo SVD è tutto denso comunque
        )
    else:
        preprocess = ColumnTransformer(
            transformers=[
                ("title_lsa",   title_lsa,   "title"),
                ("article_lsa", article_lsa, "article"),
                ("source",      source_ohe,  ["source"]),
                ("num",         StandardScaler(), numeric_cols),
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
        stop_words="english",
        ngram_range=(1, 4),
        min_df=2,
        max_df=0.8,
        norm="l2",
        sublinear_tf=True,
    )

    article_vec = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 3),
        min_df=6,
        max_df=0.9,
        norm="l2",
        sublinear_tf=True,
    )

    article_char_vec = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=10,
        max_df=0.9,
        sublinear_tf=True,
        norm="l2",
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

    article_char_pipe = Pipeline(
        steps=[
            ("tfidf", article_char_vec),
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
            ("article_char", article_char_pipe, "article"),
            ("source",  source_ohe,  ["source"]),
            ("num",     "passthrough", numeric_cols),
        ],
        remainder="drop",
    )

    return preprocess


def build_preprocess_nn():
    source_ohe = OneHotEncoder(handle_unknown="ignore")

    title_vec = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 4),
        min_df=3,
        max_df=0.8,
        norm="l2",
        sublinear_tf=True,
    )

    article_vec = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 3),
        min_df=7,
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

    num_scaler = StandardScaler(with_mean=False)

    preprocess = ColumnTransformer(
        transformers=[
            ("title",   title_vec,   "title"),
            ("article", article_vec, "article"),
            ("source",  source_ohe,  ["source"]),
            ("num",     num_scaler,  numeric_cols),
        ],
        remainder="drop",
    )
    return preprocess


def build_preprocess(model_name, big=False):
    if model_name == "linear_svm":
        return build_preprocess_linear(big)
        # return build_preprocess_linear_svd(big)
    elif model_name == "xgboost":
        return build_preprocess_xgb()
    elif model_name == "nn":
        return build_preprocess_nn()
    else:
        raise ValueError(f"Unknown model: {model_name}")
