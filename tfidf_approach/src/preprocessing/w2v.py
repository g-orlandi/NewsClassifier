import re
import html
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from gensim.models import Word2Vec

def tokenizer(s):
    lemmatizer = WordNetLemmatizer()

    s = s.apply(html.unescape)
    s = s.str.lower()

    HTML_NOISE_WORDS = [
        "http", "https", "www", "com",
        "rss", "feed", "feeds",
        "img", "src", "href",
        "reuters", "border", "said", "new",
        "yahoo", "yimg", "jpg", "jpeg", "png", "gif",
        "width", "height", "align", "alt",
        "photo", "clear", "left", "right",
        "sig",
        "dailynews", "csmonitor", "feedburner",
        "yeartoken",
        "monday", "tuesday", "wednesday", "thursday",
        "friday", "saturday", "sunday",
        "president", "people", "world"
    ]

    data = []

    pat = r"(?u)\b(" + "|".join(map(re.escape, HTML_NOISE_WORDS)) + r")\b"
    s = s.str.replace(pat, " ", regex=True)
    s = s.str.replace(r"\s+", " ", regex=True).str.strip()

    stop_words = set(stopwords.words('english'))

    s = s.str.replace(r"\d+", " ", regex=True)
    s = s.str.replace(r"[^\w\s]", " ", regex=True)

    for article in s:
        tokens = word_tokenize(article)
        filtered_tokens = [
            lemmatizer.lemmatize(word)
            for word in tokens
            if word not in stop_words
        ]
        data.append(filtered_tokens)

    return data

import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer

def _fit_tfidf_idf(docs_tokens):
    # docs_tokens: list[list[str]]
    # usa solo IDF (stabile), TF verrà gestito a livello documento
    vec = TfidfVectorizer(
        analyzer=lambda x: x,
        lowercase=False,
        token_pattern=None,
        min_df=2
    )
    vec.fit(docs_tokens)
    idf = dict(zip(vec.get_feature_names_out(), vec.idf_))
    return idf

def _doc_embed(tokens, wv, vector_size, idf=None, weighting="mean"):
    if not tokens:
        return np.zeros(vector_size, dtype=np.float32), 1.0, 0  # oov_rate=1, n_tokens=0

    vecs = []
    ws = []
    oov = 0

    if weighting == "tfidf" and idf is None:
        raise ValueError("weighting='tfidf' requires idf dict")

    for t in tokens:
        if t in wv:
            vecs.append(wv[t])
            if weighting == "tfidf":
                ws.append(idf.get(t, 1.0))
            else:
                ws.append(1.0)
        else:
            oov += 1

    n = len(tokens)
    if not vecs:
        return np.zeros(vector_size, dtype=np.float32), oov / max(n, 1), n

    V = np.vstack(vecs).astype(np.float32)
    W = np.asarray(ws, dtype=np.float32)
    emb = (V * W[:, None]).sum(axis=0) / (W.sum() + 1e-12)

    # normalizzazione utile per modelli lineari
    norm = np.linalg.norm(emb)
    if norm > 0:
        emb = emb / norm

    return emb.astype(np.float32), oov / max(n, 1), n

def w2v_prep(df, fit_models: bool, vector_size=100, path="w2v.model",
             weighting="tfidf", add_style=True):
    df = df.copy()

    title_tokens = tokenizer(df["title"])
    article_tokens = tokenizer(df["article"])

    all_tokens = title_tokens + article_tokens

    # fit Word2Vec SOLO quando richiesto
    if fit_models:
        model = Word2Vec(
            sentences=all_tokens,
            vector_size=vector_size,
            window=8,
            sg=1,
            min_count=2,
            negative=10,
            sample=1e-3,
            epochs=10,
            workers=5,
            seed=42,
        )
        model.save(path)
    else:
        model = Word2Vec.load(path)

    # fit IDF SOLO sul corpus (tipicamente development)
    idf = None
    if weighting == "tfidf":
        idf = _fit_tfidf_idf(all_tokens)

    # embeddings + feature diagnostiche
    title_emb = np.zeros((len(df), vector_size), dtype=np.float32)
    article_emb = np.zeros((len(df), vector_size), dtype=np.float32)

    title_oov = np.zeros(len(df), dtype=np.float32)
    article_oov = np.zeros(len(df), dtype=np.float32)
    title_len = np.zeros(len(df), dtype=np.float32)
    article_len = np.zeros(len(df), dtype=np.float32)

    for i, (tt, at) in enumerate(zip(title_tokens, article_tokens)):
        e_t, o_t, n_t = _doc_embed(tt, model.wv, vector_size, idf=idf, weighting=weighting)
        e_a, o_a, n_a = _doc_embed(at, model.wv, vector_size, idf=idf, weighting=weighting)
        title_emb[i] = e_t
        article_emb[i] = e_a
        title_oov[i], article_oov[i] = o_t, o_a
        title_len[i], article_len[i] = n_t, n_a

    title_cols = [f"title_w2v_{i}" for i in range(vector_size)]
    article_cols = [f"article_w2v_{i}" for i in range(vector_size)]

    title_df = pd.DataFrame(title_emb, columns=title_cols, index=df.index)
    article_df = pd.DataFrame(article_emb, columns=article_cols, index=df.index)

    df = df.drop(columns=["title", "article"])
    df = pd.concat([df, title_df, article_df], axis=1)

    if add_style:
        df["title_oov_rate"] = title_oov
        df["article_oov_rate"] = article_oov
        df["title_n_tokens"] = title_len
        df["article_n_tokens"] = article_len

    return df
