import re
import html
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from gensim.models import Word2Vec


def w2v_prep(df, fit_models: bool, vector_size=100, path="w2v.model"):
    df = df.copy()
    vector_size = 100
    title_tokens = tokenizer(df["title"])
    article_tokens = tokenizer(df["article"])

    all_tokens = title_tokens + article_tokens
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

    title_emb = np.vstack([document_embedding(doc, model) for doc in title_tokens]).astype(np.float32)
    article_emb = np.vstack([document_embedding(doc, model) for doc in article_tokens]).astype(np.float32)

    title_cols = [f"title_w2v_{i}" for i in range(vector_size)]
    article_cols = [f"article_w2v_{i}" for i in range(vector_size)]

    title_df = pd.DataFrame(title_emb, columns=title_cols, index=df.index)
    article_df = pd.DataFrame(article_emb, columns=article_cols, index=df.index)

    df = df.drop(columns=["title", "article"])
    df = pd.concat([df, title_df, article_df], axis=1)

    return df


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

def document_embedding(tokens, model):
    vectors = []
    for word in tokens:
        if word in model.wv:
            vectors.append(model.wv[word])

    if len(vectors) == 0:
        # documento senza parole nel vocabolario
        return np.zeros(model.vector_size)

    return np.mean(vectors, axis=0)