"""
Text cleaning utilities for titles and articles.

This module provides:
- a wrapper to clean the dataset text columns and add basic length features
- simple and advanced text cleaning functions
- number and pattern normalization (years, money, percentages, units, etc.)
"""

import re
import html
from bs4 import BeautifulSoup

def text_cleaner_wrapper(df):
    """
    Wrapper that manage title and article + introduce two others features:
    - len_article
    - len_title
    """
    
    df["title"] = df["title"].fillna("")
    df["article"] = df["article"].fillna("")
    df['len_article'] = df['article'].str.len()
    df['len_title'] = df['title'].str.len()

    df['title'] = clean_text(df['title'])
    df['article'] = clean_text(df['article'])

    df.loc[df['article'].str.len() < 5, "article"] = ""
    
    df['title'] = clean_number(df['title'])
    df['article'] = clean_number(df['article'])
    return df

def clean_text(s):
    """ 
    Basic cleaner: unescape HTML, lowercase, remove very common web tokens
    """
    s = s.apply(html.unescape)
    s = s.str.lower()

    HTML_NOISE_WORDS = ["http", "https", "www", "com",]

    pat = r"(?u)\b(" + "|".join(map(re.escape, HTML_NOISE_WORDS)) + r")\b"
    s = s.str.replace(pat, " ", regex=True)
    s = s.str.replace(r"\s+", " ", regex=True).str.strip()
    return s

def clean_number(s):
    """
    Replace number patterns (years, money, %, units, scores, etc.) with tokens
    """
    s = s.str.replace(r"\d+", " ", regex=True)

    year_pattern = r"\b(?:18|19|20)\d{2}\b"

    pct_pattern = (
        r"\b\d+(?:\.\d+)?\s*%\b"
        r"|%\s*\d+(?:\.\d+)?\b"
        r"|\b\d+(?:\.\d+)?\s*percent(?:age)?s?\b"
        r"|\b\d+(?:\.\d+)?\s*[-–]\s*\d+(?:\.\d+)?\s*(?:%|percent(?:age)?s?)\b"
    )

    unit_pattern = r"\b\d+(?:\.\d+)?\s?(?:GB|MB|TB|MP|kg|g|km|m|cm|mm)\b"

    money_pattern = r"(?<!\w)((\d+(?:[.,]\d*)?[$€£]+)|([$€£]+\d+(?:[.,]\d*)?))(?:[MB])?(?!\w)"

    score_pattern = r"\b\d+\s?[-–]\s?\d+\b"
    quarter_pattern = r"\b(?:[1-4]Q|Q[1-4])\b"
    ord_pattern = r"\b\d+(?:st|nd|rd|th)\b"

    num_pattern = r"(?<![A-Za-z])\d{1,3}(?:,\d{3})*(?:\.\d+)?(?![A-Za-z])"
    iso_pattern = r"\b\d{4,5}:\d{4}\b"

    replacements = [
        (iso_pattern,     "ISOTOKEN"),
        (quarter_pattern, "QUARTERTOKEN"),
        (score_pattern,   "SCORETOKEN"),
        (pct_pattern,     "PCTTOKEN"),
        (money_pattern,   "MONEYTOKEN"),
        (unit_pattern,    "UNITTOKEN"),
        (year_pattern,    "YEARTOKEN"),
        (ord_pattern,     "ORDTOKEN"),
        (num_pattern,     ""),
    ]

    for pattern, token in replacements:
        s = s.str.replace(pattern, token, regex=True, flags=re.IGNORECASE)

    return s

# =============================================================================================

""" 
This part below contains functions and utilities that implement aggressive text cleaning.
Experimental results showed that this approach led to worse performance, so this code is not fully refined.
"""
URL_LIKE = re.compile(r'^\s*(https?|ftp)?\s*:\s*//', re.I)

def _strip_html_one(x):
    if x is None:
        return ""
    x = str(x)

    if URL_LIKE.match(x):
        return x

    return BeautifulSoup(x, "html.parser").get_text(separator=" ")

def clean_text_advanced(s):
    s = s.apply(html.unescape)

    s = s.apply(_strip_html_one)
    s = s.str.lower()

    NOISE_WORDS = [
        "http", "https", "www", "com", "img",
        # "rss", "feed", "feeds",
        # "reuters", "border", "said", "new",
        # "yahoo", "yimg", "jpg", "jpeg", "png", "gif",
        # "width", "height", "align", "alt",
        # "photo", "clear", "left", "right",
        # "sig",
        # "dailynews", "csmonitor", "feedburner",
        # "yeartoken", 
        # "monday", "tuesday", "wednesday", "thursday", "friday"
        # "president", "people", "world"
    ]

    pat = r"(?u)\b(" + "|".join(map(re.escape, NOISE_WORDS)) + r")\b"
    s = s.str.replace(pat, " ", regex=True)
    s = s.str.replace(r"\s+", " ", regex=True).str.strip()

    return s




