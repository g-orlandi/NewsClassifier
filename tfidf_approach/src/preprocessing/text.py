import re
import html


def text_cleaner_wrapper(df):
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
    s = s.apply(html.unescape)
    s = s.str.lower()

    HTML_NOISE_WORDS = [
        "http", 
        "https", "www", "com",
        # "rss", "feed", "feeds",
        # "img", "src", "href",
        # "reuters", "border", "said", "new",
        # "yahoo", "yimg", "jpg", "jpeg", "png", "gif",
        # "width", "height", "align", "alt",
        # "photo", "clear", "left", "right",
        # "sig",
        # "dailynews", "csmonitor", "feedburner",
        # "yeartoken", 
        # "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
        # "president", "people", "world"
    ]

    pat = r"(?u)\b(" + "|".join(map(re.escape, HTML_NOISE_WORDS)) + r")\b"
    s = s.str.replace(pat, " ", regex=True)
    s = s.str.replace(r"\s+", " ", regex=True).str.strip()
    return s

def clean_number(s):
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