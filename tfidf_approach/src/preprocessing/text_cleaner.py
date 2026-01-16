import re
import html

def clean_text(s):
    s = s.apply(html.unescape)
    s = s.str.lower()
    return s


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

def text_cleaner_wrapper(df):
    df["source"] = df["source"].fillna("MISSING")
    df["title"] = df["title"].fillna("")
    df["article"] = df["article"].fillna("")

    df['title'] = clean_text(df['title'])
    df['article'] = clean_text(df['article'])
    df.loc[df['article'].str.len() < 5, "article"] = ""
    
    df['title'] = clean_number(df['title'])
    df['article'] = clean_number(df['article'])
    

    return df