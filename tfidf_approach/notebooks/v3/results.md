# Baseline V3 — Final TF-IDF Refinement

## Objective

Refine **Baseline V2** through:
- more expressive feature usage (full source encoding, article inclusion),
- differentiated preprocessing pipelines for linear and tree-based models,
- explicit handling of duplicated observations,
- extended hyperparameter search.

This version represents the **final and most complete TF-IDF-based configuration** explored in this project.

---

## Feature Set

### Time Features

- Robust timestamp parsing:
  - invalid placeholders coerced to missing (`NaT`)
- Explicit missingness indicator:
  - `timestamp_missing` ∈ {0,1}
- Cyclical encodings:
  - `hour_sin`, `hour_cos` (hour-of-day)
  - `month_sin`, `month_cos` (month-of-year)
- Calendar feature:
  - `is_weekend` ∈ {0,1}
- Handling of missing timestamps:
  - all derived time features set to 0
  - `timestamp_missing = 1`

---

### Source
- **Full One-Hot Encoding** of all observed sources
- Unknown sources handled safely at inference time

---

### Title

#### Text Preprocessing

- Unicode normalization
- HTML entity unescaping
- Regex-based normalization of informative numeric patterns:
  - years, percentages, currency amounts, units, scores, quarters, ordinals, ISO-like tokens
- Numeric patterns mapped to semantic placeholder tokens:
  - `YEARTOKEN`, `PCTTOKEN`, `MONEYTOKEN`, `UNITTOKEN`,
    `SCORETOKEN`, `QUARTERTOKEN`, `ORDTOKEN`, `NUMTOKEN`

#### Vectorization

- TF-IDF vectorizer with:
  - `ngram_range = (1, 2)`
  - `min_df = 7`
  - `max_df = 0.8`
  - `sublinear_tf = True`

---

### Article

- **Included for the first time**
- Lightweight preprocessing only:
  - HTML unescape
  - lowercasing
  - numeric normalization
- Articles with length < 5 characters removed

This choice intentionally avoids aggressive cleaning to preserve semantic content.

---

### Model-Specific Preprocessing

- **Two distinct preprocessing pipelines**:
  - one for **Linear SVM**
  - one for **XGBoost**
- Separate vectorizers for:
  - title
  - article
- For XGBoost specifically:
  - **Truncated SVD** applied to TF-IDF representations
  - dimensionality reduction used to control computational cost and noise

Implementation details are fully contained in the corresponding `.py` files.

---

### Data Quality Handling

- Dedicated preprocessing and analysis of **duplicated observations**
- Conflicting or redundant samples handled explicitly prior to training

---

## Hyperparameter Search

- Optimization framework: **Optuna**
- Search space: version 1
- Trials:
  - `OPTUNA_TRIALS = 20` for XGBoost
  - `OPTUNA_TRIALS = 10` for Linear SVM
- Cross-validation:
  - `StratifiedKFold`
  - `OPTUNA_KSPLITS = 3`

---

## Models Evaluated

- Linear SVM
- XGBoost

---

## Results

- Best model on development set (macro-F1):
  - *[to be filled / comparable to V2]*

- Public evaluation score:
  - *[to be filled / no significant improvement over V2]*

Despite the substantially increased modeling and preprocessing complexity, performance gains over **Baseline V2** were marginal and, in some configurations, negative.

---

## Outcome and Final Assessment

Across Baseline V1, V2, and V3, the following pattern consistently emerged:

- Incremental preprocessing and feature engineering yield **diminishing returns**
- More aggressive cleaning and normalization can **degrade performance**
- Different model families converge to similar results
- Confusion matrices exhibit **systematic, structured errors**, not random noise

These observations indicate a **clear performance plateau** for TF-IDF-based representations in this task.

The remaining errors are attributed to **semantic overlap between classes**, which cannot be resolved through additional preprocessing or classical bag-of-words modeling.

**Baseline V3 therefore marks the closure of the TF-IDF approach**, motivating a transition toward **embedding-based representations**, explored in the subsequent `embedding_approach/`.

