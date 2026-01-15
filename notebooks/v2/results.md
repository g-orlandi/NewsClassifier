- Baseline_V2
  - Objective: refine Baseline_V1 by improving robustness and expressiveness of time, title and page_rank preprocessing 
  - Features:
    - Time:
      - Timestamp parsing with invalid placeholders coerced to missing (`NaT`)
      - Missingness indicator:
        - `timestamp_missing` ∈ {0,1}
      - Cyclical encodings:
        - `hour_sin`, `hour_cos` from hour-of-day
        - `month_sin`, `month_cos` from month-of-year
      - Calendar features:
        - `is_weekend` ∈ {0,1}
      - For missing timestamps:
        - all derived time features set to 0
        - `timestamp_missing = 1`
    - Source:
      - Not changed
    - Title:
      - Text preprocessing:
        - Unicode normalization and HTML entity unescaping
        - Regex-based normalization of informative numeric patterns:
          - years, percentages, currency, units, scores, quarters, ordinals, ISO-like tokens
        - Numeric patterns mapped to semantic placeholder tokens
          (e.g. `YEARTOKEN`, `PCTTOKEN`, `MONEYTOKEN`, `UNITTOKEN`, `SCORETOKEN`, `QUARTERTOKEN`, `ORDTOKEN`, `NUMTOKEN`)
      - TF-IDF vectorizer:
        - `ngram_range = (1, 3)`
        - custom token filtering:
          - tokens composed only of digits removed
          - tokens with length ≤ 2 removed
          - curated whitelist of informative short tokens and acronyms (e.g. `us`, `uk`, `eu`, `un`, `f1`, `g7`, `g8`)
    - Article:
      - still not used
    - PageRank:
      - Ordinal compression via discrete mapping:
        - `{2 → 0, 3 → 0, 4 → 1, 5 → 2}`
  - Hyperparameter search:
    - Optuna search space: version 1
    - `OPTUNA_TRIALS = 10`
    - `OPTUNA_KSPLITS = 3`
    - Cross-validation: StratifiedKFold
  - Models tried:
    - Linear SVM
    - XGBoost
  - Best model on development set (F1-macro):
    - XGBoost
  - Result on public evaluation set:
    - **0.645**

  - All best configs:
   {'linear_svm': {'C': 6.663520926844875, 'max_iter': 5000},
 'xgboost': {'objective': 'multi:softprob',
  'num_class': 7,
  'tree_method': 'hist',
  'n_jobs': -1,
  'learning_rate': 0.14300309514812126,
  'n_estimators': 700,
  'max_depth': 5,
  'min_child_weight': 3,
  'subsample': 0.7803692476011997,
  'colsample_bytree': 0.5815469732683555,
  'gamma': 0.05727022330707218,
  'reg_alpha': 0.2336308089104474,
  'reg_lambda': 1.7931095438689504}}

  - All results:
  {'Precision': {'linear_svm': 0.6363895756829131,
  'xgboost': 0.6529303291604106},
 'Recall': {'linear_svm': 0.5940743209348611, 'xgboost': 0.6222883670816721},
 'Fbeta': {'linear_svm': 0.6002013979153824, 'xgboost': 0.6348600760443152},
 'Accuracy': {'linear_svm': 0.6325625, 'xgboost': 0.653125},
 'f1-macro': {'linear_svm': 0.6002013979153824, 'xgboost': 0.6348600760443152},
 'time': {'linear_svm': 54.50900173187256, 'xgboost': 1749.9664335250854}}
