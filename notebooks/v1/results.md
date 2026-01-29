- Baseline_V1
  - Objective: establish a fast and reliable first baseline before full-article modeling
  - Features:
    - Time:
      - dayofweek, month, hour, year
      - missing timestamps replaced with placeholder "1970-01-01 00:00:00"
    - Source:
      - One-Hot Encoding of the top 50 most frequent sources (including NaN mapped to "MISSING")
      - all other sources mapped to an all-zero vector
    - Title:
      - TF-IDF vectorizer with min_df = 10 and stop_words = "english"
      - tokens composed only of digits removed
      - tokens with length ≤ 2 removed
    - Article:
      - not used
    - PageRank:
      - used as provided (no transformation)
  - Hyperparameter search:
    - Optuna search space: version 0 (light configuration)
    - OPTUNA_TRIALS = 10
    - OPTUNA_KSPLITS = 3 (StratifiedKFold)
  - Models tried:
    - Logistic regression, XGboost, naive bayes classifier, linear_svm
  - Best model on development set (F1-macro):
    - Xgboost
  - Result on public evaluation set:
    - **0.640**

  - All best configs:
  {'logistic_regression': {'solver': 'saga',
    'l1_ratio': 0,
    'C': 7.064102061998609,
    'max_iter': 5000},
  'naive_bayes': {'alpha': 0.10292465492640979},
  'xgboost': {'objective': 'multi:softprob',
    'num_class': 7,
    'eval_metric': 'mlogloss',
    'tree_method': 'hist',
    'n_jobs': -1,
    'learning_rate': 0.1241198606157316,
    'n_estimators': 800,
    'max_depth': 5,
    'min_child_weight': 1,
    'subsample': 0.8,
    'colsample_bytree': 0.6,
    'gamma': 0.0,
    'reg_alpha': 0.0,
    'reg_lambda': 1.0},
  'linear_svm': {'C': 0.4213625248742607, 'max_iter': 5000}}

  - All results:
  {'Precision': {'logistic_regression': 0.5515214321043703,
  'naive_bayes': 0.603212822166865,
  'xgboost': 0.6526365764076042,
  'linear_svm': 0.5955405386459546},
 'Recall': {'logistic_regression': 0.3354984757606628,
  'naive_bayes': 0.6114429384743388,
  'xgboost': 0.621907124791118,
  'linear_svm': 0.5108767180131853},
 'Fbeta': {'logistic_regression': 0.3319578883232342,
  'naive_bayes': 0.5892080577793984,
  'xgboost': 0.6347364017954045,
  'linear_svm': 0.5300040885231924},
 'Accuracy': {'logistic_regression': 0.4563125,
  'naive_bayes': 0.6173125,
  'xgboost': 0.649875,
  'linear_svm': 0.574625},
 'f1-macro': {'logistic_regression': 0.3319578883232342,
  'naive_bayes': 0.5892080577793984,
  'xgboost': 0.6347364017954045,
  'linear_svm': 0.5300040885231924},
 'time': {'logistic_regression': 1041.1600322723389,
  'naive_bayes': 6.735179424285889,
  'xgboost': 898.2960603237152,
  'linear_svm': 26.49807095527649}}