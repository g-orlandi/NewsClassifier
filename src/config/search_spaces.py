HYPERPARAMS_V0 = {
    "logistic_regression": {
        "solver": {"fixed": "saga"},
        "l1_ratio": {"fixed": 0},
        "C": {"range": (0.1, 10.0), "type": "logfloat"},
        "max_iter": {"fixed": 5000},
    },
    "naive_bayes": {
        "alpha": {"range": (0.1, 5.0), "type": "logfloat"}
    },
    "linear_svm": {
        "C": {"range": (0.1, 10.0), "type": "logfloat"},
        "max_iter": {"fixed": 5000},
        "class_weight": {"fixed": {0:1.0, 1:1.0, 2:1.0, 3:2.0, 4:1.0, 5:2.0, 6:1.5}},
    },
}

MODELS_SEARCH = {
    "logistic_regression": {
        "solver": {"fixed": "saga"},
        "penalty": {"fixed": "l2"},
        "C": {"range": (0.05, 8.0), "type": "logfloat"},
        "tol": {"range": [1e-2, 1e-3], "type":"categorical"},
        "max_iter": {"fixed": 2000},
        "class_weight": {"fixed": {0:1.0, 5:1.8, 2:2.1, 1:2.2, 3:2.4, 4:2.7, 6:7.6}},
    },
    "naive_bayes": {
        "alpha": {"range": (1e-3, 10.0), "type": "logfloat"}
    },
    "linear_svm": {
        # "C": {"range": (0.01, 0.15), "type": "logfloat"},
        "C": {"range": (0.005, 3), "type": "logfloat"},
        "max_iter": {"fixed": 5000},
        "dual": {"fixed": True},
    },
    "sgd": {
        "loss": {"range": ["hinge", "log_loss"], "type": "categorical"},

        # regolarizzazione (alpha è la più importante)
        "alpha": {"range": (1e-6, 1e-3), "type": "logfloat"},
        "penalty": {"fixed": "l2"},

        # ottimizzazione
        "learning_rate": {"range": ["optimal", "adaptive"], "type": "categorical"},
        "eta0": {"range": (1e-3, 1e-1), "type": "logfloat"},     # usata se adaptive
        "average": {"range": [True, False], "type": "categorical"},

        "max_iter": {"fixed": 3000},
        "tol": {"range": (1e-5, 1e-3), "type": "logfloat"},

        # opzionale ma spesso utile
        "early_stopping": {"range": [True, False], "type": "categorical"},
        "n_iter_no_change": {"range": [5, 10], "type": "categorical"},
    },
    "ridge": {
        "alpha": {"range": (1e-2, 20.0), "type": "logfloat"},
        "class_weight": {"fixed": {0:1.0, 5:1.8, 2:2.1, 1:2.2, 3:2.4, 4:2.7, 6:7.6}},
    }
}


# CLASS_WEIGHT_CHOICES = [
#     {0:0.6,1:1.2,2:1.2,3:2.6,4:1.3,5:3.0,6:2.0},
#     {0:0.4,1:1.2,2:1.3,3:2.8,4:1.4,5:3.4,6:2.2},
#     {0:0.5,1:1.2,2:1.2,3:2.4,4:1.3,5:3.8,6:2.0},
#     {0:0.8,1:1.1,2:1.1,3:2.0,4:1.2,5:2.4,6:1.6},
#     {0:0.6,1:1.3,2:1.3,3:2.5,4:1.4,5:2.8,6:1.8},
#     {0:1.0,1:2.2,2:2.1,3:2.4,4:2.7,5:1.8,6:7.6},
#     {0:1.0,1:2.2,2:2.1,3:2.4,4:2.7,5:1.8,6:4.0},

#     {0:0.5,1:1.2,2:1.2,3:2.6,4:1.3,5:3.2,6:2.5},
#     {0:0.6,1:1.2,2:1.2,3:2.6,4:1.3,5:3.2,6:3.0},
#     {0:0.7,1:1.2,2:1.2,3:2.5,4:1.3,5:3.0,6:2.8},
#     {0:0.6,1:1.2,2:1.2,3:2.8,4:1.3,5:3.4,6:3.0},
#     {0:0.7,1:1.2,2:1.2,3:2.4,4:1.3,5:3.2,6:2.6},

#     {0:0.5,1:1.1,2:1.1,3:3.0,4:1.2,5:3.8,6:2.5},
#     {0:0.6,1:1.1,2:1.1,3:2.9,4:1.2,5:3.6,6:2.8},
#     {0:0.7,1:1.1,2:1.1,3:2.7,4:1.2,5:3.4,6:3.0},

#     {0:0.8,1:1.2,2:1.2,3:2.2,4:1.3,5:2.6,6:2.0},
#     {0:0.9,1:1.2,2:1.2,3:2.3,4:1.3,5:2.8,6:2.2}
# ]

CLASS_WEIGHT_CHOICES = [
    # balanced-ish (baseline inverse-freq around)
    {0: 1.00, 3: 2.00, 5: 1.70},
    {0: 1.00, 3: 2.30, 5: 1.80},
    {0: 1.00, 3: 2.60, 5: 2.00},

    # penalizza 0 (se 0 domina troppo)
    {0: 0.90, 3: 2.30, 5: 1.80},
    {0: 0.80, 3: 2.40, 5: 1.90},
    {0: 0.70, 3: 2.60, 5: 2.10},

    # più focus su 3 (se 3 è quella più sacrificata)
    {0: 0.90, 3: 2.80, 5: 1.80},
    {0: 0.80, 3: 3.00, 5: 1.90},
    {0: 0.70, 3: 3.20, 5: 2.00},

    # più focus su 5 (se 5 viene confusa spesso con 0/3)
    {0: 0.90, 3: 2.20, 5: 2.20},
    {0: 0.80, 3: 2.30, 5: 2.40},
    {0: 0.70, 3: 2.40, 5: 2.60},

    # aggressivo (solo se macro-F1 sulle minoritarie è pessimo)
    {0: 0.60, 3: 3.20, 5: 2.40},
    {0: 0.60, 3: 3.50, 5: 2.60},
    {0: 0.50, 3: 3.80, 5: 2.80},
]


PREP_SEARCH = {
  "title_vec": {
    "include": {"range": [True, False], "type": "categorical"},
    "ngram_range": {"range": ["1,1", "1,2"], "type": "categorical"},
    "min_df": {"range": [2, 3], "type": "categorical"},
    "max_df": {"range": [0.6, 0.8], "type": "categorical"},
    "sublinear_tf": {"fixed": True},
    "norm": {"fixed": "l2"},
  },

  "article_vec": {
    "ngram_range": {"range": ["1,2", "1,3"], "type": "categorical"},
    "min_df": {"range": [2, 3, 6], "type": "categorical"},      # 2 è spesso il boost per 3/5
    "max_df": {"range": [0.8, 0.9], "type": "categorical"},
    "sublinear_tf": {"fixed": True},
    "norm": {"fixed": "l2"},
  },

  "article_char_vec": {
    "ngram_range": {"range": ["3,5", "3,6"], "type": "categorical"},
    "min_df": {"range": [2, 3, 5], "type": "categorical"},
    "max_df": {"range": [0.75, 0.85], "type": "categorical"},
    "sublinear_tf": {"fixed": True},
    "norm": {"fixed": "l2"},
  },
  "big":  {"flag": {"range": [True, False], "type": "categorical"}}
}

SVD_SEARCH = {
  "k_title":   {"range": [50,  100, 200], "type": "categorical"},
  "k_article": {"range": [200, 300, 400], "type": "categorical"},
  "k_char":    {"range": [100, 150, 200, 350], "type": "categorical"},
  "source_cat":{"range": [2, 20, 50], "type": "categorical"},   # 0 = off
}
