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
    "xgboost": {
        "objective": {"fixed": "multi:softprob"},
        "num_class": {"fixed": 7},
        "eval_metric": {"fixed": "mlogloss"},
        "tree_method": {"fixed": "hist"},
        "n_jobs": {"fixed": -1},
        "learning_rate": {"range": (0.03, 0.2), "type": "logfloat"},
        "n_estimators": {"range": [300, 600, 1000, 1500], "type": "categorical"},
        "max_depth": {"range": (2, 4), "type": "int"},
        "min_child_weight": {"range": (1, 20), "type": "int"},
        "subsample": {"fixed": 0.8},
        "colsample_bytree": {"fixed": 0.6},
        "gamma": {"range": (0.0, 2.0), "type": "float"},
        "reg_alpha": {"range": (1e-8, 1e-1), "type": "logfloat"},
        "reg_lambda": {"range": (0.1, 50.0), "type": "logfloat"},
    },
}

HYPERPARAMS_V1 = {
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
    "xgboost": {
        "objective": {"fixed": "multi:softprob"},
        "num_class": {"fixed": 7},
        "tree_method": {"fixed": "hist"},
        "n_jobs": {"fixed": -1},
        "learning_rate": {"range": (0.03, 0.15), "type": "logfloat"},
        "n_estimators": {"range": [400, 800, 1200], "type": "categorical"}, 
        "max_depth": {"range": [2, 3, 4], "type": "categorical"},
        "min_child_weight": {"range": [5, 10, 20], "type": "categorical"},
        "subsample": {"range": [0.8, 0.9], "type": "categorical"},
        "colsample_bytree": {"range": [0.2, 0.35, 0.5], "type": "categorical"},
        "gamma": {"range": [0.0, 0.2, 0.5], "type": "categorical"},
        "reg_alpha": {"range": (1e-4, 1e-1), "type": "logfloat"},
        "reg_lambda": {"range": (0.5, 5.0), "type": "logfloat"},
    },
    "linear_svm": {
        "C": {"range": (0.01, 0.15), "type": "logfloat"},
        "max_iter": {"fixed": 5000},
        "dual": {"fixed": True},
    },
    "sgd": {
        "loss": {
            "range": ["hinge", "log_loss"], "type": "categorical"
        },
        "alpha": {
            "range": (1e-6, 1e-3),
            "type": "logfloat"
        },
        "penalty": {
            "fixed": "l2"
        },
        "class_weight": {"fixed": {0:1.0, 5:1.8, 2:2.1, 1:2.2, 3:2.4, 4:2.7, 6:7.6}},
        "max_iter": {"fixed": 3000}
    },
    "ridge": {
        "alpha": {"range": (1e-2, 20.0), "type": "logfloat"},
        "class_weight": {"fixed": {0:1.0, 5:1.8, 2:2.1, 1:2.2, 3:2.4, 4:2.7, 6:7.6}},
    }
}


CLASS_WEIGHT_CHOICES = [
    {0:0.6,1:1.2,2:1.2,3:2.6,4:1.3,5:3.0,6:2.0},
    {0:0.4,1:1.2,2:1.3,3:2.8,4:1.4,5:3.4,6:2.2},
    {0:0.5,1:1.2,2:1.2,3:2.4,4:1.3,5:3.8,6:2.0},
    {0:0.8,1:1.1,2:1.1,3:2.0,4:1.2,5:2.4,6:1.6},
    {0:0.6,1:1.3,2:1.3,3:2.5,4:1.4,5:2.8,6:1.8},
    {0:1.0,1:2.2,2:2.1,3:2.4,4:2.7,5:1.8,6:7.6},
    {0:1.0,1:2.2,2:2.1,3:2.4,4:2.7,5:1.8,6:4.0},

    {0:0.5,1:1.2,2:1.2,3:2.6,4:1.3,5:3.2,6:2.5},
    {0:0.6,1:1.2,2:1.2,3:2.6,4:1.3,5:3.2,6:3.0},
    {0:0.7,1:1.2,2:1.2,3:2.5,4:1.3,5:3.0,6:2.8},
    {0:0.6,1:1.2,2:1.2,3:2.8,4:1.3,5:3.4,6:3.0},
    {0:0.7,1:1.2,2:1.2,3:2.4,4:1.3,5:3.2,6:2.6},

    {0:0.5,1:1.1,2:1.1,3:3.0,4:1.2,5:3.8,6:2.5},
    {0:0.6,1:1.1,2:1.1,3:2.9,4:1.2,5:3.6,6:2.8},
    {0:0.7,1:1.1,2:1.1,3:2.7,4:1.2,5:3.4,6:3.0},

    {0:0.8,1:1.2,2:1.2,3:2.2,4:1.3,5:2.6,6:2.0},
    {0:0.9,1:1.2,2:1.2,3:2.3,4:1.3,5:2.8,6:2.2}
]