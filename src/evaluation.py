from sklearn.model_selection import train_test_split
from scipy import sparse

from .preprocessing import *
from . import models
from .hyperparams_opt import optuna_hyp_opt

def evaluate(columns_to_drop, models_name=['logistic_regression', 'naive_bayes', 'xgboost', 'linear_svm']):
    prep = Preprocessor()
    prep.full_prep()
    news_df = prep.df


    X = news_df.drop(columns=(['y'] + columns_to_drop))
    X = sparse.csr_matrix(X.values)

    y = news_df['y']

    X_train_valid, X_test, y_train_valid, y_test = train_test_split(X, y, test_size=0.2, stratify=y, seed=SEED)


    all_models_results = {}
    all_hyperparams = {}

    if isinstance(models_name, str):
        models_name = [models_name]

    for model_name in models_name:
        function_name = model_name + "_performances"

        function = getattr(models, function_name)

        # TO-DO: eventually scale
        hyperparams = optuna_hyp_opt(model_name, function, X_train_valid, y_train_valid)

        result = function(
            hyperparams, 
            X_train_valid, X_test, y_train_valid, y_test
        )

        all_models_results[model_name] = result
        all_hyperparams[model_name] = hyperparams

    models_results_df = pd.DataFrame(all_models_results)

    return models_results_df, all_hyperparams

