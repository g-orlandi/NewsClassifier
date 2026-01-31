"""
Global configuration file for the project.
"""

SEED = 1000

# Paths to datasets
DEVELOPMENT_PATH = "/home/giovanni/Projects/universita/NewsClassifier/data/development.csv"
EVALUATION_PATH = "/home/giovanni/Projects/universita/NewsClassifier/data/evaluation.csv"

# Path to store submission files
SUBMISSION_PATH = "/home/giovanni/Projects/universita/NewsClassifier/submissions"

# Optuna settings
OPTUNA_KSPLITS = 2
OPTUNA_TRIALS = 2

# Missing value pattern used in CSV files
NAN_PATTERNS = ["\\N", "0000-00-00 00:00:00"]

# Default (better) pipeline configuration of preprocessor
DEFAULT_PIPELINE_CONFIG = {
    'title_vec': {
        'ngram_range': (1,2),
        'min_df': 2,
        'max_df': 0.8,
        'norm': 'l2',
        'sublinear_tf': True
    },
    'article_vec': {
        'ngram_range': (1,3),
        'min_df': 6,
        'max_df': 0.9,
        'norm': 'l2',
        'sublinear_tf': True 
    },
    'article_char_vec': {
        'ngram_range': (3,5),
        'min_df': 5,
        'max_df': 0.9,
        'norm': 'l2',
        'sublinear_tf': True
    }
}