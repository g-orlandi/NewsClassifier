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
OPTUNA_KSPLITS = 3
OPTUNA_TRIALS = 100

# Missing value pattern used in CSV files
NAN_PATTERN = "\\N"