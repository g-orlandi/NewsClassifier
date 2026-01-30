# NewsClassifier

Author: Giovanni Orlandi  
Date: 2026-01-31

---

## Contents of this page

- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Structure](#structure)

---

## Introduction

The goal of this project is to address a news article classification task, as described in the official [assignment](docs/DS_ML_Lab__Project_Assignment_Winter_2026).

The task is approached using a classical NLP pipeline based on TF–IDF representations and linear classifiers, with extensive hyperparameter optimization.

A detailed description of the methodology, experimental setup and results is provided in the final [report](docs/report.pdf).

---

## Installation

1. Create directory for new project:

```bash
mkdir news_classifier/
```

2. Create and activate a virtualenv for the project:

```bash
mkvirtualenv news_clf
workon news_clf
```

or

```bash
mkdir ~/.virtualenvs
python3 -m venv ~/.virtualenvs/news_clf
source ~/.virtualenvs/news_clf/bin/activate
```

3. Clone the repository

```bash
cd news_classifier/
git clone git@gitlab.com:univeristy3/observa.git
```

4. Update the virtualenv:

```bash
pip install -r src/requirements.txt
```

---

## Usage

The main entry points of the project are defined in [run.py](src/evaluation/run.py).

#### optimize_and_evaluate

This function implements the complete experimental workflow:

- split the DEVELOPMENT dataset using a holdout strategy (train–validation vs test)
- perform hyperparameter tuning on the train–validation set
- retrain the best model on the full train–validation data
- report final performance metrics on the held-out test set

Parameters:
- **models_name**: list of model identifiers to evaluate
- **also_weights** (default: False): whether to include class-weight configurations in the search space
- **also_pipe** (default: False): whether to include preprocessing (TF–IDF) hyperparameters in the optimization
- **svd** (default: False): whether to apply dimensionality reduction via SVD

Output:
- evaluation metrics
- best hyperparameters found during optimization

---

#### performance

This function evaluates a single model with a fixed set of hyperparameters.

It performs a single holdout split on the DEVELOPMENT dataset and reports
the corresponding performance.

Parameters:
- **model_name**: identifier of the model to evaluate
- **hyperparams**: dictionary of fixed model hyperparameters
- **prep_params**: TF–IDF preprocessing parameters

Output:
- evaluation metrics
- model predictions
- indices of the test split

---

#### produce_submissions

This function trains a model on the full DEVELOPMENT dataset and generates
predictions for the EVALUATION dataset.

The predictions are stored in the specified output file and can be directly
used for leaderboard submission.

Parameters:
- **model_name**: identifier of the model
- **hyperparams**: dictionary of model hyperparameters
- **prep_params**: TF–IDF preprocessing parameters
- **output_filename**: path to the output file

Output:
- submission dataframe containing predictions for the evaluation set

---

## Structure

.
├── data/                 # datasets used in the task
├── docs/                 # assignment and final report pdfs
├── notebooks/            # Jupyter notebooks
│   ├── archive/          # deprecated / experimental notebooks
│   ├── eda/              # exploratory data analysis
│   ├── further_experiments/  # advanced experiments (e.g. two-stage models, calibration)
│   └── models_test/      # model selection and testing
├── src/                  # core source code
│   ├── config/
│   │   ├── constants.py  # global configuration file for the project.
│   │   └── search_spaces.py # hyperparameter search spaces and shared tuning presets.
│   ├── evaluation/
│   │   ├── calibration.py # utilities for probability calibration and threshold selection (NOTE: not fully refined)
│   │   ├── metrics.py     # utility for compute evaluation metrics for classification models
│   │   └── run.py         # main functions to run experiments
│   ├── optimization/
│   │   ├── big_tuning.py  # utilities to search for the best hyperparameters on the whole development dataset
│   │   ├── optuna.py      # hyperparameters optimizer with Optuna
│   │   └── params_utils.py  # helper functions to build hyperparameters search space for Optuna
│   ├── preprocessing/
│   │   ├── pipeline.py    # preprocessing pipeline builder
│   │   ├── safe.py        # basic preprocessing utilities used before vectorization and model training
│   │   └── text.py        # text cleaning utilities for titles and articles
│   ├── models.py          # wrapper to train different classification models and get predictions/metrics
│   ├── utils.py           # utility functions used across the project
│   └── requirements.txt
└── submissions/          # leaderboard submissions