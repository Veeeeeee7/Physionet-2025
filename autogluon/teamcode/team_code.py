#!/usr/bin/env python

# Edit this script to add your team's code. Some functions are *required*, but you can edit most parts of the required functions,
# change or remove non-required functions, and add your own functions.

################################################################################
#
# Optional libraries, functions, and variables. You can change or remove them.
#
################################################################################

import joblib
import numpy as np
import os
import sys

from helper_code import *

from sktime.transformations.panel.rocket import MiniRocketMultivariate
import pandas as pd
from autogluon.tabular import TabularPredictor
from autogluon.core.metrics import make_scorer
from sktime.utils import mlflow_sktime

################################################################################
#
# Required functions. Edit these functions to add your code, but do not change the arguments for the functions.
#
################################################################################

# Train your models. This function is *required*. You should edit this function to add your code, but do *not* change the arguments
# of this function. If you do not train one of the models, then you can return None for the model.

# Train your model.
def train_model(data_folder, model_folder, verbose):
    # Remove existing model directories if they exist
    if os.path.exists(os.path.join(model_folder, 'minirocket')):
        os.system(f'rm -rf {os.path.join(model_folder, "minirocket")}')
    if os.path.exists(os.path.join(model_folder, 'autogluon')):
        os.system(f'rm -rf {os.path.join(model_folder, "autogluon")}')

    # Find the data files.
    if verbose:
        print('Finding the Challenge data...')

    records = find_records(data_folder)
    num_records = len(records)

    if num_records == 0:
        raise FileNotFoundError('No data were provided.')

    # Extract the features and labels from the data.
    if verbose:
        print('Extracting features and labels from the data...')

    features = np.zeros((num_records, 8428), dtype=np.float64)
    labels = np.zeros(num_records, dtype=bool)

    # Instantiate MiniRocket Transformer
    num_kernels = 84 * 100
    random_state = 42
    minirocket = MiniRocketMultivariate(num_kernels=num_kernels, random_state=random_state)

    # Instantiate Autogluon
    autogluon_challenge_scorer = make_scorer(name='challenge_score', score_func=compute_challenge_score, optimum=1, greater_is_better=True, needs_proba=True, needs_threshold=False)
    autogluon = TabularPredictor(problem_type='binary', label='chagas', eval_metric=autogluon_challenge_scorer, path=os.path.join(model_folder, 'autogluon'), verbosity=2)
    time_limit = 54000
    memory = 64
    hyperparameters = {
        'NN_TORCH': {},
        'GBM': [
            {'extra_trees': True, 'ag_args': {'name_suffix': 'XT'}},
            {},
            {
                "learning_rate": 0.03,
                "num_leaves": 128,
                "feature_fraction": 0.9,
                "min_data_in_leaf": 3,
                "ag_args": {
                    "name_suffix": "Large",
                    "priority": 0,
                    "hyperparameter_tune_kwargs": None,
                },
            },
        ],
        'CAT': {},
        'XGB': {},
        # 'FASTAI': {},
        'RF': [
            {'criterion': 'gini', 'ag_args': {'name_suffix': 'Gini', 'problem_types': ['binary', 'multiclass']}},
            {'criterion': 'entropy', 'ag_args': {'name_suffix': 'Entr', 'problem_types': ['binary', 'multiclass']}},
            {'criterion': 'squared_error', 'ag_args': {'name_suffix': 'MSE', 'problem_types': ['regression']}},
        ],
        'XT': [
            {'criterion': 'gini', 'ag_args': {'name_suffix': 'Gini', 'problem_types': ['binary', 'multiclass']}},
            {'criterion': 'entropy', 'ag_args': {'name_suffix': 'Entr', 'problem_types': ['binary', 'multiclass']}},
            {'criterion': 'squared_error', 'ag_args': {'name_suffix': 'MSE', 'problem_types': ['regression']}},
        ],
        'KNN': [
            {'weights': 'uniform', 'ag_args': {'name_suffix': 'Unif'}},
            {'weights': 'distance', 'ag_args': {'name_suffix': 'Dist'}},
        ],
    }

    # Iterate over the records.
    for i in range(num_records):
        if verbose:
            width = len(str(num_records))
            print(f'- {i+1:>{width}}/{num_records}: {records[i]}...')

        record = os.path.join(data_folder, records[i])
        features[i] = extract_features(record, minirocket)
        labels[i] = load_label(record)

    # Train the models.
    if verbose:
        print('Training the model on the data...')
    # Combine features and labels into one DataFrame
    df = pd.DataFrame(features)
    df['chagas'] = labels
    autogluon.fit(train_data=df, fit_strategy='parallel', memory_limit=memory, time_limit=time_limit, presets='best_quality', hyperparameters=hyperparameters)

    # Save the model.
    if verbose:
        print('Saving the model...')
    mlflow_sktime.save_model(sktime_model=minirocket, path=os.path.join(model_folder, 'minirocket'))

    if verbose:
        print('Done.')
        print()

# Load your trained models. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function. If you do not train one of the models, then you can return None for the model.
def load_model(model_folder, verbose):
    model = []
    model.append(mlflow_sktime.load_model(model_uri=os.path.join(model_folder, 'minirocket')))
    model.append(TabularPredictor.load(os.path.join(model_folder, 'autogluon')))
    return model

# Run your trained model. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function.
def run_model(record, model, verbose):
    # Load the model.
    minirocket = model[0]
    autogluon = model[1]

    # Extract the features.
    features = extract_features(record, minirocket)
    features = features.reshape(1, -1)

    df = pd.DataFrame(features)

    # Get the model outputs.
    binary_output = autogluon.predict(df).to_numpy()
    probability_output = autogluon.predict_proba(df).to_numpy()[:, 1]

    return binary_output, probability_output

################################################################################
#
# Optional functions. You can change or remove these functions and/or add new functions.
#
################################################################################

# Extract your features.
def extract_features(record, minirocket):
    header = load_header(record)
    age = get_age(header)
    sex = get_sex(header)

    one_hot_encoding_sex = np.zeros(3, dtype=bool)
    if sex == 'Female':
        one_hot_encoding_sex[0] = 1
    elif sex == 'Male':
        one_hot_encoding_sex[1] = 1
    else:
        one_hot_encoding_sex[2] = 1

    signal, fields = load_signals(record)

    target_length = 4096
    total_padding = target_length - len(signal)
    padding = total_padding // 2
    padded_signal = np.pad(signal, ((padding, total_padding - padding), (0, 0)), 'constant', constant_values=(0, 0))

    transformed_features = minirocket.fit_transform(padded_signal)
    ecg_features = transformed_features.to_numpy().flatten()

    # TO-DO: Update to compute per-lead features. Check lead order and update and use functions for reordering leads as needed.

    signal_mean = np.nanmean(signal, axis=0)
    signal_std = np.nanstd(signal, axis=0)

    features = np.concatenate((np.array([age]), one_hot_encoding_sex, ecg_features, signal_mean, signal_std))

    return np.asarray(features, dtype=np.float32)
