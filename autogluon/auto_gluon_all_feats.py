import argparse
import os
import sys
import datetime

import numpy as np
import pandas as pd

import shutil
import glob
import wfdb

from sklearn.model_selection import KFold
from sktime.transformations.panel.rocket import MiniRocketMultivariate
from sktime.utils import mlflow_sktime 

from autogluon.tabular import TabularPredictor
from autogluon.core.metrics import make_scorer

from helper_code import compute_accuracy, compute_auc, compute_challenge_score, compute_f_measure, is_nan, find_records


def get_parser():
    description = "MiniRocket with RandomForest Classifier and Cross Validation"
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-d', '--data_folder', type=str, help="Path to the data file")
    parser.add_argument('-o', '--output_folder', type=str, help="Path to the output folder")
    parser.add_argument('-f', '--features_file', type=str, help="Path to the demographic features file")
    return parser

def run():
    records = [os.path.splitext(record)[0] for record in glob.glob(os.path.join(data_folder, '*.hea'))]
    if len(records) > 200:
        records = np.random.choice(records, 200, replace=False)

    log(f'Found {len(records)} records')

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    scores = []

    autogluon_challenge_scorer = make_scorer(name='challenge_score', score_func=compute_challenge_score, optimum=1, greater_is_better=True, needs_proba=True, needs_threshold=False)

    for i, (train_index, test_index) in enumerate(kf.split(records)):
        log(f'Fold {i + 1}')
        train_records = [records[i] for i in train_index]
        test_records = [records[i] for i in test_index]
            
        minirocket_path = os.path.join(output_folder, f'models/model_minirocket_{start_time}')
        autogluon_path = os.path.join(output_folder, f'models/model_autogluon_{start_time}')

        if os.path.exists(autogluon_path):
            shutil.rmtree(autogluon_path)
        if os.path.exists(minirocket_path):
            shutil.rmtree(minirocket_path)

        os.makedirs(minirocket_path, exist_ok=True)
        os.makedirs(autogluon_path, exist_ok=True)

        minirocket = MiniRocketMultivariate(num_kernels=num_kernels, random_state=random_state)
        autogluon = TabularPredictor(label='Chagas label', eval_metric=autogluon_challenge_scorer, path=autogluon_path)

        train_model(train_records, minirocket, autogluon)

        challenge_score, auroc, auprc, accuracy, f_measure = test_model(test_records, minirocket, autogluon)

        scores.append([challenge_score, auroc, auprc, accuracy, f_measure])

    mlflow_sktime.save_model(sktime_model=minirocket, path=minirocket_path)

    scores = np.array(scores)
    mean_scores = np.mean(scores, axis=0)
    log(f'Mean Challenge Score: {mean_scores[0]}')
    log(f'Mean AUROC: {mean_scores[1]}')
    log(f'Mean AUPRC: {mean_scores[2]}')
    log(f'Mean Accuracy: {mean_scores[3]}')
    log(f'Mean F-Measure: {mean_scores[4]}')

def train_model(train_records, minirocket, autogluon):
    extracted_features = []
    log('Extracting train features via MiniRocket')
    for batch_start in range(0, len(train_records), batch_size):
        log(f'Batch {batch_start // batch_size + 1} (records {batch_start + 1} to {min(batch_start + batch_size, len(train_records))})')
        batch_train_records = train_records[batch_start:min(batch_start + batch_size, len(train_records))]
        extracted_features.append(extract_features(batch_train_records, minirocket))

    combined_features = pd.concat(extracted_features, axis=0)

    log('Training Autogluon model')
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

    autogluon.fit(train_data=combined_features, fit_strategy='parallel', memory=memory, presets='high_quality' hyperparameters=hyperparameters)

def test_model(test_records, minirocket, autogluon):
    binary_outputs = []
    probability_outputs = []
    labels = []
    for batch_start in range(0, len(test_records), batch_size):
        log(f'Batch {batch_start // batch_size + 1} (records {batch_start} to {min(batch_start + batch_size, len(test_records)) - 1})')
        batch_test_records = test_records[batch_start:min(batch_start + batch_size, len(test_records))]

        log('Extracting test features via MiniRocket')
        batch_extracted_features = extract_features(batch_test_records, minirocket)

        log('Predicting with Autogluon model')
        batch_binary_outputs = autogluon.predict(batch_extracted_features).to_numpy()
        batch_probability_outputs = autogluon.predict_proba(batch_extracted_features).to_numpy()[:, 1]
        batch_labels = batch_extracted_features['Chagas label'].to_numpy()

        binary_outputs.extend(batch_binary_outputs)
        probability_outputs.extend(batch_probability_outputs)
        labels.extend(batch_labels)

    challenge_score = compute_challenge_score(labels, probability_outputs)
    auroc, auprc = compute_auc(labels, probability_outputs)
    accuracy = compute_accuracy(labels, binary_outputs)
    f_measure = compute_f_measure(labels, binary_outputs)

    log(f'Best Model: {autogluon.model_best}')
    log(f'Challenge Score: {challenge_score}')
    log(f'AUROC: {auroc}')
    log(f'AUPRC: {auprc}')
    log(f'Accuracy: {accuracy}')
    log(f'F-Measure: {f_measure}')

    return challenge_score, auroc, auprc, accuracy, f_measure

def extract_features(records, minirocket):
    header_files = records
    signal_files = [file.replace('.hea', '') for file in header_files]

    extracted_ecg_features = extract_ecg_features(signal_files, minirocket)
    extracted_demographic_features = extract_demographic_features(header_files)
    
    extracted_features = pd.concat([extracted_ecg_features, extracted_demographic_features], axis=1)

    return extracted_features

def extract_ecg_features(signal_files, minirocket):
    signals = [wfdb.rdsamp(signal)[0] for signal in signal_files]
    reshaped_signals = reshape_signals(signals)
    padded_signals = pad_signals(reshaped_signals, (4096, 12))
    preprocess_signals = pd.DataFrame(padded_signals)
    
    extracted_ecg_features = pd.DataFrame(minirocket.fit_transform(preprocess_signals))

    return extracted_ecg_features

def extract_demographic_features(header_files):
    demographic_features = pd.DataFrame()
    for header_file in header_files:
        header = wfdb.rdheader(header_file)
        labels = []
        values = []
        for comment in header.comments:
            label = comment.split(": ")[0]
            value = comment.split(": ")[1]
            labels.append(label)
            values.append(value)

        demographic_features = pd.concat([demographic_features, pd.DataFrame([values], columns=labels)], ignore_index=True)

    return demographic_features

def reshape_signals(signals):
    reshaped_signals = np.empty((len(signals), signals[0].shape[1]), dtype=object)

    for i in range(len(signals)):
        for j in range(signals[i].shape[1]):
            reshaped_signals[i, j] = pd.Series(signals[i][:, j]) # reshaping from (# of records, num_samples, num_leads) to (# of records, num_leads) where each entry is a pd.Series of length num_samples

    return reshaped_signals
    
def pad_signals(signals, target_shape):
    padded_signals = np.empty(signals.shape, dtype=object)
    for i in range(signals.shape[0]):
        for j in range(signals.shape[1]):
            signal = signals[i, j]
            if signal.shape[0] < target_shape[0]:
                pad_total = target_shape[0] - signal.shape[0]
                left_pad = pad_total // 2
                right_pad = pad_total - left_pad
                padded_signal = np.pad(signal, (left_pad, right_pad), mode='constant', constant_values=0)
                padded_signals[i, j] = pd.Series(padded_signal)
            else:
                padded_signals[i, j] = pd.Series(signal[:target_shape[0]])
    return padded_signals




def log(message):
    with open(f'{os.path.join(output_folder, log_file_name)}', 'a') as f:
        f.write(f'{message}\n')
    print(message)

start_time = datetime.datetime.now().strftime("%Y-%-m-%d_%H:%M:%S")
log_file_name = f'logs/log_{start_time}'
n_splits = 3
num_kernels = 84 * 100
n_estimators = 12
max_leaf_nodes = 34
random_state = 42
batch_size = 64
data_folder = None
output_folder = None
features_file = None
model_folder = None
memory = 10


if __name__ == "__main__":
    # parse arguments
    parser = get_parser()
    args = parser.parse_args()
    data_folder = args.data_folder
    output_folder = args.output_folder
    features_file = args.features_file

    # make folders
    os.makedirs(os.path.join(output_folder, 'models'), exist_ok=True)
    os.makedirs(os.path.join(output_folder, 'models/old/'), exist_ok=True)
    os.makedirs(os.path.join(output_folder, 'logs'), exist_ok=True)
    os.makedirs(os.path.join(output_folder, 'logs/old/'), exist_ok=True)

    # move old logs and models
    for file_name in os.listdir(os.path.join(output_folder, 'logs')):
        file_path = os.path.join(os.path.join(output_folder, 'logs'), file_name)
        if os.path.isfile(file_path) and file_name.startswith('log_'):
            shutil.move(file_path, os.path.join(output_folder, 'logs/old/', file_name))
    for file_name in os.listdir(os.path.join(output_folder, 'models')):
        file_path = os.path.join(os.path.join(output_folder, 'models'), file_name)
        if file_name.startswith('model_'):
            print(file_path)
            shutil.copytree(file_path, os.path.join(output_folder, 'models/old', '_'.join(file_name.split('_')[2:]), file_name), dirs_exist_ok=True)
            shutil.rmtree(file_path)

    run()