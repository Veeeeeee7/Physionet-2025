import argparse
import os
import sys

from helper_code import compute_accuracy, compute_auc, compute_challenge_score, compute_f_measure, is_nan

from sklearn.model_selection import KFold
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sktime.transformations.panel.rocket import MiniRocketMultivariate
from sktime.utils import mlflow_sktime 
import shutil
import glob
import wfdb
from datetime import datetime

def get_parser():
    description = "MiniRocket with RandomForest Classifier and Cross Validation"
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-d', '--data_folder', type=str, help="Path to the data file")
    parser.add_argument('-o', '--output_folder', type=str, help="Path to the output folder")
    parser.add_argument('-m', '--model_folder', type=str, help="Path to the model folder")
    return parser

def run(args):
    prepare_log_file(os.path.join(args.output_folder, log_file_name))

    log("Loading signals and labels", args.output_folder)
    signals_and_labels = load_signals_and_labels(args.data_folder, args.output_folder)
    # signals_and_labels = load_signals_and_labels(args.data_folder, args.output_folder, 200)

    y = signals_and_labels['labels']
    X = signals_and_labels.drop('labels', axis=1)

    num_records = X.shape[0]

    if signals_and_labels.empty:
        raise FileNotFoundError("No data found in the data folder")
    else:
        log(f'Found {num_records} records', args.output_folder)

    kf = KFold(n_splits=2, shuffle=True, random_state=42)
    scores = []

    for i, (train_index, test_index) in enumerate(kf.split(X)):
        log(f'Running fold {i+1} of 5', args.output_folder)

        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        train_model(X_train, y_train, args.output_folder, args.model_folder)

        challenge_score, auroc, auprc, accuracy, f_measure = test_model(X_test, y_test, args.output_folder, args.model_folder)
        scores.append((challenge_score, auroc, auprc, accuracy, f_measure))

        log(f'Challenge Score: {challenge_score:.3f}\nAUROC: {auroc:.3f}\nAUPRC: {auprc:.3f}\nAccuracy: {accuracy:.3f}\nF-measure: {f_measure:.3f}', args.output_folder)
    
    avg_scores = np.mean(scores, axis=0)
    log(f'Average Challenge Score: {avg_scores[0]:.3f}\nAverage AUROC: {avg_scores[1]:.3f}\nAverage AUPRC: {avg_scores[2]:.3f}\nAverage Accuracy: {avg_scores[3]:.3f}\nAverage F-measure: {avg_scores[4]:.3f}', args.output_folder)

    with open(os.path.join(args.output_folder, 'cross_validation_results.txt'), 'w') as f:
        f.write(f'Average Challenge Score: {avg_scores[0]:.3f}\nAverage AUROC: {avg_scores[1]:.3f}\nAverage AUPRC: {avg_scores[2]:.3f}\nAverage Accuracy: {avg_scores[3]:.3f}\nAverage F-measure: {avg_scores[4]:.3f}')


def train_model(X, y, output_folder, model_folder):
    num_records = X.shape[0]
    log(f'Training model on {num_records} records', output_folder)

    log(f'Transforming Train Signals with Mini-Rocket', output_folder)
    X_ecg = X.iloc[:, :12]
    mini_rocket = MiniRocketMultivariate(num_kernels=num_kernels, random_state=random_state)
    X_transformed = pd.DataFrame()
    for i in range(0, num_records, batch_size):
        log(f'Transforming signals {i+1} to {min(i+batch_size, num_records)} out of {num_records}', output_folder)
        X_batch = X_ecg.iloc[i:i+batch_size]
        X_batch_transformed = mini_rocket.fit_transform(X_batch)
        X_transformed = pd.concat([X_transformed, pd.DataFrame(X_batch_transformed)], ignore_index=True)

    # print(X.reset_index(drop=True).iloc[:, 12:].shape)
    # print(X_transformed.reset_index(drop=True))
    X = pd.concat([X_transformed.reset_index(drop=True), X.reset_index(drop=True).iloc[:, 12:]], axis=1)

    X.columns = [''] * len(X.columns)

    mini_rocket_path = os.path.join(model_folder, 'minirocket/')
    if os.path.exists(mini_rocket_path):
        shutil.rmtree(mini_rocket_path)
    mlflow_sktime.save_model(sktime_model=mini_rocket, path=mini_rocket_path)

    log(f'Fitting RF', output_folder)
    rf = RandomForestClassifier(n_estimators=n_estimators, max_leaf_nodes=max_leaf_nodes, random_state=random_state)
    for i in range(0, num_records, batch_size):
        log(f'Fitting RF on records {i+1} to {min(i+batch_size, num_records)} out of {num_records}', output_folder)
        X_batch = X.iloc[i:i+batch_size]
        y_batch = y.iloc[i:i+batch_size]
        rf.fit(X_batch, y_batch)

    rf_path = os.path.join(model_folder, 'random_forest/')
    if os.path.exists(rf_path):
        shutil.rmtree(rf_path)
    os.makedirs(rf_path)
    joblib.dump(rf, rf_path + 'random_forest.joblib')

def test_model(X, y, output_folder, model_folder):
    num_records = X.shape[0]
    log(f'Testing model on {num_records} records', output_folder)

    mini_rocket_path = os.path.join(model_folder, 'minirocket/')
    mini_rocket = mlflow_sktime.load_model(model_uri=mini_rocket_path)

    log(f'Transforming Test Signals with Mini-Rocket', output_folder)
    X_ecg = X.iloc[:, :12]
    X_transformed = pd.DataFrame()
    for i in range(0, num_records, batch_size):
        log(f'Transforming signals {i+1} to {min(i+batch_size, num_records)} out of {num_records}', output_folder)
        X_batch = X_ecg.iloc[i:i+batch_size]
        X_batch_transformed = mini_rocket.fit_transform(X_batch)
        X_transformed = pd.concat([X_transformed, pd.DataFrame(X_batch_transformed)], ignore_index=True)

    X = pd.concat([X_transformed.reset_index(drop=True), X.reset_index(drop=True).iloc[:, 12:]], axis=1)
    X.columns = [''] * len(X.columns)


    rf_path = os.path.join(model_folder, 'random_forest/random_forest.joblib')
    rf = joblib.load(rf_path)

    binary_outputs = np.zeros(num_records)
    probability_outputs = np.zeros(num_records)
    for i in range(0, num_records, batch_size):
        batch_end = min(i + batch_size, num_records)
        log(f'Predicting on records {i+1} to {batch_end} out of {num_records}', output_folder)

        batch_X = X.iloc[i:batch_end]
        
        batch_binary = rf.predict(batch_X)
        batch_probability = rf.predict_proba(batch_X)[:, 1]
        
        batch_binary = np.where(np.isnan(batch_binary), 0, batch_binary)
        batch_probability = np.where(np.isnan(batch_probability), 0, batch_probability)
        
        binary_outputs[i:batch_end] = batch_binary
        probability_outputs[i:batch_end] = batch_probability

    y = y.to_numpy()

    challenge_score = compute_challenge_score(y, probability_outputs)
    auroc, auprc = compute_auc(y, probability_outputs)
    accuracy = compute_accuracy(y, binary_outputs)
    f_measure = compute_f_measure(y, binary_outputs)

    return challenge_score, auroc, auprc, accuracy, f_measure
    
def log(message, output_folder):
    with open(os.path.join(output_folder, log_file_name), 'a') as f:
        f.write(message + '\n')

def prepare_log_file(log_file):
    if os.path.exists(log_file):
        os.remove(log_file)

def load_signals_and_labels(data_folder, output_folder):
    header_files = glob.glob(os.path.join(data_folder, '*.hea'))
    record_names = [os.path.splitext(file)[0] for file in header_files]
    
    df = pd.DataFrame()
    for i in range(0, len(record_names), batch_size):
        log(f'Loading signals {i+1} to {min(i+batch_size, len(record_names))} out of {len(record_names)}', output_folder)
        batch_records = record_names[i:i + batch_size]
        
        signals = [wfdb.rdsamp(record)[0] for record in batch_records]
        reshaped_signals = reshape_signals(signals)
        padded_signals = pad_signals(reshaped_signals, (4096, 12))
        
        demographic_features = extract_demographic_features(batch_records)
        labels = load_labels(batch_records)
        
        df_batch = pd.DataFrame(padded_signals)
        df_batch = pd.concat([df_batch, demographic_features], axis=1)
        df_batch['labels'] = labels

        df = pd.concat([df, df_batch], ignore_index=True)
    return df

def load_signals_and_labels(data_folder, output_folder, n):
    header_files = glob.glob(os.path.join(data_folder, '*.hea'))
    record_names = [os.path.splitext(file)[0] for file in header_files]
    
    if len(record_names) < n:
        raise ValueError(f"Requested {n} records, but only {len(record_names)} available.")
    
    selected_records = np.random.choice(record_names, n, replace=False)
    
    df = pd.DataFrame()
    for i in range(0, len(selected_records), batch_size):
        log(f'Loading signals {i+1} to {min(i+batch_size, len(selected_records))} out of {len(selected_records)}', output_folder)
        batch_records = selected_records[i:i + batch_size]
        
        signals = [wfdb.rdsamp(record)[0] for record in batch_records]
        reshaped_signals = reshape_signals(signals)
        padded_signals = pad_signals(reshaped_signals, (4096, 12))
        
        demographic_features = extract_demographic_features(batch_records)
        labels = load_labels(batch_records)
        
        df_batch = pd.DataFrame(padded_signals)
        df_batch = pd.concat([df_batch, demographic_features], axis=1)
        df_batch['labels'] = labels

        df = pd.concat([df, df_batch], ignore_index=True)
    return df

def load_labels(record_names):
    labels = []
    for record in record_names:
        header = wfdb.rdheader(record)
        chagas_label = None
        for comment in header.comments:
            if comment.startswith("Chagas label: "):
                chagas_label = comment.split("Chagas label: ")[1].strip().lower() == 'true'
                labels.append(1 if chagas_label else 0)
                break

    return labels


def extract_demographic_features(record_names):
    demographic_features = pd.DataFrame()
    for file in record_names:
        header = wfdb.rdheader(file)
        labels = []
        values = []
        for comment in header.comments:
            if 'Chagas label' not in comment:
                label = comment.split(": ")[0]
                value = comment.split(": ")[1]
                labels.append(label)
                values.append(value)

        demographic_features = pd.concat([demographic_features, pd.DataFrame([values], columns=labels)], ignore_index=True)
    
    if 'Sex' in demographic_features.columns:
        demographic_features['Sex'] = demographic_features['Sex'].apply(lambda x: 0 if x == 'Female' else 1 if x == 'Male' else 2)

    if 'Source' in demographic_features.columns:
        demographic_features = demographic_features.drop(columns=['Source'])

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


log_file_name = f'log_{datetime.now().strftime("%Y-%-m-%d_%H:%M:%S")}'
num_kernels = 84 * 100
n_estimators = 12
max_leaf_nodes = 34
random_state = 42
batch_size = 64


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    run(args)