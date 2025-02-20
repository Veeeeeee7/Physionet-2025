import argparse
import os
import sys

from helper_code import *

from sklearn.model_selection import KFold
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sktime.transformations.panel.rocket import MiniRocketMultivariate
from sktime.utils import mlflow_sktime 
import shutil
import glob


def process_all_signals(data_folder):
    header_files = glob.glob(os.path.join(data_folder, '*.hea'))

    record_names = [os.path.splitext(file)[0] for file in header_files]

    ALL_SIGNALS = [wfdb.rdsamp(record)[0] for record in record_names]

    return ALL_SIGNALS


# Parse arguments.
def get_parser():
    description = 'KFold cross validation.'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-d', '--data_folder', type=str, required=True)
    parser.add_argument('-o', '--output_folder', type=str, required=True)
    return parser

def log_message(message, output_folder):
    with open(os.path.join(output_folder, 'progress.txt'), 'a') as f:
        f.write(message + '\n')

def run(args):
    # ALL_SIGNALS = process_all_signals(args.data_folder)

    progress_file = os.path.join(args.output_folder, 'progress.txt')
    if os.path.exists(progress_file):
        os.remove(progress_file)
    else:
        open(progress_file, 'w').close()

    records = find_records(args.data_folder)
    num_records = len(records)

    if num_records == 0:
        raise FileNotFoundError('No data were provided.')
    else:
        log_message(f'Found {num_records} records', args.output_folder)

    # Perform KFold cross validation.
    kf = KFold(n_splits=5, shuffle=True, random_state=0)
    scores = []

    for i, (train_index, test_index) in enumerate(kf.split(records)):
        log_message(f'Fold {i+1} of 5', args.output_folder)

        # Train the model.
        train_records = [records[j] for j in train_index]
        model = train_model(train_records, args.data_folder, args.output_folder)

        # Test the model.
        test_records = [records[j] for j in test_index]
        challenge_score, auroc, auprc, accuracy, f_measure = test_model(model, test_records, args.data_folder, args.output_folder)
        scores.append((challenge_score, auroc, auprc, accuracy, f_measure))

        output_string = f'Challenge Score: {challenge_score:.3f}\nAUROC: {auroc:.3f}\nAUPRC: {auprc:.3f}\nAccuracy: {accuracy:.3f}\nF-measure: {f_measure:.3f}'
        log_message(output_string, args.output_folder)

    avg_scores = np.mean(scores, axis=0)
    output_string = f'Average Challenge Score: {avg_scores[0]:.3f}\nAverage AUROC: {avg_scores[1]:.3f}\nAverage AUPRC: {avg_scores[2]:.3f}\nAverage Accuracy: {avg_scores[3]:.3f}\nAverage F-measure: {avg_scores[4]:.3f}'
    log_message(output_string, args.output_folder)
    with open(os.path.join(args.output_folder, 'cross_validation_results.txt'), 'w') as f:
        f.write(output_string)

def train_model(records, data_folder, output_folder, ALL_SIGNALS):
    num_records = len(records)
    if num_records == 0:
        raise FileNotFoundError('No data were provided.')
    else:
        log_message(f'Training with {num_records} records', output_folder)

    num_kernels = 84 * 100
    n_estimators = 12
    max_leaf_nodes = 34
    random_state = 56
    batch_size = 2000

    features = np.zeros((num_records, num_kernels + 28), dtype=np.float64)
    labels = np.zeros(num_records, dtype=bool)

    mini_rocket = MiniRocketMultivariate(num_kernels=num_kernels, max_dilations_per_kernel=32, random_state=random_state)

    for i in range(0, num_records, batch_size):
        batch_records = records[i:i + batch_size]
        log_message(f"Processing records {i+1} to {i + len(batch_records)}", output_folder)

        signals_batch = []
        features_batch = []
        labels_batch = []
        for record in batch_records:
            signals, fields = load_signals(os.path.join(data_folder, record))
            signals_batch.append(signals)
            features_batch.append(extract_features(os.path.join(data_folder, record), signals))
            labels_batch.append(load_label(os.path.join(data_folder, record)))
        
        # Reshape the signals
        reshaped_signals_batch = pd.DataFrame(reshape_signals(signals_batch))
        
        # Get the first signal's shape and pad all signals accordingly
        padded_signals_batch = pad_signals(reshaped_signals_batch, [4096,0])

        # Extract ECG features from the padded signals
        ecg_features_batch = extract_ecg_features(padded_signals_batch, mini_rocket)

        for j in range(len(batch_records)):
            features[i + j] = np.concatenate((ecg_features_batch.iloc[j], features_batch[j]))
            labels[i + j] = labels_batch[j]
        

    model_path = 'mini_rocket_model'
    if os.path.exists(model_path):
        shutil.rmtree(model_path)
    mlflow_sktime.save_model(sktime_model=mini_rocket, path=model_path)

    log_message('Fitting base model', output_folder)

    base_model = RandomForestClassifier(
        n_estimators=n_estimators, max_leaf_nodes=max_leaf_nodes, random_state=random_state).fit(features, labels)


    # Sample weighting (similar to focal loss as focal loss isn't available for random forest in sklearn)
    gamma = 1
    alpha = 0.5

    pred_probs = base_model.predict_proba(features)[:, 1]

    pt = np.where(labels == 1, pred_probs, 1 - pred_probs)
    focal_weights = (1 - pt) ** gamma
    if alpha is not None:
        class_weights = np.where(labels == 1, alpha, 1 - alpha)
        focal_weights *= class_weights

    log_message('Fitting final model', output_folder)

    final_model = RandomForestClassifier(
        n_estimators=n_estimators, max_leaf_nodes=max_leaf_nodes, random_state=random_state
    ).fit(features, labels, sample_weight=focal_weights)

    return final_model

def reshape_signals(signals):
    reshaped_signals = np.empty((len(signals), signals[0].shape[1]), dtype=object)

    for i in range(len(signals)):
        for j in range(signals[i].shape[1]):
            reshaped_signals[i, j] = pd.Series(signals[i][:, j]) # reshaping from (# of records, num_samples, num_leads) to (# of records, num_leads) where each entry is a pd.Series of length num_samples

    return reshaped_signals
    
def pad_signals(signals, target_shape):
    padded_signals = signals.copy()
    for i in range(signals.shape[0]):
        for j in range(signals.shape[1]):
            signal = signals.iloc[i, j]
            if signal.shape[0] < target_shape[0]:
                pad_total = target_shape[0] - signal.shape[0]
                left_pad = pad_total // 2
                right_pad = pad_total - left_pad
                padded_signal = np.pad(signal, (left_pad, right_pad), mode='constant', constant_values=0)
                padded_signals.iloc[i, j] = pd.Series(padded_signal)
            else:
                padded_signals.iloc[i, j] = signal[:target_shape[0]]
    return padded_signals

def run_model(record, model):
    model_path = 'mini_rocket_model'
    loaded_model = mlflow_sktime.load_model(model_uri=model_path)

    signals, fields = load_signals(record)

    features = extract_features(record, signals)
    reshaped_signals = reshape_signals([signals])
    padded_signals = pad_signals(pd.DataFrame(reshaped_signals), [4096, 0])
    signals = padded_signals

    ecg_features = loaded_model.transform(signals)
    features = np.concatenate((ecg_features.iloc[0], features))

    features = features.reshape(1, -1)

    binary_output = model.predict(features)[0]
    probability_output = model.predict_proba(features)[0][1]

    return binary_output, probability_output

def test_model(model, records, data_folder, output_folder):
    num_records = len(records)

    if num_records == 0:
        raise FileNotFoundError('No data were provided.')
    else:
        log_message(f'Testing with {num_records} records', output_folder)

    labels = np.zeros(num_records)
    binary_outputs = np.zeros(num_records)
    probability_outputs = np.zeros(num_records)
    for i, record in enumerate(records):
        if i % 2000 == 0:
            log_message(f'Testing records {i+1} to {min(i+2000, num_records)}', output_folder)
        try:
            binary_output, probability_output = run_model(os.path.join(data_folder, record), model)
        except:
            binary_output, probability_output = float('nan'), float('nan')
            log_message(f'Error processing record {record}', output_folder)

        # binary_output, probability_output = run_model(os.path.join(data_folder, record), model)

        binary_outputs[i] = binary_output
        probability_outputs[i] = probability_output

        label_filename = os.path.join(data_folder, record)
        label = load_label(label_filename)
        labels[i] = label

        if not is_nan(binary_output):
            binary_outputs[i] = binary_output
        else:
            binary_outputs[i] = 0
        if not is_nan(probability_output):
            probability_outputs[i] = probability_output
        else:
            probability_outputs[i] = 0

    challenge_score = compute_challenge_score(labels, probability_outputs)
    auroc, auprc = compute_auc(labels, probability_outputs)
    accuracy = compute_accuracy(labels, binary_outputs)
    f_measure = compute_f_measure(labels, binary_outputs)

    return challenge_score, auroc, auprc, accuracy, f_measure

def extract_ecg_features(signals, model):
    model.fit(signals)
    return model.transform(signals)

def extract_features(record, signal):
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

    num_leads = signal.shape[1]
    lead_means = np.zeros(num_leads)
    lead_stds = np.zeros(num_leads)

    for i in range(num_leads):
        lead_signal = signal[:, i]
        num_finite_samples = np.sum(np.isfinite(lead_signal))
        if num_finite_samples > 0:
            lead_means[i] = np.nanmean(lead_signal)
        else:
            lead_means[i] = 0.0
        if num_finite_samples > 1:
            lead_stds[i] = np.nanstd(lead_signal)
        else:
            lead_stds[i] = 0.0

    features = np.concatenate(([age], one_hot_encoding_sex, lead_means, lead_stds))
    return np.asarray(features, dtype=np.float32)

def save_model(model_folder, model, fold):
    d = {'model': model}
    filename = os.path.join(model_folder, f'model{fold}.sav')
    joblib.dump(d, filename, protocol=0)

def load_model(model_folder, fold):
    model_filename = os.path.join(model_folder, f'model{fold}.sav')
    model = joblib.load(model_filename)
    return model

if __name__ == '__main__':
    run(get_parser().parse_args(sys.argv[1:]))
