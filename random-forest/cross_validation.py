import argparse
import os
import sys

from helper_code import *

from sklearn.model_selection import KFold
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# Parse arguments.
def get_parser():
    description = 'KFold cross validation.'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-d', '--data_folder', type=str, required=True)
    parser.add_argument('-o', '--output_folder', type=str, required=True)
    return parser

def run(args):
    records = find_records(args.data_folder)
    num_records = len(records)

    if num_records == 0:
        raise FileNotFoundError('No data were provided.')
    else:
        print(f'Found {num_records} records')

    # Perform KFold cross validation.
    kf = KFold(n_splits=5, shuffle=True, random_state=0)
    scores = []

    for i, (train_index, test_index) in enumerate(kf.split(records)):
        print(f'Fold {i+1} of 5')

        # Train the model.
        train_records = [records[j] for j in train_index]
        model = train_model(train_records, args.data_folder)

        # Test the model.
        test_records = [records[j] for j in test_index]
        challenge_score, auroc, auprc, accuracy, f_measure= test_model(model, test_records, args.data_folder)
        scores.append((challenge_score, auroc, auprc, accuracy, f_measure))

    avg_scores = np.mean(scores, axis=0)
    output_string = f'Average Challenge Score: {avg_scores[0]:.4f}\nAverage AUROC: {avg_scores[1]:.4f}\nAverage AUPRC: {avg_scores[2]:.4f}\nAverage Accuracy: {avg_scores[3]:.4f}\nAverage F-measure: {avg_scores[4]:.4f}'
    print(output_string)
    with open(os.path.join(args.output_folder, 'cross_validation_results.txt'), 'w') as f:
        f.write(output_string)

def train_model(records, data_folder):
    num_records = len(records)
    if num_records == 0:
        raise FileNotFoundError('No data were provided.')
    else:
        print(f'Training with {num_records} records')

    features = np.zeros((num_records, 28), dtype=np.float64)
    labels = np.zeros(num_records, dtype=bool)

    for i in range(num_records):
        record = os.path.join(data_folder, records[i])
        features[i] = extract_features(record)
        labels[i] = load_label(record)

    n_estimators = 12
    max_leaf_nodes = 34
    random_state = 56

    model = RandomForestClassifier(
        n_estimators=n_estimators, max_leaf_nodes=max_leaf_nodes, random_state=random_state).fit(features, labels)

    return model

def run_model(record, model):
    features = extract_features(record)
    features = features.reshape(1, -1)

    binary_output = model.predict(features)[0]
    probability_output = model.predict_proba(features)[0][1]

    return binary_output, probability_output

def test_model(model, records, data_folder):
    num_records = len(records)

    if num_records == 0:
        raise FileNotFoundError('No data were provided.')
    else:
        print(f'Testing with {num_records} records')

    labels = np.zeros(num_records)
    binary_outputs = np.zeros(num_records)
    probability_outputs = np.zeros(num_records)
    for i, record in enumerate(records):
        try:
            binary_output, probability_output = run_model(os.path.join(data_folder, record), model) ### Teams: Implement this function!!!
        except:
            binary_output, probability_output = float('nan'), float('nan')

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

def extract_features(record):
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

    # Assuming the signal has shape (num_samples, num_leads)
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