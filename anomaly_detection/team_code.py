#!/usr/bin/env python

# Edit this script to add your team's code. Some functions are *required*, but you can edit most parts of the required functions,
# change or remove non-required functions, and add your own functions.

################################################################################
#
# Optional libraries, functions, and variables. You can change or remove them.
#
################################################################################

from sre_compile import isstring
import numpy as np
import os
import sys

from helper_code import *
from features import *

import torch
from fairseq_signals.models import build_model_from_checkpoint
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import minmax_scale
import pandas as pd
import time

################################################################################
#
# Required functions. Edit these functions to add your code, but do not change the arguments for the functions.
#
################################################################################

# Train your models. This function is *required*. You should edit this function to add your code, but do *not* change the arguments
# of this function. If you do not train one of the models, then you can return None for the model.

def log(message, file):
    with open(file, 'a') as f:
        f.write(message + '\n')

# Train your model.
def train_model(data_folder, model_folder, verbose):
    log_file = '/Users/victorli/Documents/GitHub/Physionet-2025/anomaly_detection/log.txt'
    if os.path.exists(log_file):
        os.remove(log_file)
    # Create the log file
    with open(log_file, 'w') as f:
        f.write('')

    # Find the data files.
    if verbose:
        log('Finding the Challenge data...', log_file)

    records = find_records(data_folder)
    num_records = len(records)

    if num_records == 0:
        raise FileNotFoundError('No data were provided.')

    # Extract the features and labels from the data.
    if verbose:
        log('Extracting features and labels from the data...', log_file)

    features = np.zeros((num_records, 3293), dtype=np.float32)
    labels = np.zeros(num_records, dtype=bool)

    # Instantiate Foundation Model
    model_pretrained = build_model_from_checkpoint(
        checkpoint_path='ckpts/mimic_iv_ecg_physionet_pretrained.pt'
    ).to('mps')
    model_pretrained.eval()


    start_time = time.time()
    for i in range(num_records):
        if verbose and i % 10000 == 0:
            width = len(str(num_records))
            log(f'- {i+1:>{width}}/{num_records}: {records[i]}...', log_file)

        record = os.path.join(data_folder, records[i])
        features[i] = extract_features(record, model_pretrained)
        labels[i] = load_label(record)
    end_time = time.time()
    elapsed_time = end_time - start_time
    if verbose:
        log(f'Feature extraction and labeling completed in {elapsed_time:.2f} seconds.', log_file)

    # Train the models.
    if verbose:
        log('Training the model on the data...', log_file)
    # Combine features and labels into one DataFrame
    df = pd.DataFrame(features)
    df.rename(columns={0: 'sample_weight'}, inplace=True)
    df['chagas'] = labels

    train_df = df.sample(frac=0.8, random_state=42)
    val_df = df.drop(train_df.index)
    val_labels = val_df['chagas'].to_numpy()
    val_df = val_df.drop(columns=['chagas'])
    val_df = val_df.drop(columns=['sample_weight'])

    start_time = time.time()
    iforest = IsolationForest(random_state=42, n_estimators=100, contamination=0.05, max_samples=100)
    iforest.fit(X=train_df.drop(columns=['sample_weight', 'chagas']).to_numpy(), y=train_df['chagas'], sample_weight=train_df['sample_weight'].to_numpy())
    end_time = time.time()
    elapsed_time = end_time - start_time
    if verbose:
        log(f'Model training completed in {elapsed_time:.2f} seconds.', log_file)

    raw_scores = iforest.decision_function(val_df)  
    proba_inlier = minmax_scale(raw_scores)
    proba_outlier = 1.0 - proba_inlier
    val_proba = proba_outlier

    val_predictions = iforest.predict(val_df)
    val_predictions = np.where(val_predictions == 1, False, True)
    challenge_score = compute_challenge_score(val_labels, val_proba)
    auroc, auprc = compute_auc(val_labels, val_proba)
    accuracy = compute_accuracy(val_labels, val_predictions)
    f_measure = compute_f_measure(val_labels, val_predictions)

    output_string = \
        f'Challenge score: {challenge_score:.3f}\n' + \
        f'AUROC: {auroc:.3f}\n' \
        f'AUPRC: {auprc:.3f}\n' + \
        f'Accuracy: {accuracy:.3f}\n' \
        f'F-measure: {f_measure:.3f}\n'
    
    # Save the evaluation metrics to a file
    scores_file = "/Users/victorli/Documents/GitHub/Physionet-2025/anomaly_detection/scores.txt"
    with open(scores_file, 'w') as f:
        f.write(output_string)

    if verbose:
        log(output_string, log_file)

# Load your trained models. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function. If you do not train one of the models, then you can return None for the model.
def load_model(model_folder, verbose):
    # model = TabularPredictor.load(os.path.join(model_folder, 'autogluon'))
    model = None
    return model

# Run your trained model. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function.
def run_model(record, model, verbose):
    # Load the model.
    autogluon = model
    foundation_model = build_model_from_checkpoint(
        checkpoint_path='ckpts/mimic_iv_ecg_physionet_pretrained.pt'
    )

    # Extract the features.
    features = extract_features(record, foundation_model)
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

def get_source(header):
    source, has_source = get_variable(header, '# Source:')
    if not has_source:
        source = 'Unknown'
    elif isstring(source):
        source = source.strip()
    return source

# Extract your features.
def extract_features(record, foundation_model):
    header = load_header(record)
    age = get_age(header)
    sex = get_sex(header)
    source = get_source(header)
    sampling_frequency = get_sampling_frequency(header)

    if source == 'CODE-15%':
        sample_weight = 0.1
    else:
        sample_weight = 1.0


    one_hot_encoding_sex = np.zeros(3, dtype=bool)
    if sex == 'Female':
        one_hot_encoding_sex[0] = 1
    elif sex == 'Male':
        one_hot_encoding_sex[1] = 1
    else:
        one_hot_encoding_sex[2] = 1

    signal, fields = load_signals(record)

    signal_means = np.zeros((12, 1), dtype=np.float32)
    signal_stds = np.zeros((12, 1), dtype=np.float32)
    signal_num_peaks = np.zeros((12, 1), dtype=np.float32)
    signal_avg_intervals = np.zeros((12, 1), dtype=np.float32)
    signal_sdnns = np.zeros((12, 1), dtype=np.float32)
    signal_rmssds = np.zeros((12, 1), dtype=np.float32)
    signal_freqs = np.zeros((12, 100), dtype=np.float32)
    signal_amplitudes = np.zeros((12, 100), dtype=np.float32)
    signal_energies = np.zeros((12, 4), dtype=np.float32)
    for lead in range(12):
        signal_mean = np.mean(signal[:, lead])
        signal_std = np.std(signal[:, lead])
        signal_r_peaks = compute_r_peaks(signal[:, lead], fs=sampling_frequency)
        signal_rr_intervals = compute_rr_intervals(signal_r_peaks, fs=sampling_frequency)
        signal_sdnn, signal_rmssd = compute_hrv(signal_rr_intervals)
        signal_freq, signal_amplitude = compute_fft(signal[:, lead], fs=sampling_frequency)
        signal_energy = compute_wavelet_energy(signal[:, lead])

        signal_num_peak = len(signal_r_peaks)
        signal_avg_interval = np.mean(np.diff(signal_r_peaks)) / sampling_frequency

        signal_means[lead] = signal_mean
        signal_stds[lead] = signal_std
        signal_num_peaks[lead] = signal_num_peak
        signal_avg_intervals[lead] = signal_avg_interval
        signal_sdnns[lead] = signal_sdnn
        signal_rmssds[lead] = signal_rmssd
        signal_freqs[lead] = signal_freq
        signal_amplitudes[lead] = signal_amplitude
        signal_energies[lead] = signal_energy

    target_length = 4096
    if len(signal) > target_length:
        padded_signal = signal[:target_length]
    else:
        total_padding = target_length - len(signal)
        padding = total_padding // 2
        padded_signal = np.pad(signal, ((padding, total_padding - padding), (0, 0)), 'constant', constant_values=(0, 0))

    x = torch.from_numpy(padded_signal.T).float().to('mps')
    x = x.unsqueeze(0)
    transformed_features = foundation_model(source=x)['features'].mean(dim=1).to('cpu')
    ecg_features = transformed_features.detach().numpy().flatten()

    signal_features = np.concatenate((signal_means.flatten(), signal_stds.flatten(), signal_num_peaks.flatten(),
                                       signal_avg_intervals.flatten(), signal_sdnns.flatten(), signal_rmssds.flatten(),
                                       signal_freqs.flatten(), signal_amplitudes.flatten(), signal_energies.flatten()))

    features = np.concatenate((np.array([sample_weight]), np.array([age]), one_hot_encoding_sex, ecg_features, signal_features))

    return np.asarray(features, dtype=np.float32)

