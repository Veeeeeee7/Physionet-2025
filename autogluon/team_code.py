#!/usr/bin/env python

# Edit this script to add your team's code. Some functions are *required*, but you can edit most parts of the required functions,
# change or remove non-required functions, and add your own functions.

################################################################################
#
# Optional libraries, functions, and variables. You can change or remove them.
#
################################################################################

from sre_compile import isstring
import joblib
import numpy as np
import os
import sys

from helper_code import *
from features import *

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
    # if os.path.exists(os.path.join(model_folder, 'minirocket')):
    #     os.system(f'rm -rf {os.path.join(model_folder, "minirocket")}')
    # if os.path.exists(os.path.join(model_folder, 'autogluon')):
    #     os.system(f'rm -rf {os.path.join(model_folder, "autogluon")}')

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

    features = np.zeros((num_records, 3029), dtype=np.float32)
    labels = np.zeros(num_records, dtype=bool)

    # Instantiate MiniRocket Transformer
    num_kernels = 84 * 6
    random_state = 42
    minirocket = MiniRocketMultivariate(num_kernels=num_kernels, random_state=random_state)

    # Instantiate Autogluon
    autogluon_challenge_scorer = make_scorer(name='challenge_score', score_func=compute_challenge_score, optimum=1, greater_is_better=True, needs_proba=True, needs_threshold=False)
    autogluon = TabularPredictor(problem_type='binary', label='chagas', eval_metric=autogluon_challenge_scorer, path=os.path.join(model_folder, 'autogluon'), verbosity=2)
    time_limit = 86400
    memory = 60

    # Iterate over the records.
    for i in range(num_records):
        if verbose:
            width = len(str(num_records))
            print(f'- {i+1:>{width}}/{num_records}: {records[i]}...')

        flip = False
        if i == 0:
            flip = True
        record = os.path.join(data_folder, records[i])
        features[i] = extract_features(record, minirocket, flip)
        labels[i] = load_label(record)

    # Train the models.
    if verbose:
        print('Training the model on the data...')
    # Combine features and labels into one DataFrame
    sample_weights = features[:, 0]
    features = features[:, 1:]
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    df = pd.DataFrame(features)
    # df['sample_weight'] = sample_weights
    df['chagas'] = labels
    autogluon.fit(
        train_data=df, 
        fit_strategy='sequential', 
        num_cpus=16, 
        num_gpus=0, 
        memory_limit=memory, 
        save_bag_folds=True, 
        ag_args_fit={'max_memory_usage_ratio': 0.75},
        ag_args_ensemble={'fold_fitting_strategy': 'sequential_local'},
        time_limit=time_limit, 
        presets='best_quality')
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

def get_source(header):
    source, has_source = get_variable(header, '# Source:')
    if not has_source:
        source = 'Unknown'
    elif isstring(source):
        source = source.strip()
    return source

# Extract your features.
def extract_features(record, minirocket, flip):
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

    signal = (signal - signal_means.flatten()) / (signal_stds.flatten() + 1e-6)
    
    target_length = 4096
    if len(signal) > target_length:
        padded_signal = signal[:target_length]
    else:
        total_padding = target_length - len(signal)
        padding = total_padding // 2
        padded_signal = np.pad(signal, ((padding, total_padding - padding), (0, 0)), 'constant', constant_values=(0, 0))

    if flip:
        transformed_features = minirocket.fit(padded_signal)
    transformed_features = minirocket.transform(padded_signal)
    ecg_features = transformed_features.to_numpy().flatten()

    signal_features = np.concatenate((signal_means.flatten(), signal_stds.flatten(), signal_num_peaks.flatten(),
                                    signal_avg_intervals.flatten(), signal_sdnns.flatten(), signal_rmssds.flatten(),
                                    signal_freqs.flatten(), signal_amplitudes.flatten(), signal_energies.flatten()))

    features = np.concatenate((np.array([sample_weight]), np.array([age]), one_hot_encoding_sex, ecg_features, signal_features))

    return np.asarray(features, dtype=np.float32)
