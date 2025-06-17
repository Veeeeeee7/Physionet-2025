from tsai.all import *
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from sklearn.model_selection import train_test_split
from helper_code import *
import numpy as np
import pandas as pd
from datetime import datetime
import pytz
import random
import gc
import torch
import time
from concurrent.futures import ThreadPoolExecutor
import os

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

    target_length = 4096
    if len(signal) > target_length:
        padded_signal = signal[:target_length]
    else:
        total_padding = target_length - len(signal)
        padding = total_padding // 2
        padded_signal = np.pad(signal, ((padding, total_padding - padding), (0, 0)), 'constant', constant_values=(0, 0))

    return padded_signal, age, sex

def process_record(record):
    record = os.path.join(data_folder, record)
    record_signals, record_age, record_sex = extract_features(record)
    record_label = load_label(record)
    return record_signals.T, record_age, record_sex, record_label

def log(message):
    with open('/Users/victorli/Documents/GitHub/Physionet-2025/tsai/log_tcn.txt', 'a') as f:
        f.write(f"{message} \n")
        print(message)

if os.path.exists('/Users/victorli/Documents/GitHub/Physionet-2025/tsai/log_tcn.txt'):
    os.remove('/Users/victorli/Documents/GitHub/Physionet-2025/tsai/log_tcn.txt')

pst = pytz.timezone('US/Pacific')
start_time = datetime.now(pst).strftime('%Y-%m-%d %H:%M:%S %Z')
log(f"Script started at: {start_time}")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
log(f"Using device: {device}")

my_setup()

data_folder = '/Users/victorli/Documents/GitHub/Physionet-2025-ECGFM-Submission/training_data'
records = find_records(data_folder)
random.shuffle(records)
train_records, test_records = train_test_split(records, test_size=0.2, random_state=42)


num_train_records = len(train_records)
num_test_records = len(test_records)
log(f"Number of training records: {num_train_records}")
log(f"Number of test records: {num_test_records}")
ecg_signals = []
age = []
sex = []
labels = []


batch_size = 4
inner_batch_size = 2
num_workers = 1
num_epochs = 3
lr = 1e-3
archs = [
    # (LSTM, {}),                # LSTM (Hochreiter, 1997)
    # (GRU, {}),                 # GRU (Cho, 2014)
    # (MLP, {}),                 # MLP - Multilayer Perceptron (Wang, 2016)
    # (FCN, {}),                 # FCN - Fully Convolutional Network (Wang, 2016)
    # (ResNet, {}),              # ResNet - Residual Network (Wang, 2016)
    # (LSTM_FCN, {}),            # LSTM-FCN (Karim, 2017)
    # (GRU_FCN, {}),             # GRU-FCN (Elsayed, 2018)
    # (mWDN, {'levels': 4}),     # mWDN - Multilevel wavelet decomposition network (Wang, 2018)
    (TCN, {}),                 # TCN - Temporal Convolutional Network (Bai, 2018)
    # (MLSTM_FCN, {}),           # MLSTM-FCN - Multivariate LSTM-FCN (Karim, 2019)
    # (InceptionTime, {}),       # InceptionTime (Fawaz, 2019)
    # (XceptionTime, {}),        # XceptionTime (Rahimian, 2019)
    # (ResCNN, {}),              # ResCNN - 1D-ResCNN (Zou, 2019)
    # missing c_out (TabModel, {}),            # TabModel - modified from fastai’s TabularModel
    # takes very long? or doesn't work? (OmniScaleCNN, {}),        # OmniScaleCNN - Omni-Scale 1D-CNN (Tang, 2020)
    # not enough gpu memory (TST, {}),                 # TST - Time Series Transformer (Zerveas, 2020)
    # missing c_out (TabTransformer, {}),      # TabTransformer (Huang, 2020)
    # takes too much memory (TSiT, {}),                # TSiT - Adapted from ViT (Dosovitskiy, 2020)
    # (MiniRocket, {}),          # MiniRocket (Dempster, 2021)
    # (XCM, {}),                 # XCM - An Explainable Convolutional Neural Network (Fauvel, 2021)
    # (gMLP, {}),                # gMLP - Gated Multilayer Perceptron (Liu, 2021)
    # missing seq_len (TSPerceiver, {}),         # TSPerceiver - Adapted from Perceiver IO (Jaegle, 2021)
    # missing c_out (GatedTabTransformer, {}), # GatedTabTransformer (Cholakov, 2022)
    # not enough gpu memory (TSSequencerPlus, {}),     # TSSequencerPlus - Adapted from Sequencer (Tatsunami, 2022)
    # not enough gpu memory (PatchTST, {})             # PatchTST (Nie, 2022)
]

for i, (arch, k) in enumerate(archs):
    log(f"********************** {arch.__name__} **********************")
    for split in range(1):
        log(f"------------------- Fold {split} -------------------")
        log(f"Training {arch.__name__}")
        start = time.time()
        model = create_model(arch, c_in=12, c_out=2, seq_len=4096, **k)
        # load_model(model=model, device=device, opt=None, file=f'/Users/victorli/Documents/GitHub/Physionet-2025/tsai/{arch.__name__}', with_opt=False)
        model.train()
        for epoch in range(num_epochs):
            log(f"Epoch {epoch + 1}/{num_epochs} for {arch.__name__}")
            for batch_index in range((num_train_records + batch_size - 1) // batch_size):
                batch_start = time.time()
                start_index = batch_index * batch_size
                end_index = min((batch_index + 1) * batch_size, num_train_records)
                records_batch = train_records[start_index:end_index]

                # ecg_signals = []
                # age = []
                # sex = []
                # labels = []

                with ThreadPoolExecutor() as executor:
                    results = list(executor.map(process_record, records_batch))

                ecg_signals, age, sex, labels = zip(*results)

                # oversampling positive samples
                ecg_signals = list(ecg_signals)
                age = list(age)
                sex = list(sex)
                labels = list(labels)

                for i in range(len(labels) - 1, -1, -1):
                    if labels[i] == 1:
                        for j in range(9):
                            ecg_signals.append(ecg_signals[i])
                            age.append(age[i])
                            sex.append(sex[i])
                            labels.append(labels[i])
                            
                X = np.array(ecg_signals)
                y = np.array(labels)

                splits = (list(np.arange(0, len(X))), [])
                train_dsets = TSDatasets(X, y, splits=splits, inplace=True)
                train_dls  = TSDataLoaders.from_dsets(train_dsets.train, train_dsets.valid, bs=[inner_batch_size, 0], shuffle=True, num_workers=num_workers)

                # label smoothing
                learn = Learner(train_dls, model, loss_func=FocalLoss(smoothing=0.01)) #FBeta is basically F1
                learn.fit_one_cycle(1, lr)
                avg_loss = round(np.sum(np.array(learn.recorder.losses)) / len(learn.recorder.losses), 6)
                batch_elapsed = time.time() - batch_start
                log(f"Processing batch {batch_index + 1}/{(num_train_records + batch_size - 1) // batch_size}, start_index: {start_index}, end_index: {end_index-1}, avg_loss: {avg_loss}, time: {batch_elapsed} seconds")

                del ecg_signals
                del age
                del sex
                del labels
                del X
                del y
                del splits
                del train_dsets
                del train_dls
                del learn
                gc.collect()
                torch.cuda.empty_cache()

            log('\n')

            if epoch == num_epochs - 1:
                save_model(model=model, opt=None, file=f'/Users/victorli/Documents/GitHub/Physionet-2025/tsai/{arch.__name__}', with_opt=False)

        elapsed = time.time() - start
        log(f"Training time for {arch.__name__}: {elapsed} seconds")

        log(f"Evaluating {arch.__name__}")
        start = time.time()
        test_probas = np.zeros((num_test_records, 2))
        test_targets = np.zeros((num_test_records,))
        test_preds = np.zeros((num_test_records,))

        model = create_model(arch, c_in=12, c_out=2, seq_len=4096, **k)
        load_model(model=model, device=device, opt=None, file=f'/Users/victorli/Documents/GitHub/Physionet-2025/tsai/{arch.__name__}', with_opt=False)
        model.eval()

        for i in range(num_test_records):
            record = test_records[i]
            ecg_signal, age, sex, label = process_record(record)
            test_proba = model(torch.tensor(ecg_signal, dtype=torch.float32).unsqueeze(0).to(device)).detach().cpu().numpy()
            test_pred = np.argmax(test_proba, axis=1)
            test_probas[i] = test_proba
            test_targets[i] = label
            test_preds[i] = test_pred

            if (i + 1) % 1000 == 0:
                log(f"Evaluated {i + 1}/{num_test_records} records")

            del ecg_signal
            gc.collect()
            torch.cuda.empty_cache()

        
        elapsed = time.time() - start
        log(f"Test Accuracy: {accuracy_score(test_targets, test_preds)}")
        log(f"Test AUROC: {roc_auc_score(test_targets, test_preds)}")
        log(f"Test F1: {f1_score(test_targets, test_preds)}")
        log(f"Test Challenge Score: {compute_challenge_score(test_targets, test_probas[:, 1])}")
        log(f"Evaluation time for {arch.__name__}: {elapsed} seconds")