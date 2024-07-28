# data_preprocessing.py

import os
import numpy as np
import pandas as pd
import pywt
import mne
from sklearn.decomposition import PCA
from mne.preprocessing import ICA
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import gc

# Mount Google Drive
'''from google.colab import drive
drive.mount('/content/drive')
'''

# Directory paths
edf_dir_path = '/content/drive/MyDrive/Epileptic_Seizure_Classification_Project/Dataset/chb-mit-scalp-eeg-database-1.0.0/chb07'
output_dir_path = '/content/drive/MyDrive/Epileptic_Seizure_Classification_Project/cnn_Extracted_Features_/chb07'

os.makedirs(output_dir_path, exist_ok=True)

# List all EDF files
edf_files = [f for f in sorted(os.listdir(edf_dir_path)) if f.endswith('.edf') and f in ['chb07_13.edf', 'chb07_12.edf', 'chb07_19.edf', 'chb07_15.edf']]

# Initialize ICA and PCA parameters
ica = ICA(n_components=23, random_state=97, max_iter=400)
pca = PCA(n_components=16)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the CNN model
class ModifiedCNN(nn.Module):
    def __init__(self):
        super(ModifiedCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(16 * (23 // 2 // 2 // 2), 128)
        self.fc2 = nn.Linear(128, 16)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

cnn_model = ModifiedCNN().to(device)

# Function to check for seizures in the file name
def has_seizure(file_name):
    seizure_files = ['chb07_13.edf', 'chb07_12.edf', 'chb07_19.edf', 'chb07_15.edf']
    return file_name in seizure_files

# Function to process a single EDF file
def process_edf_file(edf_file):
    edf_file_path = os.path.join(edf_dir_path, edf_file)

    # Load the EDF file
    raw = mne.io.read_raw_edf(edf_file_path, preload=True)

    # Get signal parameters
    signal_length = raw.n_times
    sfreq = raw.info['sfreq']
    n_channels = len(raw.ch_names)
    signal_labels = raw.ch_names

    epoch_length_samples = int(360 * sfreq)
    num_epochs = signal_length // epoch_length_samples

    batch_size = 20  # Number of epochs to process in one batch
    cnn_features = []

    for batch_start in range(0, num_epochs, batch_size):
        batch_end = min(batch_start + batch_size, num_epochs)
        batch_data = []

        for epoch_idx in range(batch_start, batch_end):
            start_idx = epoch_idx * epoch_length_samples
            end_idx = (epoch_idx + 1) * epoch_length_samples
            if end_idx > signal_length:
                end_idx = signal_length

            signals = raw.get_data(start=start_idx, stop=end_idx)

            info = mne.create_info(ch_names=signal_labels, sfreq=sfreq, ch_types='eeg')
            raw_epoch = mne.io.RawArray(signals, info)
            raw_epoch.filter(1., 50., fir_design='firwin')

            scales = range(1, 128)
            cwt_data = np.zeros((n_channels, len(scales), signals.shape[1]))

            signals_torch = torch.tensor(raw_epoch.get_data(), dtype=torch.float32).to(device)
            for i in range(n_channels):
                cwt_result, _ = pywt.cwt(signals_torch[i].cpu().numpy(), scales, 'morl')
                cwt_data[i, :, :] = cwt_result

            cwt_data = torch.tensor(cwt_data, dtype=torch.float32).to(device)

            if torch.any(torch.isnan(cwt_data)) or torch.any(torch.isinf(cwt_data)):
                cwt_data = torch.nan_to_num(cwt_data)

            cwt_data_reshaped = cwt_data.view(n_channels * len(scales), signals.shape[1]).T

            if torch.any(torch.isnan(cwt_data_reshaped)) or torch.any(torch.isinf(cwt_data_reshaped)):
                cwt_data_reshaped = torch.nan_to_num(cwt_data_reshaped)

            ica.fit(raw_epoch)

            scaler = StandardScaler()
            cwt_data_normalized = scaler.fit_transform(cwt_data_reshaped.cpu().numpy())

            pca_data = pca.fit_transform(cwt_data_normalized)
            batch_data.append(pca_data)

            print(f"File {edf_file} - Epoch {epoch_idx + 1}/{num_epochs}: Explained variance ratio by first 20 components: {pca.explained_variance_ratio_}")

            del raw_epoch
            del cwt_data
            del cwt_data_reshaped
            del cwt_data_normalized
            gc.collect()

        batch_data_array = np.vstack(batch_data)
        pca_data_tensor = torch.tensor(batch_data_array, dtype=torch.float32).unsqueeze(2).to(device)

        dataset = TensorDataset(pca_data_tensor)
        data_loader = DataLoader(dataset, batch_size=32, shuffle=False)

        cnn_model.eval()
        with torch.no_grad():
            for batch in data_loader:
                inputs = batch[0].permute(0, 2, 1)
                outputs = cnn_model(inputs)
                cnn_features.append(outputs.cpu().numpy())

    cnn_features_array = np.vstack(cnn_features)

    del raw
    del cnn_features
    del batch_data
    del batch_data_array
    gc.collect()

    final_pca = PCA(n_components=16)
    reduced_features = final_pca.fit_transform(cnn_features_array)

    reduced_features_quantized = np.round(reduced_features * 1e6).astype(np.int32)

    return reduced_features_quantized

# Process all EDF files and save the extracted features
for edf_file in edf_files:
    features = process_edf_file(edf_file)
    has_seizure_label = 1 if has_seizure(edf_file) else 0
    features_with_label = np.column_stack((features, np.full((features.shape[0], 1), has_seizure_label)))
    output_file_path = os.path.join(output_dir_path, f'cnn_features_{os.path.splitext(edf_file)[0]}.csv')
    pd.DataFrame(features_with_label).to_csv(output_file_path, index=False, header=False)
    print(f"Processed and saved features for file {edf_file}.")

print("Processing and feature extraction complete for all EDF files.")
