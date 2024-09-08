import os
import gc
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import mne
import pywt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

# Initialize ICA and PCA parameters
ica = mne.preprocessing.ICA(n_components=23, random_state=97, max_iter=400)
pca = PCA(n_components=16)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ModifiedCNN(nn.Module):
    def __init__(self):
        super(ModifiedCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1)  # Additional conv layer
        self.fc1 = nn.Linear(16 * (23 // 2 // 2 // 2), 128)  # Adjust input size
        self.fc2 = nn.Linear(128, 16)  # Adjust output size to 16

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))  # Additional conv layer
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

cnn_model = ModifiedCNN().to(device)

def process_edf_file(edf_file_path):
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

            # Compute CWT using PyTorch
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

            # Apply ICA (still on CPU)
            ica.fit(raw_epoch)

            # Normalize the data before PCA
            scaler = StandardScaler()
            cwt_data_normalized = scaler.fit_transform(cwt_data_reshaped.cpu().numpy())

            # Apply PCA
            pca_data = pca.fit_transform(cwt_data_normalized)
            batch_data.append(pca_data)

            print(f"Processed Epoch {epoch_idx + 1}/{num_epochs}")

            # Clear variables to free memory
            del raw_epoch
            del cwt_data
            del cwt_data_reshaped
            del cwt_data_normalized
            gc.collect()

        batch_data_array = np.vstack(batch_data)
        pca_data_tensor = torch.tensor(batch_data_array, dtype=torch.float32).unsqueeze(2).to(device)

        # Create DataLoader for the PCA data
        dataset = TensorDataset(pca_data_tensor)
        data_loader = DataLoader(dataset, batch_size=32, shuffle=False)

        # Extract features using the CNN model
        cnn_model.eval()
        with torch.no_grad():
            for batch in data_loader:
                inputs = batch[0].permute(0, 2, 1)  # Change the shape to (batch_size, channels, length)
                outputs = cnn_model(inputs)
                cnn_features.append(outputs.cpu().numpy())

    cnn_features_array = np.vstack(cnn_features)

    del raw
    del cnn_features
    del batch_data
    del batch_data_array
    gc.collect()

    # Further reduce the features using PCA
    final_pca = PCA(n_components=16)
    reduced_features = final_pca.fit_transform(cnn_features_array)

    # Quantize the features to 32-bit integers
    reduced_features_quantized = np.round(reduced_features * 1e6).astype(np.int32)

    return reduced_features_quantized
