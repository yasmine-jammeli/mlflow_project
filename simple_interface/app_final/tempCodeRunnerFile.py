import streamlit as st
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import mne
import pywt
import pickle
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from scipy.stats import skew, kurtosis
from statsmodels.robust.scale import mad
from scipy.signal import welch

# CNN model class for processing
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

cnn_model = ModifiedCNN().to("cpu")

# EEG File Processing function
def process_edf_file(edf_file, edf_dir_path):
    edf_file_path = os.path.join(edf_dir_path, edf_file)
    raw = mne.io.read_raw_edf(edf_file_path, preload=True)
    signal_length = raw.n_times
    sfreq = raw.info['sfreq']
    n_channels = len(raw.ch_names)
    signal_labels = raw.ch_names

    epoch_length_samples = int(360 * sfreq)
    num_epochs = signal_length // epoch_length_samples

    batch_size = 20
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

            signals_torch = torch.tensor(raw_epoch.get_data(), dtype=torch.float32).to("cpu")
            for i in range(n_channels):
                cwt_result, _ = pywt.cwt(signals_torch[i].cpu().numpy(), scales, 'morl')
                cwt_data[i, :, :] = cwt_result

            cwt_data = torch.tensor(cwt_data, dtype=torch.float32).to("cpu")

            if torch.any(torch.isnan(cwt_data)) or torch.any(torch.isinf(cwt_data)):
                cwt_data = torch.nan_to_num(cwt_data)

            cwt_data_reshaped = cwt_data.view(n_channels * len(scales), signals.shape[1]).T

            if torch.any(torch.isnan(cwt_data_reshaped)) or torch.any(torch.isinf(cwt_data_reshaped)):
                cwt_data_reshaped = torch.nan_to_num(cwt_data_reshaped)

            ica = mne.preprocessing.ICA(n_components=23, random_state=97, max_iter=400)
            ica.fit(raw_epoch)

            scaler = StandardScaler()
            cwt_data_normalized = scaler.fit_transform(cwt_data_reshaped.cpu().numpy())

            pca = PCA(n_components=16)
            pca_data = pca.fit_transform(cwt_data_normalized)
            batch_data.append(pca_data)

        batch_data_array = np.vstack(batch_data)
        pca_data_tensor = torch.tensor(batch_data_array, dtype=torch.float32).unsqueeze(2).to("cpu")
        dataset = TensorDataset(pca_data_tensor)
        data_loader = DataLoader(dataset, batch_size=32, shuffle=False)

        cnn_model.eval()
        with torch.no_grad():
            for batch in data_loader:
                inputs = batch[0].permute(0, 2, 1)
                outputs = cnn_model(inputs)
                cnn_features.append(outputs.cpu().numpy())

    cnn_features_array = np.vstack(cnn_features)
    final_pca = PCA(n_components=16)
    reduced_features = final_pca.fit_transform(cnn_features_array)
    reduced_features_quantized = np.round(reduced_features * 1e6).astype(np.int32)

    return reduced_features_quantized

# Feature extraction function for prediction
def extract_features(file_path):
    df = pd.read_csv(file_path)
    features = df.iloc[:, :-1]  # Drop the last column (if it contains the label)
    
    feature_stats = []
    for col in features.columns:
        col_values = features[col].values
        feature_stats.extend([
            np.mean(col_values), np.std(col_values), np.min(col_values),
            np.max(col_values), skew(col_values), kurtosis(col_values),
            np.median(col_values), mad(col_values), np.var(col_values),
            np.ptp(col_values), np.percentile(col_values, 25),
            np.percentile(col_values, 50), np.percentile(col_values, 75),
        ])
        
    return feature_stats[:39]

# Streamlit Interface
st.title('EEG File Processing and Prediction')

# Step 1: Upload the EEG (EDF) File for Processing
uploaded_file = st.file_uploader("Upload an EEG (EDF) file", type="edf")
if uploaded_file is not None:
    with st.spinner('Processing the EEG file...'):
        edf_dir = 'temp/'  # Ensure the directory exists
        os.makedirs(edf_dir, exist_ok=True)
        edf_path = os.path.join(edf_dir, uploaded_file.name)
        with open(edf_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())

        features = process_edf_file(uploaded_file.name, edf_dir)
        processed_file_path = os.path.join(edf_dir, f'cnn_features_{os.path.splitext(uploaded_file.name)[0]}.csv')
        pd.DataFrame(features).to_csv(processed_file_path, index=False, header=False)

        st.success('EEG file processed successfully!')
        st.download_button(
            label="Download Processed File",
            data=open(processed_file_path, 'rb').read(),
            file_name=f'cnn_features_{os.path.splitext(uploaded_file.name)[0]}.csv'
        )

# Step 2: Upload the Processed CSV File for Prediction
uploaded_csv = st.file_uploader("Upload the processed EEG features (CSV) file", type="csv")
if uploaded_csv is not None:
    with st.spinner('Predicting using the processed features...'):
        csv_dir = 'temp/'  # Ensure this directory exists
        os.makedirs(csv_dir, exist_ok=True)
        csv_path = os.path.join(csv_dir, uploaded_csv.name)
        with open(csv_path, 'wb') as f:
            f.write(uploaded_csv.getbuffer())

        extracted_features = extract_features(csv_path)
        extracted_features_df = pd.DataFrame([extracted_features])

        # Load pre-trained model
        model_path = 'C:/Users/pc/Desktop/Nesrin master/simple_interface/model.pkl'  # Update the path as needed
        with open(model_path, 'rb') as file:
            xgb_model = pickle.load(file)

        # Standardize features
        scaler = StandardScaler()
        X_new_scaled = scaler.fit_transform(extracted_features_df)

        # Predict
        y_pred = xgb_model.predict(X_new_scaled)

        # Display prediction and information
        if y_pred[0] == 0:
            st.error('Your file does not contain seizure data.')
        else:
            st.success(f'Predicted Label: {y_pred[0]}')
