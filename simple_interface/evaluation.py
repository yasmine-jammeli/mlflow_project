import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis
from statsmodels import robust
from scipy.signal import welch
import pywt
from sklearn.preprocessing import StandardScaler
import pickle

# Define paths
model_path = '/path/to/your/model.pkl'  # Update this path
scaler_path = '/path/to/your/scaler.pkl'  # Update this path if needed

# Function to calculate zero-crossing rate
def zero_crossing_rate(data):
    return ((data[:-1] * data[1:]) < 0).sum()

# Function to calculate entropy
def entropy(data):
    hist, bin_edges = np.histogram(data, bins=10, density=True)
    hist = hist[np.nonzero(hist)]  # Remove zero entries
    return -np.sum(hist * np.log(hist))

# Function to calculate autocorrelation
def autocorrelation(data, lag=1):
    return np.corrcoef(np.array([data[:-lag], data[lag:]]))[0, 1]

# Function to calculate Power Spectral Density (PSD)
def psd(data, fs=256):
    freqs, psd_values = welch(data, fs)
    return psd_values

# Function to calculate band power
def band_power(data, fs=256, band=(0.5, 30)):
    freqs, psd_values = welch(data, fs)
    band_freqs = np.logical_and(freqs >= band[0], freqs <= band[1])
    return np.sum(psd_values[band_freqs])

# Function to calculate wavelet coefficients
def wavelet_coeffs(data, wavelet='db4', level=4):
    coeffs = pywt.wavedec(data, wavelet, level=level)
    return coeffs

# Function to extract features from a file including frequency and time-frequency domain features
def extract_features(file_path):
    df = pd.read_csv(file_path)
    features = df.iloc[:, :-1]  # Drop the last column (if it contains the label)
    
    # Extract selected statistical features
    feature_stats = []
    for col in features.columns:
        col_values = features[col].values
        feature_stats.extend([
            np.mean(col_values),
            np.std(col_values),
            np.min(col_values),
            np.max(col_values),
            skew(col_values),
            kurtosis(col_values),
            np.median(col_values),
            robust.scale.mad(col_values),  # Median absolute deviation
            np.var(col_values),
            np.ptp(col_values),  # Peak-to-peak (range)
            np.percentile(col_values, 25),  # 25th percentile
            np.percentile(col_values, 50),  # 50th percentile
            np.percentile(col_values, 75),  # 75th percentile,
            zero_crossing_rate(col_values),  # Zero crossing rate
            entropy(col_values)  # Entropy
        ])
        
    return feature_stats[:39]

def evaluate_features(features_csv_path):
    # Extract features from the new CSV file
    new_features = extract_features(features_csv_path)

    # Load the saved model
    with open(model_path, 'rb') as file:
        model = pickle.load(file)

    # Load and apply the scaler
    with open(scaler_path, 'rb') as file:
        scaler = pickle.load(file)

    # Save the extracted features to a new CSV file
    df_features = pd.DataFrame([new_features])
    df_features_scaled = scaler.transform(df_features)

    # Predict using the loaded model
    y_pred = model.predict(df_features_scaled)
    
    return y_pred[0]
